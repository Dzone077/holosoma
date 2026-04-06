from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import replace

import holosoma.config_values.terrain as terrain_values

import tyro
from loguru import logger

import math
import yaml
import torch
import torch.nn as nn

from holosoma.utils.rotations import quat_rotate_inverse
from fast_td3 import load_policy
from dataclasses import dataclass
from typing import Optional

try:
    from holosoma.remote_control_service import RemoteControlService
    CONTROLLER_AVAILABLE = True
except ImportError:
    CONTROLLER_AVAILABLE = False
    logger.error("RemoteControlService not available. Controller input will be disabled.")

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import (
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG
from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization


@dataclass(frozen=True)
class ExternalFastTD3Cfg:
    enabled: bool = False
    checkpoint_pt: Optional[str] = None   # path to your .pt
    yaml_path: Optional[str] = None       # path to your YAML
    max_steps: int = 5000                 # policy steps (not physics steps)
    fast_eval: bool = False               # reduce rendering/debug overhead for faster wall-clock playback
    render_interval: Optional[int] = None # optional override; higher means fewer render calls


def _parse_bool(value: str) -> bool:
    value_l = value.strip().lower()
    if value_l in {"1", "true", "yes", "y", "on"}:
        return True
    if value_l in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _apply_external_fallback_overrides(eval_cfg: ExperimentConfig, args: list[str]) -> ExperimentConfig:
    """Apply a minimal subset of CLI overrides without Tyro parser generation.

    This fallback is used when Tyro cannot build parser for ExperimentConfig due to
    list[SceneFileConfig] parser constraints in some versions/setups.
    """
    cfg = eval_cfg
    i = 0
    unknown_args: list[str] = []

    while i < len(args):
        token = args[i]

        # Support --key=value form
        if token.startswith("--") and "=" in token:
            key, value = token.split("=", 1)
            i += 1
        else:
            key = token
            if i + 1 < len(args) and args[i + 1] and not args[i + 1].startswith("--"):
                value = args[i + 1]
                i += 2
            else:
                value = None
                i += 1

        if key == "--terrain" and value is not None:
            if value not in terrain_values.DEFAULTS:
                raise ValueError(
                    f"Unknown terrain preset '{value}'. Available: {sorted(terrain_values.DEFAULTS.keys())}"
                )
            cfg = replace(cfg, terrain=terrain_values.DEFAULTS[value])
        elif key == "--training.num-envs" and value is not None:
            cfg = replace(cfg, training=replace(cfg.training, num_envs=int(value)))
        elif key == "--eval-overrides.randomize-tiles" and value is not None:
            spawn = replace(cfg.terrain.terrain_term.spawn, randomize_tiles=_parse_bool(value))
            terrain_term = replace(cfg.terrain.terrain_term, spawn=spawn)
            cfg = replace(cfg, terrain=replace(cfg.terrain, terrain_term=terrain_term))
        else:
            unknown_args.append(token)

    if unknown_args:
        logger.warning(f"Ignored unsupported fallback CLI overrides: {unknown_args}")

    return cfg


def _apply_external_runtime_tuning(eval_cfg: ExperimentConfig, external_cfg: ExternalFastTD3Cfg) -> ExperimentConfig:
    """Apply runtime-only performance tuning for external eval.

    This does not change policy/control math. It only changes rendering/debug settings
    so wall-clock playback can be closer to real-time.
    """
    sim_cfg = eval_cfg.simulator.config.sim
    target_render_interval = external_cfg.render_interval

    if external_cfg.fast_eval and target_render_interval is None:
        # A good default: render once per control step.
        target_render_interval = sim_cfg.control_decimation

    if target_render_interval is None:
        return eval_cfg

    target_render_interval = max(1, int(target_render_interval))
    tuned_sim = replace(sim_cfg, render_interval=target_render_interval)
    tuned_init = replace(
        eval_cfg.simulator.config,
        sim=tuned_sim,
        debug_viz=(False if external_cfg.fast_eval else eval_cfg.simulator.config.debug_viz),
    )
    tuned_simulator = replace(eval_cfg.simulator, config=tuned_init)

    logger.info(
        "External eval runtime tuning: "
        f"render_interval={target_render_interval}, debug_viz={tuned_init.debug_viz}"
    )
    return replace(eval_cfg, simulator=tuned_simulator)


FASTTD3_POLICY_DOF_NAMES_20 = [
    "Left_Shoulder_Pitch","Left_Shoulder_Roll","Left_Elbow_Pitch","Left_Elbow_Yaw",
    "Right_Shoulder_Pitch","Right_Shoulder_Roll","Right_Elbow_Pitch","Right_Elbow_Yaw",
    "Left_Hip_Pitch","Left_Hip_Roll","Left_Hip_Yaw","Left_Knee_Pitch","Left_Ankle_Pitch","Left_Ankle_Roll",
    "Right_Hip_Pitch","Right_Hip_Roll","Right_Hip_Yaw","Right_Knee_Pitch","Right_Ankle_Pitch","Right_Ankle_Roll",
]


def apply_fasttd3_yaml_overrides_to_cfg(eval_cfg: ExperimentConfig, y: dict) -> ExperimentConfig:
    sim_dt = float(y["sim"]["dt"])  # 0.002
    fps = int(round(1.0 / sim_dt))  # 500
    control_decimation = int(y["control"]["decimation"])  # 10

    # Increase PhysX solver iterations for more stable contacts
    new_physx = replace(
        eval_cfg.simulator.config.sim.physx,
        num_position_iterations=16,  # Was 8, increase for stability
        num_velocity_iterations=8,   # Was 4, increase for stability
        bounce_threshold_velocity=0.2,  # Lower to reduce bouncy contacts
    )
    logger.info("PhysX solver: 16 position iters, 8 velocity iters, bounce_threshold=0.2")

    new_sim_engine = replace(
        eval_cfg.simulator.config.sim,
        fps=fps,
        control_decimation=control_decimation,
        physx=new_physx,
    )
    new_sim_init = replace(eval_cfg.simulator.config, sim=new_sim_engine)
    new_simulator = replace(eval_cfg.simulator, config=new_sim_init)

    # Keep terrain settings from the loaded Holosoma eval config by default.
    # Users can still override via CLI, e.g. --terrain.terrain-term.mesh-type PLANE.
    new_terrain = eval_cfg.terrain

    kp = y["control"]["stiffness"]
    kd = y["control"]["damping"]

    # Exact joint names from your error output + policy DOFs
    all_dof_names = [
        'AAHead_yaw', 'Left_Shoulder_Pitch', 'Right_Shoulder_Pitch', 'Waist', 'Head_pitch',
        'Left_Shoulder_Roll', 'Right_Shoulder_Roll', 'Left_Hip_Pitch', 'Right_Hip_Pitch',
        'Left_Elbow_Pitch', 'Right_Elbow_Pitch', 'Left_Hip_Roll', 'Right_Hip_Roll',
        'Left_Elbow_Yaw', 'Right_Elbow_Yaw', 'Left_Hip_Yaw', 'Right_Hip_Yaw',
        'Left_Wrist_Pitch', 'Right_Wrist_Pitch', 'Left_Knee_Pitch', 'Right_Knee_Pitch',
        'Left_Wrist_Yaw', 'Right_Wrist_Yaw', 'Left_Ankle_Pitch', 'Right_Ankle_Pitch',
        'Left_Hand_Roll', 'Right_Hand_Roll', 'Left_Ankle_Roll', 'Right_Ankle_Roll'
    ]

    # Your 20 policy DOFs only (no waist/head/wrist/hand)
    policy_dofs = FASTTD3_POLICY_DOF_NAMES_20

    # Build full stiffness/damping maps with exact joint-name keys (no regex)
    stiffness = {}
    damping = {}
    stiffness = {
        "Left_Shoulder_Pitch": 20.0,
        "Left_Shoulder_Roll": 20.0,
        "Left_Elbow_Pitch": 20.0,
        "Left_Elbow_Yaw": 20.0,
        "Right_Shoulder_Pitch": 20.0,
        "Right_Shoulder_Roll": 20.0,
        "Right_Elbow_Pitch": 20.0,
        "Right_Elbow_Yaw": 20.0,
        "Left_Hip_Pitch": 50.0,
        "Left_Hip_Roll": 50.0,
        "Left_Hip_Yaw": 50.0,
        "Left_Knee_Pitch": 50.0,
        "Left_Ankle_Pitch": 30.0,
        "Left_Ankle_Roll": 30.0,
        "Right_Hip_Pitch": 50.0,
        "Right_Hip_Roll": 50.0,
        "Right_Hip_Yaw": 50.0,
        "Right_Knee_Pitch": 50.0,
        "Right_Ankle_Pitch": 30.0,
        "Right_Ankle_Roll": 30.0,
        # ZERO stiffness for non-policy DOFs - they don't exist in MuJoCo model
        # With Kp=0, they won't oscillate and will just be passive
        "AAHead_yaw": 0.0,
        "Waist": 0.0,
        "Head_pitch": 0.0,
        "Left_Wrist_Pitch": 0.0,
        "Right_Wrist_Pitch": 0.0,
        "Left_Wrist_Yaw": 0.0,
        "Right_Wrist_Yaw": 0.0,
        "Left_Hand_Roll": 0.0,
        "Right_Hand_Roll": 0.0,
    }

    damping = {
        # Arm damping - matching MuJoCo exactly
        "Left_Shoulder_Pitch": 2.0,
        "Left_Shoulder_Roll": 2.0,
        "Left_Elbow_Pitch": 2.0,
        "Left_Elbow_Yaw": 2.0,
        "Right_Shoulder_Pitch": 2.0,
        "Right_Shoulder_Roll": 2.0,
        "Right_Elbow_Pitch": 2.0,
        "Right_Elbow_Yaw": 2.0,
        "Left_Hip_Pitch": 3.0,
        "Left_Hip_Roll": 3.0,
        "Left_Hip_Yaw": 3.0,
        "Left_Knee_Pitch": 3.0,
        "Left_Ankle_Pitch": 1.0,
        "Left_Ankle_Roll": 1.0,
        "Right_Hip_Pitch": 3.0,
        "Right_Hip_Roll": 3.0,
        "Right_Hip_Yaw": 3.0,
        "Right_Knee_Pitch": 3.0,
        "Right_Ankle_Pitch": 1.0,
        "Right_Ankle_Roll": 1.0,
        # Zero for non-policy DOFs
        "AAHead_yaw": 0.0,
        "Waist": 0.0,
        "Head_pitch": 0.0,
        "Left_Wrist_Pitch": 0.0,
        "Right_Wrist_Pitch": 0.0,
        "Left_Wrist_Yaw": 0.0,
        "Right_Wrist_Yaw": 0.0,
        "Left_Hand_Roll": 0.0,
        "Right_Hand_Roll": 0.0,
    }

    logger.info("Built stiffness/damping maps with exact joint names (no regex)")
    logger.info(f"Policy DOFs stiffness: {dict(sorted({k: v for k, v in stiffness.items() if v > 0}.items()))}")

    new_control = replace(
        eval_cfg.robot.control,
        action_scale=float(y["control"]["action_scale"]),
        stiffness=stiffness,  # Exact joint-name dict, no regex keys
        damping=damping,      # Exact joint-name dict, no regex keys
        action_scales_by_effort_limit_over_p_gain=(
            eval_cfg.robot.control.action_scales_by_effort_limit_over_p_gain
            if hasattr(eval_cfg.robot.control, "action_scales_by_effort_limit_over_p_gain")
            else False
        ),
    )
    # Grab the pos and rot from the YAML
    yaml_pos = tuple(y["init_state"]["pos"]) 
    yaml_rot = tuple(y["init_state"]["rot"])

    new_init_state = replace(
        eval_cfg.robot.init_state,
        pos=yaml_pos,
        rot=yaml_rot,
        default_joint_angles=dict(y["init_state"]["default_joint_angles"]),
    )

    # CRITICAL: Override dof_effort_limit_list to match MuJoCo actuatorfrcrange
    # These are used when creating the articulation, so must be set in config!
    # MuJoCo values from XML: Ankle_Pitch=24, Ankle_Roll=15, Knee=65, Hip_Roll=45
    mujoco_effort_limits = list(eval_cfg.robot.dof_effort_limit_list)  # Make a copy

    # Map joint names to their MuJoCo effort limits
    mujoco_effort_map = {
        "Left_Hip_Roll": 45.0, "Right_Hip_Roll": 45.0,  # Was 30!
        "Left_Knee_Pitch": 65.0, "Right_Knee_Pitch": 65.0,  # Was 60!
        "Left_Ankle_Pitch": 24.0, "Right_Ankle_Pitch": 24.0,  # Was 12!
        "Left_Ankle_Roll": 15.0, "Right_Ankle_Roll": 15.0,  # Was 12!
    }

    for i, dof_name in enumerate(eval_cfg.robot.dof_names):
        if dof_name in mujoco_effort_map:
            old_val = mujoco_effort_limits[i]
            new_val = mujoco_effort_map[dof_name]
            mujoco_effort_limits[i] = new_val
            logger.info(f"Effort limit override: {dof_name}: {old_val} -> {new_val}")

    # CRITICAL: Override armature and friction ONLY for policy-controlled DOFs
    # Non-policy DOFs (head, waist, wrists) need some armature/friction to stay stable
    # since they have Kp=0, Kd=0 and would otherwise flop around
    num_dofs = len(eval_cfg.robot.dof_names)

    # Start with original values
    mujoco_armature = list(eval_cfg.robot.dof_armature_list)
    mujoco_joint_friction = list(eval_cfg.robot.dof_joint_friction_list)

    # Convert policy DOF names to lowercase for matching
    policy_dofs_lower = {name.lower() for name in FASTTD3_POLICY_DOF_NAMES_20}

    for i, dof_name in enumerate(eval_cfg.robot.dof_names):
        # Check if this DOF is policy-controlled (compare lowercase)
        dof_core = dof_name.replace("_joint", "").lower()

        if dof_core in policy_dofs_lower:
            mujoco_armature[i] = 0.0
            mujoco_joint_friction[i] = 0.0
            logger.info(f"Policy DOF {dof_name}: armature=0.0, friction=0.0")
        else:
            # Keep original values for non-policy DOFs (head, waist, wrists)
            logger.info(f"Non-policy DOF {dof_name}: keeping armature={mujoco_armature[i]:.4f}, friction={mujoco_joint_friction[i]:.4f}")

    new_robot = replace(
        eval_cfg.robot,
        control=new_control,
        init_state=new_init_state,
        dof_effort_limit_list=mujoco_effort_limits,
        dof_armature_list=mujoco_armature,
        dof_joint_friction_list=mujoco_joint_friction,
    )
    logger.info("Effort limits updated to match MuJoCo actuatorfrcrange")
    logger.info("Armature/friction set to 0.0 for policy DOFs, kept original for non-policy DOFs (waist, wrists)")

    # Create a DISABLED randomization config - sets up required attributes but no actual randomization
    disabled_randomization = RandomizationManagerCfg(
        setup_terms={
            # These setup the required attributes but with randomization disabled
            "push_randomizer_state": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
                params={"push_interval_s": [999999, 999999], "max_push_vel": [0.0, 0.0], "enabled": False},
            ),
            "setup_action_delay_buffers": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:setup_action_delay_buffers",
                params={"ctrl_delay_step_range": [0, 0], "enabled": False},
            ),
            "setup_torque_rfi": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:setup_torque_rfi",
                params={"enabled": False, "rfi_lim": 0.0},
            ),
            "setup_dof_pos_bias": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:setup_dof_pos_bias",
                params={"dof_pos_bias_range": [0.0, 0.0], "enabled": False},
            ),
            "actuator_randomizer_state": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
                params={"kp_range": [1.0, 1.0], "kd_range": [1.0, 1.0], "rfi_lim_range": [1.0, 1.0],
                        "enable_pd_gain": False, "enable_rfi_lim": False},
            ),
            "mass_randomizer": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:randomize_mass_startup",
                params={"enable_link_mass": False, "link_mass_range": [1.0, 1.0],
                        "enable_base_mass": False, "added_mass_range": [0.0, 0.0]},
            ),
            "randomize_friction_startup": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:randomize_friction_startup",
                params={"friction_range": [1.0, 1.0], "enabled": False},
            ),
            "randomize_base_com_startup": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:randomize_base_com_startup",
                params={"base_com_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}, "enabled": False},
            ),
        },
        reset_terms={
            "push_randomizer_state": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"),
            "actuator_randomizer_state": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState"),
            "randomize_push_schedule": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:randomize_push_schedule"),
            "randomize_action_delay": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:randomize_action_delay"),
            "randomize_dof_state": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:randomize_dof_state",
                params={"joint_pos_scale_range": [1.0, 1.0], "joint_pos_bias_range": [0.0, 0.0],
                        "joint_vel_range": [0.0, 0.0], "randomize_dof_pos_bias": False},
            ),
            "configure_torque_rfi": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:configure_torque_rfi"),
        },
        step_terms={
            "push_randomizer_state": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"),
            "apply_pushes": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.locomotion:apply_pushes"),
        },
    )
    logger.info("DISABLING all domain randomization for external policy testing")
    return replace(eval_cfg, simulator=new_simulator, robot=new_robot, terrain=new_terrain, randomization=disabled_randomization)

def build_default20_from_yaml(y: dict, device: torch.device) -> torch.Tensor:
    m = y["init_state"]["default_joint_angles"]
    default_scalar = float(m.get("default", 0.0))
    vals = []
    for name in FASTTD3_POLICY_DOF_NAMES_20:
        vals.append(float(m.get(name, default_scalar)))
    return torch.tensor(vals, device=device).view(1, 20)  # (1,20)

def _indices_from_dof_names(all_dof_names: list[str], wanted: list[str], device: str) -> torch.Tensor:
    idxs = []
    for n in wanted:
        if n not in all_dof_names:
            raise KeyError(f"DOF '{n}' not found in env.dof_names")
        idxs.append(all_dof_names.index(n))
    return torch.tensor(idxs, device=device, dtype=torch.long)

class FastTD3Actor(nn.Module):
    def __init__(self, actor_state_dict: dict[str, torch.Tensor]):
        super().__init__()
        net_linear_indices = sorted(
            {int(k.split(".")[1]) for k in actor_state_dict if k.startswith("net.") and k.endswith(".weight")}
        )
        trunk: list[nn.Module] = []
        for j, idx in enumerate(net_linear_indices):
            w = actor_state_dict[f"net.{idx}.weight"]
            trunk.append(nn.Linear(w.shape[1], w.shape[0]))
            if j != len(net_linear_indices) - 1:
                trunk.append(nn.ReLU())
        self.net = nn.Sequential(*trunk)

        w_mu = actor_state_dict["fc_mu.0.weight"]
        self.fc_mu = nn.Sequential(nn.Linear(w_mu.shape[1], w_mu.shape[0]))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fc_mu(self.net(obs))


class _DistLike:
    def __init__(self, loc: torch.Tensor):
        self.loc = loc


class _Policy:
    def __init__(self, actor: nn.Module, obs_normalizer: nn.Module):
        self.actor = actor
        self.obs_normalizer = obs_normalizer

    def to(self, device: str):
        self.actor = self.actor.to(device)
        self.obs_normalizer = self.obs_normalizer.to(device)
        return self

    def eval(self):
        self.actor = self.actor.eval()
        self.obs_normalizer = self.obs_normalizer.eval()
        return self

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        norm_obs = self.obs_normalizer(obs, update=False) if not isinstance(self.obs_normalizer, nn.Identity) else obs
        return _DistLike(self.actor(norm_obs))


def load_policy2(checkpoint_path: str, device: str):
    """Local FastTD3 loader to match MuJoCo path without importing fast_td3."""
    try:
        torch_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        torch_checkpoint = torch.load(checkpoint_path, map_location="cpu")

    args = torch_checkpoint.get("args", {})
    agent = args.get("agent", "fasttd3")

    if agent == "fasttd3":
        n_obs = torch_checkpoint["actor_state_dict"]["net.0.weight"].shape[-1]
        n_act = torch_checkpoint["actor_state_dict"]["fc_mu.0.weight"].shape[0]
    elif agent == "fasttd3_simbav2":
        n_obs = torch_checkpoint["actor_state_dict"]["embedder.w.w.weight"].shape[-1] - 1
        n_act = torch_checkpoint["actor_state_dict"]["predictor.mean_bias"].shape[0]
        raise ValueError(
            f"Agent {agent} is not supported by local eval loader yet. "
            "Please use a fasttd3 checkpoint."
        )
    else:
        raise ValueError(f"Agent {agent} not supported")

    actor_sd = torch_checkpoint["actor_state_dict"]
    actor_sd = {k: v for k, v in actor_sd.items() if k not in {"noise_scales", "std_min", "std_max"}}

    actor = FastTD3Actor(actor_sd)
    actor.load_state_dict(actor_sd, strict=True)

    obs_norm_state = torch_checkpoint.get("obs_normalizer_state", None)
    if not obs_norm_state:
        obs_normalizer: nn.Module = nn.Identity()
    else:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device="cpu")
        obs_normalizer.load_state_dict(obs_norm_state)

    policy = _Policy(actor=actor, obs_normalizer=obs_normalizer).to(device).eval()
    try:
        policy = torch.compile(policy)
    except Exception:
        pass
    return policy

@torch.no_grad()
def evaluate_external_fasttd3(env, device: str, cfg: ExternalFastTD3Cfg):
    assert cfg.checkpoint_pt and cfg.yaml_path

    # Load YAML
    with open(cfg.yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    assert y["env"]["num_observations"] == 71
    assert y["env"]["num_actions"] == 20

    decimation = int(y["control"]["decimation"])
    clip_actions = float(y["normalization"]["clip_actions"])

    norm_gravity = float(y["normalization"]["gravity"])
    norm_lin_vel = float(y["normalization"]["lin_vel"])
    norm_ang_vel = float(y["normalization"]["ang_vel"])
    norm_dof_pos = float(y["normalization"]["dof_pos"])
    norm_dof_vel = float(y["normalization"]["dof_vel"])

    # gait
    sim_dt = float(y["sim"]["dt"])
    gf = 0.5 * (float(y["commands"]["gait_frequency"][0]) + float(y["commands"]["gait_frequency"][1]))
    gait_process = torch.zeros((env.num_envs,), device=device)

    # FastTD3 deploy loader initializes modules/buffers on CPU by default.
    # Move policy + obs normalizer to the same device as env observations.
    policy = load_policy(cfg.checkpoint_pt)
    policy = policy.to(device).eval()
    try:
        policy = torch.compile(policy)
    except Exception:
        logger.warning("torch.compile(policy) failed; continuing without compilation.")

    # Initialize controller input if available
    rc_service = None
    if CONTROLLER_AVAILABLE:
        try:
            rc_service = RemoteControlService()
            logger.info("✓ Controller initialized via RemoteControlService")
        except Exception as e:
            logger.warning(f"Failed to initialize controller: {e}. Running without controller input.")

    # Map joint indices
    idx20 = _indices_from_dof_names(list(env.dof_names), FASTTD3_POLICY_DOF_NAMES_20, device=device)

    # right after idx20 is computed
    for i, name in enumerate(FASTTD3_POLICY_DOF_NAMES_20):
        print(f"policy[{i:02d}] {name:>22s} -> holosoma_dof_index={idx20[i].item()} env.dof_names[{idx20[i].item()}]={env.dof_names[idx20[i].item()]}")

    # CRITICAL: Verify left/right leg joints are not swapped
    left_hip_pitch_policy_idx = FASTTD3_POLICY_DOF_NAMES_20.index("Left_Hip_Pitch")
    right_hip_pitch_policy_idx = FASTTD3_POLICY_DOF_NAMES_20.index("Right_Hip_Pitch")
    left_hip_isaac_idx = idx20[left_hip_pitch_policy_idx].item()
    right_hip_isaac_idx = idx20[right_hip_pitch_policy_idx].item()

    if "Left" not in env.dof_names[left_hip_isaac_idx]:
        logger.error(f"CRITICAL BUG: Policy's Left_Hip_Pitch maps to Isaac Lab joint '{env.dof_names[left_hip_isaac_idx]}' which is NOT a left joint!")
    if "Right" not in env.dof_names[right_hip_isaac_idx]:
        logger.error(f"CRITICAL BUG: Policy's Right_Hip_Pitch maps to Isaac Lab joint '{env.dof_names[right_hip_isaac_idx]}' which is NOT a right joint!")

    logger.info("✓ Joint mapping verified: Left/Right legs are correctly mapped")

    default20 = build_default20_from_yaml(y, device)          # (1,20)
    default20 = default20.repeat(env.num_envs, 1)             # (N,20)

    obs_dict = env.reset_all()

    # --- THE PD GAIN OVERRIDE HACK (Holosoma Native) ---
    if hasattr(env, "p_gains") and hasattr(env, "d_gains"):
        
        # 1. Lock the Non-Policy Joints (Waist, Head, Wrists)
        idx_non_policy = [i for i in range(env.num_dof) if i not in idx20.tolist()]
        if env.p_gains.ndim == 1:
            env.p_gains[idx_non_policy] = 200.0
            env.d_gains[idx_non_policy] = 10.0
        else:
            env.p_gains[:, idx_non_policy] = 200.0
            env.d_gains[:, idx_non_policy] = 10.0
            
        # 2. Inject Exact Policy Gains from MuJoCo
        policy_stiffness = {
            "Left_Shoulder_Pitch": 20.0, "Left_Shoulder_Roll": 20.0, "Left_Elbow_Pitch": 20.0, "Left_Elbow_Yaw": 20.0,
            "Right_Shoulder_Pitch": 20.0, "Right_Shoulder_Roll": 20.0, "Right_Elbow_Pitch": 20.0, "Right_Elbow_Yaw": 20.0,
            "Left_Hip_Pitch": 50.0, "Left_Hip_Roll": 50.0, "Left_Hip_Yaw": 50.0, "Left_Knee_Pitch": 50.0, "Left_Ankle_Pitch": 30.0, "Left_Ankle_Roll": 30.0,
            "Right_Hip_Pitch": 50.0, "Right_Hip_Roll": 50.0, "Right_Hip_Yaw": 50.0, "Right_Knee_Pitch": 50.0, "Right_Ankle_Pitch": 30.0, "Right_Ankle_Roll": 30.0,
        }
        policy_damping = {
            "Left_Shoulder_Pitch": 2.0, "Left_Shoulder_Roll": 2.0, "Left_Elbow_Pitch": 2.0, "Left_Elbow_Yaw": 2.0,
            "Right_Shoulder_Pitch": 2.0, "Right_Shoulder_Roll": 2.0, "Right_Elbow_Pitch": 2.0, "Right_Elbow_Yaw": 2.0,
            "Left_Hip_Pitch": 3.0, "Left_Hip_Roll": 3.0, "Left_Hip_Yaw": 3.0, "Left_Knee_Pitch": 3.0, "Left_Ankle_Pitch": 1.0, "Left_Ankle_Roll": 1.0,
            "Right_Hip_Pitch": 3.0, "Right_Hip_Roll": 3.0, "Right_Hip_Yaw": 3.0, "Right_Knee_Pitch": 3.0, "Right_Ankle_Pitch": 1.0, "Right_Ankle_Roll": 1.0,
        }
        
        for i, name in enumerate(FASTTD3_POLICY_DOF_NAMES_20):
            holosoma_idx = idx20[i]
            if env.p_gains.ndim == 1:
                env.p_gains[holosoma_idx] = policy_stiffness[name]
                env.d_gains[holosoma_idx] = policy_damping[name]
            else:
                env.p_gains[:, holosoma_idx] = policy_stiffness[name]
                env.d_gains[:, holosoma_idx] = policy_damping[name]
                
        logger.info("Successfully forced explicit PD gains for ALL joints directly onto the environment!")

        # CRITICAL: Override torque limits to match MuJoCo MJCF
        logger.info("Overriding torque limits to match MuJoCo...")
        mujoco_torque_limits = {
            "Left_Shoulder_Pitch": 18.0, "Left_Shoulder_Roll": 18.0, "Left_Elbow_Pitch": 18.0, "Left_Elbow_Yaw": 18.0,
            "Right_Shoulder_Pitch": 18.0, "Right_Shoulder_Roll": 18.0, "Right_Elbow_Pitch": 18.0, "Right_Elbow_Yaw": 18.0,
            "Left_Hip_Pitch": 45.0, "Left_Hip_Roll": 45.0, "Left_Hip_Yaw": 30.0, "Left_Knee_Pitch": 65.0,
            "Left_Ankle_Pitch": 24.0, "Left_Ankle_Roll": 15.0,
            "Right_Hip_Pitch": 45.0, "Right_Hip_Roll": 45.0, "Right_Hip_Yaw": 30.0, "Right_Knee_Pitch": 65.0,
            "Right_Ankle_Pitch": 24.0, "Right_Ankle_Roll": 15.0,
        }
        if hasattr(env, "torque_limits"):
            for i, name in enumerate(FASTTD3_POLICY_DOF_NAMES_20):
                holosoma_idx = idx20[i]
                if name in mujoco_torque_limits:
                    env.torque_limits[holosoma_idx] = mujoco_torque_limits[name]
            logger.info("Torque limits overridden to match MuJoCo!")
        else:
            logger.warning("Could not find torque_limits on environment - limits may not match MuJoCo!")

        # Skip warmup - PD gains should be active immediately in IsaacLab
        logger.info("PD gains set, skipping warmup")
    else:
        logger.error("Could not find p_gains or d_gains on the environment!")
    # ---------------------------------------------------

    # CRITICAL DIAGNOSTIC: Verify env.default_dof_pos matches YAML default20
    if hasattr(env, "default_dof_pos"):
        env_defaults_20 = env.default_dof_pos[0, idx20] if env.default_dof_pos.ndim > 1 else env.default_dof_pos[idx20]
        yaml_defaults_20 = default20[0]
        max_diff = (env_defaults_20 - yaml_defaults_20).abs().max().item()
        logger.info(f"=== DEFAULT DOF POSITION CHECK ===")
        logger.info(f"Max diff between env.default_dof_pos and YAML defaults: {max_diff:.6f}")
        if max_diff > 0.01:
            logger.error("CRITICAL: env.default_dof_pos does NOT match YAML defaults! This will cause PD target mismatch!")
            for i, name in enumerate(FASTTD3_POLICY_DOF_NAMES_20):
                env_val = env_defaults_20[i].item()
                yaml_val = yaml_defaults_20[i].item()
                if abs(env_val - yaml_val) > 0.001:
                    logger.error(f"  {name}: env={env_val:.4f}, yaml={yaml_val:.4f}, diff={env_val-yaml_val:.4f}")
        else:
            logger.info("OK: env.default_dof_pos matches YAML defaults")
        logger.info(f"==================================")

    # Verify action scales are 1.0 (uniform scaling like MuJoCo)
    if hasattr(env, "action_scales"):
        action_scales_20 = env.action_scales[idx20] if env.action_scales.ndim == 1 else env.action_scales[0, idx20]
        logger.info(f"=== ACTION SCALES CHECK ===")
        logger.info(f"action_scales[idx20] min={action_scales_20.min().item():.4f}, max={action_scales_20.max().item():.4f}")
        if not torch.allclose(action_scales_20, torch.ones_like(action_scales_20)):
            logger.warning("action_scales are NOT uniform 1.0 - this may differ from MuJoCo!")
        logger.info(f"===========================")

    # --- No settle phase - policy controls from start ---
    actions_prev20 = torch.zeros((env.num_envs, 20), device=device)

    logger.info("Using DIRECT PD TORQUE CONTROL (bypassing holosoma action pipeline)")
    logger.info("This matches MuJoCo validation exactly: torque = Kp*(target-pos) - Kd*vel")

    print(type(env))
    print(type(env.simulator))
    print([a for a in dir(env.simulator) if "target" in a.lower() or "command" in a.lower()])

    # === CRITICAL: Verify PD control parameters match MuJoCo validation ===
    logger.info("=== PD CONTROL PARAMETER VERIFICATION ===")

    # Check control type and action_scales_by_effort_limit_over_p_gain
    control_cfg = env.robot_config.control
    logger.info(f"control_type: {control_cfg.control_type}")
    logger.info(f"action_scale (from config): {control_cfg.action_scale}")
    if hasattr(control_cfg, 'action_scales_by_effort_limit_over_p_gain'):
        logger.info(f"action_scales_by_effort_limit_over_p_gain: {control_cfg.action_scales_by_effort_limit_over_p_gain}")
        if control_cfg.action_scales_by_effort_limit_over_p_gain:
            logger.error("action_scales_by_effort_limit_over_p_gain is TRUE! This will cause non-uniform action scaling!")

    # Check action_scales
    logger.info(f"action_scales[idx20]: min={env.action_scales[idx20].min().item():.4f}, max={env.action_scales[idx20].max().item():.4f}")
    if not torch.allclose(env.action_scales[idx20], torch.ones(20, device=device)):
        logger.error("ACTION_SCALES ARE NOT ALL 1.0! This will break policy transfer!")
        for i, name in enumerate(FASTTD3_POLICY_DOF_NAMES_20):
            logger.error(f"  {name}: action_scale={env.action_scales[idx20[i]].item():.4f}")

    # Check _kp_scale and _kd_scale from action manager
    if hasattr(env, "action_manager"):
        active_terms = env.action_manager.active_terms
        # Handle both list and dict
        term_items = active_terms.items() if isinstance(active_terms, dict) else enumerate(active_terms)
        for term_key, term in term_items:
            term_name = term_key if isinstance(term_key, str) else term.__class__.__name__
            if hasattr(term, "_kp_scale") and hasattr(term, "_kd_scale"):
                kp_scale = term._kp_scale
                kd_scale = term._kd_scale
                # Handle both 1D and 2D tensors
                if kp_scale.ndim == 2:
                    kp_vals = kp_scale[0, idx20]
                    kd_vals = kd_scale[0, idx20]
                else:
                    kp_vals = kp_scale[idx20]
                    kd_vals = kd_scale[idx20]
                logger.info(f"[{term_name}] _kp_scale[idx20]: min={kp_vals.min().item():.4f}, max={kp_vals.max().item():.4f}")
                logger.info(f"[{term_name}] _kd_scale[idx20]: min={kd_vals.min().item():.4f}, max={kd_vals.max().item():.4f}")
                if not torch.allclose(kp_vals, torch.ones(20, device=device)):
                    logger.error("_kp_scale IS NOT 1.0! Actuator randomization may still be active!")
                if not torch.allclose(kd_vals, torch.ones(20, device=device)):
                    logger.error("_kd_scale IS NOT 1.0! Actuator randomization may still be active!")

    # Verify torque_limits match MuJoCo ctrlrange
    logger.info("Torque limits vs MuJoCo ctrlrange:")
    mujoco_ctrlrange = {
        "Left_Shoulder_Pitch": 18, "Left_Shoulder_Roll": 18, "Left_Elbow_Pitch": 18, "Left_Elbow_Yaw": 18,
        "Right_Shoulder_Pitch": 18, "Right_Shoulder_Roll": 18, "Right_Elbow_Pitch": 18, "Right_Elbow_Yaw": 18,
        "Left_Hip_Pitch": 45, "Left_Hip_Roll": 45, "Left_Hip_Yaw": 30, "Left_Knee_Pitch": 65,
        "Left_Ankle_Pitch": 24, "Left_Ankle_Roll": 15,
        "Right_Hip_Pitch": 45, "Right_Hip_Roll": 45, "Right_Hip_Yaw": 30, "Right_Knee_Pitch": 65,
        "Right_Ankle_Pitch": 24, "Right_Ankle_Roll": 15,
    }
    for i, name in enumerate(FASTTD3_POLICY_DOF_NAMES_20):
        holosoma_limit = env.torque_limits[idx20[i]].item()
        mujoco_limit = mujoco_ctrlrange.get(name, "?")
        match = "✓" if abs(holosoma_limit - mujoco_limit) < 0.1 else "✗"
        logger.info(f"  {name}: holosoma={holosoma_limit:.1f}, mujoco={mujoco_limit}, {match}")

    # Verify p_gains and d_gains match YAML
    logger.info("PD gains verification:")
    expected_kp = {"Shoulder": 20, "Elbow": 20, "Hip": 50, "Knee": 50, "Ankle": 30}
    expected_kd = {"Shoulder": 2, "Elbow": 2, "Hip": 3, "Knee": 3, "Ankle": 1}
    for i, name in enumerate(FASTTD3_POLICY_DOF_NAMES_20):
        kp = env.p_gains[idx20[i]].item()
        kd = env.d_gains[idx20[i]].item()
        # Find expected values
        exp_kp = exp_kd = None
        for key in expected_kp:
            if key in name:
                exp_kp = expected_kp[key]
                exp_kd = expected_kd[key]
                break
        match_kp = "✓" if exp_kp and abs(kp - exp_kp) < 0.1 else "✗"
        match_kd = "✓" if exp_kd and abs(kd - exp_kd) < 0.1 else "✗"
        logger.info(f"  {name}: Kp={kp:.1f} (exp={exp_kp}) {match_kp}, Kd={kd:.1f} (exp={exp_kd}) {match_kd}")

    logger.info("==========================================")
    # === END PD CONTROL VERIFICATION ===

    # === ZERO INITIAL VELOCITY AND FIX ORIENTATION TO MATCH MUJOCO RESET ===
    logger.info("Zeroing initial velocity and fixing orientation to match MuJoCo reset state...")
    env_ids = torch.arange(env.num_envs, device=device)

    # Set robot to perfectly upright orientation (identity quaternion)
    # Isaac Lab uses wxyz format: [1, 0, 0, 0] = perfectly upright
    root_state = env.simulator._robot.data.root_state_w.clone()
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # wxyz quaternion
    root_state[:, 7:13] = 0.0  # Zero all velocities (linear and angular)
    env.simulator._robot.write_root_state_to_sim(root_state, env_ids)

    # Zero joint velocities
    zero_dof_vel = torch.zeros((env.num_envs, env.num_dof), device=device)
    env.simulator._robot.write_joint_state_to_sim(
        env.simulator.dof_pos,  # Keep current positions
        zero_dof_vel,           # Zero velocities
        None,                   # No specific joint IDs
        env_ids
    )

    # Write state to sim WITHOUT stepping physics (stepping adds velocity from gravity)
    env.simulator.scene.write_data_to_sim()
    # Just update the scene data cache, don't advance physics
    env.simulator.scene.update(dt=0.0)

    # Verify velocity is now zero
    robot_data = env.simulator._robot.data
    root_ang_vel = robot_data.root_ang_vel_w[0].cpu().numpy()
    root_lin_vel = robot_data.root_lin_vel_w[0].cpu().numpy()
    dof_vel = env.simulator.dof_vel[0].cpu().numpy()
    logger.info(f"After initial zeroing - Root lin vel: {root_lin_vel}, ang vel: {root_ang_vel}")
    logger.info(f"After initial zeroing - Joint vels (first 5): {dof_vel[:5]}")
    logger.info(f"After initial zeroing - Joint vels max: {abs(dof_vel).max():.6f}")

    # Check if orientation is truly identity
    root_quat_wxyz = robot_data.root_quat_w[0].cpu().numpy()
    logger.info(f"After initial zeroing - Root quat wxyz: {root_quat_wxyz}")
    # === END VELOCITY ZEROING ===

    for step in range(cfg.max_steps):

        # --- state ---
        q_full = env.simulator.dof_pos
        dq_full = env.simulator.dof_vel
        q20 = q_full[:, idx20]
        dq20 = dq_full[:, idx20]

        if step == 0:
            logger.info(f"max |q20-default20| = {(q20-default20).abs().max().item():.4f}")

        # Read fresh state directly from robot data (not cached env properties)
        # After scene.update(), robot.data has the latest state
        robot_data = env.simulator._robot.data

        # Root quaternion: Isaac Lab uses wxyz, MuJoCo/our obs uses xyzw
        root_quat_wxyz = robot_data.root_quat_w  # (num_envs, 4) in wxyz
        base_quat_xyzw = torch.cat([root_quat_wxyz[:, 1:4], root_quat_wxyz[:, 0:1]], dim=1)  # Convert to xyzw

        # Angular velocity in world frame
        ang_vel_world = robot_data.root_ang_vel_w  # (num_envs, 3)
        base_ang_vel = quat_rotate_inverse(base_quat_xyzw, ang_vel_world, w_last=True)

        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=device).view(1, 3).repeat(env.num_envs, 1)
        projected_gravity = quat_rotate_inverse(base_quat_xyzw, gravity_world, w_last=True)

        # --- command source ---
        # Prefer manager commands because that is the standard command path used by the env.
        # --- command source ---
        # Prefer manager commands because that is the standard command path used by the env.
        if hasattr(env, "command_manager") and hasattr(env.command_manager, "commands"):
            vx = env.command_manager.commands[:, 0]
            vy = env.command_manager.commands[:, 1]
            vyaw = (
                env.command_manager.commands[:, 2]
                if env.command_manager.commands.shape[1] > 2
                else torch.zeros((env.num_envs,), device=device)
            )
        elif hasattr(env.simulator, "commands"):
            vx = env.simulator.commands[:, 0]
            vy = env.simulator.commands[:, 1]
            vyaw = env.simulator.commands[:, 2] if env.simulator.commands.shape[1] > 2 else torch.zeros_like(vx)
        else:
            vx = torch.zeros((env.num_envs,), device=device)
            vy = torch.zeros((env.num_envs,), device=device)
            vyaw = torch.zeros((env.num_envs,), device=device)

        # Override with controller input if available
        if rc_service is not None:
            try:
                controller_vx = float(rc_service.get_vx_cmd())
                controller_vy = float(rc_service.get_vy_cmd())
                controller_vyaw = float(rc_service.get_vyaw_cmd())

                # Controller is authoritative in external mode: always override,
                # including zero commands after reset, to avoid random command-manager values.
                vx = torch.full((env.num_envs,), controller_vx, device=device)
                vy = torch.full((env.num_envs,), controller_vy, device=device)
                vyaw = torch.full((env.num_envs,), controller_vyaw, device=device)
                if step == 0 or step % 100 == 0:
                    logger.info(
                        f"[Step {step}] Controller commands: vx={controller_vx:.3f}, "
                        f"vy={controller_vy:.3f}, vyaw={controller_vyaw:.3f}"
                    )
            except Exception as e:
                logger.warning(f"Error reading controller input at step {step}: {e}")

        cmd = torch.stack([vx, vy, vyaw], dim=1)

        moving = (cmd.abs().sum(dim=1) > 0).float()
        gait_freq = moving * 1.5  # or whatever you used in MuJoCo training
        #gait_freq = 0

        # Use CURRENT gait_process for observations (MuJoCo line 115-116)
        # Update happens AFTER physics steps (MuJoCo line 133)

        # Zero out gait signal when standing still (like MuJoCo does)
        gait_active = (gait_freq > 1.0e-8).float().unsqueeze(1)
        cos_g = torch.cos(2.0 * math.pi * gait_process) * gait_active.squeeze()
        sin_g = torch.sin(2.0 * math.pi * gait_process) * gait_active.squeeze()

        # --- build obs[71] in your FastTD3 layout ---
        # NOTE: default20 should come from your YAML default_joint_angles. Here we use env.default_dof_pos slice,
        # which is Holosoma’s defaults. If you need *exactly* your YAML defaults, load and use them instead.

        obs = torch.zeros((env.num_envs, 71), device=device)
        obs[:, 0:3] = projected_gravity * norm_gravity
        obs[:, 3:6] = base_ang_vel * norm_ang_vel
        obs[:, 6] = vx * norm_lin_vel
        obs[:, 7] = vy * norm_lin_vel
        obs[:, 8] = vyaw * norm_ang_vel
        obs[:, 9] = cos_g   # COS first (like MuJoCo)
        obs[:, 10] = sin_g  # SIN second (like MuJoCo)
        obs[:, 11:31] = (q20 - default20) * norm_dof_pos
        obs[:, 31:51] = dq20 * norm_dof_vel
        obs[:, 51:71] = actions_prev20
        # --- actor ---
        dist = policy.act(obs)
        act20 = torch.clamp(dist.loc.detach(), -clip_actions, clip_actions)
        # before embedding into act29
        #print(env.robot_config.control.action_scale)
        # --- embed 20->29 and step env ---
        action_scale = float(y["control"]["action_scale"])  # usually 1.0

        # network output is typically in [-clip_actions, clip_actions] already
        delta20 = torch.clamp(act20, -clip_actions, clip_actions)

        if step == 0:
            logger.info("=== STEP 0 OBSERVATION CHECK ===")
            logger.info(f"[Raw Quat xyzw] {base_quat_xyzw[0].cpu().numpy()}")
            logger.info(f"[Projected Gravity] {projected_gravity[0].cpu().numpy()} (EXPECTED: [0, 0, -1] if flat)")
            logger.info(f"[Base Ang Vel] {base_ang_vel[0].cpu().numpy()} (EXPECTED: near 0)")
            logger.info(f"[Root Ang Vel World] {ang_vel_world[0].cpu().numpy()} (EXPECTED: near 0)")

            # Print full 71-dim observation vector for comparison with MuJoCo
            obs_np = obs[0].cpu().numpy()
            logger.info("=== FULL 71-DIM OBSERVATION VECTOR (ISAAC LAB) ===")
            logger.info(f"obs[0:3]   proj_gravity: {obs_np[0:3]}")
            logger.info(f"obs[3:6]   base_ang_vel: {obs_np[3:6]}")
            logger.info(f"obs[6:9]   commands:     {obs_np[6:9]}")
            logger.info(f"obs[9:11]  gait:         {obs_np[9:11]}")
            logger.info(f"obs[11:31] dof_pos_delta: {obs_np[11:31]}")
            logger.info(f"obs[31:51] dof_vel:       {obs_np[31:51]}")
            logger.info(f"obs[51:71] prev_actions:  {obs_np[51:71]}")
            logger.info("--- RAW NUMPY ARRAY (copy for comparison) ---")
            import numpy as np
            np.set_printoptions(precision=8, suppress=True, linewidth=200)
            logger.info(f"ISAAC_OBS = np.array({obs_np.tolist()})")
            logger.info("=================================================")

            # Let's check the first 5 joints against their defaults
            for i in range(5):
                name = FASTTD3_POLICY_DOF_NAMES_20[i]
                diff = q20[0, i] - default20[0, i]
                logger.info(f"[{name}] Live: {q20[0, i]:.4f} | Default: {default20[0, i]:.4f} | Diff: {diff:.4f}")
            logger.info("==================================")
            action_scales = env.action_scales[idx20]
            logger.info(
                "FastTD3 effective action scale (20-DOF) min/max: "
                f"{action_scales.min().item():.4f}/{action_scales.max().item():.4f}"
            )
            logger.info(
                "FastTD3 raw action (20-DOF) min/max: "
                f"{delta20.min().item():.4f}/{delta20.max().item():.4f}"
            )
            logger.info(
                "FastTD3 scaled offset (20-DOF) min/max: "
                f"{(delta20 * action_scales).min().item():.4f}/"
                f"{(delta20 * action_scales).max().item():.4f}"
            )

        # Print observations at multiple steps for debugging
        if step in [0, 10, 20, 30, 60]:
            logger.info(f"=== STEP {step} FULL OBSERVATION (compare with MuJoCo) ===")
            obs_np = obs[0].cpu().numpy()
            logger.info(f"obs[0:3] projected_gravity*{norm_gravity}: {obs_np[0:3]}")
            logger.info(f"  RAW projected_gravity: {projected_gravity[0].cpu().numpy()}")
            logger.info(f"obs[3:6] base_ang_vel*{norm_ang_vel}: {obs_np[3:6]}")
            logger.info(f"  RAW base_ang_vel: {base_ang_vel[0].cpu().numpy()}")
            logger.info(f"obs[6:9] commands*norm: vx={obs_np[6]}, vy={obs_np[7]}, vyaw={obs_np[8]}")
            logger.info(f"  RAW commands: vx={vx[0].item()}, vy={vy[0].item()}, vyaw={vyaw[0].item()}")
            logger.info(f"obs[9:11] gait: cos={obs_np[9]}, sin={obs_np[10]}")
            logger.info(f"obs[11:31] dof_pos delta (first 5): {obs_np[11:16]}")
            logger.info(f"  RAW q20 (first 5): {q20[0, :5].cpu().numpy()}")
            logger.info(f"  default20 (first 5): {default20[0, :5].cpu().numpy()}")
            logger.info(f"obs[31:51] dof_vel*{norm_dof_vel} (first 5): {obs_np[31:36]}")
            logger.info(f"  RAW dq20 (first 5): {dq20[0, :5].cpu().numpy()}")
            logger.info(f"obs[51:71] prev_actions (first 5): {obs_np[51:56]}")
            logger.info(f"POLICY OUTPUT delta20 (first 5): {delta20[0, :5].cpu().numpy()}")
            logger.info(f"  clipped between [-{clip_actions}, {clip_actions}]")
            logger.info("="*60)
            logger.info(f"obs[3:6] base_ang_vel: {obs_np[3:6]}")
            logger.info(f"obs[6:9] commands (vx,vy,vyaw): {obs_np[6:9]}")
            logger.info(f"obs[9:11] gait (cos,sin): {obs_np[9:11]}")
            logger.info(f"obs[11:31] dof_pos - default (first 5): {obs_np[11:16]}")
            logger.info(f"obs[31:51] dof_vel (first 5): {obs_np[31:36]}")
            logger.info(f"obs[51:71] prev_actions (first 5): {obs_np[51:56]}")
            logger.info(f"Full action output: {act20[0].cpu().numpy()}")
            logger.info("Run MuJoCo with same setup and compare these values!")
            logger.info("==============================================")

            # 5. Check Live Effort Limits (Torque limits) - Compare with MuJoCo
            logger.info("=== VERIFYING TORQUES & LIMITS (compare with MuJoCo MJCF) ===")
            mujoco_torque_limits = {
                "Left_Shoulder_Pitch": 18, "Left_Shoulder_Roll": 18, "Left_Elbow_Pitch": 18, "Left_Elbow_Yaw": 18,
                "Right_Shoulder_Pitch": 18, "Right_Shoulder_Roll": 18, "Right_Elbow_Pitch": 18, "Right_Elbow_Yaw": 18,
                "Left_Hip_Pitch": 45, "Left_Hip_Roll": 45, "Left_Hip_Yaw": 30, "Left_Knee_Pitch": 65,
                "Left_Ankle_Pitch": 24, "Left_Ankle_Roll": 15,
                "Right_Hip_Pitch": 45, "Right_Hip_Roll": 45, "Right_Hip_Yaw": 30, "Right_Knee_Pitch": 65,
                "Right_Ankle_Pitch": 24, "Right_Ankle_Roll": 15,
            }
            if hasattr(env, "torque_limits"):
                live_effort_limits = env.torque_limits[idx20]
                logger.info("Joint | MuJoCo | Isaac Lab | Match?")
                for i, name in enumerate(FASTTD3_POLICY_DOF_NAMES_20):
                    mj_limit = mujoco_torque_limits.get(name, "?")
                    isaac_limit = live_effort_limits[i].item()
                    match = "✓" if abs(mj_limit - isaac_limit) < 0.1 else "✗ MISMATCH!"
                    logger.info(f"{name}: {mj_limit} | {isaac_limit:.1f} | {match}")
            else:
                logger.error("Wait, torque_limits disappeared?!")
            # 6. Check Live PD Gains and Action Filters
            # 6. Check Live PD Gains and Action Filters (BULLETPROOF VERSION)
            logger.info("=== VERIFYING PD GAINS AND FILTERS ===")
            robot = None
            if hasattr(env, "scene") and "robot" in env.scene.keys():
                robot = env.scene["robot"]
            elif hasattr(env, "robot"):
                robot = env.robot

            if robot is not None and hasattr(robot, "data"):
                # IsaacLab versions change these variable names constantly. Let's find it dynamically:
                stiff_attr = next((a for a in dir(robot.data) if "stiff" in a.lower()), None)
                damp_attr = next((a for a in dir(robot.data) if "damp" in a.lower()), None)

                if stiff_attr and damp_attr:
                    live_stiffness = getattr(robot.data, stiff_attr)
                    live_damping = getattr(robot.data, damp_attr)

                    # Handle either 1D or 2D tensor shapes
                    if len(live_stiffness.shape) > 1:
                        stiff_vals = live_stiffness[0, idx20]
                        damp_vals = live_damping[0, idx20]
                    else:
                        stiff_vals = live_stiffness[idx20]
                        damp_vals = live_damping[idx20]

                    hip_idx = FASTTD3_POLICY_DOF_NAMES_20.index('Left_Hip_Pitch')
                    logger.info(f"[PD Gains] Found '{stiff_attr}' - Left_Hip_Pitch LIVE Stiffness: {stiff_vals[hip_idx].item():.2f} (Expected: 50.00)")
                    logger.info(f"[PD Gains] Found '{damp_attr}' - Left_Hip_Pitch LIVE Damping:   {damp_vals[hip_idx].item():.2f} (Expected: 3.00)")
                    
                    if abs(stiff_vals[hip_idx].item() - 50.0) > 1.0:
                        logger.error("DANGER: Your stiffness overrides were ignored! The robot is using the wrong springs!")
                else:
                    logger.warning(f"Could not find stiffness/damping keys. Available data: {[k for k in dir(robot.data) if not k.startswith('_')]}")

            # Safe Action Filter check (Handles both lists and dicts)
            if hasattr(env, "action_manager"):
                active_terms = env.action_manager.active_terms
                term_list = active_terms if isinstance(active_terms, list) else active_terms.values()

                for term in term_list:
                    term_name = term.__class__.__name__
                    if hasattr(term, "cfg"):
                        filter_active = getattr(term.cfg, "use_filter", False) or hasattr(term.cfg, "filter_model")
                        logger.info(f"[Action Filter] Term '{term_name}' has action filter active: {filter_active}")
                        if filter_active:
                            logger.error("DANGER: You have an action filter slowing down your policy!")

        target20 = default20 + delta20 * action_scale

        actions_prev20 = delta20  # Use raw action for observation history (matches MuJoCo)

        if step == 10:
            logger.info("=== CRITICAL ACTION FLOW CHECK ===")
            logger.info(f"delta20 (network output) range: [{delta20.min().item():.3f}, {delta20.max().item():.3f}]")
            logger.info(f"action_scale from YAML: {action_scale}")
            logger.info(f"env.action_scales[idx20] range: [{env.action_scales[idx20].min().item():.3f}, {env.action_scales[idx20].max().item():.3f}]")
            logger.info(f"Effective action after env scaling: [{(delta20 * env.action_scales[idx20]).min().item():.3f}, {(delta20 * env.action_scales[idx20]).max().item():.3f}]")
            logger.info("Using DIRECT PD TORQUE CONTROL (bypassing env.step action pipeline)")

        # ========================================================================
        # DIRECT PD TORQUE CONTROL (matches MuJoCo validation exactly)
        # CRITICAL: PD control must run at EVERY physics step (500Hz), not just policy step (50Hz)
        # MuJoCo recomputes torques every step with fresh state - we must do the same!
        # ========================================================================

        # Build full DOF targets (29 DOF) - non-policy DOFs stay at default
        dof_targets_full = env.default_dof_pos.clone()
        dof_targets_full[:, idx20] = target20

        # Step physics for each decimation substep
        # CRITICAL: Recompute PD torques at EVERY substep with fresh state (like MuJoCo)
        for substep_idx in range(decimation):
            # Get FRESH state at this substep
            dof_pos_full = env.simulator.dof_pos
            dof_vel_full = env.simulator.dof_vel

            # Compute PD torques with current state
            # This is EXACTLY like MuJoCo: dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel
            torques_full = (
                env.p_gains * (dof_targets_full - dof_pos_full)
                - env.d_gains * dof_vel_full
            )

            # Clip torques to limits (like MuJoCo's actuator_ctrlrange clipping)
            torques_full = torch.clamp(torques_full, -env.torque_limits, env.torque_limits)

            if step in [0, 10, 20, 30, 60] and substep_idx == 0:
                logger.info(f"=== DIRECT PD TORQUE CHECK (step {step}, substep 0) ===")
                logger.info(f"target20[0,:5]: {target20[0,:5].cpu().numpy()}")
                logger.info(f"dof_pos[idx20][0,:5]: {dof_pos_full[0,idx20[:5]].cpu().numpy()}")
                logger.info(f"dof_vel[idx20][0,:5]: {dof_vel_full[0,idx20[:5]].cpu().numpy()}")
                logger.info(f"p_gains[idx20][:5]: {env.p_gains[idx20[:5]].cpu().numpy()}")
                logger.info(f"d_gains[idx20][:5]: {env.d_gains[idx20[:5]].cpu().numpy()}")
                logger.info(f"torques[idx20][0,:5]: {torques_full[0,idx20[:5]].cpu().numpy()}")
                logger.info(f"torque_limits[idx20][:5]: {env.torque_limits[idx20[:5]].cpu().numpy()}")
                logger.info(f"torques min/max (all 20): [{torques_full[0,idx20].min().item():.3f}, {torques_full[0,idx20].max().item():.3f}]")
                logger.info("=========================================")

            # Apply torques and step physics once
            env.simulator.apply_torques_at_dof(torques_full)
            env.simulator.simulate_at_each_physics_step()

        # Update gait process AFTER physics steps (MuJoCo line 133)
        # This happens once per policy step, after all decimation substeps
        gait_process = torch.fmod(gait_process + (sim_dt * decimation) * gait_freq, 1.0)

        # We bypass env.step(), so we must run termination + reset manually.
        # Mirror the key pieces of BaseTask._post_physics_step() needed for reset behavior.
        if hasattr(env, "_refresh_sim_tensors"):
            env._refresh_sim_tensors()
        if hasattr(env, "episode_length_buf"):
            env.episode_length_buf += 1
        if hasattr(env, "_update_counters_each_step"):
            env._update_counters_each_step()
        if hasattr(env, "_pre_compute_observations_callback"):
            env._pre_compute_observations_callback()
        if hasattr(env, "_update_tasks_callback"):
            env._update_tasks_callback()
        if hasattr(env, "_check_termination"):
            env._check_termination()

        if hasattr(env, "reset_buf"):
            env_ids_to_reset = env.reset_buf.nonzero(as_tuple=False).flatten()
            if env_ids_to_reset.numel() > 0:
                logger.info(f"Resetting {env_ids_to_reset.numel()} env(s) after termination/fall: {env_ids_to_reset.tolist()}")
                env.reset_envs_idx(env_ids_to_reset)

                # Avoid stale/random velocity commands after reset in manager buffers.
                if hasattr(env, "command_manager") and hasattr(env.command_manager, "commands"):
                    env.command_manager.commands[env_ids_to_reset, :3] = 0.0
                if hasattr(env.simulator, "commands"):
                    env.simulator.commands[env_ids_to_reset, :3] = 0.0

                if hasattr(env, "_get_envs_to_refresh") and hasattr(env, "_refresh_envs_after_reset"):
                    refresh_env_ids = env._ensure_long_tensor(env._get_envs_to_refresh())
                    if refresh_env_ids.numel() > 0:
                        env._refresh_envs_after_reset(refresh_env_ids)

                # Keep external-eval local buffers aligned with reset environments.
                actions_prev20[env_ids_to_reset] = 0.0
                gait_process[env_ids_to_reset] = 0.0


def run_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
):
    # Use shared simulation environment setup
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    exported_policy_dir_path = os.path.join(checkpoint_dir, "exported")
    os.makedirs(exported_policy_dir_path, exist_ok=True)
    exported_policy_name = checkpoint_path.split("/")[-1]  # example: model_5000.pt
    exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")  # example: model_5000.onnx

    if tyro_config.training.export_onnx:
        exported_onnx_path = os.path.join(exported_policy_dir_path, exported_onnx_name)
        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
            )

        algo.export(onnx_file_path=exported_onnx_path)  # type: ignore[attr-defined]
        logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

    algo.evaluate_policy(
        max_eval_steps=tyro_config.training.max_eval_steps,
    )

    # Cleanup simulation app
    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()

    # Parse checkpoint config first
    checkpoint_cfg, remaining_args = tyro.cli(
        CheckpointConfig, return_unknown_args=True, add_help=False
    )

    # Parse external config next (consumes --enabled/--checkpoint-pt/--yaml-path/--max-steps)
    external_cfg, remaining_args = tyro.cli(
        ExternalFastTD3Cfg, args=remaining_args, return_unknown_args=True, add_help=False
    )

    if external_cfg.enabled:
        if checkpoint_cfg.checkpoint is None:
            raise ValueError("External mode requires --checkpoint (Holosoma ckpt) to source ExperimentConfig defaults.")
        if external_cfg.checkpoint_pt is None or external_cfg.yaml_path is None:
            raise ValueError("External mode requires --checkpoint-pt and --yaml-path.")

        with open(external_cfg.yaml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)

        saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
        eval_cfg = saved_cfg.get_eval_config()

        eval_cfg = apply_fasttd3_yaml_overrides_to_cfg(eval_cfg, y)

        try:
            overwritten_tyro_config = tyro.cli(
                ExperimentConfig,
                default=eval_cfg,
                args=remaining_args,
                description="Overriding config on top of what's loaded (external fasttd3 eval).",
                config=TYRO_CONIFG,
            )
        except BaseException as e:
            # Tyro may raise SystemExit on parser-generation failures (not caught by Exception).
            logger.warning(
                "Tyro parser failed in external eval mode; "
                f"falling back to manual CLI overrides. Cause: {type(e).__name__}: {e}"
            )
            overwritten_tyro_config = _apply_external_fallback_overrides(eval_cfg, remaining_args)

        overwritten_tyro_config = _apply_external_runtime_tuning(overwritten_tyro_config, external_cfg)

        # Debug nested actuator config
        control = overwritten_tyro_config.robot.control
        if hasattr(control, 'actuators'):
            logger.info(f"control.actuators: {control.actuators}")
            if hasattr(control.actuators, 'joint_names_expr'):
                logger.info(f"actuators.joint_names_expr keys: {list(control.actuators.joint_names_expr.keys())}")
        robot_cfg = overwritten_tyro_config.robot
        if hasattr(robot_cfg, 'actuators'):
            logger.info(f"robot.actuators: {robot_cfg.actuators}")
            if hasattr(robot_cfg.actuators, 'joint_names_expr'):
                logger.info(f"robot.actuators.joint_names_expr keys: {list(robot_cfg.actuators.joint_names_expr.keys())}")

        env, device, simulation_app = setup_simulation_environment(overwritten_tyro_config)
        evaluate_external_fasttd3(env, device, external_cfg)

        if simulation_app:
            close_simulation_app(simulation_app)
        return

    # --- Normal Holosoma eval path (requires Holosoma checkpoint with experiment_config) ---
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    run_eval_with_tyro(overwritten_tyro_config, checkpoint_cfg, saved_cfg, saved_wandb_path)


if __name__ == "__main__":
    main()