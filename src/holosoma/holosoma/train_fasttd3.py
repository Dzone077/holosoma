"""FastTD3 standalone training script for holosoma.

This script trains T1 robot locomotion using the FastTD3 algorithm,
producing checkpoint files compatible with both MuJoCo Playground and Isaac Lab.

Usage:
    python train_fasttd3.py --help
    python train_fasttd3.py --num_envs 1024 --total_timesteps 100000
"""

from __future__ import annotations

import os
import random
import time

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
from tensordict import TensorDict

from holosoma.agents.fast_td3.fast_td3 import Actor, Critic
from holosoma.agents.fast_td3.fast_td3_env import FastTD3Env, create_fasttd3_env
from holosoma.agents.fast_td3.fast_td3_utils import (
    EmpiricalNormalization,
    SimpleReplayBuffer,
    get_ddp_state_dict,
    mark_step,
    save_params,
)

torch.set_float32_matmul_precision("high")


# MuJoCo-compatible default joint angles for FastTD3 training
# These MUST match the defaults used in MuJoCo Playground / booster_gym T1.yaml
# to ensure policy transfer compatibility
MUJOCO_COMPATIBLE_DEFAULT_JOINT_ANGLES = {
    "Left_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.25,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.5,
    "Right_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.25,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.5,
    "Left_Hip_Pitch": -0.2,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.4,
    "Left_Ankle_Pitch": -0.25,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.2,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.4,
    "Right_Ankle_Pitch": -0.25,
    "Right_Ankle_Roll": 0.0,
    "default": 0.0,  # For any unlisted joints
}


@dataclass
class FastTD3Args:
    """FastTD3 training hyperparameters for T1 locomotion."""

    # Environment
    config_path: str = ""
    """Path to a holosoma experiment config YAML (optional)."""
    seed: int = 1
    """Random seed."""
    torch_deterministic: bool = True
    """If True, enable torch deterministic mode."""
    cuda: bool = True
    """Use CUDA if available."""
    device_rank: int = 0
    """GPU device rank."""

    # Training
    num_envs: int = 1024
    """Number of parallel environments."""
    total_timesteps: int = 100000
    """Total training timesteps."""
    learning_starts: int = 100
    """Timestep to start learning."""
    buffer_size: int = 2048
    """Replay buffer size per environment (reduced from 8192 for memory)."""
    batch_size: int = 8192
    """Batch size for training (reduced from 32768 for memory)."""
    num_updates: int = 2
    """Number of gradient updates per step."""

    # TD3 specific
    gamma: float = 0.97
    """Discount factor."""
    tau: float = 0.1
    """Target network soft update coefficient."""
    policy_noise: float = 0.001
    """Policy smoothing noise."""
    noise_clip: float = 0.5
    """Noise clip range."""
    policy_frequency: int = 2
    """Delayed policy update frequency."""
    num_steps: int = 1
    """N-step returns (not used currently, kept for compatibility)."""

    # Exploration
    std_min: float = 0.001
    """Minimum exploration noise std."""
    std_max: float = 0.4
    """Maximum exploration noise std."""

    # Networks
    actor_hidden_dim: int = 512
    """Actor hidden layer dimension."""
    critic_hidden_dim: int = 1024
    """Critic hidden layer dimension."""
    init_scale: float = 0.01
    """Initial weight scale for actor output layer."""

    # Distributional RL
    num_atoms: int = 101
    """Number of atoms for distributional critic."""
    v_min: float = -10.0
    """Minimum value for critic support."""
    v_max: float = 10.0
    """Maximum value for critic support."""
    use_cdq: bool = True
    """Use clipped double Q-learning."""

    # Learning rates
    actor_learning_rate: float = 3e-4
    """Actor learning rate."""
    critic_learning_rate: float = 3e-4
    """Critic learning rate."""
    actor_learning_rate_end: float = 3e-4
    """Actor LR at end of training (for cosine annealing)."""
    critic_learning_rate_end: float = 3e-4
    """Critic LR at end of training (for cosine annealing)."""
    weight_decay: float = 0.1
    """Weight decay for AdamW."""

    # Normalization
    obs_normalization: bool = True
    """Enable observation normalization."""

    # Compile
    compile: bool = True
    """Use torch.compile for faster training."""
    compile_mode: str = "reduce-overhead"
    """torch.compile mode."""

    # Logging
    use_wandb: bool = False
    """Use Weights & Biases logging."""
    project: str = "holosoma-fasttd3"
    """W&B project name."""
    exp_name: str = "t1_locomotion"
    """Experiment name."""
    eval_interval: int = 5000
    """Evaluation interval (0 to disable)."""
    save_interval: int = 5000
    """Checkpoint save interval."""

    # Environment config
    action_scale: float = 1.0
    """Action scaling factor."""
    gait_freq: float = 1.5
    """Default gait frequency in Hz."""
    use_direct_pd: bool = True
    """Use direct PD torque control (matches MuJoCo/eval_agent.py physics exactly)."""
    headless: bool = True
    """Run in headless mode (no rendering)."""

    # Misc
    measure_burnin: int = 10
    """Steps before measuring speed."""

    # Checkpoint
    checkpoint_path: str = ""
    """Path to checkpoint to resume from (optional)."""


def get_default_experiment_config() -> Any:
    """Get default experiment config for T1 locomotion."""
    from holosoma.config_types.experiment import ExperimentConfig
    from holosoma.config_values.loco.t1.fast_td3 import t1_loco_fasttd3

    return t1_loco_fasttd3


def setup_environment(args: FastTD3Args) -> tuple[Any, str, Any]:
    """Setup the holosoma environment.

    Returns:
        Tuple of (environment, device_string, simulation_app)
    """
    from holosoma.config_types.experiment import ExperimentConfig
    from holosoma.utils.sim_utils import setup_simulation_environment

    # Load or create experiment config
    if args.config_path:
        # Load config from file
        from holosoma.utils.eval_utils import load_saved_experiment_config

        config = load_saved_experiment_config(args.config_path)
    else:
        # Use default T1 locomotion config
        config = get_default_experiment_config()

    # Apply CLI overrides
    from dataclasses import replace

    training_cfg = replace(
        config.training,
        num_envs=args.num_envs,
        seed=args.seed,
        headless=args.headless,
        torch_deterministic=args.torch_deterministic,
    )
    config = replace(config, training=training_cfg)

    # Device selection
    if not args.cuda:
        device = "cpu"
    elif torch.cuda.is_available():
        device = f"cuda:{args.device_rank}"
    else:
        device = "cpu"

    # Setup simulation environment
    env, device, simulation_app = setup_simulation_environment(config, device=device)

    return env, device, simulation_app, config


def main():
    """Main training loop."""
    args = tyro.cli(FastTD3Args)
    print(f"FastTD3 Training Args:\n{args}")

    run_name = f"t1_fasttd3_{args.exp_name}_{args.seed}"

    # Wandb setup
    if args.use_wandb:
        import wandb

        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Device
    if not args.cuda:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_rank}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create environment
    print("Setting up environment...")
    base_env, device_str, simulation_app, config = setup_environment(args)

    # Wrap with FastTD3 interface
    # CRITICAL: Pass MuJoCo-compatible default joint angles to ensure policy transfer!
    env = create_fasttd3_env(
        env=base_env,
        device=device_str,
        action_scale=args.action_scale,
        gait_freq=args.gait_freq,
        default_joint_angles=MUJOCO_COMPATIBLE_DEFAULT_JOINT_ANGLES,
        use_direct_pd=args.use_direct_pd,
    )

    print("Using MuJoCo-compatible default joint angles for policy transfer compatibility")
    if args.use_direct_pd:
        print("Using DIRECT PD torque control (matches MuJoCo/eval_agent.py physics exactly)")

    n_obs = env.num_obs  # 71
    n_act = env.num_actions  # 20
    n_critic_obs = n_obs  # Symmetric observations

    print(f"Environment: num_envs={env.num_envs}, n_obs={n_obs}, n_act={n_act}")

    # Observation normalizer
    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(shape=n_critic_obs, device=device)
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    # Actor and Critic networks
    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=args.num_envs,
        device=device,
        init_scale=args.init_scale,
        hidden_dim=args.actor_hidden_dim,
        std_min=args.std_min,
        std_max=args.std_max,
    )

    qnet = Critic(
        n_obs=n_critic_obs,
        n_act=n_act,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        hidden_dim=args.critic_hidden_dim,
        device=device,
    )
    qnet_target = Critic(
        n_obs=n_critic_obs,
        n_act=n_act,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        hidden_dim=args.critic_hidden_dim,
        device=device,
    )
    qnet_target.load_state_dict(qnet.state_dict())

    # Optimizers
    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=args.critic_learning_rate,
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=args.actor_learning_rate,
        weight_decay=args.weight_decay,
    )

    # Learning rate schedulers
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=args.total_timesteps,
        eta_min=args.critic_learning_rate_end,
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=args.actor_learning_rate_end,
    )

    # Replay buffer
    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=False,
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,
    )

    # TD3 hyperparameters
    policy_noise = args.policy_noise
    noise_clip = args.noise_clip
    action_low, action_high = -1.0, 1.0

    # Update functions
    def update_critic(data: TensorDict, logs_dict: TensorDict) -> TensorDict:
        """Critic update step."""
        observations = data["observations"]
        next_observations = data["next"]["observations"]
        critic_observations = observations
        next_critic_observations = next_observations
        actions = data["actions"]
        rewards = data["next"]["rewards"]
        dones = data["next"]["dones"].bool()
        truncations = data["next"]["truncations"].bool()
        bootstrap = (truncations | ~dones).float()

        # Defensive checks for NaN/inf values
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print(f"WARNING: Invalid rewards detected: NaN={torch.isnan(rewards).sum()}, Inf={torch.isinf(rewards).sum()}")
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)

        if torch.isnan(observations).any() or torch.isinf(observations).any():
            print(f"WARNING: Invalid observations detected")
            observations = torch.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)

        if torch.isnan(next_observations).any() or torch.isinf(next_observations).any():
            print(f"WARNING: Invalid next_observations detected")
            next_observations = torch.nan_to_num(next_observations, nan=0.0, posinf=100.0, neginf=-100.0)

        # Target policy smoothing
        clipped_noise = torch.randn_like(actions)
        clipped_noise = clipped_noise.mul(policy_noise).clamp(-noise_clip, noise_clip)

        next_state_actions = (actor(next_observations) + clipped_noise).clamp(
            action_low, action_high
        )
        discount = args.gamma ** data["next"]["effective_n_steps"]

        # Ensure discount is valid
        discount = discount.clamp(0.0, 1.0)

        with torch.no_grad():
            qf1_next_target_projected, qf2_next_target_projected = qnet_target.projection(
                next_critic_observations,
                next_state_actions,
                rewards,
                bootstrap,
                discount,
            )
            qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
            qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)

            if args.use_cdq:
                qf_next_target_dist = torch.where(
                    qf1_next_target_value.unsqueeze(1) < qf2_next_target_value.unsqueeze(1),
                    qf1_next_target_projected,
                    qf2_next_target_projected,
                )
                qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
            else:
                qf1_next_target_dist, qf2_next_target_dist = (
                    qf1_next_target_projected,
                    qf2_next_target_projected,
                )

        qf1, qf2 = qnet(critic_observations, actions)
        qf1_loss = -torch.sum(
            qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
        ).mean()
        qf2_loss = -torch.sum(
            qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
        ).mean()
        qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        q_optimizer.step()

        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf1_next_target_value.max().detach()
        logs_dict["qf_min"] = qf1_next_target_value.min().detach()
        return logs_dict

    def update_actor(data: TensorDict, logs_dict: TensorDict) -> TensorDict:
        """Actor update step."""
        critic_observations = data["observations"]

        qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
        qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
        qf2_value = qnet.get_value(F.softmax(qf2, dim=1))

        if args.use_cdq:
            qf_value = torch.minimum(qf1_value, qf2_value)
        else:
            qf_value = (qf1_value + qf2_value) / 2.0

        actor_loss = -qf_value.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optimizer.step()

        logs_dict["actor_loss"] = actor_loss.detach()
        return logs_dict

    @torch.no_grad()
    def soft_update(src: nn.Module, tgt: nn.Module, tau: float):
        """Soft update target network."""
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    # Compile if enabled
    if args.compile:
        compile_mode = args.compile_mode
        update_critic = torch.compile(update_critic, mode=compile_mode)
        update_actor = torch.compile(update_actor, mode=compile_mode)
        policy = torch.compile(actor.explore, mode=None)
        normalize_obs = obs_normalizer.forward
    else:
        policy = actor.explore
        normalize_obs = obs_normalizer.forward

    # Load checkpoint if resuming
    global_step = 0
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        qnet.load_state_dict(checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(checkpoint["qnet_target_state_dict"])
        if checkpoint.get("obs_normalizer_state"):
            obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
        if checkpoint.get("critic_obs_normalizer_state"):
            critic_obs_normalizer.load_state_dict(checkpoint["critic_obs_normalizer_state"])
        global_step = checkpoint.get("global_step", 0)
        print(f"Resumed from checkpoint at step {global_step}")

    # Reset environment
    obs = env.reset()
    dones = None

    # Training loop
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    start_time = None
    measure_burnin = 0

    print("Starting training...")

    while global_step < args.total_timesteps:
        mark_step()
        logs_dict = TensorDict()

        if start_time is None and global_step >= args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        # Collect experience
        with torch.no_grad():
            norm_obs = normalize_obs(obs)
            actions = policy(obs=norm_obs, dones=dones)

        next_obs, rewards, dones, infos = env.step(actions.float())
        truncations = infos["time_outs"]

        # Defensive checks for NaN/Inf in outputs
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print(f"WARNING Step {global_step}: NaN/Inf in rewards! NaN={torch.isnan(rewards).sum()}, Inf={torch.isinf(rewards).sum()}")
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=10.0, neginf=-10.0)

        if torch.isnan(next_obs).any() or torch.isinf(next_obs).any():
            print(f"WARNING Step {global_step}: NaN/Inf in observations!")
            next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=100.0, neginf=-100.0)

        # Store true next observations (handle autoreset)
        true_next_obs = torch.where(
            dones[:, None] > 0,
            infos["observations"]["raw"]["obs"],
            next_obs,
        )

        # Add to replay buffer
        transition = TensorDict(
            {
                "observations": obs,
                "actions": actions.clone(),
                "next": {
                    "observations": true_next_obs,
                    "rewards": rewards,
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(env.num_envs,),
            device=device,
        )
        rb.extend(transition)

        obs = next_obs

        # Training updates
        if global_step > args.learning_starts:
            for i in range(args.num_updates):
                data = rb.sample(max(1, args.batch_size // args.num_envs))

                # Normalize observations
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(data["next"]["observations"])

                # Critic update
                logs_dict = update_critic(data, logs_dict)

                # Delayed policy update
                if args.num_updates > 1:
                    if i % args.policy_frequency == 1:
                        logs_dict = update_actor(data, logs_dict)
                else:
                    if global_step % args.policy_frequency == 0:
                        logs_dict = update_actor(data, logs_dict)

                # Soft update target network
                soft_update(qnet, qnet_target, args.tau)

            # Logging
            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed:.1f} sps")

                if args.use_wandb:
                    import wandb

                    logs = {
                        "speed": speed,
                        "frame": global_step * args.num_envs,
                        "env_rewards": rewards.mean().item(),
                        "qf_loss": logs_dict.get("qf_loss", torch.tensor(0.0)).mean().item(),
                        "actor_loss": logs_dict.get("actor_loss", torch.tensor(0.0)).mean().item(),
                        "qf_max": logs_dict.get("qf_max", torch.tensor(0.0)).mean().item(),
                        "qf_min": logs_dict.get("qf_min", torch.tensor(0.0)).mean().item(),
                        "critic_lr": q_scheduler.get_last_lr()[0],
                        "actor_lr": actor_scheduler.get_last_lr()[0],
                    }
                    wandb.log(logs, step=global_step)

            # Detailed reward term logging every 1000 steps
            if global_step % 1000 == 0:
                reward_mgr = env._env.reward_manager
                print(f"\n=== Step {global_step} - Reward Term Analysis ===")
                print(f"Total reward: mean={rewards.mean().item():.4f}, min={rewards.min().item():.4f}, max={rewards.max().item():.4f}")
                print("-" * 60)

                # Compute each reward term and show its contribution
                for term_name, term_cfg in zip(reward_mgr._term_names, reward_mgr._term_cfgs):
                    try:
                        if term_name in reward_mgr._term_instances:
                            instance = reward_mgr._term_instances[term_name]
                            rew_raw = instance(env._env, **term_cfg.params)
                        else:
                            func = reward_mgr._term_funcs[term_name]
                            rew_raw = func(env._env, **term_cfg.params)

                        # Calculate weighted contribution (with dt)
                        dt = env._env.dt
                        weighted = rew_raw * term_cfg.weight * dt

                        print(f"  {term_name:25s} | raw: {rew_raw.mean().item():8.4f} | weight: {term_cfg.weight:8.4f} | contrib: {weighted.mean().item():8.4f}")
                    except Exception as e:
                        print(f"  {term_name:25s} | ERROR: {e}")

                print("=" * 60 + "\n")

            # Save checkpoint
            if args.save_interval > 0 and global_step > 0 and global_step % args.save_interval == 0:
                os.makedirs("models", exist_ok=True)
                save_params(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    obs_normalizer,
                    critic_obs_normalizer,
                    args,
                    f"models/{run_name}_{global_step}.pt",
                )

        global_step += 1
        actor_scheduler.step()
        q_scheduler.step()
        pbar.update(1)

    # Save final checkpoint
    os.makedirs("models", exist_ok=True)
    save_params(
        global_step,
        actor,
        qnet,
        qnet_target,
        obs_normalizer,
        critic_obs_normalizer,
        args,
        f"models/{run_name}_final.pt",
    )
    print(f"Training complete. Final checkpoint saved to models/{run_name}_final.pt")

    # Cleanup
    if simulation_app is not None:
        from holosoma.utils.sim_utils import close_simulation_app

        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
