"""FastTD3 observation builder for Isaac Lab.

Constructs 71-dimensional observations matching MuJoCo Playground format exactly.
This ensures checkpoint compatibility between the two simulators.

Observation layout (71 dimensions):
- obs[0:3]   = projected_gravity * 1.0
- obs[3:6]   = base_ang_vel (body frame) * 0.25
- obs[6:9]   = [vx*2.0, vy*2.0, vyaw*0.25]
- obs[9:11]  = [cos(gait), sin(gait)]  # COS first!
- obs[11:31] = (dof_pos_20 - default_20) * 1.0
- obs[31:51] = dof_vel_20 * 0.05
- obs[51:71] = last_actions_20
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from holosoma.utils.rotations import quat_rotate_inverse

if TYPE_CHECKING:
    from holosoma.envs.base_task.base_task import BaseTask


# The 20 policy-controlled DOFs (excludes head, waist, wrists)
FASTTD3_POLICY_DOF_NAMES_20 = [
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

# Normalization constants - MUST match MuJoCo T1.yaml
NORM_GRAVITY = 1.0
NORM_ANG_VEL = 1.0   # Updated from 0.25 to match T1.yaml
NORM_LIN_VEL = 1.0   # Updated from 2.0 to match T1.yaml
NORM_DOF_POS = 1.0
NORM_DOF_VEL = 0.1   # Updated from 0.05 to match T1.yaml


class FastTD3ObservationBuilder:
    """Constructs 71-dim observations matching MuJoCo Playground format.

    This class extracts state from a holosoma environment and builds observations
    in the exact format expected by FastTD3 policies trained in MuJoCo Playground.
    """

    def __init__(
        self,
        env: BaseTask,
        device: str | torch.device,
        default_joint_angles: dict[str, float] | None = None,
    ):
        """Initialize observation builder.

        Args:
            env: Holosoma environment with simulator and robot data.
            device: Torch device for tensors.
            default_joint_angles: Optional dict of default angles per joint.
                If None, uses environment defaults.
        """
        self.env = env
        self.device = device
        self.num_envs = env.num_envs

        # Build DOF index mapping from env.dof_names to policy DOFs
        self.idx20 = self._build_dof_indices()

        # Build default joint positions for the 20 policy DOFs
        self.default20 = self._load_default_positions(default_joint_angles)

    def _build_dof_indices(self) -> torch.Tensor:
        """Map policy DOF names to environment DOF indices."""
        dof_names = list(self.env.dof_names)
        indices = []

        for name in FASTTD3_POLICY_DOF_NAMES_20:
            if name not in dof_names:
                raise KeyError(
                    f"DOF '{name}' not found in env.dof_names. "
                    f"Available: {dof_names}"
                )
            indices.append(dof_names.index(name))

        return torch.tensor(indices, device=self.device, dtype=torch.long)

    def _load_default_positions(
        self, default_angles: dict[str, float] | None
    ) -> torch.Tensor:
        """Load default joint positions for policy DOFs.

        Args:
            default_angles: Optional dict mapping joint names to default angles.
                If None, extracts from environment default_dof_pos.

        Returns:
            Tensor of shape [1, 20] with default positions.
        """
        if default_angles is not None:
            # Build from provided dict
            vals = []
            default_scalar = float(default_angles.get("default", 0.0))
            for name in FASTTD3_POLICY_DOF_NAMES_20:
                vals.append(float(default_angles.get(name, default_scalar)))
            return torch.tensor(vals, device=self.device).view(1, 20)
        else:
            # Extract from environment
            if hasattr(self.env, "default_dof_pos"):
                env_defaults = self.env.default_dof_pos
                if env_defaults.ndim > 1:
                    return env_defaults[0:1, self.idx20]
                else:
                    return env_defaults[self.idx20].unsqueeze(0)
            else:
                # Fallback to zeros
                return torch.zeros(1, 20, device=self.device)

    def compute(
        self,
        gait_process: torch.Tensor,
        last_actions: torch.Tensor,
        commands: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute 71-dim observation tensor.

        Args:
            gait_process: Gait phase in [0, 1) for each env. Shape [num_envs].
            last_actions: Previous policy actions. Shape [num_envs, 20].
            commands: Velocity commands [vx, vy, vyaw]. Shape [num_envs, 3].
                If None, reads from env.command_manager.commands.

        Returns:
            Observation tensor of shape [num_envs, 71].
        """
        # Get robot state from simulator
        q_full = self.env.simulator.dof_pos  # [num_envs, num_dof]
        dq_full = self.env.simulator.dof_vel  # [num_envs, num_dof]

        # Extract 20 policy DOFs
        q20 = q_full[:, self.idx20]  # [num_envs, 20]
        dq20 = dq_full[:, self.idx20]  # [num_envs, 20]

        # Get root state
        robot_data = self.env.simulator._robot.data

        # Root quaternion: Isaac Lab uses wxyz, convert to xyzw for quat_rotate_inverse
        root_quat_wxyz = robot_data.root_quat_w  # [num_envs, 4] in wxyz
        base_quat_xyzw = torch.cat(
            [root_quat_wxyz[:, 1:4], root_quat_wxyz[:, 0:1]], dim=1
        )  # Convert to xyzw

        # Angular velocity in world frame -> body frame
        ang_vel_world = robot_data.root_ang_vel_w  # [num_envs, 3]
        base_ang_vel = quat_rotate_inverse(base_quat_xyzw, ang_vel_world, w_last=True)

        # Projected gravity (rotate world gravity into body frame)
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        gravity_world = gravity_world.view(1, 3).expand(self.num_envs, 3)
        projected_gravity = quat_rotate_inverse(
            base_quat_xyzw, gravity_world, w_last=True
        )

        # Get velocity commands
        if commands is not None:
            vx = commands[:, 0]
            vy = commands[:, 1]
            vyaw = commands[:, 2]
        elif hasattr(self.env, "command_manager") and hasattr(
            self.env.command_manager, "commands"
        ):
            cmd = self.env.command_manager.commands
            vx = cmd[:, 0]
            vy = cmd[:, 1]
            vyaw = cmd[:, 2] if cmd.shape[1] > 2 else torch.zeros_like(vx)
        else:
            vx = torch.zeros(self.num_envs, device=self.device)
            vy = torch.zeros(self.num_envs, device=self.device)
            vyaw = torch.zeros(self.num_envs, device=self.device)

        # ====================================================================
        # Gait phase: Use the command_manager's LocomotionGait state
        # This ensures the observation phase matches the reward phase!
        # ====================================================================
        gait_state = None
        if hasattr(self.env, "command_manager"):
            gait_state = self.env.command_manager.get_state("locomotion_gait")

        if gait_state is not None and gait_state.phase is not None:
            # Use phase[:, 0] (first leg's phase) - already in [-π, π]
            # LocomotionGait already handles standing still (sets phase to π)
            phi = gait_state.phase[:, 0]
            cos_g = torch.cos(phi)
            sin_g = torch.sin(phi)
        else:
            # Fallback to passed gait_process if no command_manager
            # (shouldn't happen in normal FastTD3 training)
            gait_rad = gait_process * 2.0 * math.pi
            cos_g = torch.cos(gait_rad)
            sin_g = torch.sin(gait_rad)

            # Standing still: use phase=π → cos(π)=-1, sin(π)=0
            # This matches MuJoCo behavior (NOT zeroing out!)
            cmd_norm = torch.sqrt(vx**2 + vy**2 + vyaw**2)
            standing = (cmd_norm <= 0.01)
            cos_g = torch.where(standing, torch.full_like(cos_g, -1.0), cos_g)
            sin_g = torch.where(standing, torch.zeros_like(sin_g), sin_g)

        # Expand default20 to batch size
        default20_batch = self.default20.expand(self.num_envs, 20)

        # Build observation tensor
        obs = torch.zeros((self.num_envs, 71), device=self.device)

        obs[:, 0:3] = projected_gravity * NORM_GRAVITY
        obs[:, 3:6] = base_ang_vel * NORM_ANG_VEL
        obs[:, 6] = vx * NORM_LIN_VEL
        obs[:, 7] = vy * NORM_LIN_VEL
        obs[:, 8] = vyaw * NORM_ANG_VEL
        obs[:, 9] = cos_g  # COS first (like MuJoCo)
        obs[:, 10] = sin_g  # SIN second
        obs[:, 11:31] = (q20 - default20_batch) * NORM_DOF_POS
        obs[:, 31:51] = dq20 * NORM_DOF_VEL
        obs[:, 51:71] = last_actions

        return obs

    def get_policy_dof_indices(self) -> torch.Tensor:
        """Return the DOF indices for the 20 policy-controlled joints."""
        return self.idx20

    def get_default_positions(self) -> torch.Tensor:
        """Return the default joint positions for policy DOFs."""
        return self.default20


class GaitPhaseTracker:
    """Tracks gait phase for each environment.

    Updates phase each step based on gait frequency.
    Resets phase for done environments.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        default_gait_freq: float = 1.5,
    ):
        """Initialize gait phase tracker.

        Args:
            num_envs: Number of parallel environments.
            device: Torch device.
            default_gait_freq: Default gait frequency in Hz.
        """
        self.num_envs = num_envs
        self.device = device
        self.default_gait_freq = default_gait_freq

        # Gait process in [0, 1) - fraction of gait cycle completed
        self.gait_process = torch.zeros(num_envs, device=device)

    def update(
        self,
        dt: float,
        dones: torch.Tensor | None = None,
        commands: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update gait phase for one step.

        Args:
            dt: Timestep in seconds.
            dones: Done flags [num_envs] - reset phase for done envs.
            commands: Velocity commands [num_envs, 3] to determine if moving.

        Returns:
            Current gait_process tensor [num_envs].
        """
        # Determine gait frequency based on whether robot is moving
        if commands is not None:
            cmd_norm = torch.sqrt(
                commands[:, 0] ** 2 + commands[:, 1] ** 2 + commands[:, 2] ** 2
            )
            moving = (cmd_norm > 0.01).float()
            gait_freq = moving * self.default_gait_freq
        else:
            gait_freq = torch.full(
                (self.num_envs,), self.default_gait_freq, device=self.device
            )

        # Update phase
        self.gait_process = (self.gait_process + gait_freq * dt) % 1.0

        # Reset phase for done environments
        if dones is not None and dones.any():
            self.gait_process = torch.where(
                dones.bool(), torch.zeros_like(self.gait_process), self.gait_process
            )

        return self.gait_process

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset gait phase for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, resets all.
        """
        if env_ids is None:
            self.gait_process.zero_()
        else:
            self.gait_process[env_ids] = 0.0
