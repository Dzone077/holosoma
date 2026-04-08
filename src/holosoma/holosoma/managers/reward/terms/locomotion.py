"""Reward terms for locomotion tasks.

These terms are migrated from LeggedRobotBase._reward_* methods to be
compatible with the reward manager system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma.managers.observation.terms.locomotion import (
    base_forward_vector,
    get_base_ang_vel,
    get_base_lin_vel,
    get_projected_gravity,
    gravity_vector,
)
from holosoma.utils.rotations import (
    quat_apply,
    quat_rotate_batched,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager


def _expected_foot_height(phi: torch.Tensor, swing_height: float) -> torch.Tensor:
    """Expected foot height from gait phase using a cubic Bézier profile."""

    def cubic_bezier_interpolation(y_start: torch.Tensor, y_end: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phi + torch.pi) / (2 * torch.pi)
    stance = cubic_bezier_interpolation(torch.zeros_like(x), torch.full_like(x, swing_height), 2 * x)
    swing = cubic_bezier_interpolation(torch.full_like(x, swing_height), torch.zeros_like(x), 2 * x - 1)
    return torch.where(x <= 0.5, stance, swing)


# ================================================================================================
# Termination Rewards
# ================================================================================================


def termination(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Terminal reward/penalty for early termination (excluding timeouts).

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return (env.reset_buf * ~env.time_out_buf).float()


# ================================================================================================
# Penalty Rewards
# ================================================================================================


def penalty_action_rate(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Uses self-managed buffers (env._reward_action / env._reward_prev_action)
    when available (set by step_direct_pd), falling back to action_manager
    for the standard env.step() path.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    if hasattr(env, '_reward_action') and hasattr(env, '_reward_prev_action'):
        actions = env._reward_action
        prev_actions = env._reward_prev_action
    else:
        actions = env.action_manager.action
        prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def penalty_orientation(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize non-flat base orientation.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    projected = get_projected_gravity(env)
    return torch.sum(torch.square(projected[:, :2]), dim=1)


def penalty_feet_ori(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize feet orientation deviation from flat.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    left_quat = env.simulator._rigid_body_rot[:, env.feet_indices[0]]
    gravity = gravity_vector(env)
    left_gravity = quat_rotate_inverse(left_quat, gravity, w_last=True)
    right_quat = env.simulator._rigid_body_rot[:, env.feet_indices[1]]
    right_gravity = quat_rotate_inverse(right_quat, gravity, w_last=True)
    return (
        torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
        + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5
    )


# ================================================================================================
# Limit Rewards
# ================================================================================================


def limits_dof_pos(env: LeggedRobotLocomotionManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# ================================================================================================
# Tracking and Task Rewards
# ================================================================================================


def tracking_lin_vel(env, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes combined).

    Uses exponential reward: exp(-(ex² + ey²) / sigma).
    NOTE: This combines both axes into one exp(), which differs from MuJoCo's
    per-axis tracking. Use ``tracking_lin_vel_x`` / ``tracking_lin_vel_y`` for
    exact MuJoCo T1 joystick parity.

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - get_base_lin_vel(env)[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / tracking_sigma)


def tracking_lin_vel_x(env, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward tracking of x-axis linear velocity command.

    Per-axis version matching MuJoCo joystick ``_reward_tracking_lin_vel_axis(0)``.

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    lin_vel = get_base_lin_vel(env)
    err = torch.square(commands[:, 0] - lin_vel[:, 0])
    return torch.exp(-err / tracking_sigma)


def tracking_lin_vel_y(env, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward tracking of y-axis linear velocity command.

    Per-axis version matching MuJoCo joystick ``_reward_tracking_lin_vel_axis(1)``.

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    lin_vel = get_base_lin_vel(env)
    err = torch.square(commands[:, 1] - lin_vel[:, 1])
    return torch.exp(-err / tracking_sigma)


def tracking_ang_vel(env, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw).

    Uses exponential reward: exp(-error / sigma)

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    ang_vel = get_base_ang_vel(env)
    ang_vel_error = torch.square(commands[:, 2] - ang_vel[:, 2])
    return torch.exp(-ang_vel_error / tracking_sigma)


def penalty_ang_vel_xy(env) -> torch.Tensor:
    """Penalize xy axes base angular velocity.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    ang_vel = get_base_ang_vel(env)
    return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


def penalty_close_feet_xy(env, close_feet_threshold: float = 0.05) -> torch.Tensor:
    """Penalize when feet are too close together in xy plane (binary).

    NOTE: This is a simplified binary version. Use ``penalty_feet_distance``
    for the exact MuJoCo T1 joystick continuous formulation.

    Args:
        env: The environment instance
        close_feet_threshold: Minimum distance threshold between feet

    Returns:
        Reward tensor [num_envs]
    """
    left_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[0], :2]
    right_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[1], :2]

    # Get base orientation
    base_forward = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)
    base_yaw = torch.atan2(base_forward[:, 1], base_forward[:, 0])

    # Calculate perpendicular distance in base-local coordinates
    feet_distance = torch.abs(
        torch.cos(base_yaw) * (left_foot_xy[:, 1] - right_foot_xy[:, 1])
        - torch.sin(base_yaw) * (left_foot_xy[:, 0] - right_foot_xy[:, 0])
    )

    # Return penalty when feet are too close
    return (feet_distance < close_feet_threshold).float()


def penalty_feet_distance(
    env,
    min_distance: float = 0.2,
    max_penalty: float = 0.1,
) -> torch.Tensor:
    """Continuous feet-distance penalty matching MuJoCo ``_cost_feet_distance``.

    Returns ``clip(min_distance - lateral_dist, 0, max_penalty)``.
    Zero when feet are at least ``min_distance`` apart; linearly grows as they
    get closer, capped at ``max_penalty``.

    Args:
        env: The environment instance
        min_distance: Desired minimum lateral distance between feet (m)
        max_penalty: Maximum per-step penalty value

    Returns:
        Reward tensor [num_envs]
    """
    left_foot_pos = env.simulator._rigid_body_pos[:, env.feet_indices[0]]
    right_foot_pos = env.simulator._rigid_body_pos[:, env.feet_indices[1]]

    base_forward = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)
    base_yaw = torch.atan2(base_forward[:, 1], base_forward[:, 0])

    feet_distance = torch.abs(
        torch.cos(base_yaw) * (left_foot_pos[:, 1] - right_foot_pos[:, 1])
        - torch.sin(base_yaw) * (left_foot_pos[:, 0] - right_foot_pos[:, 0])
    )

    return torch.clamp(min_distance - feet_distance, min=0.0, max=max_penalty)


def base_height(
    env, desired_base_height: float = 0.89, zero_vel_penalty_scale: float = 1.0, stance_penalty_scale: float = 1.0
) -> torch.Tensor:
    """Penalize base height away from target.

    Args:
        env: The environment instance
        desired_base_height: Target base height
        zero_vel_penalty_scale: Multiplier for base height penalty when robot has zero velocity commands
        stance_penalty_scale: Multiplier for base height penalty when robot is in stance mode

    Returns:
        Reward tensor [num_envs]
    """
    # Get base heights from terrain manager
    base_heights = env.terrain_manager.get_state("locomotion_terrain").base_heights

    # Fallback: if base_heights is NaN (terrain not initialized), use absolute robot Z position
    # This ensures the robot still gets feedback about maintaining height
    if torch.isnan(base_heights).any():
        # Use the robot's base Z position directly (on flat terrain, this is approximately the height)
        base_heights = env.simulator.robot_root_states[:, 2]

    base_height_penalty = torch.square(base_heights - desired_base_height)

    # Apply stronger penalty for zero velocity commands if configured
    if zero_vel_penalty_scale != 1.0:
        commands = env.command_manager.commands
        zero_vel_mask = torch.norm(commands[:, :2], dim=1) < 0.1
        base_height_penalty = torch.where(
            zero_vel_mask, base_height_penalty * zero_vel_penalty_scale, base_height_penalty
        )

    # Apply stronger penalty for stance mode if configured (used in decoupled locomotion)
    if stance_penalty_scale != 1.0 and hasattr(env, "stance_mask"):
        base_height_penalty = torch.where(
            env.stance_mask, base_height_penalty * stance_penalty_scale, base_height_penalty
        )

    return base_height_penalty


def feet_phase(env, swing_height: float = 0.08, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward for tracking desired foot height based on gait phase.

    Based on MuJoCo Playground's G1 implementation (continuous height tracking).
    NOTE: This is NOT the same as MuJoCo T1 joystick.py's ``_reward_feet_swing``
    which is a simpler binary contact-based reward. Use ``feet_swing`` below for
    exact T1 joystick parity.

    Args:
        env: The environment instance
        swing_height: Maximum height during swing phase
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    # Get foot heights (relative to terrain)
    foot_z_left = env.terrain_manager.get_state("locomotion_terrain").feet_heights[:, 0]
    foot_z_right = env.terrain_manager.get_state("locomotion_terrain").feet_heights[:, 1]

    # Calculate expected foot heights based on phase
    gait_state = env.command_manager.get_state("locomotion_gait")
    rz_left = _expected_foot_height(gait_state.phase[:, 0], swing_height)
    rz_right = _expected_foot_height(gait_state.phase[:, 1], swing_height)

    # Calculate height tracking errors
    error_left = torch.square(foot_z_left - rz_left)
    error_right = torch.square(foot_z_right - rz_right)

    # Combine errors and apply exponential reward
    total_error = error_left + error_right

    return torch.exp(-total_error / tracking_sigma)


def feet_swing(env, swing_period: float = 0.2) -> torch.Tensor:
    """Binary swing reward matching MuJoCo T1 joystick.py ``_reward_feet_swing``.

    During a time window around each foot's expected swing phase, reward +1 if
    the foot is NOT in contact with the ground.  No height tracking — just
    "be airborne during your swing window".

    The gait cycle is divided as:
      - Left  foot swings at ~25% of the cycle (gait ∈ [0.15, 0.35] for period=0.2)
      - Right foot swings at ~75% of the cycle (gait ∈ [0.65, 0.85] for period=0.2)

    Args:
        env: The environment instance
        swing_period: Fraction of the gait cycle that counts as swing window
            (MuJoCo default 0.2 → each foot has a 20% swing window)

    Returns:
        Reward tensor [num_envs]
    """
    gait_state = env.command_manager.get_state("locomotion_gait")
    # Use phase[:, 0] (left foot phase) — same as MuJoCo's phase[0]
    phi = gait_state.phase[:, 0]  # in [-π, π]

    # Convert to [0, 1) fraction of cycle — same as MuJoCo:
    #   gait = fmod(phase[0] + π, 2π) / (2π)
    gait = torch.fmod(phi + torch.pi, 2 * torch.pi) / (2 * torch.pi)

    half_window = 0.5 * swing_period

    left_swing = torch.abs(gait - 0.25) < half_window
    right_swing = torch.abs(gait - 0.75) < half_window

    # Contact detection from contact forces
    contact_left = env.simulator.contact_forces[:, env.feet_indices[0], 2] > 1.0
    contact_right = env.simulator.contact_forces[:, env.feet_indices[1], 2] > 1.0

    # Reward: +1 per foot that is airborne during its swing window
    reward = (left_swing & ~contact_left).float() + (right_swing & ~contact_right).float()

    return reward


def pose(
    env,
    pose_weights: list[float],
) -> torch.Tensor:
    """Reward for maintaining default pose.

    Penalizes deviation from default joint positions with weighted importance.

    Args:
        env: The environment instance
        pose_weights: List of weights for each DOF (must match num_dof)

    Returns:
        Reward tensor [num_envs]
    """
    # Get current joint positions
    qpos = env.simulator.dof_pos

    # Convert pose_weights to tensor
    weights = torch.tensor(pose_weights, device=env.device, dtype=torch.float32)

    # Calculate squared deviation from default pose
    # Use env.default_dof_pos which is already set up from robot config
    pose_error = torch.square(qpos - env.default_dof_pos)

    # Weight and sum the errors
    weighted_error = pose_error * weights.unsqueeze(0)

    return torch.sum(weighted_error, dim=1)


def penalty_stumble(env) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return torch.any(
        torch.norm(env.simulator.contact_forces[:, env.feet_indices, :2], dim=2)
        > 4 * torch.abs(env.simulator.contact_forces[:, env.feet_indices, 2]),
        dim=1,
    )


def penalty_foothold(env, foothold_epsilon: float = 0.01) -> torch.Tensor:
    """Sampling-based foothold penalty.

    For each foot in contact, sample a grid of points on the sole, transform to world,
    read terrain height at those XY, compute depth d_ij = z_sample - terrain_z, and count
    samples with d_ij < epsilon. Sum over both feet.

    Args:
        env: The environment instance
        foothold_epsilon: Threshold for foothold depth penalty

    Returns:
        Reward tensor [num_envs]
    """
    # Contact mask per foot
    contact = env.simulator.contact_forces[:, env.feet_indices, 2] > 1.0  # [E,2]
    if not (contact.any()):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # Accumulator
    penalty = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    for foot_idx_local in range(2):
        # Skip if no env has contact on this foot to save work
        if not contact[:, foot_idx_local].any():
            continue
        rb_idx = env.feet_indices[foot_idx_local]
        foot_pos_w = env.simulator._rigid_body_pos[:, rb_idx, :]  # [E,3]
        foot_quat_w = env.simulator._rigid_body_rot[:, rb_idx, :]  # [E,4]

        # Use precomputed sample points in the foot frame
        pts_local = env.foot_samples_local[foot_idx_local].unsqueeze(0).repeat(env.num_envs, 1, 1)

        # Rotate to world and translate
        pts_world = quat_rotate_batched(foot_quat_w, pts_local) + foot_pos_w.unsqueeze(1)

        # Query terrain height at those XY positions
        terrain_h = env._get_terrain_heights_at_points_world(pts_world)

        # Depth: world z minus terrain height
        depth = pts_world[:, :, 2] - terrain_h  # [E,S]

        # Indicator for d_ij > epsilon, only for envs with this foot in contact
        bad = (depth > foothold_epsilon).float()
        bad *= contact[:, foot_idx_local].unsqueeze(1).float()

        penalty += torch.sum(bad, dim=1)

    return penalty / env.num_foot_samples


def alive(env) -> torch.Tensor:
    """Reward for staying alive.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return torch.ones(env.num_envs, dtype=torch.float, device=env.device)


# ================================================================================================
# FastTD3/MuJoCo Playground Compatible Rewards
# ================================================================================================


def penalty_torques(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize total squared torques.

    Uses env._reward_torques when available (set by step_direct_pd),
    falling back to action_manager for the standard env.step() path.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    if hasattr(env, '_reward_torques'):
        torques = env._reward_torques
    else:
        torques = env.action_manager.applied_torques
    return torch.sum(torch.square(torques), dim=1)


def penalty_energy(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize positive mechanical power (torque * velocity when positive).

    This is effectively energy consumption penalty.

    Uses env._reward_torques when available (set by step_direct_pd),
    falling back to action_manager for the standard env.step() path.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    if hasattr(env, '_reward_torques'):
        torques = env._reward_torques
    else:
        torques = env.action_manager.applied_torques
    dof_vel = env.simulator.dof_vel
    power = torques * dof_vel
    positive_power = torch.where(power > 0, power, torch.zeros_like(power))
    return torch.sum(positive_power, dim=1)


def penalty_torque_tiredness(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize torques relative to their limits.

    sum((torque / torque_limit)^2), clamped to 1.0 max ratio.

    Uses env._reward_torques when available (set by step_direct_pd),
    falling back to action_manager for the standard env.step() path.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    if hasattr(env, '_reward_torques'):
        torques = env._reward_torques
    else:
        torques = env.action_manager.applied_torques
    torque_limits = env.torque_limits

    # Compute ratio and clamp
    ratio = torch.abs(torques) / (torque_limits + 1e-6)
    ratio = torch.clamp(ratio, max=1.0)

    return torch.sum(torch.square(ratio), dim=1)


def penalty_lin_vel_z(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize vertical (z-axis) linear velocity.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    lin_vel = get_base_lin_vel(env)
    return torch.square(lin_vel[:, 2])


def penalty_dof_vel(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize total joint velocity magnitude.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    dof_vel = env.simulator.dof_vel
    return torch.sum(torch.square(dof_vel), dim=1)


def penalty_dof_acc(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize joint accelerations.

    Uses finite differences from last velocity.  Stores the previous-step
    velocity buffer on ``env`` as ``_reward_last_dof_vel`` so it persists
    across calls even when the simulator does not track it.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    dof_vel = env.simulator.dof_vel

    if not hasattr(env, "_reward_last_dof_vel"):
        # First call – initialise buffer and return zero penalty
        env._reward_last_dof_vel = dof_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)

    if hasattr(env.simulator, "config") and hasattr(env.simulator.config, "sim"):
        dt = 1.0 / env.simulator.config.sim.fps * env.simulator.config.sim.control_decimation
    else:
        dt = 0.02

    dof_acc = (dof_vel - env._reward_last_dof_vel) / dt
    env._reward_last_dof_vel = dof_vel.clone()
    return torch.sum(torch.square(dof_acc), dim=1)


def penalty_root_acc(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize root link acceleration.

    Stores previous-step root velocities on ``env`` as
    ``_reward_last_root_lin_vel`` / ``_reward_last_root_ang_vel``.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    lin_vel = env.simulator._robot.data.root_lin_vel_w
    ang_vel = env.simulator._robot.data.root_ang_vel_w

    if not hasattr(env, "_reward_last_root_lin_vel"):
        env._reward_last_root_lin_vel = lin_vel.clone()
        env._reward_last_root_ang_vel = ang_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)

    if hasattr(env.simulator, "config") and hasattr(env.simulator.config, "sim"):
        dt = 1.0 / env.simulator.config.sim.fps * env.simulator.config.sim.control_decimation
    else:
        dt = 0.02

    lin_acc = (lin_vel - env._reward_last_root_lin_vel) / dt
    ang_acc = (ang_vel - env._reward_last_root_ang_vel) / dt

    env._reward_last_root_lin_vel = lin_vel.clone()
    env._reward_last_root_ang_vel = ang_vel.clone()

    return torch.sum(torch.square(lin_acc), dim=1) + torch.sum(torch.square(ang_acc), dim=1)


def penalty_feet_slip(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize feet slipping (velocity while in contact).

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    # Get foot velocities (use _rigid_body_vel which is the linear velocity)
    left_foot_vel = env.simulator._rigid_body_vel[:, env.feet_indices[0], :2]
    right_foot_vel = env.simulator._rigid_body_vel[:, env.feet_indices[1], :2]

    # Get contact status
    contact_left = env.simulator.contact_forces[:, env.feet_indices[0], 2] > 1.0
    contact_right = env.simulator.contact_forces[:, env.feet_indices[1], 2] > 1.0

    # Compute slip penalty
    left_slip = torch.sum(torch.square(left_foot_vel), dim=1) * contact_left.float()
    right_slip = torch.sum(torch.square(right_foot_vel), dim=1) * contact_right.float()

    return left_slip + right_slip


def penalty_feet_roll(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize feet roll angle deviation from flat.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    # Get foot orientations and compute roll from gravity projection
    left_quat = env.simulator._rigid_body_rot[:, env.feet_indices[0]]
    right_quat = env.simulator._rigid_body_rot[:, env.feet_indices[1]]

    # Project gravity into foot frames
    gravity = gravity_vector(env)
    left_gravity = quat_rotate_inverse(left_quat, gravity, w_last=True)
    right_gravity = quat_rotate_inverse(right_quat, gravity, w_last=True)

    # Roll is deviation from purely vertical gravity (z=-1)
    # When flat, gravity should be [0, 0, -1] in foot frame
    left_roll = torch.atan2(left_gravity[:, 1], -left_gravity[:, 2])
    right_roll = torch.atan2(right_gravity[:, 1], -right_gravity[:, 2])

    return torch.square(left_roll) + torch.square(right_roll)


def _feet_yaw(env) -> torch.Tensor:
    """Extract yaw angles of both feet from their world-frame quaternions.

    Returns:
        Tensor of shape [num_envs, 2] with [left_yaw, right_yaw] in radians.
    """
    yaws = []
    for i in range(2):
        quat = env.simulator._rigid_body_rot[:, env.feet_indices[i]]  # [E, 4] wxyz
        # Extract yaw = atan2(R21, R11) from quaternion
        # For wxyz quaternion: w=q0, x=q1, y=q2, z=q3
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        # R[1,0] = 2(xy + wz),  R[0,0] = 1 - 2(y² + z²)
        r10 = 2.0 * (x * y + w * z)
        r00 = 1.0 - 2.0 * (y * y + z * z)
        yaws.append(torch.atan2(r10, r00))
    return torch.stack(yaws, dim=1)


def _base_yaw(env) -> torch.Tensor:
    """Extract base (trunk) yaw angle.

    Returns:
        Tensor of shape [num_envs] with yaw in radians.
    """
    base_forward = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)
    return torch.atan2(base_forward[:, 1], base_forward[:, 0])


def penalty_feet_yaw_diff(env) -> torch.Tensor:
    """Penalize yaw angle difference between left and right foot.

    Matches MuJoCo joystick ``_cost_feet_yaw_diff``.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    feet_yaw = _feet_yaw(env)
    diff = torch.fmod(feet_yaw[:, 1] - feet_yaw[:, 0] + torch.pi, 2 * torch.pi) - torch.pi
    return torch.square(diff)


def penalty_feet_yaw_mean(env) -> torch.Tensor:
    """Penalize mean foot yaw deviating from base yaw.

    Matches MuJoCo joystick ``_cost_feet_yaw_mean``.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    feet_yaw = _feet_yaw(env)
    mean_yaw = feet_yaw.mean(dim=1)
    base_y = _base_yaw(env)
    err = torch.fmod(base_y - mean_yaw + torch.pi, 2 * torch.pi) - torch.pi
    return torch.square(err)


def penalty_collision_feet(env) -> torch.Tensor:
    """Penalize left and right feet colliding with each other.

    Approximation of MuJoCo joystick ``_cost_collision`` which checks
    ``geoms_colliding(left_foot, right_foot)``.  In Isaac Lab we detect this
    when both feet have contact AND are within a small distance of each other.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]  (0 or 1)
    """
    left_pos = env.simulator._rigid_body_pos[:, env.feet_indices[0], :3]
    right_pos = env.simulator._rigid_body_pos[:, env.feet_indices[1], :3]
    dist = torch.norm(left_pos - right_pos, dim=1)

    # Both feet in contact AND very close (< ~23cm, sum of two foot half-extents)
    contact_left = env.simulator.contact_forces[:, env.feet_indices[0], 2] > 1.0
    contact_right = env.simulator.contact_forces[:, env.feet_indices[1], 2] > 1.0
    return (contact_left & contact_right & (dist < 0.23)).float()


def _get_dof_indices(env, joint_names: list[str]) -> torch.Tensor:
    """Return DOF indices for given joint names, looked up from env.dof_names."""
    dof_list = list(env.dof_names)
    indices = [dof_list.index(n) for n in joint_names if n in dof_list]
    return torch.tensor(indices, device=env.device, dtype=torch.long)


def penalty_arm_roll(
    env: LeggedRobotLocomotionManager,
    target_left: float = -1.25,
    target_right: float = 1.25,
) -> torch.Tensor:
    """Penalize shoulder roll deviation from neutral pose.

    Matches MuJoCo joystick _cost_arm: keeps arms folded at sides.

    Args:
        env: The environment instance
        target_left: Target angle for Left_Shoulder_Roll (rad)
        target_right: Target angle for Right_Shoulder_Roll (rad)

    Returns:
        Reward tensor [num_envs]
    """
    idx = _get_dof_indices(env, ["Left_Shoulder_Roll", "Right_Shoulder_Roll"])
    if len(idx) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    q = env.simulator.dof_pos[:, idx]
    return (q[:, 0] - target_left) ** 2 + (q[:, 1] - target_right) ** 2


def penalty_arm_pitch(
    env: LeggedRobotLocomotionManager,
    target: float = 0.0,
) -> torch.Tensor:
    """Penalize shoulder pitch deviation from neutral pose.

    Matches MuJoCo joystick _cost_arm_pitch.

    Args:
        env: The environment instance
        target: Target angle for both shoulder pitch joints (rad)

    Returns:
        Reward tensor [num_envs]
    """
    idx = _get_dof_indices(env, ["Left_Shoulder_Pitch", "Right_Shoulder_Pitch"])
    if len(idx) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    q = env.simulator.dof_pos[:, idx]
    return (q[:, 0] - target) ** 2 + (q[:, 1] - target) ** 2


def penalty_arm_yaw(
    env: LeggedRobotLocomotionManager,
    target: float = 0.0,
) -> torch.Tensor:
    """Penalize elbow yaw deviation from neutral pose.

    Matches MuJoCo joystick _cost_arm_yaw (uses Left/Right_Elbow_Pitch for yaw-like terms).

    Args:
        env: The environment instance
        target: Target angle for both elbow pitch joints (rad)

    Returns:
        Reward tensor [num_envs]
    """
    idx = _get_dof_indices(env, ["Left_Elbow_Pitch", "Right_Elbow_Pitch"])
    if len(idx) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    q = env.simulator.dof_pos[:, idx]
    return (q[:, 0] - target) ** 2 + (q[:, 1] - target) ** 2


def penalty_elbow_yaw(
    env: LeggedRobotLocomotionManager,
    target_left: float = -0.5,
    target_right: float = 0.5,
) -> torch.Tensor:
    """Penalize elbow yaw deviation from default pose.

    Matches MuJoCo joystick _cost_elbow_pitch (Left_Elbow_Yaw / Right_Elbow_Yaw).

    Args:
        env: The environment instance
        target_left: Target for Left_Elbow_Yaw (rad)
        target_right: Target for Right_Elbow_Yaw (rad)

    Returns:
        Reward tensor [num_envs]
    """
    idx = _get_dof_indices(env, ["Left_Elbow_Yaw", "Right_Elbow_Yaw"])
    if len(idx) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    q = env.simulator.dof_pos[:, idx]
    return (q[:, 0] - target_left) ** 2 + (q[:, 1] - target_right) ** 2