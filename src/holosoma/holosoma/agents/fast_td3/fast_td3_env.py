"""FastTD3 environment wrapper for holosoma.

Wraps a holosoma BaseTask environment to provide the FastTD3-compatible interface.
Handles observation construction, action mapping, and gait phase tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.agents.fast_td3.fast_td3_obs import (
    FASTTD3_POLICY_DOF_NAMES_20,
    FastTD3ObservationBuilder,
    GaitPhaseTracker,
)

# ============================================================================
# MuJoCo-Compatible Joint Limits for 20 Policy DOFs
# These MUST match MuJoCo's t2_mjx_feetonly_lowdim_collision2.xml joint limits exactly!
# CRITICAL: Several joints have ASYMMETRIC limits in MuJoCo
# ============================================================================
MUJOCO_JOINT_LIMITS_20 = {
    # Arms - Elbow Yaw has ASYMMETRIC limits in MuJoCo!
    "Left_Shoulder_Pitch": (-3.31, 1.22),
    "Left_Shoulder_Roll": (-1.74, 1.57),
    "Left_Elbow_Pitch": (-2.27, 2.27),
    "Left_Elbow_Yaw": (-2.44, 0.0),  # CRITICAL: Only negative in MuJoCo!
    "Right_Shoulder_Pitch": (-3.31, 1.22),
    "Right_Shoulder_Roll": (-1.57, 1.74),
    "Right_Elbow_Pitch": (-2.27, 2.27),
    "Right_Elbow_Yaw": (0.0, 2.44),  # CRITICAL: Only positive in MuJoCo!
    # Legs - Hip Roll has asymmetric inner limits
    "Left_Hip_Pitch": (-1.8, 1.57),
    "Left_Hip_Roll": (-0.3, 1.57),  # MuJoCo inner limit -0.3
    "Left_Hip_Yaw": (-1.0, 1.0),
    "Left_Knee_Pitch": (0.0, 2.34),
    "Left_Ankle_Pitch": (-0.87, 0.35),
    "Left_Ankle_Roll": (-0.44, 0.44),
    "Right_Hip_Pitch": (-1.8, 1.57),
    "Right_Hip_Roll": (-1.57, 0.3),  # MuJoCo inner limit 0.3
    "Right_Hip_Yaw": (-1.0, 1.0),
    "Right_Knee_Pitch": (0.0, 2.34),
    "Right_Ankle_Pitch": (-0.87, 0.35),
    "Right_Ankle_Roll": (-0.44, 0.44),
}

if TYPE_CHECKING:
    from holosoma.envs.base_task.base_task import BaseTask


class FastTD3Env:
    """Environment wrapper providing FastTD3-compatible interface.

    This wrapper sits between the FastTD3 training loop and a holosoma environment.
    It handles:
    - Constructing 71-dim observations in FastTD3 format
    - Mapping 20-action policy outputs to full robot DOFs
    - Tracking gait phase state
    - Tracking last actions for observation history

    Interface:
    - reset() -> obs[num_envs, 71]
    - step(actions[num_envs, 20]) -> (obs, rewards, dones, info)
    """

    def __init__(
        self,
        env: BaseTask,
        device: str | torch.device,
        default_joint_angles: dict[str, float] | None = None,
        action_scale: float = 1.0,
        gait_freq: float = 1.5,
        clip_actions: float = 100.0,
        use_direct_pd: bool = True,
    ):
        """Initialize FastTD3 environment wrapper.

        Args:
            env: Holosoma BaseTask environment.
            device: Torch device.
            default_joint_angles: Optional default angles for policy DOFs.
            action_scale: Scale factor for policy actions (default 1.0).
            gait_freq: Default gait frequency in Hz (default 1.5).
            clip_actions: Action clipping bounds (default 100.0).
            use_direct_pd: If True, use direct PD torque control matching MuJoCo/eval_agent.py.
                          If False, use env.step() (default True for MuJoCo compatibility).
        """
        self._env = env
        self.device = device
        self.action_scale = action_scale
        self.clip_actions = clip_actions
        self.use_direct_pd = use_direct_pd

        # FastTD3 interface properties
        self.num_envs = env.num_envs
        self.num_obs = 71  # Fixed for FastTD3 locomotion
        self.num_actions = 20  # 20 policy-controlled DOFs
        self.asymmetric_obs = False  # No critic observations
        self.num_privileged_obs = 0
        self.seed = 0

        # Get max episode steps from environment
        if hasattr(env, "max_episode_length"):
            self.max_episode_steps = env.max_episode_length
        elif hasattr(env.simulator, "config") and hasattr(
            env.simulator.config, "sim"
        ):
            # Estimate from episode length and dt
            self.max_episode_steps = 1000  # Default fallback
        else:
            self.max_episode_steps = 1000

        # Observation builder
        self.obs_builder = FastTD3ObservationBuilder(
            env=env,
            device=device,
            default_joint_angles=default_joint_angles,
        )

        # Gait phase tracker
        self.gait_tracker = GaitPhaseTracker(
            num_envs=self.num_envs,
            device=device,
            default_gait_freq=gait_freq,
        )

        # Last actions buffer for observation
        self._last_actions = torch.zeros(
            self.num_envs, self.num_actions, device=device
        )

        # Full action buffer for stepping the environment
        self._full_actions = torch.zeros(
            self.num_envs, env.num_dof, device=device
        )

        # Get simulation dt
        if hasattr(env.simulator, "config") and hasattr(env.simulator.config, "sim"):
            self._dt = 1.0 / env.simulator.config.sim.fps
            self._control_decimation = env.simulator.config.sim.control_decimation
            self._control_dt = self._dt * self._control_decimation
        else:
            self._dt = 0.002  # Default 500Hz
            self._control_decimation = 10
            self._control_dt = 0.02

        # Store the DOF indices for the 20 policy joints
        self._idx20 = self.obs_builder.get_policy_dof_indices()

        # ====================================================================
        # Build MuJoCo-compatible joint limits for the 20 policy DOFs
        # CRITICAL: These constrain the position targets during training
        # to prevent the policy from learning to use impossible joint ranges
        # ====================================================================
        lower_limits_20 = []
        upper_limits_20 = []
        for joint_name in FASTTD3_POLICY_DOF_NAMES_20:
            limits = MUJOCO_JOINT_LIMITS_20[joint_name]
            lower_limits_20.append(limits[0])
            upper_limits_20.append(limits[1])
        self._joint_lower_limits_20 = torch.tensor(
            lower_limits_20, dtype=torch.float32, device=device
        )
        self._joint_upper_limits_20 = torch.tensor(
            upper_limits_20, dtype=torch.float32, device=device
        )

        # Store default positions for all DOFs
        # Start with environment defaults for non-policy joints
        if hasattr(env, "default_dof_pos"):
            self._default_dof_pos = env.default_dof_pos.clone()
        else:
            self._default_dof_pos = torch.zeros(
                self.num_envs, env.num_dof, device=device
            )

        # CRITICAL: Override policy joint defaults with passed values
        # This ensures action targets match MuJoCo evaluation convention
        if default_joint_angles is not None:
            policy_defaults = self.obs_builder.get_default_positions()  # [1, 20]
            self._default_dof_pos[:, self._idx20] = policy_defaults.expand(self.num_envs, -1)

        # CRITICAL FIX: IsaacSim quaternion proxy doesn't handle broadcasting correctly
        # Expand base_init_state to (num_envs, 13) to avoid shape mismatch during reset
        if hasattr(env, "base_init_state") and env.base_init_state.dim() == 1:
            # Save original for reference
            self._original_base_init_state = env.base_init_state.clone()
            # Replace with expanded version
            env.base_init_state = env.base_init_state.unsqueeze(0).expand(
                self.num_envs, -1
            ).contiguous()

            # Monkey-patch _reset_root_states to handle indexing correctly
            # This is necessary because the original code doesn't index base_init_state
            original_reset_root_states = env._reset_root_states

            def patched_reset_root_states(env_ids, target_root_states=None):
                """Patched version that indexes into expanded base_init_state."""
                from holosoma.utils.torch_utils import torch_rand_float

                if target_root_states is not None:
                    env.simulator.robot_root_states[env_ids] = target_root_states
                    env.simulator.robot_root_states[env_ids, :3] += env.terrain_manager.get_state(
                        "locomotion_terrain"
                    ).env_origins[env_ids]
                else:
                    # Index into the expanded base_init_state
                    env.simulator.robot_root_states[env_ids] = env.base_init_state[env_ids]
                    env.simulator.robot_root_states[env_ids, :3] += env.terrain_manager.get_state(
                        "locomotion_terrain"
                    ).env_origins[env_ids]

                    # Apply randomized XY offset if custom_origins
                    if env.terrain_manager.get_state("locomotion_terrain").custom_origins:
                        spawn_cfg = env.terrain_manager.cfg.terrain_term.spawn
                        xy_offsets = torch_rand_float(
                            -spawn_cfg.xy_offset_range, spawn_cfg.xy_offset_range, (len(env_ids), 2), device=str(env.device)
                        )

                        if spawn_cfg.query_terrain_height:
                            current_xy = env.simulator.robot_root_states[env_ids, :2]
                            new_xy = current_xy + xy_offsets
                            terrain_state = env.terrain_manager.get_state("locomotion_terrain")
                            terrain_heights = terrain_state.query_terrain_heights(
                                new_xy,
                                use_grid_sampling=spawn_cfg.use_grid_sampling,
                                grid_size=spawn_cfg.grid_size,
                                grid_spacing=spawn_cfg.grid_spacing,
                            )
                            robot_base_height = env.robot_config.init_state.pos[2]
                            new_z = terrain_heights + robot_base_height
                            new_xyz = torch.cat([new_xy, new_z.unsqueeze(1)], dim=1)
                            env.simulator.robot_root_states[env_ids, :3] = new_xyz
                        else:
                            env.simulator.robot_root_states[env_ids, :2] += xy_offsets

                    # base velocities
                    env.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(
                        -0.5, 0.5, (len(env_ids), 6), device=str(env.device)
                    )

            # Replace the method
            env._reset_root_states = patched_reset_root_states

    @property
    def dof_names(self):
        """Return DOF names from underlying environment."""
        return self._env.dof_names

    def reset(self, random_start_init: bool = True) -> torch.Tensor:
        """Reset all environments.

        Args:
            random_start_init: If True, randomize initial episode lengths
                to decorrelate episode horizons.

        Returns:
            Observations tensor of shape [num_envs, 71].
        """
        # Reset underlying environment
        self._env.reset_all()

        # Reset internal state
        self._last_actions.zero_()
        self.gait_tracker.reset()

        # Optionally decorrelate episode horizons
        if random_start_init and hasattr(self._env, "episode_length_buf"):
            self._env.episode_length_buf = torch.randint_like(
                self._env.episode_length_buf, high=int(self.max_episode_steps)
            )

        # Compute initial observations
        gait_process = self.gait_tracker.gait_process
        obs = self.obs_builder.compute(
            gait_process=gait_process,
            last_actions=self._last_actions,
        )

        return obs

    def reset_with_critic_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset and return both actor and critic observations.

        For symmetric training, critic obs is the same as actor obs.
        """
        obs = self.reset()
        return obs, obs.clone()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment with 20-DOF policy actions.

        Args:
            actions: Policy actions of shape [num_envs, 20].

        Returns:
            obs: Next observations [num_envs, 71]
            rewards: Reward tensor [num_envs]
            dones: Done flags [num_envs]
            info: Dict with 'time_outs' and 'observations'
        """
        # Use direct PD control for MuJoCo-compatible physics
        if self.use_direct_pd:
            return self.step_direct_pd(actions)

        # Original env.step() based implementation (legacy, less accurate)
        # Clip actions
        actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # Scale actions
        scaled_actions = actions * self.action_scale

        # Compute target positions for policy DOFs
        target20 = self._default_dof_pos[:, self._idx20] + scaled_actions

        # Clamp to MuJoCo-compatible joint limits
        target20 = torch.clamp(
            target20,
            min=self._joint_lower_limits_20,
            max=self._joint_upper_limits_20,
        )

        # Map 20 actions to full DOF space
        # Non-policy joints use their default positions as targets
        self._full_actions.copy_(self._default_dof_pos)
        self._full_actions[:, self._idx20] = target20

        # Step the underlying environment
        obs_dict, rewards, dones, extras = self._env.step(
            {"actions": self._full_actions}
        )

        # Get commands for gait phase update
        if hasattr(self._env, "command_manager"):
            commands = self._env.command_manager.commands
        else:
            commands = None

        # Update gait phase
        gait_process = self.gait_tracker.update(
            dt=self._control_dt,
            dones=dones,
            commands=commands,
        )

        # CRITICAL FIX: Update last_actions BEFORE computing observation
        # This ensures obs[51:71] contains the action that was just taken (actions_t),
        # not the action from 2 steps ago (actions_{t-1}).
        # This matches MuJoCo validation behavior where obs uses the previous action output.
        self._last_actions.copy_(actions)

        # Reset last actions for done environments (they're starting new episodes)
        if dones.any():
            done_mask = dones.bool()
            self._last_actions[done_mask] = 0.0

        # Build FastTD3 observations (now with correctly timed last_actions)
        obs = self.obs_builder.compute(
            gait_process=gait_process,
            last_actions=self._last_actions,
            commands=commands,
        )

        # Build info dict matching FastTD3 expected format
        time_outs = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
        info = {
            "time_outs": time_outs,
            "observations": {
                "critic": obs.clone(),  # Symmetric observations
                "raw": {
                    "obs": obs,
                    "critic_obs": obs.clone(),
                },
            },
        }

        return obs, rewards, dones.long(), info

    def step_direct_pd(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment using direct PD torque control (like eval_agent.py).

        This method bypasses env.step() and manually computes PD torques at each
        physics substep, exactly matching MuJoCo's behavior where torques are
        recomputed every simulation step with fresh state.

        Args:
            actions: Policy actions of shape [num_envs, 20].

        Returns:
            obs: Next observations [num_envs, 71]
            rewards: Reward tensor [num_envs]
            dones: Done flags [num_envs]
            info: Dict with 'time_outs' and 'observations'
        """
        env = self._env

        # Clip actions
        actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # Track actions for penalty_action_rate (step_direct_pd bypasses action_manager)
        if not hasattr(env, '_reward_prev_action'):
            env._reward_prev_action = torch.zeros_like(actions)
            env._reward_action = torch.zeros_like(actions)
        env._reward_prev_action.copy_(env._reward_action)
        env._reward_action.copy_(actions)

        # Scale actions
        scaled_actions = actions * self.action_scale

        # Compute position targets for the 20 policy DOFs
        target20 = self._default_dof_pos[:, self._idx20] + scaled_actions

        # ====================================================================
        # CRITICAL: Clamp target positions to MuJoCo-compatible joint limits!
        # Without this, the policy learns to use joint ranges that are
        # impossible in MuJoCo (e.g., Left_Elbow_Yaw > 0, Right_Elbow_Yaw < 0)
        # ====================================================================
        target20 = torch.clamp(
            target20,
            min=self._joint_lower_limits_20,
            max=self._joint_upper_limits_20,
        )

        # Build full DOF targets (all DOFs) - non-policy DOFs stay at default
        dof_targets_full = self._default_dof_pos.clone()
        dof_targets_full[:, self._idx20] = target20

        # ========================================================================
        # DIRECT PD TORQUE CONTROL (matches MuJoCo/eval_agent.py exactly)
        # CRITICAL: PD control must run at EVERY physics step, not just policy step
        # ========================================================================
        for substep_idx in range(self._control_decimation):
            # Get FRESH state at this substep
            dof_pos_full = env.simulator.dof_pos
            dof_vel_full = env.simulator.dof_vel

            # Compute PD torques with current state
            # This is EXACTLY like MuJoCo: stiffness * (target - pos) - damping * vel
            torques_full = (
                env.p_gains * (dof_targets_full - dof_pos_full)
                - env.d_gains * dof_vel_full
            )

            # Clip torques to limits (like MuJoCo's actuator_ctrlrange clipping)
            torques_full = torch.clamp(torques_full, -env.torque_limits, env.torque_limits)

            # Apply torques and step physics once
            env.simulator.apply_torques_at_dof(torques_full)
            env.simulator.simulate_at_each_physics_step()

        # Store final-substep torques for reward functions (penalty_torques, etc.)
        env._reward_torques = torques_full

        # ========================================================================
        # POST-PHYSICS: Termination, reset, and reward computation
        # Mirror the key pieces of BaseTask._post_physics_step()
        # ========================================================================

        # Refresh simulation tensors
        if hasattr(env, "_refresh_sim_tensors"):
            env._refresh_sim_tensors()

        # Update episode length buffer
        if hasattr(env, "episode_length_buf"):
            env.episode_length_buf += 1

        # Update counters and callbacks
        if hasattr(env, "_update_counters_each_step"):
            env._update_counters_each_step()
        if hasattr(env, "_pre_compute_observations_callback"):
            env._pre_compute_observations_callback()
        if hasattr(env, "_update_tasks_callback"):
            env._update_tasks_callback()

        # Check termination
        if hasattr(env, "_check_termination"):
            env._check_termination()

        # Get done flags and time_outs
        if hasattr(env, "reset_buf"):
            dones = env.reset_buf.clone()
        else:
            dones = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        time_outs = torch.zeros_like(dones, dtype=torch.bool)
        if hasattr(env, "time_out_buf"):
            time_outs = env.time_out_buf.clone()

        # Compute rewards using the reward manager
        if hasattr(env, "reward_manager"):
            rewards = env.reward_manager.compute(dt=self._control_dt)
        else:
            rewards = torch.zeros(self.num_envs, device=self.device)

        # Handle environment resets
        env_ids_to_reset = dones.nonzero(as_tuple=False).flatten()
        if env_ids_to_reset.numel() > 0:
            env.reset_envs_idx(env_ids_to_reset)

            # NOTE: Do NOT zero commands here — reset_envs_idx already calls
            # command_manager.reset() which resamples fresh commands (including
            # stand_prob handling). Zeroing here would override the resampled
            # commands and cause the gait phase to get stuck at pi (standing).

            # Refresh environments after reset
            if hasattr(env, "_get_envs_to_refresh") and hasattr(env, "_refresh_envs_after_reset"):
                refresh_env_ids = env._ensure_long_tensor(env._get_envs_to_refresh())
                if refresh_env_ids.numel() > 0:
                    env._refresh_envs_after_reset(refresh_env_ids)

            # Reset gait tracker for done environments
            self.gait_tracker.reset(env_ids_to_reset)

            # Reset action tracking buffers for done environments
            if hasattr(env, '_reward_action'):
                env._reward_action[env_ids_to_reset] = 0.0
                env._reward_prev_action[env_ids_to_reset] = 0.0

        # Get commands for gait phase update
        if hasattr(env, "command_manager"):
            commands = env.command_manager.commands
        else:
            commands = None

        # Update gait phase AFTER physics steps (like MuJoCo)
        gait_process = self.gait_tracker.update(
            dt=self._control_dt,
            dones=dones,
            commands=commands,
        )

        # Build FastTD3 observations BEFORE updating last_actions
        # so that obs[51:71] contains action_{t-1} (like MuJoCo), not action_t
        obs = self.obs_builder.compute(
            gait_process=gait_process,
            last_actions=self._last_actions,
            commands=commands,
        )

        # Update last_actions AFTER obs computation (matches MuJoCo timing)
        self._last_actions.copy_(actions)

        # Reset last actions for done environments
        if dones.any():
            done_mask = dones.bool()
            self._last_actions[done_mask] = 0.0

        # Build info dict
        info = {
            "time_outs": time_outs,
            "observations": {
                "critic": obs.clone(),
                "raw": {
                    "obs": obs,
                    "critic_obs": obs.clone(),
                },
            },
        }

        return obs, rewards, dones.long(), info

    def get_episode_rewards(self) -> torch.Tensor | None:
        """Get episode reward sums if tracked by underlying environment."""
        if hasattr(self._env, "extras") and "episode_sums" in self._env.extras:
            return self._env.extras["episode_sums"].get("rew", None)
        return None

    @property
    def reward_manager(self):
        """Access underlying reward manager for logging."""
        return getattr(self._env, "reward_manager", None)

    @property
    def p_gains(self):
        """Access PD gains from underlying environment."""
        return getattr(self._env, "p_gains", None)

    @property
    def d_gains(self):
        """Access PD gains from underlying environment."""
        return getattr(self._env, "d_gains", None)

    @property
    def torque_limits(self):
        """Access torque limits from underlying environment."""
        return getattr(self._env, "torque_limits", None)


def create_fasttd3_env(
    env: BaseTask,
    device: str,
    action_scale: float = 1.0,
    gait_freq: float = 1.5,
    default_joint_angles: dict[str, float] | None = None,
    use_direct_pd: bool = True,
) -> FastTD3Env:
    """Factory function to create FastTD3 environment wrapper.

    Args:
        env: Holosoma BaseTask environment.
        device: Torch device string.
        action_scale: Scale factor for actions.
        gait_freq: Default gait frequency.
        default_joint_angles: Optional default joint angles dict.
        use_direct_pd: If True, use direct PD torque control for MuJoCo compatibility.

    Returns:
        FastTD3Env wrapper instance.
    """
    return FastTD3Env(
        env=env,
        device=device,
        action_scale=action_scale,
        gait_freq=gait_freq,
        default_joint_angles=default_joint_angles,
        use_direct_pd=use_direct_pd,
    )