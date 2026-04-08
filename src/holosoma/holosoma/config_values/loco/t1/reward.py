"""Locomotion reward presets for the T1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

t1_29dof_loco = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-10.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=1.0,
            params={},
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={
                "pose_weights": [
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                ],
            },
            tags=["penalty_curriculum"],
        ),
    },
)

t1_29dof_loco_fast_sac = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-10.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={
                "pose_weights": [
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                ],
            },
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=10.0,
            params={},
        ),
    },
)

__all__ = ["t1_29dof_loco", "t1_29dof_loco_fast_sac", "t1_20dof_fast_td3"]


# FastTD3 reward configuration matching MuJoCo Playground joystick.py
# NOTE: This uses 20 DOF policy joints (excludes head, waist, wrists)
t1_20dof_fast_td3 = RewardManagerCfg(
    only_positive_rewards=False,  # Clip negative rewards like MuJoCo Playground
    terms={
        # ================================================================
        # Positive rewards — matches MuJoCo joystick.py exactly
        # ================================================================
        # survival: 0.25
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=0.25,
            params={},
        ),
        # tracking_lin_vel_x: 2.0  (per-axis, NOT combined)
        "tracking_lin_vel_x": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel_x",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        # tracking_lin_vel_y: 2.0
        "tracking_lin_vel_y": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel_y",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        # tracking_ang_vel: 2.0
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        # feet_swing: 3.0 — binary contact-based reward
        "feet_swing": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_swing",
            weight=3.0,
            params={"swing_period": 0.2},
        ),
        # ================================================================
        # Penalties — matches MuJoCo joystick.py scales exactly
        # MuJoCo comment format: "original_value / divisor" where applicable
        # ================================================================
        # base_height: -20.0
        "base_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:base_height",
            weight=-20.0,
            params={"desired_base_height": 0.68},
        ),
        # orientation: -5.0
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-5.0,
            params={},
        ),
        # torques: -2.0e-4 / 2 = -1.0e-4
        "penalty_torques": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_torques",
            weight=-1.0e-4,
            params={},
        ),
        # torque_tiredness: -1.0e-2 / 2 = -5.0e-3
        "penalty_torque_tiredness": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_torque_tiredness",
            weight=-5.0e-3,
            params={},
        ),
        # power: -2.0e-3 / 2 = -1.0e-3
        "penalty_energy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_energy",
            weight=-1.0e-3,
            params={},
        ),
        # lin_vel_z: -2.0
        "penalty_lin_vel_z": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_lin_vel_z",
            weight=-2.0,
            params={},
        ),
        # ang_vel_xy: -0.2
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-0.2,
            params={},
        ),
        # dof_vel: -1.0e-4
        "penalty_dof_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_dof_vel",
            weight=-1.0e-4,
            params={},
        ),
        # dof_acc: -1.0e-7
        "penalty_dof_acc": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_dof_acc",
            weight=-1.0e-7,
            params={},
        ),
        # root_acc: -1.0e-4
        "penalty_root_acc": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_root_acc",
            weight=-1.0e-4,
            params={},
        ),
        # action_rate: -1.0 / 2 = -0.5
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-0.5,
            params={},
        ),
        # dof_pos_limits: -1.0
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:limits_dof_pos",
            weight=-1.0,
            params={"soft_dof_pos_limit": 0.95},
        ),
        # collision (feet-to-feet): -1.0 * 10.0 = -10.0
        "penalty_collision_feet": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_collision_feet",
            weight=-10.0,
            params={},
        ),
        # feet_slip: -0.1
        "penalty_feet_slip": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_slip",
            weight=-0.1,
            params={},
        ),
        # feet_yaw_diff: -1.0
        "penalty_feet_yaw_diff": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_yaw_diff",
            weight=-1.0,
            params={},
        ),
        # feet_yaw_mean: -1.0
        "penalty_feet_yaw_mean": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_yaw_mean",
            weight=-1.0,
            params={},
        ),
        # feet_roll: -0.1 * 10.0 = -1.0
        "penalty_feet_roll": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_roll",
            weight=-1.0,
            params={},
        ),
        # feet_distance: -1.0 * 10.0 = -10.0  (continuous, not binary)
        "penalty_feet_distance": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_distance",
            weight=-10.0,
            params={"min_distance": 0.2, "max_penalty": 0.1},
        ),
        # arm: -2.0 (shoulder roll)
        "penalty_arm_roll": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_arm_roll",
            weight=-2.0,
            params={"target_left": -1.25, "target_right": 1.25},
        ),
        # arm_pitch: -2.0
        "penalty_arm_pitch": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_arm_pitch",
            weight=-2.0,
            params={"target": 0.0},
        ),
        # arm_yaw: -2.0 (elbow pitch)
        "penalty_arm_yaw": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_arm_yaw",
            weight=-2.0,
            params={"target": 0.0},
        ),
        # elbow_pitch: -2.0 (elbow yaw)
        "penalty_elbow_yaw": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_elbow_yaw",
            weight=-2.0,
            params={"target_left": -0.5, "target_right": 0.5},
        ),
        # NOTE: collision_hand (-10.0) and stop (0.0) from MuJoCo are omitted.
        # collision_hand requires hand-leg pair contact detection not available in Isaac Lab.
        # stop is disabled (weight=0.0) in MuJoCo.
    },
)