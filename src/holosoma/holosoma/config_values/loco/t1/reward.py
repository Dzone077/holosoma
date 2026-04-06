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
    only_positive_rewards=True,  # Clip negative rewards like MuJoCo Playground
    terms={
        # Positive rewards (tracking)
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=0.25,  # survival
            params={},
        ),
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,  # Increased from 1.0 to encourage forward movement
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=2.0,  # Higher weight for ang vel
            params={"tracking_sigma": 0.25},
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=3.0,  # feet_swing equivalent
            params={"swing_height": 0.08, "tracking_sigma": 0.25},
        ),
        # Penalties (negative weights)
        "base_height": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:base_height",
            weight=-20.0,
            params={"desired_base_height": 0.68},
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-5.0,
            params={},
        ),
        "penalty_torques": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_torques",
            weight=-1.0e-4,  # -2.0e-4 / 2
            params={},
        ),
        "penalty_torque_tiredness": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_torque_tiredness",
            weight=-5.0e-3,  # -1.0e-2 / 2
            params={},
        ),
        "penalty_energy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_energy",
            weight=-1.0e-3,  # -2.0e-3 / 2 (power)
            params={},
        ),
        "penalty_lin_vel_z": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_lin_vel_z",
            weight=-2.0,
            params={},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-0.2,
            params={},
        ),
        "penalty_dof_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_dof_vel",
            weight=-1.0e-4,
            params={},
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-0.1,  # Reduced from -0.5 to encourage more movement
            params={},
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:limits_dof_pos",
            weight=-1.0,
            params={"soft_dof_pos_limit": 0.95},
        ),
        "penalty_feet_slip": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_slip",
            weight=-0.1,
            params={},
        ),
        "penalty_feet_roll": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_roll",
            weight=-1.0,  # -0.1 * 10
            params={},
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,  # -1.0 * 10 (feet_distance)
            params={"close_feet_threshold": 0.2},
        ),
    },
)
