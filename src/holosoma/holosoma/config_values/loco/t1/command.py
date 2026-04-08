"""Locomotion command presets for the T1 robot."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg

t1_29dof_command = CommandManagerCfg(
    params={
        "locomotion_command_resampling_time": 10.0,
    },
    setup_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
            params={
                # 1.5 Hz gait (period=0.667s) matches MuJoCo Playground default.
                # Randomization width of 0.25 Hz covers MuJoCo's [1.25, 1.75] Hz range.
                "gait_period": 0.6667,
                "gait_period_randomization_width": 0.25,
                "randomize_phase": True,
            },
        ),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
            params={
                "command_ranges": {
                    "lin_vel_x": [-1.0, 1.0],
                    "lin_vel_y": [-1.0, 1.0],
                    "ang_vel_yaw": [-1.0, 1.0],
                    "heading": [-3.14, 3.14],
                },
                "stand_prob": 0.2,
            },
        ),
    },
    reset_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
        ),
    },
    step_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
        ),
    },
)

__all__ = ["t1_29dof_command"]