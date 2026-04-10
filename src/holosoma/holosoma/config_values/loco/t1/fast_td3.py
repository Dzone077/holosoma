"""FastTD3 experiment configuration for T1 robot locomotion.

This module provides experiment configurations optimized for FastTD3 training,
matching MuJoCo Playground settings for checkpoint compatibility.
"""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.config_types.video import VideoConfig
from holosoma.config_types.simulator import (
    PhysxConfig,
    SimEngineConfig,
    SimulatorConfig,
    SimulatorInitConfig,
)
from holosoma.config_values import (
    action,
    algo,
    command,
    observation,
    randomization,
    robot,
    simulator,
    termination,
    terrain,
)
from holosoma.config_values.loco.t1.curriculum import t1_29dof_curriculum_fasttd3
from holosoma.config_values.loco.t1.reward import t1_20dof_fast_td3


# Logger config with video recording enabled
# Record every 2000 episodes (~100,000 training steps with ~50 step avg episode length)
# Adjust the interval based on your actual average episode length:
#   desired_step_interval / avg_steps_per_episode = interval_value
logger_with_video = DisabledLoggerConfig(
    headless_recording=True,
    base_dir="logs",
#    video=VideoConfig(
#        enabled=True,
#        interval=2000,  # Adjust this value based on your needs
#        width=640,
#        height=360,
#        record_env_id=0,
#    ),
)


# IsaacSim simulator config tuned for FastTD3 (500Hz physics, 50Hz control)
isaacsim_fasttd3 = SimulatorConfig(
    _target_="holosoma.simulator.isaacsim.isaacsim.IsaacSim",
    _recursive_=False,
    config=SimulatorInitConfig(
        name="isaacsim",
        sim=SimEngineConfig(
            fps=500,  # 500Hz physics like MuJoCo Playground
            control_decimation=10,  # 50Hz control (500/10)
            substeps=1,
            physx=PhysxConfig(
                solver_type=1,
                num_position_iterations=16,  # Higher for stability
                num_velocity_iterations=8,
                bounce_threshold_velocity=0.2,
            ),
            render_mode="human",
            render_interval=10,
        ),
        contact_sensor_history_length=3,
    ),
)

# IsaacGym simulator config tuned for FastTD3
isaacgym_fasttd3 = SimulatorConfig(
    _target_="holosoma.simulator.isaacgym.isaacgym.IsaacGym",
    _recursive_=False,
    config=SimulatorInitConfig(
        name="isaacgym",
        sim=SimEngineConfig(
            fps=500,  # 500Hz physics
            control_decimation=10,  # 50Hz control
            substeps=1,
            physx=PhysxConfig(
                solver_type=1,
                num_position_iterations=16,
                num_velocity_iterations=8,
                bounce_threshold_velocity=0.2,
            ),
        ),
        contact_sensor_history_length=3,
    ),
)


# T1 locomotion FastTD3 experiment (for IsaacSim)
t1_loco_fasttd3 = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(
        project="holosoma-fasttd3",
        name="t1_20dof_fasttd3",
        num_envs=1024,
        headless=True,
    ),
    # Note: FastTD3 has its own training loop, so algo is not actively used
    # but ExperimentConfig requires it, so we provide a placeholder
    algo=algo.ppo,
    simulator=isaacsim_fasttd3,
    robot=robot.t1_29dof_waist_wrist,
    terrain=terrain.terrain_locomotion_plane,  # Flat + stairs terrain
    observation=observation.t1_29dof_loco_single_wolinvel,
    action=action.t1_29dof_joint_pos,
    termination=termination.t1_29dof_termination,
    randomization=randomization.t1_29dof_randomization,
    command=command.t1_29dof_command,
    curriculum=t1_29dof_curriculum_fasttd3,  # Minimal curriculum (tracking only)
    reward=t1_20dof_fast_td3,
    logger=logger_with_video,  # Enable video recording
)


# T1 locomotion FastTD3 experiment (for IsaacGym)
t1_loco_fasttd3_isaacgym = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(
        project="holosoma-fasttd3",
        name="t1_20dof_fasttd3_isaacgym",
        num_envs=1024,
        headless=True,
    ),
    algo=algo.ppo,  # Placeholder, FastTD3 uses its own training loop
    simulator=isaacgym_fasttd3,
    robot=robot.t1_29dof_waist_wrist,
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.t1_29dof_loco_single_wolinvel,
    action=action.t1_29dof_joint_pos,
    termination=termination.t1_29dof_termination,
    randomization=randomization.t1_29dof_randomization,
    command=command.t1_29dof_command,
    curriculum=t1_29dof_curriculum_fasttd3,
    reward=t1_20dof_fast_td3,
    logger=logger_with_video,  # Enable video recording
)


__all__ = [
    "t1_loco_fasttd3",
    "t1_loco_fasttd3_isaacgym",
    "isaacsim_fasttd3",
    "isaacgym_fasttd3",
]
