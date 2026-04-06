"""FastTD3 agent module for holosoma.

This module provides FastTD3 training for locomotion, compatible with
MuJoCo Playground checkpoints for cross-simulator policy transfer.
"""

from holosoma.agents.fast_td3.fast_td3 import Actor, Critic
from holosoma.agents.fast_td3.fast_td3_env import FastTD3Env, create_fasttd3_env
from holosoma.agents.fast_td3.fast_td3_obs import (
    FASTTD3_POLICY_DOF_NAMES_20,
    FastTD3ObservationBuilder,
    GaitPhaseTracker,
)
from holosoma.agents.fast_td3.fast_td3_utils import (
    EmpiricalNormalization,
    SimpleReplayBuffer,
    load_policy,
    save_params,
)

__all__ = [
    # Networks
    "Actor",
    "Critic",
    # Environment
    "FastTD3Env",
    "create_fasttd3_env",
    # Observation
    "FastTD3ObservationBuilder",
    "GaitPhaseTracker",
    "FASTTD3_POLICY_DOF_NAMES_20",
    # Utilities
    "EmpiricalNormalization",
    "SimpleReplayBuffer",
    "load_policy",
    "save_params",
]
