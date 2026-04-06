"""FastTD3 training utilities.

Contains replay buffer, observation normalization, and checkpoint save/load functions.
These utilities are copied from FastTD3-main to ensure checkpoint compatibility.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist

from tensordict import TensorDict


class SimpleReplayBuffer(nn.Module):
    """Simple replay buffer that stores transitions in a circular buffer.

    Supports n-step returns and asymmetric observations.
    """

    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int = 0,
        asymmetric_obs: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        device: torch.device | str = None,
    ):
        """Initialize replay buffer.

        Args:
            n_env: Number of parallel environments.
            buffer_size: Maximum size of buffer per environment.
            n_obs: Observation dimension.
            n_act: Action dimension.
            n_critic_obs: Critic observation dimension (if asymmetric).
            asymmetric_obs: Whether to store separate critic observations.
            n_steps: Number of steps for multi-step returns.
            gamma: Discount factor for multi-step returns.
            device: Torch device.
        """
        super().__init__()

        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.asymmetric_obs = asymmetric_obs
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        # Allocate buffers
        self.observations = torch.zeros(
            (n_env, buffer_size, n_obs), device=device, dtype=torch.float
        )
        self.actions = torch.zeros(
            (n_env, buffer_size, n_act), device=device, dtype=torch.float
        )
        self.rewards = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.float
        )
        self.dones = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.truncations = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.long
        )
        self.next_observations = torch.zeros(
            (n_env, buffer_size, n_obs), device=device, dtype=torch.float
        )

        if asymmetric_obs:
            self.critic_observations = torch.zeros(
                (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
            )
            self.next_critic_observations = torch.zeros(
                (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
            )

        self.ptr = 0

    @torch.no_grad()
    def extend(self, tensor_dict: TensorDict):
        """Add a batch of transitions to the buffer."""
        observations = tensor_dict["observations"]
        actions = tensor_dict["actions"]
        rewards = tensor_dict["next"]["rewards"]
        dones = tensor_dict["next"]["dones"]
        truncations = tensor_dict["next"]["truncations"]
        next_observations = tensor_dict["next"]["observations"]

        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = observations
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.truncations[:, ptr] = truncations
        self.next_observations[:, ptr] = next_observations

        if self.asymmetric_obs:
            critic_observations = tensor_dict["critic_observations"]
            next_critic_observations = tensor_dict["next"]["critic_observations"]
            self.critic_observations[:, ptr] = critic_observations
            self.next_critic_observations[:, ptr] = next_critic_observations

        self.ptr += 1

    @torch.no_grad()
    def sample(self, batch_size: int) -> TensorDict:
        """Sample a batch of transitions from the buffer.

        Returns:
            TensorDict with sampled transitions.
        """
        # Sample indices
        indices = torch.randint(
            0,
            min(self.buffer_size, self.ptr),
            (self.n_env, batch_size),
            device=self.device,
        )

        # Gather observations
        obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
        act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

        observations = torch.gather(self.observations, 1, obs_indices).reshape(
            self.n_env * batch_size, self.n_obs
        )
        next_observations = torch.gather(
            self.next_observations, 1, obs_indices
        ).reshape(self.n_env * batch_size, self.n_obs)
        actions = torch.gather(self.actions, 1, act_indices).reshape(
            self.n_env * batch_size, self.n_act
        )
        rewards = torch.gather(self.rewards, 1, indices).reshape(
            self.n_env * batch_size
        )
        dones = torch.gather(self.dones, 1, indices).reshape(self.n_env * batch_size)
        truncations = torch.gather(self.truncations, 1, indices).reshape(
            self.n_env * batch_size
        )
        effective_n_steps = torch.ones_like(dones)

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                    "effective_n_steps": effective_n_steps,
                },
            },
            batch_size=self.n_env * batch_size,
        )

        if self.asymmetric_obs:
            critic_obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)
            critic_observations = torch.gather(
                self.critic_observations, 1, critic_obs_indices
            ).reshape(self.n_env * batch_size, self.n_critic_obs)
            next_critic_observations = torch.gather(
                self.next_critic_observations, 1, critic_obs_indices
            ).reshape(self.n_env * batch_size, self.n_critic_obs)
            out["critic_observations"] = critic_observations
            out["next"]["critic_observations"] = next_critic_observations

        return out


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values.

    This is used for observation normalization during training.
    """

    def __init__(
        self,
        shape: int | tuple,
        device: torch.device | str,
        eps: float = 1e-2,
        until: int | None = None,
    ):
        """Initialize normalizer.

        Args:
            shape: Shape of input values (excluding batch dimension).
            device: Torch device.
            eps: Small value for numerical stability.
            until: If specified, stops updating after this many samples.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device

        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, center: bool = True, update: bool = True
    ) -> torch.Tensor:
        """Normalize input tensor.

        Args:
            x: Input tensor to normalize.
            center: If True, subtract mean.
            update: If True and in training mode, update running statistics.

        Returns:
            Normalized tensor.
        """
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(
                f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}"
            )

        if self.training and update:
            self.update(x)

        if center:
            return (x - self._mean) / (self._std + self.eps)
        else:
            return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor):
        """Update running statistics with new batch."""
        if self.until is not None and self.count >= self.until:
            return

        if dist.is_available() and dist.is_initialized():
            # Multi-GPU: synchronize statistics
            local_batch_size = x.shape[0]
            world_size = dist.get_world_size()
            global_batch_size = world_size * local_batch_size

            x_shifted = x - self._mean
            local_sum_shifted = torch.sum(x_shifted, dim=0, keepdim=True)
            local_sum_sq_shifted = torch.sum(x_shifted.pow(2), dim=0, keepdim=True)

            stats_to_sync = torch.cat([local_sum_shifted, local_sum_sq_shifted], dim=0)
            dist.all_reduce(stats_to_sync, op=dist.ReduceOp.SUM)
            global_sum_shifted, global_sum_sq_shifted = stats_to_sync

            batch_mean_shifted = global_sum_shifted / global_batch_size
            batch_var = (
                global_sum_sq_shifted / global_batch_size - batch_mean_shifted.pow(2)
            )
            batch_mean = batch_mean_shifted + self._mean
        else:
            global_batch_size = x.shape[0]
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

        new_count = self.count + global_batch_size

        # Update mean (Welford's online algorithm)
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (global_batch_size / new_count))

        # Update variance
        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * global_batch_size
        M2 = m_a + m_b + delta2.pow(2) * (self.count * global_batch_size / new_count)
        self._var.copy_(M2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse normalization."""
        return y * (self._std + self.eps) + self._mean


def cpu_state(sd: dict) -> dict:
    """Move state dict to CPU without blocking compute stream."""
    return {k: v.detach().to("cpu", non_blocking=True) for k, v in sd.items()}


def get_ddp_state_dict(model: nn.Module) -> dict:
    """Get state dict from model, handling DDP wrapper if present."""
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def load_ddp_state_dict(model: nn.Module, state_dict: dict):
    """Load state dict into model, handling DDP wrapper if present."""
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def save_params(
    global_step: int,
    actor: nn.Module,
    qnet: nn.Module,
    qnet_target: nn.Module,
    obs_normalizer: nn.Module,
    critic_obs_normalizer: nn.Module | None,
    args: Any,
    save_path: str,
):
    """Save model parameters and training configuration to disk.

    Saves in the MuJoCo Playground compatible format.

    Args:
        global_step: Current training step.
        actor: Actor network.
        qnet: Q-network.
        qnet_target: Target Q-network.
        obs_normalizer: Observation normalizer.
        critic_obs_normalizer: Critic observation normalizer (can be None).
        args: Training arguments (dataclass or namespace).
        save_path: Path to save checkpoint.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert args to dict if needed
    if hasattr(args, "__dict__"):
        args_dict = vars(args) if not hasattr(args, "_asdict") else args._asdict()
    else:
        args_dict = dict(args)

    save_dict = {
        "actor_state_dict": cpu_state(get_ddp_state_dict(actor)),
        "qnet_state_dict": cpu_state(get_ddp_state_dict(qnet)),
        "qnet_target_state_dict": cpu_state(get_ddp_state_dict(qnet_target)),
        "obs_normalizer_state": (
            cpu_state(obs_normalizer.state_dict())
            if hasattr(obs_normalizer, "state_dict")
            else None
        ),
        "critic_obs_normalizer_state": (
            cpu_state(critic_obs_normalizer.state_dict())
            if critic_obs_normalizer is not None
            and hasattr(critic_obs_normalizer, "state_dict")
            else None
        ),
        "args": args_dict,
        "global_step": global_step,
    }
    torch.save(save_dict, save_path, _use_new_zipfile_serialization=True)
    print(f"Saved checkpoint to {save_path}")


def load_policy(checkpoint_path: str, device: str = "cuda"):
    """Load a trained policy from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Device to load model on.

    Returns:
        Policy wrapper with .act() method for inference.
    """
    from holosoma.agents.fast_td3.fast_td3 import Actor

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    args = checkpoint.get("args", {})
    agent = args.get("agent", "fasttd3")

    if agent != "fasttd3":
        raise ValueError(f"Agent type '{agent}' not supported. Expected 'fasttd3'.")

    # Infer network dimensions from saved weights
    n_obs = checkpoint["actor_state_dict"]["net.0.weight"].shape[-1]
    n_act = checkpoint["actor_state_dict"]["fc_mu.0.weight"].shape[0]

    # Get hidden dim from first layer
    hidden_dim = checkpoint["actor_state_dict"]["net.0.weight"].shape[0]

    # Create actor
    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=1,  # Will be overwritten
        init_scale=0.01,
        hidden_dim=hidden_dim,
        device=device,
    )

    # Load weights (excluding noise scales)
    state_dict = {
        k: v
        for k, v in checkpoint["actor_state_dict"].items()
        if k not in {"noise_scales", "std_min", "std_max"}
    }
    actor.load_state_dict(state_dict, strict=False)

    # Load observation normalizer
    obs_norm_state = checkpoint.get("obs_normalizer_state")
    if obs_norm_state:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        obs_normalizer.load_state_dict(obs_norm_state)
    else:
        obs_normalizer = nn.Identity()

    # Create policy wrapper
    class PolicyWrapper:
        def __init__(self, actor, obs_normalizer):
            self.actor = actor.to(device).eval()
            self.obs_normalizer = (
                obs_normalizer.to(device).eval()
                if not isinstance(obs_normalizer, nn.Identity)
                else obs_normalizer
            )

        def to(self, device):
            self.actor = self.actor.to(device)
            if not isinstance(self.obs_normalizer, nn.Identity):
                self.obs_normalizer = self.obs_normalizer.to(device)
            return self

        def eval(self):
            self.actor.eval()
            if not isinstance(self.obs_normalizer, nn.Identity):
                self.obs_normalizer.eval()
            return self

        @torch.no_grad()
        def act(self, obs):
            """Return action distribution with .loc attribute."""
            if not isinstance(self.obs_normalizer, nn.Identity):
                norm_obs = self.obs_normalizer(obs, update=False)
            else:
                norm_obs = obs

            action = self.actor(norm_obs)

            # Return distribution-like object
            class _Dist:
                def __init__(self, loc):
                    self.loc = loc

            return _Dist(action)

    return PolicyWrapper(actor, obs_normalizer)


@torch.no_grad()
def mark_step():
    """Call before any compiled function to mark cudagraph step."""
    torch.compiler.cudagraph_mark_step_begin()
