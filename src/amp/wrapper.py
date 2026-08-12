"""Manual-reset RSL-RL wrapper preserving true terminal AMP observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from mjlab.rl import RslRlVecEnvWrapper
from tensordict import TensorDict


def _tensor_observations(observations: Mapping[str, Any]) -> dict[str, torch.Tensor]:
  result: dict[str, torch.Tensor] = {}
  for name, value in observations.items():
    if not isinstance(value, torch.Tensor):
      raise TypeError(f"Observation group {name!r} must be concatenated")
    result[name] = value
  return result


class ManualResetAmpVecEnvWrapper(RslRlVecEnvWrapper):
  """Reset only done worlds after saving their terminal AMP state."""

  def __init__(self, env, clip_actions: float | None = None) -> None:
    if env.cfg.auto_reset:
      raise ValueError("AMP training requires env.cfg.auto_reset=False")
    super().__init__(env, clip_actions=clip_actions)

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    obs_dict, reward, terminated, truncated, env_extras = self.env.step(actions)
    done = terminated | truncated
    tensor_obs = _tensor_observations(obs_dict)
    terminal_amp = tensor_obs["amp"].detach().clone()
    extras = dict(env_extras)
    done_ids = done.nonzero(as_tuple=False).squeeze(-1)
    if done_ids.numel() > 0:
      reset_obs, reset_extras = self.env.reset(env_ids=done_ids)
      tensor_obs = _tensor_observations(reset_obs)
      if reset_extras.get("log"):
        extras.setdefault("log", {}).update(reset_extras["log"])
    extras["terminal_amp_observation"] = terminal_amp
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    batched_obs = TensorDict(batch_size=[self.num_envs])
    for name, value in tensor_obs.items():
      batched_obs.set(name, value)
    return (
      batched_obs,
      reward,
      done.to(dtype=torch.long),
      extras,
    )
