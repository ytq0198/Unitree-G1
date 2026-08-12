"""Fixed-capacity replay for policy AMP transitions."""

from __future__ import annotations

import torch

from .state import AMP_TRANSITION_DIM


class ReplayBuffer:
  """A device-local ring buffer with uniform sampling."""

  def __init__(
    self,
    capacity: int = 50_000,
    feature_dim: int = AMP_TRANSITION_DIM,
    device: str | torch.device = "cpu",
  ) -> None:
    if capacity <= 0 or feature_dim <= 0:
      raise ValueError("Replay capacity and feature_dim must be positive")
    self.capacity = capacity
    self.feature_dim = feature_dim
    self.device = torch.device(device)
    self._data = torch.empty(capacity, feature_dim, device=self.device)
    self._size = 0
    self._cursor = 0

  def __len__(self) -> int:
    return self._size

  def add(self, samples: torch.Tensor) -> None:
    samples = samples.detach().to(device=self.device, dtype=torch.float32)
    if samples.ndim != 2 or samples.shape[1] != self.feature_dim:
      raise ValueError(
        f"Replay samples must be [N, {self.feature_dim}], got {tuple(samples.shape)}"
      )
    if not torch.isfinite(samples).all():
      raise ValueError("Replay samples contain NaN or Inf")
    if samples.shape[0] >= self.capacity:
      samples = samples[-self.capacity :]
    count = samples.shape[0]
    first = min(count, self.capacity - self._cursor)
    self._data[self._cursor : self._cursor + first].copy_(samples[:first])
    remaining = count - first
    if remaining:
      self._data[:remaining].copy_(samples[first:])
    self._cursor = (self._cursor + count) % self.capacity
    self._size = min(self.capacity, self._size + count)

  def sample(self, batch_size: int) -> torch.Tensor:
    if self._size == 0:
      raise RuntimeError("Cannot sample an empty replay buffer")
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    indices = torch.randint(self._size, (batch_size,), device=self.device)
    return self._data[indices]
