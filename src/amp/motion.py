"""50 Hz expert motion loading with clip-safe transition sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.student_api import load_student_function

from .state import AMP_STATE_DIM, build_state_with


class MotionDataset:
  """Load converted G1 motion components and build student-ordered AMP states."""

  REQUIRED_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "pelvis_height",
    "projected_gravity",
    "base_lin_vel_yaw",
    "base_ang_vel_yaw",
    "key_body_pos_pelvis",
    "clip_id",
  )

  def __init__(
    self,
    path: str | Path,
    student_path: str | Path,
    device: str | torch.device = "cpu",
  ) -> None:
    self.path = Path(path)
    if not self.path.is_file():
      raise FileNotFoundError(f"AMP motion file does not exist: {self.path}")
    with np.load(self.path, allow_pickle=False) as data:
      missing = sorted(set(self.REQUIRED_KEYS) - set(data.files))
      if missing:
        raise ValueError(f"Motion file is missing keys: {missing}")
      fps = float(np.asarray(data["fps"]).item())
      if fps != 50.0:
        raise ValueError(f"AMP motion must be 50 Hz, got {fps:g} Hz")
      tensors = {
        key: torch.as_tensor(np.asarray(data[key]), dtype=torch.float32, device=device)
        for key in self.REQUIRED_KEYS
        if key not in {"fps", "clip_id"}
      }
      self.clip_id = torch.as_tensor(
        np.asarray(data["clip_id"]), dtype=torch.long, device=device
      )

    builder = load_student_function(student_path, "build_amp_state")
    self.states = build_state_with(builder=builder, **tensors)
    if self.states.ndim != 2 or self.states.shape[1] != AMP_STATE_DIM:
      raise ValueError(f"Motion AMP states must be [T, 83], got {self.states.shape}")
    if self.clip_id.shape != (self.states.shape[0],):
      raise ValueError("clip_id must contain one entry per motion frame")
    self.valid_start = torch.nonzero(
      self.clip_id[:-1] == self.clip_id[1:], as_tuple=False
    ).flatten()
    if self.valid_start.numel() == 0:
      raise ValueError("Motion file contains no within-clip adjacent transitions")

  def __len__(self) -> int:
    return int(self.states.shape[0])

  def sample_transitions(self, batch_size: int) -> torch.Tensor:
    """Sample adjacent states without crossing a clip boundary."""
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    choice = torch.randint(
      self.valid_start.numel(),
      (batch_size,),
      device=self.valid_start.device,
    )
    start = self.valid_start[choice]
    return torch.cat((self.states[start], self.states[start + 1]), dim=-1)
