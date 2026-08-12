"""AMP state and transition shape contracts."""

from __future__ import annotations

from collections.abc import Callable

import torch

AMP_STATE_DIM = 83
AMP_TRANSITION_DIM = 166
NUM_G1_JOINTS = 29
KEY_BODY_NAMES = (
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "torso_link",
)


def build_state_with(
  builder: Callable[..., torch.Tensor],
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  pelvis_height: torch.Tensor,
  projected_gravity: torch.Tensor,
  base_lin_vel_yaw: torch.Tensor,
  base_ang_vel_yaw: torch.Tensor,
  key_body_pos_pelvis: torch.Tensor,
) -> torch.Tensor:
  """Call a student builder and enforce the public 83-D contract."""
  _validate_components(
    joint_pos,
    joint_vel,
    pelvis_height,
    projected_gravity,
    base_lin_vel_yaw,
    base_ang_vel_yaw,
    key_body_pos_pelvis,
  )
  state = builder(
    joint_pos,
    joint_vel,
    pelvis_height,
    projected_gravity,
    base_lin_vel_yaw,
    base_ang_vel_yaw,
    key_body_pos_pelvis,
  )
  expected = (*joint_pos.shape[:-1], AMP_STATE_DIM)
  if tuple(state.shape) != expected:
    raise ValueError(
      f"build_amp_state() returned {tuple(state.shape)}; expected {expected}"
    )
  if not torch.isfinite(state).all():
    raise ValueError("build_amp_state() returned NaN or Inf")
  return state


def build_transition(state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
  """Concatenate adjacent 83-D states into a finite 166-D transition."""
  if state.shape != next_state.shape or state.shape[-1] != AMP_STATE_DIM:
    raise ValueError(
      "AMP transition inputs must have identical [..., 83] shapes; "
      f"got {tuple(state.shape)} and {tuple(next_state.shape)}"
    )
  transition = torch.cat((state, next_state), dim=-1)
  if not torch.isfinite(transition).all():
    raise ValueError("AMP transition contains NaN or Inf")
  return transition


def _validate_components(
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  pelvis_height: torch.Tensor,
  projected_gravity: torch.Tensor,
  base_lin_vel_yaw: torch.Tensor,
  base_ang_vel_yaw: torch.Tensor,
  key_body_pos_pelvis: torch.Tensor,
) -> None:
  batch = joint_pos.shape[:-1]
  expected = {
    "joint_pos": (*batch, NUM_G1_JOINTS),
    "joint_vel": (*batch, NUM_G1_JOINTS),
    "pelvis_height": (*batch, 1),
    "projected_gravity": (*batch, 3),
    "base_lin_vel_yaw": (*batch, 3),
    "base_ang_vel_yaw": (*batch, 3),
    "key_body_pos_pelvis": (*batch, len(KEY_BODY_NAMES), 3),
  }
  actual = {
    "joint_pos": tuple(joint_pos.shape),
    "joint_vel": tuple(joint_vel.shape),
    "pelvis_height": tuple(pelvis_height.shape),
    "projected_gravity": tuple(projected_gravity.shape),
    "base_lin_vel_yaw": tuple(base_lin_vel_yaw.shape),
    "base_ang_vel_yaw": tuple(base_ang_vel_yaw.shape),
    "key_body_pos_pelvis": tuple(key_body_pos_pelvis.shape),
  }
  for name, shape in expected.items():
    if actual[name] != shape:
      raise ValueError(f"{name} has shape {actual[name]}; expected {shape}")
