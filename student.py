"""Student formulas for the Course Project.

Edit only this file.  The task already provides G1 simulation, AMP-PPO, terrain,
route management, training, evaluation, video, and submission packaging.
"""

from __future__ import annotations

import torch


def build_amp_state(
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  pelvis_height: torch.Tensor,
  projected_gravity: torch.Tensor,
  base_lin_vel_yaw: torch.Tensor,
  base_ang_vel_yaw: torch.Tensor,
  key_body_pos_pelvis: torch.Tensor,
) -> torch.Tensor:
  """Build the 83-D AMP state used in Experiment 07, adapted here locally.

  Concatenate ``29 + 29 + 1 + 3 + 3 + 3 + 5*3`` features in that exact order.
  Do not import Experiment 07.  Return shape: ``[..., 83]``.
  """
  key_body_positions = key_body_pos_pelvis.flatten(start_dim=-2)
  return torch.cat(
    (
      joint_pos,
      joint_vel,
      pelvis_height,
      projected_gravity,
      base_lin_vel_yaw,
      base_ang_vel_yaw,
      key_body_positions,
    ),
    dim=-1,
  )


def normalize_depth(depth_m: torch.Tensor) -> torch.Tensor:
  """Clip metric depth to ``[0.1, 5.0]`` and map it linearly to ``[0, 1]``.

  Input and output shape: ``[B, 1, 60, 80]``.
  """
  clipped_depth = torch.clamp(depth_m, min=0.1, max=5.0)
  return (clipped_depth - 0.1) / (5.0 - 0.1)


def waypoint_in_body_frame(
  base_position_w: torch.Tensor,
  base_quaternion_w: torch.Tensor,
  waypoint_position_w: torch.Tensor,
) -> torch.Tensor:
  """Return the next waypoint's planar displacement in the yaw-only body frame.

  Compute ``delta_w = waypoint_w - base_w`` and rotate ``delta_w[..., :2]`` by
  negative base yaw. Inputs are ``[B, 3]``, ``[B, 4]`` in ``wxyz`` order, and
  ``[B, 3]``. Return shape: ``[B, 2]``.
  """
  w, x, y, z = base_quaternion_w.unbind(dim=-1)
  yaw = torch.atan2(
    2.0 * (w * z + x * y),
    1.0 - 2.0 * (y.square() + z.square()),
  )
  delta_xy = waypoint_position_w[..., :2] - base_position_w[..., :2]
  cos_yaw = torch.cos(yaw)
  sin_yaw = torch.sin(yaw)
  body_x = cos_yaw * delta_xy[..., 0] + sin_yaw * delta_xy[..., 1]
  body_y = -sin_yaw * delta_xy[..., 0] + cos_yaw * delta_xy[..., 1]
  return torch.stack((body_x, body_y), dim=-1)


def navigation_reward(
  progress_m: torch.Tensor,
  waypoint_reached: torch.Tensor,
  route_success: torch.Tensor,
) -> torch.Tensor:
  """Return the per-environment navigation task reward.

  Use ``4 * progress_m + 0.5 * waypoint_reached + 5 * route_success`` after
  converting the boolean indicators to the progress dtype. Return shape: ``[B]``.
  """
  dtype = progress_m.dtype
  return (
    4.0 * progress_m
    + 0.5 * waypoint_reached.to(dtype=dtype)
    + 5.0 * route_success.to(dtype=dtype)
  )


def smoothness_penalty(
  action: torch.Tensor,
  previous_action: torch.Tensor,
  previous_previous_action: torch.Tensor,
) -> torch.Tensor:
  """Return ``mean(abs(a_t - 2 a_(t-1) + a_(t-2)), dim=-1)``.

  Inputs are ``[B, 29]`` and the return shape is ``[B]``.
  """
  second_difference = action - 2.0 * previous_action + previous_previous_action
  return torch.mean(torch.abs(second_difference), dim=-1)
