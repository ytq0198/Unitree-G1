"""Pure geometry helpers shared by navigation commands and rewards."""

from __future__ import annotations

import torch


def route_position_m(
  robot_xy: torch.Tensor,
  env_origins_xy: torch.Tensor,
  route_offsets: torch.Tensor,
  route_index: torch.Tensor,
  cumulative_length: torch.Tensor,
  scene_index: torch.Tensor | None = None,
) -> torch.Tensor:
  """Project each robot onto its active route segment in path-length units."""
  previous_index = torch.clamp(route_index - 1, min=0)
  if scene_index is None:
    segment_start = route_offsets[previous_index]
    segment_end = route_offsets[route_index]
    cumulative_start = cumulative_length[previous_index]
    cumulative_end = cumulative_length[route_index]
  else:
    segment_start = route_offsets[scene_index, previous_index]
    segment_end = route_offsets[scene_index, route_index]
    cumulative_start = cumulative_length[scene_index, previous_index]
    cumulative_end = cumulative_length[scene_index, route_index]
  segment = segment_end - segment_start
  segment_length_sq = torch.sum(torch.square(segment), dim=-1).clamp_min(1.0e-8)
  relative = robot_xy - (env_origins_xy + segment_start)
  fraction = torch.sum(relative * segment, dim=-1) / segment_length_sq
  fraction = torch.clamp(fraction, 0.0, 1.0)
  segment_length = cumulative_end - cumulative_start
  return cumulative_start + fraction * segment_length
