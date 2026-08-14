"""Next-waypoint command that exposes no global route state to the actor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from src.navigation_math import forward_yaw_command, route_position_m
from src.student_api import load_student_function


class WaypointCommand(CommandTerm):
  cfg: "WaypointCommandCfg"

  def __init__(self, cfg: "WaypointCommandCfg", env) -> None:
    super().__init__(cfg, env)
    if not cfg.routes or any(len(route) < 2 for route in cfg.routes):
      raise ValueError("Every navigation route must contain at least two waypoints")
    self.robot: Entity = env.scene[cfg.entity_name]
    max_waypoints = max(len(route) for route in cfg.routes)
    self.route_offsets = torch.zeros(
      len(cfg.routes), max_waypoints, 2, device=self.device
    )
    self.cumulative_length = torch.zeros(
      len(cfg.routes), max_waypoints, device=self.device
    )
    self.final_index = torch.empty(len(cfg.routes), dtype=torch.long, device=self.device)
    for index, route_points in enumerate(cfg.routes):
      route = torch.tensor(route_points, dtype=torch.float32, device=self.device)
      offsets = route - route[:1]
      count = len(route_points)
      self.route_offsets[index, :count] = offsets
      self.route_offsets[index, count:] = offsets[-1]
      segment_lengths = torch.norm(offsets[1:] - offsets[:-1], dim=-1)
      cumulative = torch.cat(
        (torch.zeros(1, device=self.device), torch.cumsum(segment_lengths, dim=0))
      )
      self.cumulative_length[index, :count] = cumulative
      self.cumulative_length[index, count:] = cumulative[-1]
      self.final_index[index] = count - 1
    terrain = env.scene.terrain
    if len(cfg.routes) == 1:
      self.scene_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    else:
      if terrain is None or terrain.terrain_types is None:
        raise ValueError("Multiple routes require procedural terrain columns")
      self.scene_index = terrain.terrain_types.clone()
      if int(self.scene_index.max()) >= len(cfg.routes):
        raise ValueError("Terrain columns and navigation routes are misaligned")
    self.total_length = self.cumulative_length[
      self.scene_index, self.final_index[self.scene_index]
    ]
    self.route_index = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
    self.progress = torch.zeros(self.num_envs, device=self.device)
    self.path_position_m = torch.zeros(self.num_envs, device=self.device)
    self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.waypoint_reached = torch.zeros_like(self.success)
    self._body_frame = load_student_function(cfg.student_path, "waypoint_in_body_frame")
    self.metrics["route_progress"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["route_success"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def current_waypoint_w(self) -> torch.Tensor:
    offset = self.route_offsets[self.scene_index, self.route_index]
    target = self._env.scene.env_origins.clone()
    target[:, :2] += offset
    return target

  @property
  def command(self) -> torch.Tensor:
    displacement = self._body_frame(
      self.robot.data.root_link_pos_w,
      self.robot.data.root_link_quat_w,
      self.current_waypoint_w,
    )
    expected = (self.num_envs, 2)
    if tuple(displacement.shape) != expected or not torch.isfinite(displacement).all():
      raise ValueError("waypoint_in_body_frame() must return a finite [B, 2] tensor")
    distance = torch.norm(displacement, dim=-1, keepdim=True)
    if self.cfg.command_mode == "xy":
      scale = torch.clamp(
        self.cfg.max_command_speed / distance.clamp_min(1.0e-6), max=1.0
      )
      return displacement * scale
    return forward_yaw_command(
      displacement,
      max_command_speed=self.cfg.max_command_speed,
      min_turning_speed=self.cfg.min_turning_speed,
      heading_stiffness=self.cfg.heading_stiffness,
      max_yaw_rate=self.cfg.max_yaw_rate,
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.route_index[env_ids] = 1
    self.progress[env_ids] = 0.0
    self.path_position_m[env_ids] = 0.0
    self.success[env_ids] = False
    self.waypoint_reached[env_ids] = False

  def _update_command(self) -> None:
    robot_xy = self.robot.data.root_link_pos_w[:, :2]
    target_xy = self.current_waypoint_w[:, :2]
    distance = torch.norm(target_xy - robot_xy, dim=-1)
    reached = (distance <= self.cfg.waypoint_threshold) & ~self.success
    self.waypoint_reached = reached
    final_index = self.final_index[self.scene_index]
    at_final = reached & (self.route_index == final_index)
    self.success |= at_final
    advance = reached & (self.route_index < final_index)
    self.route_index[advance] += 1

    self.path_position_m = route_position_m(
      robot_xy,
      self._env.scene.env_origins[:, :2],
      self.route_offsets,
      self.route_index,
      self.cumulative_length,
      self.scene_index,
    )
    current_progress = torch.clamp(
      self.path_position_m / self.total_length, 0.0, 1.0
    )
    current_progress[self.success] = 1.0
    self.progress = torch.maximum(self.progress, current_progress)

  def _update_metrics(self) -> None:
    self.metrics["route_progress"] = self.progress
    self.metrics["route_success"] = self.success.float()


@dataclass(kw_only=True)
class WaypointCommandCfg(CommandTermCfg):
  entity_name: str
  routes: tuple[tuple[tuple[float, float], ...], ...]
  student_path: str
  waypoint_threshold: float = 0.9
  max_command_speed: float = 0.6
  max_yaw_rate: float = 0.25
  min_turning_speed: float = 0.1
  heading_stiffness: float = 0.5
  command_mode: Literal["xy", "forward_yaw"] = "xy"

  def build(self, env) -> WaypointCommand:
    return WaypointCommand(self, env)
