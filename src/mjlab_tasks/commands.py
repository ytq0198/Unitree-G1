"""Next-waypoint command that exposes no global route state to the actor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from src.navigation_math import route_position_m
from src.student_api import load_student_function


class WaypointCommand(CommandTerm):
  cfg: "WaypointCommandCfg"

  def __init__(self, cfg: "WaypointCommandCfg", env) -> None:
    super().__init__(cfg, env)
    if len(cfg.route) < 2:
      raise ValueError("Navigation route must contain at least two waypoints")
    self.robot: Entity = env.scene[cfg.entity_name]
    route = torch.tensor(cfg.route, dtype=torch.float32, device=self.device)
    self.route_offsets = route - route[:1]
    segment_lengths = torch.norm(
      self.route_offsets[1:] - self.route_offsets[:-1], dim=-1
    )
    self.cumulative_length = torch.cat(
      (torch.zeros(1, device=self.device), torch.cumsum(segment_lengths, dim=0))
    )
    self.total_length = float(self.cumulative_length[-1])
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
    offset = self.route_offsets[self.route_index]
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
    heading_error = torch.atan2(displacement[:, 1], displacement[:, 0])
    forward = torch.clamp(distance.squeeze(-1), max=self.cfg.max_command_speed)
    forward_scale = torch.clamp(torch.cos(heading_error), min=0.0)
    forward = torch.maximum(
      forward * forward_scale,
      torch.full_like(forward, self.cfg.min_turning_speed),
    )
    yaw_rate = torch.clamp(
      self.cfg.heading_stiffness * heading_error,
      -self.cfg.max_yaw_rate,
      self.cfg.max_yaw_rate,
    )
    return torch.stack((forward, yaw_rate), dim=-1)

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
    final_index = len(self.cfg.route) - 1
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
  route: tuple[tuple[float, float], ...]
  student_path: str
  waypoint_threshold: float = 0.45
  max_command_speed: float = 0.6
  max_yaw_rate: float = 0.25
  min_turning_speed: float = 0.1
  heading_stiffness: float = 0.5
  command_mode: Literal["xy", "forward_yaw"] = "xy"

  def build(self, env) -> WaypointCommand:
    return WaypointCommand(self, env)
