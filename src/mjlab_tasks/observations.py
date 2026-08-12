"""AMP, camera, navigation reward, and metric terms for the Course Project."""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from src.amp.state import KEY_BODY_NAMES, build_state_with
from src.student_api import load_student_function

from .commands import WaypointCommand


def amp_observation(
  env: ManagerBasedRlEnv,
  student_path: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  root_pos = robot.data.root_link_pos_w
  root_quat = robot.data.root_link_quat_w
  heading_quat = yaw_quat(root_quat)
  key_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids]
  relative_w = key_pos_w - root_pos.unsqueeze(1)
  heading = heading_quat.unsqueeze(1).expand(-1, len(KEY_BODY_NAMES), -1)
  relative_yaw = quat_apply_inverse(heading, relative_w)
  pelvis_height = (root_pos[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)
  builder = load_student_function(student_path, "build_amp_state")
  return build_state_with(
    builder,
    robot.data.joint_pos,
    robot.data.joint_vel,
    pelvis_height,
    robot.data.projected_gravity_b,
    quat_apply_inverse(heading_quat, robot.data.root_link_lin_vel_w),
    quat_apply_inverse(heading_quat, robot.data.root_link_ang_vel_w),
    relative_yaw,
  )


def normalized_depth(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  student_path: str,
) -> torch.Tensor:
  sensor: CameraSensor = env.scene[sensor_name]
  depth = sensor.data.depth
  if depth is None:
    raise RuntimeError(f"Camera sensor {sensor_name!r} did not produce depth")
  normalize = load_student_function(student_path, "normalize_depth")
  result = normalize(depth.permute(0, 3, 1, 2))
  expected = (env.num_envs, 1, 60, 80)
  if tuple(result.shape) != expected or not torch.isfinite(result).all():
    raise ValueError("normalize_depth() must return finite [B, 1, 60, 80] depth")
  return result


class NavigationTaskReward(ManagerTermBase):
  """Track distance reduction without leaking route state to observations."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    self.command_name = cfg.params["command_name"]
    self.step_dt = float(cfg.params["step_dt"])
    if self.step_dt <= 0.0:
      raise ValueError("navigation reward step_dt must be positive")
    self.reward_fn = load_student_function(
      cfg.params["student_path"], "navigation_reward"
    )
    self.previous_distance = torch.full(
      (env.num_envs,), float("nan"), device=env.device
    )

  def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
    command = env.command_manager.get_term(self.command_name)
    if not isinstance(command, WaypointCommand):
      raise TypeError("navigation reward requires WaypointCommand")
    robot: Entity = env.scene["robot"]
    distance = torch.norm(
      command.current_waypoint_w[:, :2] - robot.data.root_link_pos_w[:, :2],
      dim=-1,
    )
    fresh = torch.isnan(self.previous_distance)
    # RewardManager integrates rewards with dt. Express dense progress as a
    # rate so the integrated term remains the actual distance improvement.
    progress = (self.previous_distance - distance) / self.step_dt
    progress[fresh] = 0.0
    self.previous_distance = distance.detach()
    reward = self.reward_fn(
      progress,
      command.waypoint_reached,
      command.success,
    )
    if reward.shape != distance.shape or not torch.isfinite(reward).all():
      raise ValueError("navigation_reward() must return a finite [B] tensor")
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self.previous_distance[:] = float("nan")
    else:
      self.previous_distance[env_ids] = float("nan")


def student_smoothness_penalty(
  env: ManagerBasedRlEnv,
  student_path: str,
) -> torch.Tensor:
  penalty_fn = load_student_function(student_path, "smoothness_penalty")
  penalty = penalty_fn(
    env.action_manager.action,
    env.action_manager.prev_action,
    env.action_manager.prev_prev_action,
  )
  if tuple(penalty.shape) != (env.num_envs,) or not torch.isfinite(penalty).all():
    raise ValueError("smoothness_penalty() must return a finite [B] tensor")
  return penalty


def waypoint_velocity_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.5,
) -> torch.Tensor:
  """Reward stable tracking of the local velocity implied by a waypoint."""
  if std <= 0.0:
    raise ValueError("velocity tracking std must be positive")
  robot: Entity = env.scene["robot"]
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, WaypointCommand):
    raise TypeError("waypoint velocity tracking requires WaypointCommand")
  planar_error = torch.sum(
    torch.square(command.command - robot.data.root_link_lin_vel_b[:, :2]), dim=-1
  )
  vertical_error = torch.square(robot.data.root_link_lin_vel_b[:, 2])
  return torch.exp(-(planar_error + vertical_error) / std**2)


def route_complete(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, WaypointCommand):
    raise TypeError("route_complete requires WaypointCommand")
  return command.success


def route_progress_metric(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, WaypointCommand):
    raise TypeError("route progress metric requires WaypointCommand")
  return command.progress
