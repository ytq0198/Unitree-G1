from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import torch

import student
from prepare_lab7_warmstart import COMMAND_START, remove_yaw_command_column


def load_paths_module():
  path = Path(__file__).resolve().parents[1] / "src" / "paths.py"
  spec = importlib.util.spec_from_file_location("course_project_paths", path)
  if spec is None or spec.loader is None:
    raise ImportError(path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def load_navigation_math_module():
  path = Path(__file__).resolve().parents[1] / "src" / "navigation_math.py"
  spec = importlib.util.spec_from_file_location("course_navigation_math", path)
  if spec is None or spec.loader is None:
    raise ImportError(path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


route_position_m = load_navigation_math_module().route_position_m
forward_yaw_command = load_navigation_math_module().forward_yaw_command


class StudentFormulaTests(unittest.TestCase):
  def test_forward_yaw_allows_turning_in_place(self) -> None:
    command = forward_yaw_command(
      torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
      max_command_speed=0.6,
      min_turning_speed=0.0,
      heading_stiffness=0.5,
      max_yaw_rate=0.25,
    )
    torch.testing.assert_close(command[:, 0], torch.tensor([0.6, 0.0, 0.0]))
    torch.testing.assert_close(command[:, 1], torch.tensor([0.0, 0.25, 0.25]))

  def test_route_position_projects_onto_active_segment(self) -> None:
    route = torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    cumulative = torch.tensor([0.0, 10.0, 20.0])
    position = route_position_m(
      torch.tensor([[4.0, 3.0], [12.0, 4.0]]),
      torch.zeros(2, 2),
      route,
      torch.tensor([1, 2]),
      cumulative,
    )
    torch.testing.assert_close(position, torch.tensor([4.0, 14.0]))

  def test_route_position_is_continuous_across_a_turn(self) -> None:
    route = torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    cumulative = torch.tensor([0.0, 10.0, 20.0])
    robot = torch.tensor([[10.0, 0.0]])
    before = route_position_m(
      robot, torch.zeros(1, 2), route, torch.tensor([1]), cumulative
    )
    after = route_position_m(
      robot, torch.zeros(1, 2), route, torch.tensor([2]), cumulative
    )
    torch.testing.assert_close(before, after)

  def test_route_position_selects_each_environment_route(self) -> None:
    routes = torch.tensor(
      [
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
        [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
      ]
    )
    cumulative = torch.tensor([[0.0, 10.0, 20.0], [0.0, 10.0, 20.0]])
    position = route_position_m(
      torch.tensor([[4.0, 3.0], [3.0, 4.0]]),
      torch.zeros(2, 2),
      routes,
      torch.tensor([1, 1]),
      cumulative,
      torch.tensor([0, 1]),
    )
    torch.testing.assert_close(position, torch.tensor([4.0, 4.0]))

  def test_editable_file_url_decodes_spaces(self) -> None:
    path = load_paths_module()._path_from_file_url(
      "file:///mnt/localDisk3/RL%20learning/mujoco_warp"
    )
    self.assertEqual(path, Path("/mnt/localDisk3/RL learning/mujoco_warp"))

  def test_warmstart_removes_only_yaw_command_feature(self) -> None:
    source = torch.arange(286, dtype=torch.float32).unsqueeze(0)
    adapted = remove_yaw_command_column(source)
    self.assertEqual(adapted.shape, (1, 285))
    torch.testing.assert_close(adapted[..., : COMMAND_START + 2], source[..., : COMMAND_START + 2])
    torch.testing.assert_close(
      adapted[..., COMMAND_START + 2 :], source[..., COMMAND_START + 3 :]
    )

  def test_build_amp_state_shape_and_order(self) -> None:
    parts = (
      torch.full((2, 29), 1.0),
      torch.full((2, 29), 2.0),
      torch.full((2, 1), 3.0),
      torch.full((2, 3), 4.0),
      torch.full((2, 3), 5.0),
      torch.full((2, 3), 6.0),
      torch.full((2, 5, 3), 7.0),
    )
    state = student.build_amp_state(*parts)
    self.assertEqual(state.shape, (2, 83))
    torch.testing.assert_close(state[:, :29], parts[0])
    torch.testing.assert_close(state[:, -15:], parts[-1].flatten(start_dim=-2))

  def test_normalize_depth_clips_and_scales(self) -> None:
    depth = torch.tensor([[[[-1.0, 0.1, 2.55, 5.0, 8.0]]]])
    expected = torch.tensor([[[[0.0, 0.0, 0.5, 1.0, 1.0]]]])
    torch.testing.assert_close(student.normalize_depth(depth), expected)

  def test_waypoint_identity_heading(self) -> None:
    body = student.waypoint_in_body_frame(
      torch.tensor([[1.0, 2.0, 0.0]]),
      torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
      torch.tensor([[4.0, 1.0, 2.0]]),
    )
    torch.testing.assert_close(body, torch.tensor([[3.0, -1.0]]))

  def test_waypoint_rotates_world_to_body(self) -> None:
    half = math.sqrt(0.5)
    body = student.waypoint_in_body_frame(
      torch.zeros(1, 3),
      torch.tensor([[half, 0.0, 0.0, half]]),
      torch.tensor([[1.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
      body, torch.tensor([[0.0, -1.0]]), atol=1.0e-6, rtol=1.0e-6
    )

  def test_navigation_reward_converts_boolean_indicators(self) -> None:
    reward = student.navigation_reward(
      torch.tensor([0.1, -0.2]),
      torch.tensor([True, False]),
      torch.tensor([False, True]),
    )
    torch.testing.assert_close(reward, torch.tensor([0.9, 4.2]))

  def test_progress_rate_cancels_environment_dt_scaling(self) -> None:
    distance_improvement = torch.tensor([0.03, -0.01])
    step_dt = 0.02
    progress_rate = distance_improvement / step_dt
    integrated_progress_reward = 4.0 * progress_rate * step_dt
    torch.testing.assert_close(
      integrated_progress_reward, 4.0 * distance_improvement
    )

  def test_event_rates_cancel_environment_dt_scaling(self) -> None:
    step_dt = 0.02
    reward_rate = student.navigation_reward(
      torch.zeros(2),
      torch.tensor([1.0, 0.0]) / step_dt,
      torch.tensor([0.0, 1.0]) / step_dt,
    )
    torch.testing.assert_close(reward_rate * step_dt, torch.tensor([0.5, 5.0]))

  def test_smoothness_zero_for_linear_action_sequence(self) -> None:
    previous_previous = torch.zeros(2, 29)
    previous = torch.ones(2, 29)
    action = torch.full((2, 29), 2.0)
    torch.testing.assert_close(
      student.smoothness_penalty(action, previous, previous_previous),
      torch.zeros(2),
    )

  def test_smoothness_uses_mean_absolute_second_difference(self) -> None:
    previous_previous = torch.zeros(1, 29)
    previous = torch.zeros(1, 29)
    action = torch.arange(29, dtype=torch.float32).unsqueeze(0)
    expected = torch.mean(torch.abs(action), dim=-1)
    torch.testing.assert_close(
      student.smoothness_penalty(action, previous, previous_previous), expected
    )


if __name__ == "__main__":
  unittest.main()
