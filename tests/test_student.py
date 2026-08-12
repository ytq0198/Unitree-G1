from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

import student
from src.paths import _path_from_file_url


class StudentFormulaTests(unittest.TestCase):
  def test_editable_file_url_decodes_spaces(self) -> None:
    path = _path_from_file_url("file:///mnt/localDisk3/RL%20learning/mujoco_warp")
    self.assertEqual(path, Path("/mnt/localDisk3/RL learning/mujoco_warp"))

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
