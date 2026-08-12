#!/usr/bin/env python3
"""Print compact command and motion diagnostics for one policy rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tensordict import TensorDict


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--command-mode", choices=("xy", "forward_yaw"), default="xy")
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--seed", type=int, default=901)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()

  from src import workflow
  from src.mjlab_tasks.commands import WaypointCommand

  env, wrapped, runner = workflow._inference_runner(
    args.checkpoint,
    "height",
    num_envs=1,
    device=args.device,
    seed=args.seed,
    student_file=Path(__file__).resolve().parent / "student.py",
    command_mode=args.command_mode,
  )
  policy = runner.get_inference_policy(args.device)
  observations = wrapped.get_observations().to(args.device)
  command = env.command_manager.get_term("waypoint")
  if not isinstance(command, WaypointCommand):
    raise TypeError("Expected WaypointCommand")
  start_xy = command.robot.data.root_link_pos_w[0, :2].clone()
  samples = []
  try:
    with torch.inference_mode():
      for step in range(args.steps):
        action = policy(observations)
        observation_dict, _, terminated, truncated, _ = env.step(action)
        if step % 25 == 0 or bool((terminated | truncated).item()):
          xy = command.robot.data.root_link_pos_w[0, :2]
          samples.append(
            {
              "step": step + 1,
              "command": command.command[0].cpu().tolist(),
              "heading": float(command.robot.data.heading_w[0]),
              "body_velocity": command.robot.data.root_link_lin_vel_b[0, :2]
              .cpu()
              .tolist(),
              "displacement": (xy - start_xy).cpu().tolist(),
              "route_progress": float(command.progress[0]),
              "terminated": bool(terminated.item()),
              "truncated": bool(truncated.item()),
            }
          )
        if bool((terminated | truncated).item()):
          break
        observations = TensorDict(observation_dict, batch_size=[1], device=args.device)
  finally:
    wrapped.close()
  print(json.dumps(samples, indent=2))


if __name__ == "__main__":
  main()
