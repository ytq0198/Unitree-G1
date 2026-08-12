#!/usr/bin/env python3
"""Adapt a Lab 7 velocity-policy checkpoint to the navigation observation size."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


COMMAND_START = 3 + 3 + 3 + 29 + 29 + 29
LAB7_COMMAND_DIM = 3
NAVIGATION_COMMAND_DIM = 2
REMOVED_COMMAND_INDEX = COMMAND_START + NAVIGATION_COMMAND_DIM


def remove_yaw_command_column(tensor: torch.Tensor) -> torch.Tensor:
  """Map Lab 7 [vx, vy, yaw-rate] inputs to navigation [x, y] inputs."""
  return torch.cat(
    (tensor[..., :REMOVED_COMMAND_INDEX], tensor[..., REMOVED_COMMAND_INDEX + 1 :]),
    dim=-1,
  )


def adapt_state(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]):
  adapted = {name: tensor.clone() for name, tensor in target.items()}
  for name, source_tensor in source.items():
    if name not in adapted:
      continue
    target_tensor = adapted[name]
    if source_tensor.shape == target_tensor.shape:
      adapted[name] = source_tensor.clone()
      continue
    candidate = remove_yaw_command_column(source_tensor)
    if candidate.shape == target_tensor.shape:
      adapted[name] = candidate.clone()
      continue
    raise ValueError(
      f"Cannot adapt {name}: source={tuple(source_tensor.shape)}, "
      f"target={tuple(target_tensor.shape)}"
    )
  return adapted


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("lab7_checkpoint", type=Path)
  parser.add_argument("navigation_template", type=Path)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()

  source = torch.load(args.lab7_checkpoint, map_location="cpu", weights_only=False)
  target = torch.load(
    args.navigation_template, map_location="cpu", weights_only=False
  )
  for key in ("actor_state_dict", "critic_state_dict"):
    target[key] = adapt_state(source[key], target[key])
  target["amp_discriminator_state_dict"] = source[
    "amp_discriminator_state_dict"
  ]
  target["iter"] = 0
  target.pop("optimizer_state_dict", None)
  target.pop("amp_discriminator_optimizer_state_dict", None)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(target, args.output)
  print(args.output)


if __name__ == "__main__":
  main()
