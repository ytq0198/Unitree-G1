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


def remove_command_column(tensor: torch.Tensor, removed_index: int) -> torch.Tensor:
  return torch.cat(
    (tensor[..., :removed_index], tensor[..., removed_index + 1 :]), dim=-1
  )


def adapt_state(
  source: dict[str, torch.Tensor],
  target: dict[str, torch.Tensor],
  removed_index: int = REMOVED_COMMAND_INDEX,
):
  adapted = {name: tensor.clone() for name, tensor in target.items()}
  for name, source_tensor in source.items():
    if name not in adapted:
      continue
    target_tensor = adapted[name]
    if source_tensor.shape == target_tensor.shape:
      adapted[name] = source_tensor.clone()
      continue
    candidate = remove_command_column(source_tensor, removed_index)
    if candidate.shape == target_tensor.shape:
      adapted[name] = candidate.clone()
      continue
    raise ValueError(
      f"Cannot adapt {name}: source={tuple(source_tensor.shape)}, "
      f"target={tuple(target_tensor.shape)}"
    )
  return adapted


def adapt_preserving_architecture(
  source: dict[str, torch.Tensor],
  source_input_dim: int,
  removed_index: int,
) -> dict[str, torch.Tensor]:
  """Remove one observation column without changing the source MLP layout."""
  adapted = {}
  for name, tensor in source.items():
    if tensor.ndim >= 1 and tensor.shape[-1] == source_input_dim:
      adapted[name] = remove_command_column(tensor, removed_index)
    else:
      adapted[name] = tensor.clone()
  return adapted


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("lab7_checkpoint", type=Path)
  parser.add_argument("navigation_template", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument(
    "--command-layout", choices=("xy", "forward_yaw"), default="xy"
  )
  parser.add_argument("--preserve-architecture", action="store_true")
  args = parser.parse_args()

  source = torch.load(args.lab7_checkpoint, map_location="cpu", weights_only=False)
  target = torch.load(
    args.navigation_template, map_location="cpu", weights_only=False
  )
  removed_index = (
    COMMAND_START + 1
    if args.command_layout == "forward_yaw"
    else REMOVED_COMMAND_INDEX
  )
  for key in ("actor_state_dict", "critic_state_dict"):
    if args.preserve_architecture:
      source_dim = 286 if key == "actor_state_dict" else 298
      target[key] = adapt_preserving_architecture(
        source[key], source_dim, removed_index
      )
    else:
      target[key] = adapt_state(source[key], target[key], removed_index)
  if source.get("amp_discriminator_state_dict"):
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
