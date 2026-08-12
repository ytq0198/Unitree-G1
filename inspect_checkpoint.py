#!/usr/bin/env python3
"""Inspect checkpoint contents and optionally compare model parameter shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def tensor_shapes(value) -> dict[str, list[int]]:
  if not isinstance(value, dict):
    return {}
  return {
    str(name): list(tensor.shape)
    for name, tensor in value.items()
    if isinstance(tensor, torch.Tensor)
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--compare", type=Path)
  args = parser.parse_args()

  source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  state_keys = (
    "actor_state_dict",
    "critic_state_dict",
    "model_state_dict",
    "amp_discriminator_state_dict",
  )
  result: dict[str, object] = {
    "checkpoint_keys": sorted(source.keys()),
    "states": {key: tensor_shapes(source.get(key, {})) for key in state_keys},
  }
  if args.compare:
    target = torch.load(args.compare, map_location="cpu", weights_only=False)
    comparison = {}
    for key in state_keys:
      source_shapes = tensor_shapes(source.get(key, {}))
      target_shapes = tensor_shapes(target.get(key, {}))
      comparison[key] = {
        "compatible": {
          name: shape
          for name, shape in source_shapes.items()
          if target_shapes.get(name) == shape
        },
        "incompatible": {
          name: {"source": shape, "target": target_shapes.get(name)}
          for name, shape in source_shapes.items()
          if target_shapes.get(name) != shape
        },
      }
    result["comparison"] = comparison
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
