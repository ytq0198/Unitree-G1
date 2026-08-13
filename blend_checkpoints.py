#!/usr/bin/env python3
"""Interpolate actor parameters while retaining the base training state."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("base", type=Path)
  parser.add_argument("adapted", type=Path, nargs="+")
  parser.add_argument("output", type=Path)
  parser.add_argument("--alpha", type=float, required=True)
  args = parser.parse_args()
  if not 0.0 <= args.alpha <= 1.0:
    raise ValueError("alpha must be in [0, 1]")
  base = torch.load(args.base, map_location="cpu", weights_only=False)
  adapted_checkpoints = [
    torch.load(path, map_location="cpu", weights_only=False)
    for path in args.adapted
  ]
  for name, base_value in base["actor_state_dict"].items():
    adapted_values = [
      checkpoint["actor_state_dict"][name]
      for checkpoint in adapted_checkpoints
    ]
    if any(base_value.shape != value.shape for value in adapted_values):
      raise ValueError(f"Actor shape mismatch for {name}")
    if torch.is_floating_point(base_value) and "obs_normalizer" not in name:
      adapted_mean = torch.stack(adapted_values).mean(dim=0)
      base["actor_state_dict"][name] = torch.lerp(
        base_value, adapted_mean, args.alpha
      )
  base["iter"] = 0
  base.pop("optimizer_state_dict", None)
  base.pop("amp_discriminator_optimizer_state_dict", None)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(base, args.output)
  print(args.output)


if __name__ == "__main__":
  main()
