#!/usr/bin/env python3
"""Blend only the actor's waypoint-input columns between two checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

WAYPOINT_SLICE = slice(96, 98)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("base")
  parser.add_argument("adapted")
  parser.add_argument("output")
  parser.add_argument("--alpha", type=float, required=True)
  args = parser.parse_args()
  if not 0.0 <= args.alpha <= 1.0:
    raise ValueError("alpha must be in [0, 1]")

  base = torch.load(args.base, map_location="cpu")
  adapted = torch.load(args.adapted, map_location="cpu")
  result = dict(base)
  actor = {key: value.clone() for key, value in base["actor_state_dict"].items()}
  base_weight = base["actor_state_dict"]["mlp.0.weight"]
  adapted_weight = adapted["actor_state_dict"]["mlp.0.weight"]
  actor["mlp.0.weight"][:, WAYPOINT_SLICE] = (
    (1.0 - args.alpha) * base_weight[:, WAYPOINT_SLICE]
    + args.alpha * adapted_weight[:, WAYPOINT_SLICE]
  )
  result["actor_state_dict"] = actor
  result["infos"] = {
    **base.get("infos", {}),
    "waypoint_blend_alpha": args.alpha,
    "waypoint_blend_source": str(Path(args.adapted).resolve()),
  }
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(result, output)
  print(output.resolve())


if __name__ == "__main__":
  main()
