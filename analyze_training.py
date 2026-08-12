#!/usr/bin/env python3
"""Summarize TensorBoard scalars from a course-project training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


IMPORTANT_TAGS = (
  "Train/mean_reward",
  "Train/mean_episode_length",
  "Episode_Reward/navigation_task",
  "Episode_Reward/alive",
  "Episode_Reward/upright",
  "Episode_Reward/action_rate_l2",
  "Episode_Reward/student_smoothness",
  "Episode_Metrics/route_progress",
  "Metrics/waypoint/route_progress",
  "Metrics/waypoint/route_success",
  "Loss/discriminator",
  "Loss/gradient_penalty",
  "Policy/mean_noise_std",
)


def summarize(run_dir: Path) -> dict[str, dict[str, float]]:
  accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  accumulator.Reload()
  available = set(accumulator.Tags().get("scalars", ()))
  summary: dict[str, dict[str, float]] = {}
  for tag in IMPORTANT_TAGS:
    if tag not in available:
      continue
    events = accumulator.Scalars(tag)
    values = np.asarray([event.value for event in events], dtype=np.float64)
    best_index = int(np.argmax(values))
    summary[tag] = {
      "first": float(values[0]),
      "last": float(values[-1]),
      "mean_last_50": float(values[-50:].mean()),
      "max": float(values[best_index]),
      "max_step": float(events[best_index].step),
    }
  return summary


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("run_dir", type=Path)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  result = summarize(args.run_dir.resolve())
  text = json.dumps(result, indent=2, sort_keys=True)
  if args.output:
    args.output.write_text(text, encoding="utf-8")
  print(text)


if __name__ == "__main__":
  main()
