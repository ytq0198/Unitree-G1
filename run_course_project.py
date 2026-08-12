#!/usr/bin/env python3
"""Command-line entry point for smoke tests, training, evaluation, and export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "command", choices=("smoke", "train", "eval", "video", "submission")
  )
  parser.add_argument("--mode", choices=("height", "depth"), default="height")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int)
  parser.add_argument("--iterations", type=int, default=1000)
  parser.add_argument("--steps-per-env", type=int, default=24)
  parser.add_argument("--eval-steps", type=int, default=5000)
  parser.add_argument("--evaluations", type=int, default=10)
  parser.add_argument("--frames", type=int, default=150)
  parser.add_argument("--seed", type=int, default=23)
  parser.add_argument("--checkpoint")
  parser.add_argument("--init-checkpoint")
  parser.add_argument("--navigation-weight", type=float, default=1.0)
  parser.add_argument("--smoothness-weight", type=float, default=-0.05)
  parser.add_argument("--velocity-tracking-weight", type=float, default=2.0)
  parser.add_argument("--amp-scale", type=float, default=0.5)
  parser.add_argument("--learning-rate", type=float, default=1.0e-3)
  args = parser.parse_args()

  from src import workflow

  root = Path(__file__).resolve().parent
  outputs = root / "outputs"
  outputs.mkdir(exist_ok=True)
  student = root / "student.py"

  if args.command == "smoke":
    result = workflow.smoke(
      args.mode,
      num_envs=args.num_envs or (32 if args.mode == "height" else 8),
      steps=16,
      device=args.device,
      student_file=student,
    )
    path = outputs / f"smoke_{args.mode}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return

  if args.command == "train":
    run_dir = workflow.train(
      args.mode,
      num_envs=args.num_envs or (64 if args.mode == "height" else 32),
      iterations=args.iterations,
      steps_per_env=args.steps_per_env,
      device=args.device,
      seed=args.seed,
      student_file=student,
      init_checkpoint=args.init_checkpoint,
      navigation_reward_weight=args.navigation_weight,
      smoothness_reward_weight=args.smoothness_weight,
      velocity_tracking_weight=args.velocity_tracking_weight,
      amp_reward_scale=args.amp_scale,
      learning_rate=args.learning_rate,
    )
    print(run_dir)
    return

  checkpoint = (
    Path(args.checkpoint).resolve()
    if args.checkpoint
    else workflow.latest_checkpoint()
  )
  if args.command == "eval":
    metrics = workflow.evaluate(
      checkpoint,
      args.mode,
      steps=args.eval_steps,
      evaluations=args.evaluations,
      device=args.device,
      seed=args.seed,
      student_file=student,
    )
    path = outputs / f"eval_{args.mode}.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
  elif args.command == "video":
    print(
      workflow.record_video(
        checkpoint,
        args.mode,
        frames=args.frames,
        device=args.device,
        seed=args.seed,
        student_file=student,
      )
    )
  else:
    print(
      workflow.prepare_submission(
        checkpoint, args.mode, device=args.device, student_file=student
      )
    )


if __name__ == "__main__":
  main()
