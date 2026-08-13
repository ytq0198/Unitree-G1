#!/usr/bin/env python3
"""Lab7 Exp07 runner: smoke / train / eval / video / plots outside the notebook."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Headless / writable caches before importing mjlab / mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

EXP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_ROOT))

from src.paths import configure_local_sources

configure_local_sources()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "command",
    choices=["smoke", "design", "train", "eval", "video", "submission", "all"],
  )
  parser.add_argument("--mode", default="height", choices=["height", "depth"])
  parser.add_argument(
    "--phase", default="walk", choices=["pretrain", "amp_flat", "rough", "walk"]
  )
  parser.add_argument(
    "--eval-terrain", default="rough", choices=["flat", "rough"]
  )
  parser.add_argument("--device", default="cuda:1")
  parser.add_argument("--num-envs", type=int, default=None)
  parser.add_argument("--iterations", type=int, default=600)
  parser.add_argument("--steps-per-env", type=int, default=24)
  parser.add_argument("--eval-steps", type=int, default=600)
  parser.add_argument("--frames", type=int, default=600)
  parser.add_argument("--checkpoint", type=str, default=None)
  parser.add_argument("--output", type=str, default=None)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--video-seed", type=int, default=5)
  args = parser.parse_args()

  student = EXP_ROOT / "student.py"
  out = EXP_ROOT / "outputs"
  out.mkdir(parents=True, exist_ok=True)
  viz = out / "visualizations"
  viz.mkdir(parents=True, exist_ok=True)

  from src import workflow

  if args.command in ("design", "all"):
    from src.mjlab_tasks.env_cfgs import course_g1_rough_walk_env_cfg
    import matplotlib.pyplot as plt

    cfg = course_g1_rough_walk_env_cfg(args.mode, student_path=student)
    fig = workflow.plot_training_design(cfg)
    path = viz / f"training_design_{args.mode}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)

  if args.command in ("smoke", "all"):
    result = workflow.smoke(
      args.mode,
      num_envs=args.num_envs or 32,
      steps=16,
      device=args.device,
      student_file=student,
      force_termination=True,
    )
    (out / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("smoke:", result)

  if args.command in ("train", "all"):
    train_envs = args.num_envs
    if train_envs is None:
      train_envs = 64 if args.mode == "height" else 32
    run_dir = workflow.train(
      args.mode,
      num_envs=train_envs,
      iterations=args.iterations,
      steps_per_env=args.steps_per_env,
      device=args.device,
      seed=args.seed if args.seed is not None else 7,
      student_file=student,
      resume_checkpoint=args.checkpoint,
      training_phase=args.phase,
    )
    print("run_dir:", run_dir)
    (out / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")

  if args.command in ("eval", "video", "submission", "all"):
    ckpt = (
      Path(args.checkpoint)
      if args.checkpoint
      else workflow.latest_checkpoint(args.mode)
    )
    print("checkpoint:", ckpt)

  if args.command in ("eval", "all"):
    import matplotlib.pyplot as plt

    metrics = workflow.evaluate(
      ckpt,
      args.mode,
      num_envs=args.num_envs or 32,
      steps=args.eval_steps,
      device=args.device,
      student_file=student,
      evaluation_terrain=args.eval_terrain,
      model_phase=args.phase,
    )
    (out / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("metrics:", metrics)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(list(metrics.keys()), list(metrics.values()), color="#2878b5")
    ax.set_title("Exp07 held-out evaluation")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(viz / "eval_metrics.png", dpi=150)
    plt.close(fig)

  if args.command in ("video", "all"):
    default_name = "evaluation_depth.mp4" if args.mode == "depth" else "evaluation.mp4"
    video_out = Path(args.output) if args.output else out / default_name
    video_path = workflow.record_video(
      ckpt,
      args.mode,
      frames=args.frames,
      device=args.device,
      student_file=student,
      output=video_out,
      seed=args.video_seed,
    )
    print("video:", video_path)

  if args.command in ("submission", "all"):
    submission = workflow.prepare_submission(
      ckpt, args.mode, device=args.device, student_file=student
    )
    print("submission:", submission)


if __name__ == "__main__":
  main()
