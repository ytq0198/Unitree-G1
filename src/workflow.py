"""Notebook-facing training, evaluation, visualization, and export tools."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import imageio.v3 as iio
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.runner import MjlabOnPolicyRunner
from tensordict import TensorDict

from src.amp import ManualResetAmpVecEnvWrapper
from src.mjlab_tasks.commands import WaypointCommand
from src.mjlab_tasks.env_cfgs import course_g1_navigation_env_cfg
from src.mjlab_tasks.rl_cfg import course_g1_navigation_ppo_runner_cfg
from src.paths import PROJECT_ROOT
from src.terrain import (
  COURSE_PROJECT_BORDER_WIDTH,
  NavigationScene,
  sample_evaluation_seeds,
)

ObservationMode = Literal["height", "depth"]
DEFAULT_STUDENT = PROJECT_ROOT / "student.py"
LOAD_CFG = {
  "actor": True,
  "critic": True,
  "optimizer": False,
  "iteration": False,
  "rnd": False,
}

MODEL_HEIGHT = """from __future__ import annotations

from collections.abc import Mapping
import torch


class ExportedPolicy:
  observation_keys = ("actor",)

  def __init__(self, path: str, device: str) -> None:
    self.device = torch.device(device)
    self.module = torch.jit.load(path, map_location=self.device).eval()

  def predict(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
      return self.module(obs["actor"].to(self.device))


def load_policy(policy_path: str, device: str = "cpu") -> ExportedPolicy:
  return ExportedPolicy(policy_path, device)
"""

MODEL_DEPTH = """from __future__ import annotations

from collections.abc import Mapping
import torch


class ExportedPolicy:
  observation_keys = ("actor", "depth")

  def __init__(self, path: str, device: str) -> None:
    self.device = torch.device(device)
    self.module = torch.jit.load(path, map_location=self.device).eval()

  def predict(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
      actor = obs["actor"].to(self.device)
      depth = obs["depth"].to(self.device)
      return self.module(actor, [depth])


def load_policy(policy_path: str, device: str = "cpu") -> ExportedPolicy:
  return ExportedPolicy(policy_path, device)
"""


def _student_path(path: str | Path | None) -> Path:
  return Path(path or DEFAULT_STUDENT).resolve()


def _tensor_observation(observations: Mapping[str, Any], key: str) -> torch.Tensor:
  value = observations[key]
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"Observation group {key!r} must be concatenated")
  return value


def latest_checkpoint(root: str | Path | None = None) -> Path:
  """Return the newest training checkpoint under the project outputs."""
  search_root = Path(root or PROJECT_ROOT / "outputs" / "rsl_rl")
  candidates = [path for path in search_root.rglob("model_*.pt") if path.is_file()]
  if not candidates:
    raise FileNotFoundError(f"No model_*.pt checkpoint under {search_root}")
  return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def smoke(
  mode: ObservationMode = "height",
  *,
  num_envs: int = 32,
  steps: int = 16,
  device: str = "cuda:0",
  student_file: str | Path | None = None,
  force_termination: bool = True,
) -> dict[str, Any]:
  """Run the short reset/step check used in the notebook."""
  cfg = course_g1_navigation_env_cfg(mode, student_path=_student_path(student_file))
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device=device)
  resets = 0
  try:
    observations, _ = env.reset()
    amp = _tensor_observation(observations, "amp")
    if amp.shape != (num_envs, 83):
      raise RuntimeError(f"Unexpected AMP shape: {amp.shape}")
    if env.action_manager.total_action_dim != 29:
      raise RuntimeError("Navigation must expose 29 G1 joint actions")
    depth_shape = None
    if mode == "depth":
      depth = _tensor_observation(observations, "depth")
      depth_shape = tuple(depth.shape)
      if depth_shape != (num_envs, 1, 60, 80):
        raise RuntimeError(f"Unexpected depth shape: {depth_shape}")
      if not torch.isfinite(depth).all() or not torch.any(depth != 0):
        raise RuntimeError("Depth observation must be finite and non-empty")
    for step in range(steps):
      if force_termination and step == steps // 2:
        env.episode_length_buf[:] = env.max_episode_length
      action = torch.zeros(num_envs, 29, device=device)
      _, reward, terminated, truncated, _ = env.step(action)
      if not torch.isfinite(reward).all():
        raise RuntimeError("Non-finite reward")
      done = terminated | truncated
      if done.any():
        env.reset(env_ids=done.nonzero(as_tuple=False).squeeze(-1))
        resets += int(done.sum())
  finally:
    env.close()
  return {
    "mode": mode,
    "num_envs": num_envs,
    "steps": steps,
    "action_dim": 29,
    "manual_resets": resets,
    "amp_shape": (num_envs, 83),
    "depth_shape": depth_shape,
  }


def train(
  mode: ObservationMode = "height",
  *,
  num_envs: int = 64,
  iterations: int = 1000,
  steps_per_env: int = 24,
  device: str = "cuda:0",
  seed: int = 23,
  student_file: str | Path | None = None,
) -> Path:
  """Train direct 29-joint navigation AMP-PPO and return its run directory."""
  if device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(f"Requested {device}, but CUDA is not available")
  student_path = _student_path(student_file)
  env_cfg = course_g1_navigation_env_cfg(
    mode, student_path=student_path, scene_seed=seed
  )
  env_cfg.scene.num_envs = num_envs
  agent_cfg = course_g1_navigation_ppo_runner_cfg(mode, student_path=student_path)
  agent_cfg.max_iterations = iterations
  agent_cfg.num_steps_per_env = steps_per_env
  agent_cfg.save_interval = max(1, min(50, iterations))
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_dir = (
    PROJECT_ROOT / "outputs" / "rsl_rl" / agent_cfg.experiment_name / timestamp
  )
  log_dir.mkdir(parents=True, exist_ok=True)
  env = ManagerBasedRlEnv(env_cfg, device=device)
  wrapped = ManualResetAmpVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(
    wrapped, asdict(agent_cfg), log_dir=str(log_dir), device=device
  )
  try:
    runner.learn(num_learning_iterations=iterations)
  finally:
    wrapped.close()
  return log_dir


def _inference_runner(
  checkpoint: str | Path,
  mode: ObservationMode,
  *,
  num_envs: int,
  device: str,
  seed: int,
  student_file: str | Path | None,
  render_mode: str | None = None,
) -> tuple[ManagerBasedRlEnv, ManualResetAmpVecEnvWrapper, MjlabOnPolicyRunner]:
  student_path = _student_path(student_file)
  env_cfg = course_g1_navigation_env_cfg(
    mode, play=True, student_path=student_path, scene_seed=seed
  )
  env_cfg.scene.num_envs = num_envs
  agent_cfg = course_g1_navigation_ppo_runner_cfg(mode, student_path)
  env = ManagerBasedRlEnv(env_cfg, device=device, render_mode=render_mode)
  wrapped = ManualResetAmpVecEnvWrapper(env)
  runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg=LOAD_CFG)
  return env, wrapped, runner


def evaluate(
  checkpoint: str | Path,
  mode: ObservationMode = "height",
  *,
  num_envs: int = 1,
  steps: int = 5000,
  evaluations: int = 10,
  device: str = "cuda:0",
  seed: int = 101,
  student_file: str | Path | None = None,
) -> dict[str, float]:
  """Average ten independent rollouts with distinct random route starts."""
  if num_envs != 1:
    raise ValueError("Course Project evaluation uses one environment per start")
  if steps <= 0:
    raise ValueError("steps must be positive")
  evaluation_seeds = sample_evaluation_seeds(seed, evaluations)
  progress_results = []
  success_results = []
  smoothness_results = []
  for scene_seed in evaluation_seeds:
    env, wrapped, runner = _inference_runner(
      checkpoint,
      mode,
      num_envs=1,
      device=device,
      seed=scene_seed,
      student_file=student_file,
    )
    policy = runner.get_inference_policy(device)
    observations = wrapped.get_observations().to(device)
    max_progress = torch.zeros(1, device=device)
    succeeded = torch.zeros(1, dtype=torch.bool, device=device)
    previous = torch.zeros(1, 29, device=device)
    previous_previous = previous.clone()
    smoothness: list[torch.Tensor] = []
    try:
      with torch.inference_mode():
        for _ in range(steps):
          action = policy(observations)
          observation_dict, _, terminated, truncated, _ = env.step(action)
          command = env.command_manager.get_term("waypoint")
          if not isinstance(command, WaypointCommand):
            raise TypeError("Expected WaypointCommand")
          max_progress = torch.maximum(max_progress, command.progress)
          succeeded |= command.success
          smoothness.append(
            torch.mean(
              torch.abs(action - 2 * previous + previous_previous), dim=-1
            )
          )
          previous_previous = previous
          previous = action
          if bool((terminated | truncated).item()):
            break
          observations = TensorDict(
            observation_dict, batch_size=[1], device=device
          )
    finally:
      wrapped.close()
    progress_results.append(max_progress.item())
    success_results.append(float(succeeded.item()))
    smoothness_results.append(float(torch.cat(smoothness).mean()))
  return {
    "route_progress": float(np.mean(progress_results)),
    "route_success": float(np.mean(success_results)),
    "smoothness": float(np.mean(smoothness_results)),
    "random_start_evaluations": float(len(evaluation_seeds)),
  }


def record_video(
  checkpoint: str | Path,
  mode: ObservationMode = "height",
  *,
  frames: int = 150,
  device: str = "cuda:0",
  seed: int = 101,
  student_file: str | Path | None = None,
  output: str | Path | None = None,
) -> Path:
  """Record a physical navigation MP4 for inline notebook display."""
  if frames < 150:
    raise ValueError("Evaluation videos must contain at least 150 frames")
  env, wrapped, runner = _inference_runner(
    checkpoint,
    mode,
    num_envs=1,
    device=device,
    seed=seed,
    student_file=student_file,
    render_mode="rgb_array",
  )
  policy = runner.get_inference_policy(device)
  observations = wrapped.get_observations().to(device)
  images: list[np.ndarray] = []
  try:
    with torch.inference_mode():
      for _ in range(frames):
        observations, _, _, _ = wrapped.step(policy(observations))
        frame = env.render()
        if frame is None:
          raise RuntimeError("mjlab offscreen renderer returned no frame")
        images.append(np.asarray(frame).copy())
  finally:
    wrapped.close()
  output_path = Path(output or PROJECT_ROOT / "outputs" / "evaluation.mp4")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  iio.imwrite(output_path, np.stack(images), fps=50, codec="libx264")
  return output_path


def prepare_submission(
  checkpoint: str | Path,
  mode: ObservationMode = "height",
  *,
  device: str = "cuda:0",
  student_file: str | Path | None = None,
  output_dir: str | Path | None = None,
) -> Path:
  """Prepare the strict policy.pt, model.py, student.py grading folder."""
  student_path = _student_path(student_file)
  cfg = course_g1_navigation_env_cfg(mode, play=True, student_path=student_path)
  cfg.scene.num_envs = 1
  agent_cfg = course_g1_navigation_ppo_runner_cfg(mode, student_path)
  env = ManagerBasedRlEnv(cfg, device=device)
  wrapped = ManualResetAmpVecEnvWrapper(env)
  runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg=LOAD_CFG)
  build_dir = Path(output_dir or PROJECT_ROOT / "outputs" / "submission")
  if build_dir.exists():
    shutil.rmtree(build_dir)
  build_dir.mkdir(parents=True)
  try:
    runner.export_policy_to_jit(str(build_dir), filename="policy.pt")
  finally:
    wrapped.close()
  (build_dir / "model.py").write_text(
    MODEL_DEPTH if mode == "depth" else MODEL_HEIGHT, encoding="utf-8"
  )
  shutil.copy2(student_path, build_dir / "student.py")
  return build_dir


def plot_scene(scene: NavigationScene):
  """Display tile types, route, start, and goal for the generated scene."""
  import matplotlib.pyplot as plt
  from matplotlib.patches import Rectangle

  colors = {
    "pile": "#8ecae6",
    "platform_gap": "#ffb703",
    "pyramid_stairs": "#90be6d",
  }
  figure, axis = plt.subplots(figsize=(6, 6))
  inner_x = scene.rows * scene.tile_size
  inner_y = scene.cols * scene.tile_size
  border = COURSE_PROJECT_BORDER_WIDTH
  axis.add_patch(
    Rectangle(
      (-border, -border),
      inner_x + 2 * border,
      inner_y + 2 * border,
      facecolor="#495057",
      edgecolor="white",
      linewidth=1,
    )
  )
  for tile in scene.tiles:
    x = tile.row * scene.tile_size
    y = tile.col * scene.tile_size
    axis.add_patch(
      Rectangle(
        (x, y),
        scene.tile_size,
        scene.tile_size,
        facecolor=colors[tile.kind],
        edgecolor="white",
        linewidth=2,
      )
    )
    axis.text(
      x + scene.tile_size / 2,
      y + scene.tile_size / 2,
      tile.kind.replace("_", "\n"),
      ha="center",
      va="center",
      fontsize=9,
    )
  route = np.asarray(scene.route)
  axis.plot(route[:, 0], route[:, 1], "-o", color="#222222", linewidth=2.5)
  axis.scatter(*route[0], s=120, color="#2a9d8f", label="start", zorder=3)
  axis.scatter(*route[-1], s=160, color="#d62828", marker="*", label="goal", zorder=3)
  axis.set_xlim(-border, inner_x + border)
  axis.set_ylim(-border, inner_y + border)
  axis.set_aspect("equal")
  axis.set_title(
    f"Navigation terrain 70 x 70 m, route={scene.route_length:.1f} m"
  )
  axis.set_xlabel("x (m)")
  axis.set_ylabel("y (m)")
  axis.legend(loc="upper left")
  figure.tight_layout()
  return figure


__all__ = [
  "evaluate",
  "latest_checkpoint",
  "plot_scene",
  "prepare_submission",
  "record_video",
  "smoke",
  "train",
]
