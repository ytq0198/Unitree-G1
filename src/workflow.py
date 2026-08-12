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

MODEL_HEIGHT_FORWARD_YAW = """from __future__ import annotations

from collections.abc import Mapping
import torch


def adapt_waypoint(actor: torch.Tensor) -> torch.Tensor:
  actor = actor.clone()
  displacement = actor[..., 96:98]
  heading_error = torch.atan2(displacement[..., 1], displacement[..., 0])
  forward = torch.clamp(torch.norm(displacement, dim=-1), max=0.6)
  forward *= torch.clamp(torch.cos(heading_error), min=0.0)
  forward = torch.maximum(forward, torch.full_like(forward, 0.4))
  actor[..., 96] = forward
  actor[..., 97] = torch.clamp(0.5 * heading_error, -0.25, 0.25)
  return actor


class ExportedPolicy:
  observation_keys = ("actor",)

  def __init__(self, path: str, device: str) -> None:
    self.device = torch.device(device)
    self.module = torch.jit.load(path, map_location=self.device).eval()

  def predict(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
      return self.module(adapt_waypoint(obs["actor"].to(self.device)))


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
  init_checkpoint: str | Path | None = None,
  navigation_reward_weight: float = 1.0,
  smoothness_reward_weight: float = -0.05,
  velocity_tracking_weight: float = 2.0,
  gait_preservation_scale: float = 0.0,
  max_command_speed: float = 0.6,
  amp_reward_scale: float = 0.5,
  learning_rate: float = 1.0e-3,
  command_mode: str = "xy",
  run_tag: str = "",
  align_start_heading: bool = False,
  start_heading_spread: float = 0.25,
  hidden_dims: tuple[int, ...] = (256, 128),
  entropy_coef: float = 0.01,
  warmstart_std: float | None = None,
  training_pushes: bool = False,
) -> Path:
  """Train direct 29-joint navigation AMP-PPO and return its run directory."""
  if device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(f"Requested {device}, but CUDA is not available")
  student_path = _student_path(student_file)
  env_cfg = course_g1_navigation_env_cfg(
    mode,
    student_path=student_path,
    scene_seed=seed,
    navigation_reward_weight=navigation_reward_weight,
    smoothness_reward_weight=smoothness_reward_weight,
    velocity_tracking_weight=velocity_tracking_weight,
    gait_preservation_scale=gait_preservation_scale,
    max_command_speed=max_command_speed,
    command_mode=command_mode,
    align_start_heading=align_start_heading,
    start_heading_spread=start_heading_spread,
    training_pushes=training_pushes,
  )
  env_cfg.scene.num_envs = num_envs
  agent_cfg = course_g1_navigation_ppo_runner_cfg(
    mode,
    student_path=student_path,
    amp_reward_scale=amp_reward_scale,
    learning_rate=learning_rate,
    hidden_dims=hidden_dims,
    entropy_coef=entropy_coef,
  )
  agent_cfg.max_iterations = iterations
  agent_cfg.num_steps_per_env = steps_per_env
  agent_cfg.save_interval = max(1, min(50, iterations))
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if run_tag:
    timestamp = f"{timestamp}_{run_tag}"
  log_dir = (
    PROJECT_ROOT / "outputs" / "rsl_rl" / agent_cfg.experiment_name / timestamp
  )
  log_dir.mkdir(parents=True, exist_ok=True)
  env = ManagerBasedRlEnv(env_cfg, device=device)
  wrapped = ManualResetAmpVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(
    wrapped, asdict(agent_cfg), log_dir=str(log_dir), device=device
  )
  if init_checkpoint is not None:
    runner.load(str(Path(init_checkpoint).resolve()), load_cfg=LOAD_CFG)
    if warmstart_std is not None:
      if warmstart_std <= 0.0:
        raise ValueError("warmstart_std must be positive")
      runner.alg.actor.distribution.std_param.data.fill_(warmstart_std)
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
  command_mode: str = "xy",
  align_start_heading: bool = False,
  start_heading_spread: float = 0.25,
  hidden_dims: tuple[int, ...] = (256, 128),
  max_command_speed: float = 0.6,
) -> tuple[ManagerBasedRlEnv, ManualResetAmpVecEnvWrapper, MjlabOnPolicyRunner]:
  student_path = _student_path(student_file)
  env_cfg = course_g1_navigation_env_cfg(
    mode,
    play=True,
    student_path=student_path,
    scene_seed=seed,
    command_mode=command_mode,
    align_start_heading=align_start_heading,
    start_heading_spread=start_heading_spread,
    max_command_speed=max_command_speed,
  )
  env_cfg.scene.num_envs = num_envs
  agent_cfg = course_g1_navigation_ppo_runner_cfg(
    mode, student_path, hidden_dims=hidden_dims
  )
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
  command_mode: str = "xy",
  align_start_heading: bool = False,
  start_heading_spread: float = 0.25,
  hidden_dims: tuple[int, ...] = (256, 128),
  max_command_speed: float = 0.6,
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
  episode_steps_results = []
  fell_over_results = []
  time_out_results = []
  for scene_seed in evaluation_seeds:
    env, wrapped, runner = _inference_runner(
      checkpoint,
      mode,
      num_envs=1,
      device=device,
      seed=scene_seed,
      student_file=student_file,
      command_mode=command_mode,
      align_start_heading=align_start_heading,
      start_heading_spread=start_heading_spread,
      hidden_dims=hidden_dims,
      max_command_speed=max_command_speed,
    )
    policy = runner.get_inference_policy(device)
    observations = wrapped.get_observations().to(device)
    max_progress = torch.zeros(1, device=device)
    succeeded = torch.zeros(1, dtype=torch.bool, device=device)
    previous = torch.zeros(1, 29, device=device)
    previous_previous = previous.clone()
    smoothness: list[torch.Tensor] = []
    episode_steps = 0
    fell_over = False
    time_out = False
    try:
      with torch.inference_mode():
        for _ in range(steps):
          episode_steps += 1
          action = policy(observations)
          observation_dict, _, terminated, truncated, extras = env.step(action)
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
            fell_over = bool(terminated.item()) and not bool(succeeded.item())
            time_out = bool(truncated.item())
            break
          observations = TensorDict(
            observation_dict, batch_size=[1], device=device
          )
    finally:
      wrapped.close()
    progress_results.append(max_progress.item())
    success_results.append(float(succeeded.item()))
    smoothness_results.append(float(torch.cat(smoothness).mean()))
    episode_steps_results.append(float(episode_steps))
    fell_over_results.append(float(fell_over))
    time_out_results.append(float(time_out))
  return {
    "route_progress": float(np.mean(progress_results)),
    "route_success": float(np.mean(success_results)),
    "smoothness": float(np.mean(smoothness_results)),
    "episode_steps": float(np.mean(episode_steps_results)),
    "fell_over_rate": float(np.mean(fell_over_results)),
    "time_out_rate": float(np.mean(time_out_results)),
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
  command_mode: str = "xy",
  hidden_dims: tuple[int, ...] = (256, 128),
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
    command_mode=command_mode,
    hidden_dims=hidden_dims,
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
  command_mode: str = "xy",
  hidden_dims: tuple[int, ...] = (256, 128),
) -> Path:
  """Prepare the strict policy.pt, model.py, student.py grading folder."""
  student_path = _student_path(student_file)
  cfg = course_g1_navigation_env_cfg(
    mode, play=True, student_path=student_path, command_mode=command_mode
  )
  cfg.scene.num_envs = 1
  agent_cfg = course_g1_navigation_ppo_runner_cfg(
    mode, student_path, hidden_dims=hidden_dims
  )
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
  model_source = MODEL_DEPTH if mode == "depth" else MODEL_HEIGHT
  if mode == "height" and command_mode == "forward_yaw":
    model_source = MODEL_HEIGHT_FORWARD_YAW
  (build_dir / "model.py").write_text(model_source, encoding="utf-8")
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
