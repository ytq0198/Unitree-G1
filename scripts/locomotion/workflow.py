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

from src.amp import ManualResetAmpVecEnvWrapper
from src.mjlab_tasks.env_cfgs import (
  course_g1_flat_validation_env_cfg,
  course_g1_rough_traversal_env_cfg,
  course_g1_rough_walk_env_cfg,
)
from src.mjlab_tasks.rl_cfg import course_g1_amp_ppo_runner_cfg
from src.paths import EXP_ROOT

ObservationMode = Literal["height", "depth"]
DEFAULT_STUDENT = EXP_ROOT / "student.py"
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


def latest_checkpoint(
  mode: ObservationMode | None = None,
  root: str | Path | None = None,
) -> Path:
  """Return the newest checkpoint, optionally restricted by observation mode."""
  search_root = Path(root or EXP_ROOT / "outputs" / "rsl_rl")
  if mode is not None:
    search_root = search_root / f"exp07_rough_amp_{mode}"
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
  cfg = course_g1_rough_walk_env_cfg(mode, student_path=_student_path(student_file))
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device=device)
  resets = 0
  try:
    observations, _ = env.reset()
    amp = _tensor_observation(observations, "amp")
    if amp.shape != (num_envs, 83):
      raise RuntimeError(f"Unexpected AMP shape: {amp.shape}")
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
    "manual_resets": resets,
    "amp_shape": (num_envs, 83),
    "depth_shape": depth_shape,
  }


def train(
  mode: ObservationMode = "height",
  *,
  num_envs: int = 4096,
  iterations: int = 600,
  steps_per_env: int = 24,
  device: str = "cuda:0",
  seed: int = 7,
  student_file: str | Path | None = None,
  resume_checkpoint: str | Path | None = None,
  training_phase: Literal["pretrain", "amp_flat", "rough", "walk"] = "walk",
) -> Path:
  """Train AMP-PPO and return the run directory containing checkpoints."""
  if device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(f"Requested {device}, but CUDA is not available")
  student_path = _student_path(student_file)
  env_cfg = course_g1_rough_walk_env_cfg(
    mode, student_path=student_path, training_phase=training_phase
  )
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  agent_cfg = course_g1_amp_ppo_runner_cfg(
    mode, student_path=student_path, training_phase=training_phase
  )
  agent_cfg.seed = seed
  agent_cfg.max_iterations = iterations
  agent_cfg.num_steps_per_env = steps_per_env
  agent_cfg.save_interval = max(1, min(50, iterations))
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
  timestamp = f"{timestamp}_{training_phase}_seed{seed}"
  log_dir = EXP_ROOT / "outputs" / "rsl_rl" / agent_cfg.experiment_name / timestamp
  log_dir.mkdir(parents=True, exist_ok=True)
  env = ManagerBasedRlEnv(env_cfg, device=device)
  wrapped = ManualResetAmpVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(
    wrapped, asdict(agent_cfg), log_dir=str(log_dir), device=device
  )
  if resume_checkpoint is not None:
    runner.load(str(resume_checkpoint), load_cfg=LOAD_CFG)
    checkpoint_text = Path(resume_checkpoint).as_posix().lower()
    reset_transition = (
      (training_phase == "amp_flat" and "pretrain" in checkpoint_text)
      or (
        training_phase == "rough"
        and ("pretrain" in checkpoint_text or "amp_flat" in checkpoint_text)
      )
      or (
        training_phase == "walk"
        and (
          "pretrain" in checkpoint_text
          or "amp_flat" in checkpoint_text
          or "_rough_" in checkpoint_text
        )
      )
    )
    if reset_transition:
      # Mjlab restores common_step_counter independently of the iteration
      # flag.  A flat prerequisite must warm-start rough training at curriculum
      # stage zero, not inherit the final pretrain command stage.
      wrapped.unwrapped.common_step_counter = 0
      print(f"[course] reset {training_phase} curriculum counter after warm-start")
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
  student_file: str | Path | None,
  evaluation_terrain: Literal["flat", "rough"] = "rough",
  model_phase: Literal["pretrain", "amp_flat", "rough", "walk"] | None = None,
  render_mode: str | None = None,
) -> tuple[ManagerBasedRlEnv, ManualResetAmpVecEnvWrapper, MjlabOnPolicyRunner]:
  student_path = _student_path(student_file)
  if evaluation_terrain == "flat":
    env_cfg = course_g1_flat_validation_env_cfg(mode, student_path)
    environment_phase = "pretrain"
  elif evaluation_terrain == "rough":
    env_cfg = course_g1_rough_traversal_env_cfg(mode, student_path)
    environment_phase = "walk"
  else:
    raise ValueError(f"Unknown evaluation terrain: {evaluation_terrain}")
  env_cfg.scene.num_envs = num_envs
  agent_cfg = course_g1_amp_ppo_runner_cfg(
    mode, student_path, training_phase=model_phase or environment_phase
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
  num_envs: int = 32,
  steps: int = 600,
  device: str = "cuda:0",
  student_file: str | Path | None = None,
  evaluation_terrain: Literal["flat", "rough"] = "rough",
  model_phase: Literal["pretrain", "amp_flat", "rough", "walk"] | None = None,
) -> dict[str, float]:
  """Measure per-episode tracking, survival, progress, and action smoothness."""
  env, wrapped, runner = _inference_runner(
    checkpoint,
    mode,
    num_envs=num_envs,
    device=device,
    student_file=student_file,
    evaluation_terrain=evaluation_terrain,
    model_phase=model_phase,
  )
  policy = runner.get_inference_policy(device=device)
  observations = wrapped.get_observations().to(device)
  start = env.scene["robot"].data.root_link_pos_w[:, :2].clone()
  episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
  completed_distances: list[torch.Tensor] = []
  completed_durations: list[torch.Tensor] = []
  completed_early: list[torch.Tensor] = []
  linear_errors: list[torch.Tensor] = []
  angular_errors: list[torch.Tensor] = []
  upright_flags: list[torch.Tensor] = []
  smoothness: list[torch.Tensor] = []
  previous = torch.zeros(num_envs, 29, device=device)
  previous_previous = previous.clone()
  try:
    with torch.inference_mode():
      for _ in range(steps):
        action = policy(observations)
        observations, _, dones, extras = wrapped.step(action)
        done_mask = dones.bool()
        command = env.command_manager.get_command("twist")
        if not isinstance(command, torch.Tensor):
          raise TypeError("Command 'twist' must be a tensor")
        robot = env.scene["robot"]
        valid = ~done_mask
        if valid.any():
          linear_errors.append(
            torch.norm(
              command[valid, :2] - robot.data.root_link_lin_vel_b[valid, :2],
              dim=-1,
            )
          )
          angular_errors.append(
            torch.abs(
              command[valid, 2] - robot.data.root_link_ang_vel_b[valid, 2]
            )
          )
          upright_flags.append(
            (robot.data.projected_gravity_b[valid, 2] < -0.75).float()
          )
        smoothness.append(
          torch.mean(torch.abs(action - 2 * previous + previous_previous), dim=-1)
        )
        episode_steps += 1
        position = robot.data.root_link_pos_w[:, :2].clone()
        terminal_position = extras.get("terminal_root_pos_w")
        if not isinstance(terminal_position, torch.Tensor):
          raise TypeError("Wrapper did not provide terminal_root_pos_w")
        position[done_mask] = terminal_position[done_mask, :2]
        # The course route advances along world +X.  Euclidean displacement
        # would incorrectly reward a policy that walks in circles.
        distance = position[:, 0] - start[:, 0]
        if done_mask.any():
          completed_distances.append(distance[done_mask].clone())
          completed_durations.append(
            episode_steps[done_mask].float() * float(env.step_dt)
          )
          time_outs = extras.get("time_outs")
          if not isinstance(time_outs, torch.Tensor):
            time_outs = torch.zeros_like(done_mask)
          completed_early.append((~time_outs[done_mask].bool()).float())
          start[done_mask] = robot.data.root_link_pos_w[done_mask, :2]
          episode_steps[done_mask] = 0
        previous_previous = previous
        previous = action
  except Exception:
    wrapped.close()
    raise
  active_distance = (
    env.scene["robot"].data.root_link_pos_w[:, 0] - start[:, 0]
  )
  active_duration = episode_steps.float() * float(env.step_dt)
  # If the evaluation horizon lands exactly on the environment timeout, the
  # wrapper has already reset and active_duration is zero.  Do not count those
  # zero-length fresh episodes in addition to the completed episodes.
  active_mask = episode_steps > 0
  distance_parts = [*completed_distances]
  duration_parts = [*completed_durations]
  early_parts = [*completed_early]
  if active_mask.any():
    distance_parts.append(active_distance[active_mask])
    duration_parts.append(active_duration[active_mask])
    early_parts.append(
      torch.zeros(
        int(active_mask.sum().item()), dtype=torch.float32, device=device
      )
    )
  all_distances = torch.cat(distance_parts)
  all_durations = torch.cat(duration_parts)
  all_early = torch.cat(early_parts)
  distance_target = 6.0 if evaluation_terrain == "rough" else 2.0
  progress = torch.clamp(all_distances / distance_target, 0.0, 1.0)
  upright_ratio = float(torch.cat(upright_flags).mean()) if upright_flags else 0.0
  distance_ok = all_distances >= distance_target
  walking_success_mask = distance_ok & (all_early < 0.5)
  metrics = {
    "linear_velocity_error": float(torch.cat(linear_errors).mean()),
    "angular_velocity_error": float(torch.cat(angular_errors).mean()),
    "traversal_progress": float(progress.mean()),
    "traversal_success": float(distance_ok.float().mean()),
    "walking_success": float(walking_success_mask.float().mean()),
    "upright_ratio": upright_ratio,
    "mean_episode_duration_s": float(all_durations.mean()),
    "early_termination_rate": float(all_early.mean()),
    "evaluated_episodes": float(all_distances.numel()),
    "smoothness": float(torch.cat(smoothness).mean()),
  }
  wrapped.close()
  return metrics


def record_video(
  checkpoint: str | Path,
  mode: ObservationMode = "height",
  *,
  frames: int = 600,
  warmup_steps: int = 50,
  device: str = "cuda:0",
  student_file: str | Path | None = None,
  output: str | Path | None = None,
  seed: int | None = None,
) -> Path:
  """Record a smooth locomotion MP4 without reset-induced teleports.

  The AMP training wrapper resets the env immediately when ``fell_over`` or
  ``time_out`` fires. If we keep recording after that, the robot appears to
  jump across the map. For demo videos we therefore:

  1. Keep the real ``fell_over`` termination used during evaluation.
  2. Render **before** each step and stop as soon as a termination occurs.
  3. Use a side-tracking camera on ``torso_link`` so leg swing is visible.
  """
  if frames < 150:
    raise ValueError("Evaluation videos must contain at least 150 frames")
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl.runner import MjlabOnPolicyRunner
  from mjlab.viewer.viewer_config import ViewerConfig

  student_path = _student_path(student_file)
  env_cfg = course_g1_rough_traversal_env_cfg(mode, student_path=student_path)
  env_cfg.scene.num_envs = 1
  if seed is not None:
    env_cfg.seed = seed
  env_cfg.episode_length_s = max(env_cfg.episode_length_s, frames * 0.02)
  # Demo video: stay on the easiest terrain tile and do not re-randomize mid-rollout.
  if env_cfg.scene.terrain is not None:
    env_cfg.scene.terrain.max_init_terrain_level = 0
  env_cfg.events.pop("randomize_terrain", None)
  env_cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_BODY
  env_cfg.viewer.entity_name = "robot"
  env_cfg.viewer.body_name = "torso_link"
  env_cfg.viewer.distance = 4.0
  env_cfg.viewer.azimuth = 100.0
  env_cfg.viewer.elevation = -12.0
  env_cfg.viewer.max_extra_envs = 0
  env_cfg.viewer.width = 640
  env_cfg.viewer.height = 480

  agent_cfg = course_g1_amp_ppo_runner_cfg(mode, student_path, training_phase="walk")
  env = ManagerBasedRlEnv(env_cfg, device=device, render_mode="rgb_array")
  wrapped = ManualResetAmpVecEnvWrapper(env)
  runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg=LOAD_CFG)
  policy = runner.get_inference_policy(device)
  observations = wrapped.get_observations().to(device)
  images: list[np.ndarray] = []
  try:
    with torch.inference_mode():
      for _ in range(warmup_steps):
        observations, _, dones, _ = wrapped.step(policy(observations))
        if dones.any():
          observations = wrapped.get_observations().to(device)
      for _ in range(frames):
        frame = env.render()
        if frame is None:
          raise RuntimeError("mjlab offscreen renderer returned no frame")
        grav = env.scene["robot"].data.projected_gravity_b[0, 2].item()
        if grav < -0.75:
          images.append(np.asarray(frame).copy())
        observations, _, dones, _ = wrapped.step(policy(observations))
        if dones.any():
          break
        if len(images) >= frames:
          break
  finally:
    wrapped.close()
  if len(images) < 150:
    raise RuntimeError(
      f"Video too short ({len(images)} frames). Policy terminated early; "
      "try a stronger checkpoint or reduce warmup."
    )
  output_path = Path(output or EXP_ROOT / "outputs" / "evaluation.mp4")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fps = max(1, int(round(1.0 / env.step_dt)))
  iio.imwrite(output_path, np.stack(images), fps=fps, codec="libx264")
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
  # Submission export must match the Codex/rough actor (512,256,128), not walk.
  cfg = course_g1_rough_walk_env_cfg(
    mode, play=True, student_path=student_path, training_phase="rough"
  )
  cfg.scene.num_envs = 1
  agent_cfg = course_g1_amp_ppo_runner_cfg(
    mode, student_path, training_phase="rough"
  )
  env = ManagerBasedRlEnv(cfg, device=device)
  wrapped = ManualResetAmpVecEnvWrapper(env)
  runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg=LOAD_CFG)
  build_dir = Path(output_dir or EXP_ROOT / "outputs" / "submission")
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


def plot_training_design(cfg: Any):
  """Visualize reward weights and the three-stage command curriculum."""
  import matplotlib.pyplot as plt

  rewards = {
    name: float(term.weight)
    for name, term in cfg.rewards.items()
    if float(term.weight) != 0.0
  }
  curriculum = cfg.curriculum["course_command_schedule"].params["velocity_stages"]
  figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
  names = list(rewards)
  values = [rewards[name] for name in names]
  axes[0].barh(
    names, values, color=["#2878b5" if value > 0 else "#d1495b" for value in values]
  )
  axes[0].axvline(0.0, color="#222222", linewidth=0.8)
  axes[0].set_title("Reward weights")
  axes[0].grid(axis="x", alpha=0.2)
  steps = [stage["step"] for stage in curriculum]
  for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
    maxima = [max(abs(value) for value in stage[key]) for stage in curriculum]
    axes[1].step(steps, maxima, where="post", marker="o", label=key)
  axes[1].set_title("Command curriculum")
  axes[1].set_xlabel("global step")
  axes[1].legend()
  axes[1].grid(alpha=0.2)
  figure.tight_layout()
  return figure


__all__ = [
  "evaluate",
  "latest_checkpoint",
  "plot_training_design",
  "prepare_submission",
  "record_video",
  "smoke",
  "train",
]
