"""Direct 29-joint G1 physical navigation environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from mjlab.terrains import TerrainEntityCfg

from src.amp.state import KEY_BODY_NAMES
from src.terrain import (
  generate_navigation_scene,
  make_navigation_terrain_generator,
)

from .commands import WaypointCommandCfg
from .observations import (
  NavigationTaskReward,
  amp_observation,
  normalized_depth,
  route_complete,
  route_progress_metric,
  student_smoothness_penalty,
  waypoint_velocity_tracking,
)
from .safe_events import reset_joints_by_offset_batched

ObservationMode = Literal["height", "depth"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDENT_PATH = PROJECT_ROOT / "student.py"
DEFAULT_MOTION_PATH = PROJECT_ROOT / "assets" / "motions" / "G1_walk_50hz.npz"


def course_g1_navigation_env_cfg(
  observation_mode: ObservationMode = "height",
  play: bool = False,
  student_path: str | Path = DEFAULT_STUDENT_PATH,
  scene_seed: int = 23,
  random_start: bool = True,
  navigation_reward_weight: float = 1.0,
  smoothness_reward_weight: float = -0.05,
  velocity_tracking_weight: float = 2.0,
) -> ManagerBasedRlEnvCfg:
  """Create the physical navigation task without a low-level policy layer."""
  if observation_mode not in ("height", "depth"):
    raise ValueError(f"Unknown observation mode: {observation_mode}")
  student_path = str(Path(student_path).resolve())
  navigation_scene = generate_navigation_scene(
    scene_seed,
    grid_shape=(5, 5),
    tile_size=10.0,
    max_route_length=100.0,
    random_start=random_start,
  )
  cfg = unitree_g1_rough_env_cfg(play=play)
  # This full 70 m primitive-geometry scene is stable up to 64 worlds with
  # the course server's MuJoCo-Warp build. Larger batches are run as separate
  # processes/GPUs instead of increasing nworld in one process.
  cfg.scene.num_envs = 32 if play else 64
  cfg.scene.extent = 7.0
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=make_navigation_terrain_generator(navigation_scene),
    max_init_terrain_level=0,
  )
  cfg.episode_length_s = 100.0
  cfg.seed = scene_seed
  cfg.auto_reset = False
  cfg.scale_rewards_by_dt = True
  cfg.events["reset_robot_joints"].func = reset_joints_by_offset_batched
  cfg.events.pop("randomize_terrain", None)
  if play:
    cfg.events.pop("push_robot", None)

  for sensor in cfg.scene.sensors or ():
    if hasattr(sensor, "debug_vis"):
      cast(Any, sensor).debug_vis = False

  cfg.commands = {
    "waypoint": WaypointCommandCfg(
      entity_name="robot",
      route=navigation_scene.route,
      student_path=student_path,
      waypoint_threshold=0.45,
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=False,
    )
  }
  for group_name in ("actor", "critic"):
    command_term = cfg.observations[group_name].terms["command"]
    command_term.func = env_mdp.generated_commands
    command_term.params = {"command_name": "waypoint"}

  amp_cfg = SceneEntityCfg("robot", body_names=KEY_BODY_NAMES)
  cfg.observations["amp"] = ObservationGroupCfg(
    terms={
      "state": ObservationTermCfg(
        func=amp_observation,
        params={"student_path": student_path, "asset_cfg": amp_cfg},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  if observation_mode == "depth":
    cfg.observations["actor"].terms.pop("height_scan", None)
    depth_sensor = CameraSensorCfg(
      name="depth",
      camera_name=None,
      parent_body="robot/torso_link",
      pos=(0.18, 0.0, 0.12),
      quat=(0.5, -0.5, 0.5, -0.5),
      fovy=70.0,
      width=80,
      height=60,
      data_types=("depth",),
      enabled_geom_groups=(0,),
      use_shadows=False,
      use_textures=False,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_sensor,)
    cfg.observations["depth"] = ObservationGroupCfg(
      terms={
        "depth": ObservationTermCfg(
          func=normalized_depth,
          params={"sensor_name": "depth", "student_path": student_path},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )

  native_rewards = cfg.rewards
  cfg.rewards = {
    "navigation_task": RewardTermCfg(
      func=NavigationTaskReward,
      weight=navigation_reward_weight,
      params={
        "command_name": "waypoint",
        "student_path": student_path,
        "step_dt": 0.02,
      },
    ),
    "alive": RewardTermCfg(func=env_mdp.is_alive, weight=0.5),
    "waypoint_velocity_tracking": RewardTermCfg(
      func=waypoint_velocity_tracking,
      weight=velocity_tracking_weight,
      params={"command_name": "waypoint", "std": 0.5},
    ),
    "upright": native_rewards["upright"],
    "dof_pos_limits": native_rewards["dof_pos_limits"],
    "action_rate_l2": native_rewards["action_rate_l2"],
    "student_smoothness": RewardTermCfg(
      func=student_smoothness_penalty,
      weight=smoothness_reward_weight,
      params={"student_path": student_path},
    ),
  }
  if "self_collisions" in native_rewards:
    cfg.rewards["self_collisions"] = native_rewards["self_collisions"]

  native_terminations = cfg.terminations
  cfg.terminations = {
    "time_out": native_terminations["time_out"],
    "fell_over": native_terminations["fell_over"],
    "route_complete": TerminationTermCfg(
      func=route_complete,
      params={"command_name": "waypoint"},
    ),
  }
  cfg.curriculum = {}
  cfg.metrics["route_progress"] = MetricsTermCfg(
    func=route_progress_metric,
    params={"command_name": "waypoint"},
  )
  return cfg
