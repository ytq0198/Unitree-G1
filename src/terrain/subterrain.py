"""Primitive MuJoCo geometry adapter for a generated navigation scene."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)

from .scene import NavigationScene, TileKind

COURSE_PROJECT_BORDER_WIDTH = 10.0


def _box(
  body: mujoco.MjsBody,
  position: tuple[float, float, float],
  size: tuple[float, float, float],
  color: tuple[float, float, float, float],
) -> TerrainGeometry:
  geom = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=position,
    size=size,
  )
  return TerrainGeometry(geom=geom, color=color)


def _pile_geometries(
  body: mujoco.MjsBody,
  origin: tuple[float, float],
  tile_size: float,
  rng: np.random.Generator,
  challenge_scale: float,
) -> list[TerrainGeometry]:
  ox, oy = origin
  result = [
    _box(
      body,
      (ox + tile_size / 2, oy + tile_size / 2, -0.05),
      (tile_size / 2, tile_size / 2, 0.05),
      (0.35, 0.37, 0.40, 1.0),
    )
  ]
  for _ in range(36):
    half_width = float(rng.uniform(0.18, 0.35))
    x = float(rng.uniform(0.75, tile_size - 0.75))
    y = float(rng.uniform(0.75, tile_size - 0.75))
    clear_half_width = 1.0 + half_width
    if (
      abs(x - tile_size / 2) < clear_half_width
      or abs(y - tile_size / 2) < clear_half_width
    ):
      continue
    height = float(
      rng.uniform(0.05 + 0.10 * challenge_scale, 0.12 + 0.33 * challenge_scale)
    )
    result.append(
      _box(
        body,
        (ox + x, oy + y, height / 2),
        (half_width, half_width, height / 2),
        (0.30, 0.48, 0.34, 1.0),
      )
    )
  return result


def _platform_gap_geometries(
  body: mujoco.MjsBody,
  origin: tuple[float, float],
  tile_size: float,
  challenge_scale: float,
) -> list[TerrainGeometry]:
  ox, oy = origin
  center_x = ox + tile_size / 2
  center_y = oy + tile_size / 2
  platform_half = min(1.0, tile_size / 4)
  gap_width = 0.25 * challenge_scale
  gap_half = platform_half + gap_width
  ground_half = tile_size / 2
  ground_color = (0.42, 0.46, 0.52, 1.0)
  result = [
    _box(
      body,
      (center_x, center_y, -0.85),
      (ground_half, ground_half, 0.05),
      (0.12, 0.14, 0.18, 1.0),
    ),
    _box(
      body,
      (center_x, center_y, -0.04),
      (platform_half, platform_half, 0.04),
      ground_color,
    ),
  ]
  side_half = (ground_half - gap_half) / 2
  side_offset = (ground_half + gap_half) / 2
  result.extend(
    (
      _box(
        body,
        (center_x - side_offset, center_y, -0.04),
        (side_half, ground_half, 0.04),
        ground_color,
      ),
      _box(
        body,
        (center_x + side_offset, center_y, -0.04),
        (side_half, ground_half, 0.04),
        ground_color,
      ),
      _box(
        body,
        (center_x, center_y - side_offset, -0.04),
        (gap_half, side_half, 0.04),
        ground_color,
      ),
      _box(
        body,
        (center_x, center_y + side_offset, -0.04),
        (gap_half, side_half, 0.04),
        ground_color,
      ),
    )
  )
  return result


def _stairs_geometries(
  body: mujoco.MjsBody,
  origin: tuple[float, float],
  tile_size: float,
  challenge_scale: float,
) -> list[TerrainGeometry]:
  ox, oy = origin
  center_x = ox + tile_size / 2
  center_y = oy + tile_size / 2
  level_count = 5
  platform_half = min(1.0, tile_size / 4)
  step_width = (tile_size / 2 - platform_half) / level_count
  step_height = 0.08 * challenge_scale
  result = []
  for level in range(level_count):
    outer_half = tile_size / 2 - level * step_width
    inner_half = tile_size / 2 - (level + 1) * step_width
    height = level * step_height
    solid_height = height + 0.05
    ring_half_width = (outer_half - inner_half) / 2
    ring_offset = (outer_half + inner_half) / 2
    color = (0.48, 0.39 + 0.04 * level, 0.30, 1.0)
    result.extend(
      (
        _box(
          body,
          (center_x - ring_offset, center_y, height - solid_height / 2),
          (ring_half_width, outer_half, solid_height / 2),
          color,
        ),
        _box(
          body,
          (center_x + ring_offset, center_y, height - solid_height / 2),
          (ring_half_width, outer_half, solid_height / 2),
          color,
        ),
        _box(
          body,
          (center_x, center_y - ring_offset, height - solid_height / 2),
          (inner_half, ring_half_width, solid_height / 2),
          color,
        ),
        _box(
          body,
          (center_x, center_y + ring_offset, height - solid_height / 2),
          (inner_half, ring_half_width, solid_height / 2),
          color,
        ),
      )
    )
  center_height = level_count * step_height
  center_solid_height = center_height + 0.05
  result.append(
    _box(
      body,
      (center_x, center_y, center_height - center_solid_height / 2),
      (platform_half, platform_half, center_solid_height / 2),
      (0.48, 0.39 + 0.04 * level_count, 0.30, 1.0),
    )
  )
  return result


@dataclass(kw_only=True)
class NavigationSceneSubTerrainCfg(SubTerrainCfg):
  """Adapt one generated scene to mjlab's public SubTerrainCfg API."""

  scene: NavigationScene
  challenge_scale: float = 1.0

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    del difficulty
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []
    generators = {
      "pile": lambda origin: _pile_geometries(
        body, origin, self.scene.tile_size, rng, self.challenge_scale
      ),
      "platform_gap": lambda origin: _platform_gap_geometries(
        body, origin, self.scene.tile_size, self.challenge_scale
      ),
      "pyramid_stairs": lambda origin: _stairs_geometries(
        body, origin, self.scene.tile_size, self.challenge_scale
      ),
    }
    for tile in self.scene.tiles:
      kind: TileKind = tile.kind
      origin = (tile.row * self.scene.tile_size, tile.col * self.scene.tile_size)
      geometries.extend(generators[kind](origin))
    start = self.scene.route[0]
    return TerrainOutput(
      origin=np.array([start[0], start[1], 0.0]),
      geometries=geometries,
    )


def make_navigation_terrain_generator(
  scene: NavigationScene, challenge_scale: float = 1.0
) -> TerrainGeneratorCfg:
  """Return a one-patch generator containing the complete shared route scene."""
  if not 0.0 <= challenge_scale <= 1.0:
    raise ValueError("challenge_scale must be in [0, 1]")
  size = (scene.rows * scene.tile_size, scene.cols * scene.tile_size)
  return TerrainGeneratorCfg(
    seed=scene.seed,
    curriculum=False,
    size=size,
    border_width=COURSE_PROJECT_BORDER_WIDTH,
    num_rows=1,
    num_cols=1,
    color_scheme="none",
    sub_terrains={
      "navigation_scene": NavigationSceneSubTerrainCfg(
        scene=scene, size=size, challenge_scale=challenge_scale
      )
    },
  )


def make_multi_scene_terrain_generator(
  scenes: tuple[NavigationScene, ...], challenge_scale: float = 1.0
) -> TerrainGeneratorCfg:
  """Place complete navigation scenes in separate terrain columns."""
  if not scenes:
    raise ValueError("At least one navigation scene is required")
  if not 0.0 <= challenge_scale <= 1.0:
    raise ValueError("challenge_scale must be in [0, 1]")
  first = scenes[0]
  if any(
    (scene.rows, scene.cols, scene.tile_size)
    != (first.rows, first.cols, first.tile_size)
    for scene in scenes
  ):
    raise ValueError("All navigation scenes must use the same grid geometry")
  size = (first.rows * first.tile_size, first.cols * first.tile_size)
  return TerrainGeneratorCfg(
    seed=first.seed,
    curriculum=True,
    size=size,
    border_width=COURSE_PROJECT_BORDER_WIDTH,
    num_rows=1,
    num_cols=len(scenes),
    color_scheme="none",
    sub_terrains={
      f"navigation_scene_{index}": NavigationSceneSubTerrainCfg(
        scene=scene, size=size, challenge_scale=challenge_scale
      )
      for index, scene in enumerate(scenes)
    },
  )
