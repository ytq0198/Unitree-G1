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
    height = float(rng.uniform(0.15, 0.45))
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
) -> list[TerrainGeometry]:
  ox, oy = origin
  result = [
    _box(
      body,
      (ox + tile_size / 2, oy + tile_size / 2, -0.85),
      (tile_size / 2, tile_size / 2, 0.05),
      (0.12, 0.14, 0.18, 1.0),
    )
  ]
  gap = 0.45
  platform_count = 7
  platform_length = (tile_size - (platform_count - 1) * gap) / platform_count
  for index in range(platform_count):
    start = index * (platform_length + gap)
    result.append(
      _box(
        body,
        (ox + start + platform_length / 2, oy + tile_size / 2, -0.04),
        (platform_length / 2, tile_size / 2, 0.04),
        (0.42, 0.46, 0.52, 1.0),
      )
    )
  return result


def _stairs_geometries(
  body: mujoco.MjsBody,
  origin: tuple[float, float],
  tile_size: float,
) -> list[TerrainGeometry]:
  ox, oy = origin
  heights = (
    0.0,
    0.08,
    0.16,
    0.24,
    0.32,
    0.40,
    0.32,
    0.24,
    0.16,
    0.08,
    0.0,
  )
  width = tile_size / len(heights)
  result = []
  for index, height in enumerate(heights):
    solid_height = max(0.05, height + 0.05)
    result.append(
      _box(
        body,
        (
          ox + (index + 0.5) * width,
          oy + tile_size / 2,
          height - solid_height / 2,
        ),
        (width / 2, tile_size / 2, solid_height / 2),
        (0.48, 0.39 + 0.04 * index, 0.30, 1.0),
      )
    )
  return result


@dataclass(kw_only=True)
class NavigationSceneSubTerrainCfg(SubTerrainCfg):
  """Adapt one generated scene to mjlab's public SubTerrainCfg API."""

  scene: NavigationScene

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
      "pile": lambda origin: _pile_geometries(body, origin, self.scene.tile_size, rng),
      "platform_gap": lambda origin: _platform_gap_geometries(
        body, origin, self.scene.tile_size
      ),
      "pyramid_stairs": lambda origin: _stairs_geometries(
        body, origin, self.scene.tile_size
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


def make_navigation_terrain_generator(scene: NavigationScene) -> TerrainGeneratorCfg:
  """Return a one-patch generator containing the complete shared route scene."""
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
      "navigation_scene": NavigationSceneSubTerrainCfg(scene=scene, size=size)
    },
  )
