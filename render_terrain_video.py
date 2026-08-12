#!/usr/bin/env python3
"""Render a Course Project route-follow video with Open3D."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.terrain import (
  COURSE_PROJECT_BORDER_WIDTH,
  NavigationScene,
  NavigationSceneSubTerrainCfg,
  generate_navigation_scene,
)

BACKGROUND = np.array([0.035, 0.045, 0.065, 1.0], dtype=np.float32)
ROUTE_COLOR = np.array([0.95, 0.64, 0.16], dtype=np.float64)
START_COLOR = np.array([0.18, 0.78, 0.42], dtype=np.float64)
GOAL_COLOR = np.array([0.95, 0.25, 0.28], dtype=np.float64)
BORDER_COLOR = np.array([0.24, 0.26, 0.29], dtype=np.float64)


def _open3d():
  try:
    import open3d as o3d
  except ImportError as exc:
    raise RuntimeError("Open3D is required: pip install open3d") from exc
  return o3d


def _box_mesh(o3d, center, half_size, color):
  size = 2.0 * np.asarray(half_size, dtype=np.float64)
  mesh = o3d.geometry.TriangleMesh.create_box(*size)
  mesh.translate(np.asarray(center, dtype=np.float64) - 0.5 * size)
  mesh.paint_uniform_color(np.asarray(color[:3], dtype=np.float64))
  mesh.compute_vertex_normals()
  return mesh


def _cylinder_between(o3d, start, end, radius, color):
  start = np.asarray(start, dtype=np.float64)
  end = np.asarray(end, dtype=np.float64)
  vector = end - start
  length = float(np.linalg.norm(vector))
  mesh = o3d.geometry.TriangleMesh.create_cylinder(
    radius=radius, height=length, resolution=20
  )
  direction = vector / length
  z_axis = np.array([0.0, 0.0, 1.0])
  dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
  if dot < 1.0 - 1.0e-8:
    if dot < -1.0 + 1.0e-8:
      axis_angle = np.array([np.pi, 0.0, 0.0])
    else:
      axis = np.cross(z_axis, direction)
      axis /= np.linalg.norm(axis)
      axis_angle = axis * np.arccos(dot)
    mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle))
  mesh.translate(0.5 * (start + end))
  mesh.paint_uniform_color(color)
  mesh.compute_vertex_normals()
  return mesh


def _sphere(o3d, position, radius, color):
  mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=24)
  mesh.translate(np.asarray(position, dtype=np.float64))
  mesh.paint_uniform_color(color)
  mesh.compute_vertex_normals()
  return mesh


def _flat_border_mesh(o3d, inner_size: tuple[float, float], border_width: float):
  inner_x, inner_y = inner_size
  total_x = inner_x + 2.0 * border_width
  mesh = o3d.geometry.TriangleMesh()
  boxes = (
    ((inner_x / 2.0, -border_width / 2.0, -0.05), (total_x / 2.0, border_width / 2.0, 0.05)),
    (
      (inner_x / 2.0, inner_y + border_width / 2.0, -0.05),
      (total_x / 2.0, border_width / 2.0, 0.05),
    ),
    ((-border_width / 2.0, inner_y / 2.0, -0.05), (border_width / 2.0, inner_y / 2.0, 0.05)),
    (
      (inner_x + border_width / 2.0, inner_y / 2.0, -0.05),
      (border_width / 2.0, inner_y / 2.0, 0.05),
    ),
  )
  for center, half_size in boxes:
    mesh += _box_mesh(o3d, center, half_size, BORDER_COLOR)
  return mesh


def _point_along_route(
  route_xy: np.ndarray, cumulative_length: np.ndarray, distance: float
) -> np.ndarray:
  total_length = float(cumulative_length[-1])
  distance = float(np.clip(distance, 0.0, total_length))
  segment = min(
    int(np.searchsorted(cumulative_length, distance, side="right") - 1),
    len(route_xy) - 2,
  )
  segment = max(0, segment)
  segment_length = cumulative_length[segment + 1] - cumulative_length[segment]
  alpha = (distance - cumulative_length[segment]) / segment_length
  return route_xy[segment] + alpha * (route_xy[segment + 1] - route_xy[segment])


def _build_scene_geometry(
  seed: int,
  grid_shape: tuple[int, int],
  tile_size: float,
  border_width: float,
):
  o3d = _open3d()
  if grid_shape[0] != grid_shape[1]:
    raise ValueError("The Course Project terrain grid must be square")
  max_route_length = (sum(grid_shape) - 2) * tile_size
  navigation_scene = generate_navigation_scene(
    seed,
    grid_shape=grid_shape,
    tile_size=tile_size,
    max_route_length=max_route_length,
    random_start=True,
  )

  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  terrain_cfg = NavigationSceneSubTerrainCfg(
    scene=navigation_scene,
    size=(
      navigation_scene.rows * navigation_scene.tile_size,
      navigation_scene.cols * navigation_scene.tile_size,
    ),
  )
  output = terrain_cfg.function(0.0, spec, np.random.default_rng(seed))

  terrain_mesh = o3d.geometry.TriangleMesh()
  for item in output.geometries:
    if item.geom.type != mujoco.mjtGeom.mjGEOM_BOX:
      raise ValueError(f"Unsupported terrain geometry: {item.geom.type}")
    terrain_mesh += _box_mesh(o3d, item.geom.pos, item.geom.size, item.color)
  inner_size = (
    navigation_scene.rows * navigation_scene.tile_size,
    navigation_scene.cols * navigation_scene.tile_size,
  )
  terrain_mesh += _flat_border_mesh(o3d, inner_size, border_width)

  route_z = 0.65
  route = np.column_stack(
    [
      np.asarray(navigation_scene.route, dtype=np.float64),
      np.full(len(navigation_scene.route), route_z),
    ]
  )
  route_mesh = o3d.geometry.TriangleMesh()
  for start, end in zip(route[:-1], route[1:], strict=True):
    route_mesh += _cylinder_between(o3d, start, end, 0.18, ROUTE_COLOR)

  start = _sphere(o3d, route[0], 0.50, START_COLOR)
  goal = _sphere(o3d, route[-1], 0.60, GOAL_COLOR)
  return navigation_scene, terrain_mesh, route_mesh, start, goal


def _overlay(
  frame: np.ndarray,
  seed: int,
  navigation_scene: NavigationScene,
  border_width: float,
) -> np.ndarray:
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image, "RGBA")
  try:
    font = ImageFont.truetype(
      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
    )
    small = ImageFont.truetype(
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18
    )
  except OSError:
    font = ImageFont.load_default()
    small = font

  width, height = image.size
  title = "Course Project | Navigation Terrain"
  subtitle = (
    f"{navigation_scene.rows} x {navigation_scene.cols} tiles   "
    f"tile {navigation_scene.tile_size:g} x {navigation_scene.tile_size:g} m   "
    f"total {navigation_scene.rows * navigation_scene.tile_size + 2 * border_width:g} x "
    f"{navigation_scene.cols * navigation_scene.tile_size + 2 * border_width:g} m   "
    f"seed {seed}   route {navigation_scene.route_length:.1f} m"
  )
  title_right = draw.textbbox((42, 34), title, font=font)[2]
  subtitle_right = draw.textbbox((42, 70), subtitle, font=small)[2]
  panel_right = min(width - 24, max(title_right, subtitle_right) + 18)
  draw.rounded_rectangle(
    (24, 22, panel_right, 102), radius=8, fill=(10, 15, 25, 190)
  )
  draw.text((42, 34), title, font=font, fill="white")
  draw.text((42, 70), subtitle, font=small, fill=(205, 216, 231))

  legend_y = height - 48
  draw.rounded_rectangle(
    (width - 330, legend_y - 12, width - 24, height - 18),
    radius=8,
    fill=(10, 15, 25, 190),
  )
  entries = (("start", START_COLOR), ("route", ROUTE_COLOR), ("goal", GOAL_COLOR))
  x = width - 310
  for label, color in entries:
    rgb = tuple(int(value * 255) for value in color)
    draw.ellipse((x, legend_y, x + 14, legend_y + 14), fill=rgb + (255,))
    draw.text((x + 21, legend_y - 4), label, font=small, fill="white")
    x += 92
  return np.asarray(image)


def render_video(
  output_path: Path,
  *,
  seed: int,
  grid_shape: tuple[int, int],
  tile_size: float,
  border_width: float,
  camera_height: float,
  look_ahead: float,
  duration: float,
  fps: int,
  width: int,
  height: int,
) -> None:
  if (
    duration <= 0.0
    or fps <= 0
    or width <= 0
    or height <= 0
    or camera_height <= 0.0
    or look_ahead <= 0.0
    or tile_size <= 0.0
    or border_width <= 0.0
  ):
    raise ValueError("render dimensions, timing, and camera values must be positive")

  o3d = _open3d()
  navigation_scene, terrain, route, start, goal = _build_scene_geometry(
    seed, grid_shape, tile_size, border_width
  )
  rendering = o3d.visualization.rendering
  renderer = rendering.OffscreenRenderer(width, height)
  scene = renderer.scene
  scene.set_background(BACKGROUND)
  scene.set_lighting(
    rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
    np.array([-0.45, -0.55, -0.70], dtype=np.float32),
  )
  scene.scene.enable_sun_light(True)

  terrain_material = rendering.MaterialRecord()
  terrain_material.shader = "defaultLit"
  terrain_material.base_roughness = 0.76
  terrain_material.base_reflectance = 0.18

  accent_material = rendering.MaterialRecord()
  accent_material.shader = "defaultLit"
  accent_material.base_roughness = 0.45
  accent_material.base_reflectance = 0.28

  scene.add_geometry("terrain", terrain, terrain_material)
  scene.add_geometry("route", route, accent_material)
  scene.add_geometry("start", start, accent_material)
  scene.add_geometry("goal", goal, accent_material)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  frame_count = max(2, int(round(duration * fps)))
  writer = imageio.get_writer(
    str(output_path),
    fps=fps,
    codec="libx264",
    format="FFMPEG",
    macro_block_size=1,
    ffmpeg_params=["-crf", "18"],
  )
  route_xy = np.asarray(navigation_scene.route, dtype=np.float32)
  segment_lengths = np.linalg.norm(np.diff(route_xy, axis=0), axis=1)
  cumulative_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
  total_length = float(cumulative_length[-1])
  up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
  overview_frames = min(frame_count // 4, 2 * fps)
  route_frames = frame_count - overview_frames
  inner_x = navigation_scene.rows * navigation_scene.tile_size
  inner_y = navigation_scene.cols * navigation_scene.tile_size
  overview_target = np.array([inner_x / 2.0, inner_y / 2.0, 0.0], dtype=np.float32)
  try:
    for index in range(frame_count):
      if index < overview_frames:
        overview_phase = index / max(1, overview_frames - 1)
        azimuth = np.deg2rad(-45.0 + 20.0 * overview_phase)
        elevation = np.deg2rad(46.0)
        radius = 110.0
        eye = overview_target + radius * np.array(
          [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
          ],
          dtype=np.float32,
        )
        renderer.setup_camera(55.0, overview_target, eye, up)
      else:
        route_index = index - overview_frames
        distance = total_length * route_index / max(1, route_frames - 1)
        camera_distance = max(0.0, distance - 0.75 * look_ahead)
        target_distance = min(distance + 0.25 * look_ahead, total_length)
        camera_xy = _point_along_route(
          route_xy, cumulative_length, camera_distance
        )
        target_xy = _point_along_route(
          route_xy, cumulative_length, target_distance
        )
        eye = np.array(
          [camera_xy[0], camera_xy[1], camera_height], dtype=np.float32
        )
        target = np.array([target_xy[0], target_xy[1], 0.15], dtype=np.float32)
        renderer.setup_camera(52.0, target, eye, up)
      frame = np.asarray(renderer.render_to_image())
      writer.append_data(_overlay(frame, seed, navigation_scene, border_width))
      if (index + 1) % max(1, frame_count // 10) == 0:
        print(f"Rendered {index + 1}/{frame_count} frames")
  finally:
    writer.close()
    scene.clear_geometry()

  tile_kinds = ", ".join(tile.kind for tile in navigation_scene.tiles)
  print(f"Saved {output_path.resolve()}")
  print(f"Terrain tiles: {tile_kinds}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output", type=Path, default=Path("outputs/course_project_terrain_open3d.mp4")
  )
  parser.add_argument("--seed", type=int, default=23)
  parser.add_argument("--grid-shape", type=int, nargs=2, default=(5, 5))
  parser.add_argument("--tile-size", type=float, default=10.0)
  parser.add_argument("--border-width", type=float, default=COURSE_PROJECT_BORDER_WIDTH)
  parser.add_argument("--camera-height", type=float, default=6.0)
  parser.add_argument("--look-ahead", type=float, default=8.0)
  parser.add_argument("--duration", type=float, default=16.0)
  parser.add_argument("--fps", type=int, default=30)
  parser.add_argument("--resolution", type=int, nargs=2, default=(1280, 720))
  args = parser.parse_args()
  render_video(
    args.output,
    seed=args.seed,
    grid_shape=tuple(args.grid_shape),
    tile_size=args.tile_size,
    border_width=args.border_width,
    camera_height=args.camera_height,
    look_ahead=args.look_ahead,
    duration=args.duration,
    fps=args.fps,
    width=args.resolution[0],
    height=args.resolution[1],
  )


if __name__ == "__main__":
  main()
