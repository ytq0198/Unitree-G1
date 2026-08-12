"""Seeded 5x5 course-project scene, portal metadata, WFC, and route graph."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

TileKind = Literal["pile", "platform_gap", "pyramid_stairs"]
Direction = Literal["north", "east", "south", "west"]
TILE_KINDS: tuple[TileKind, ...] = (
  "pile",
  "platform_gap",
  "pyramid_stairs",
)


@dataclass(frozen=True)
class Portal:
  direction: Direction
  neighbor: tuple[int, int]
  position: tuple[float, float]


@dataclass(frozen=True)
class Tile:
  row: int
  col: int
  kind: TileKind
  portals: tuple[Portal, ...]


@dataclass(frozen=True)
class NavigationScene:
  seed: int
  rows: int
  cols: int
  tile_size: float
  tiles: tuple[Tile, ...]
  route: tuple[tuple[float, float], ...]

  @property
  def route_length(self) -> float:
    points = np.asarray(self.route, dtype=np.float64)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=-1).sum())

  @property
  def route_graph(self) -> dict[tuple[int, int], tuple[tuple[int, int], ...]]:
    return {
      (tile.row, tile.col): tuple(portal.neighbor for portal in tile.portals)
      for tile in self.tiles
    }

  def tile_at(self, row: int, col: int) -> Tile:
    return self.tiles[row * self.cols + col]


def _neighbors(
  cell: tuple[int, int], rows: int, cols: int
) -> list[tuple[Direction, tuple[int, int]]]:
  row, col = cell
  candidates: tuple[tuple[Direction, tuple[int, int]], ...] = (
    ("north", (row + 1, col)),
    ("east", (row, col + 1)),
    ("south", (row - 1, col)),
    ("west", (row, col - 1)),
  )
  return [
    (direction, neighbor)
    for direction, neighbor in candidates
    if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols
  ]


def _collapse_domains(
  domains: dict[tuple[int, int], set[TileKind]],
  rows: int,
  cols: int,
  rng: np.random.Generator,
) -> dict[tuple[int, int], TileKind] | None:
  unresolved = [cell for cell, domain in domains.items() if len(domain) > 1]
  if not unresolved:
    return {cell: next(iter(domain)) for cell, domain in domains.items()}
  min_entropy = min(len(domains[cell]) for cell in unresolved)
  candidates = sorted(cell for cell in unresolved if len(domains[cell]) == min_entropy)
  cell = candidates[int(rng.integers(len(candidates)))]
  options = list(sorted(domains[cell]))
  rng.shuffle(options)
  for choice in options:
    updated = {key: set(value) for key, value in domains.items()}
    updated[cell] = {choice}
    valid = True
    queue = deque([cell])
    while queue and valid:
      current = queue.popleft()
      if len(updated[current]) != 1:
        continue
      current_kind = next(iter(updated[current]))
      for _, neighbor in _neighbors(current, rows, cols):
        if len(updated[neighbor]) > 1 and current_kind in updated[neighbor]:
          updated[neighbor].remove(current_kind)
          if not updated[neighbor]:
            valid = False
            break
          if len(updated[neighbor]) == 1:
            queue.append(neighbor)
        elif updated[neighbor] == {current_kind}:
          valid = False
          break
    if valid:
      result = _collapse_domains(updated, rows, cols, rng)
      if result is not None:
        return result
  return None


def _route_cells(
  rows: int,
  cols: int,
  start: tuple[int, int],
  goal: tuple[int, int],
  rng: np.random.Generator,
) -> list[tuple[int, int]]:
  queue: deque[tuple[int, int]] = deque([start])
  parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
  while queue:
    cell = queue.popleft()
    if cell == goal:
      break
    neighbors = [neighbor for _, neighbor in _neighbors(cell, rows, cols)]
    rng.shuffle(neighbors)
    for neighbor in neighbors:
      if neighbor not in parent:
        parent[neighbor] = cell
        queue.append(neighbor)
  if goal not in parent:
    raise RuntimeError("Generated route graph is disconnected")
  route = []
  current: tuple[int, int] | None = goal
  while current is not None:
    route.append(current)
    current = parent[current]
  return list(reversed(route))


def _random_start_cell(
  rows: int,
  cols: int,
  goal: tuple[int, int],
  rng: np.random.Generator,
) -> tuple[int, int]:
  max_distance = (rows - 1) + (cols - 1)
  min_distance = max(1, (max_distance + 1) // 2)
  candidates = [
    (row, col)
    for row in range(rows)
    for col in range(cols)
    if abs(row - goal[0]) + abs(col - goal[1]) >= min_distance
  ]
  if not candidates:
    raise ValueError("The navigation grid needs at least two distinct route cells")
  return candidates[int(rng.integers(len(candidates)))]


def sample_evaluation_seeds(
  seed: int,
  count: int = 10,
  grid_shape: tuple[int, int] = (5, 5),
) -> tuple[int, ...]:
  """Return seeds that produce ``count`` distinct random route starts."""
  rows, cols = grid_shape
  goal = (rows - 1, cols - 1)
  candidates = {
    (row, col)
    for row in range(rows)
    for col in range(cols)
    if abs(row - goal[0]) + abs(col - goal[1])
    >= max(1, ((rows - 1) + (cols - 1) + 1) // 2)
  }
  if not 1 <= count <= len(candidates):
    raise ValueError(
      f"count must be between 1 and {len(candidates)} for grid {grid_shape}"
    )

  selected: list[int] = []
  starts: set[tuple[int, int]] = set()
  candidate_seed = seed
  while len(selected) < count:
    start = _random_start_cell(
      rows, cols, goal, np.random.default_rng(candidate_seed)
    )
    if start not in starts:
      selected.append(candidate_seed)
      starts.add(start)
    candidate_seed += 1
  return tuple(selected)


def generate_navigation_scene(
  seed: int,
  grid_shape: tuple[int, int] = (5, 5),
  tile_size: float = 10.0,
  max_route_length: float = 100.0,
  random_start: bool = True,
) -> NavigationScene:
  """Generate a course navigation scene deterministically from ``seed``."""
  rows, cols = grid_shape
  if rows <= 0 or cols <= 0:
    raise ValueError("The course navigation grid dimensions must be positive")
  if tile_size <= 0.0:
    raise ValueError("The course navigation tile size must be positive")
  rng = np.random.default_rng(seed)
  goal = (rows - 1, cols - 1)
  start = _random_start_cell(rows, cols, goal, rng) if random_start else (0, 0)
  domains = {(row, col): set(TILE_KINDS) for row in range(rows) for col in range(cols)}
  domains[start] = {"pile"}
  domains[goal] = {"pile"}
  collapsed = _collapse_domains(domains, rows, cols, rng)
  if collapsed is None:
    raise RuntimeError("WFC could not satisfy the tile adjacency constraints")

  tiles = []
  for row in range(rows):
    for col in range(cols):
      portals = []
      center_x = (row + 0.5) * tile_size
      center_y = (col + 0.5) * tile_size
      for direction, neighbor in _neighbors((row, col), rows, cols):
        nr, nc = neighbor
        position = (
          0.5 * (center_x + (nr + 0.5) * tile_size),
          0.5 * (center_y + (nc + 0.5) * tile_size),
        )
        portals.append(Portal(direction, neighbor, position))
      tiles.append(Tile(row, col, collapsed[(row, col)], tuple(portals)))

  route_cells = _route_cells(rows, cols, start, goal, rng)
  route = tuple(
    ((row + 0.5) * tile_size, (col + 0.5) * tile_size) for row, col in route_cells
  )
  scene = NavigationScene(seed, rows, cols, tile_size, tuple(tiles), route)
  if scene.route_length > max_route_length + 1.0e-6:
    raise RuntimeError(
      f"Generated route is {scene.route_length:.2f} m; limit is {max_route_length:.2f} m"
    )
  return scene
