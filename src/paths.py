"""Local source routing for the self-contained Course Project package."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = PROJECT_ROOT.parent
MJLAB_VERSION = "1.5.0"


def require_mjlab_version() -> None:
  """Require the published mjlab version used by the course."""
  try:
    installed = version("mjlab")
  except PackageNotFoundError as exc:
    raise RuntimeError(
      "mjlab==1.5.0 is required; run `pip install mjlab==1.5.0`"
    ) from exc
  if installed != MJLAB_VERSION:
    raise RuntimeError(
      f"mjlab=={MJLAB_VERSION} is required, but mjlab=={installed} is installed"
    )


def configure_local_sources() -> None:
  require_mjlab_version()
  os.environ.setdefault("MUJOCO_GL", "glfw")
  os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib")
  )
  os.environ.setdefault(
    "WARP_CACHE_PATH", str(PROJECT_ROOT / "outputs" / ".warp")
  )
  Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
  Path(os.environ["WARP_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
  text = str(PROJECT_ROOT)
  if text not in sys.path:
    sys.path.insert(0, text)


__all__ = [
  "MJLAB_VERSION",
  "PROJECT_ROOT",
  "configure_local_sources",
  "require_mjlab_version",
]
