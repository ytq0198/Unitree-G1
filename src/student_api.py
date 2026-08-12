"""Load the Course Project student file by exact path."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Callable


@lru_cache(maxsize=8)
def _load_cached(path_text: str, mtime_ns: int) -> ModuleType:
  del mtime_ns
  path = Path(path_text)
  module_name = f"course_project_student_{abs(hash(path))}_{path.stat().st_mtime_ns}"
  spec = importlib.util.spec_from_file_location(module_name, path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load student module from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def load_student_module(path: str | Path) -> ModuleType:
  resolved = Path(path).resolve()
  if not resolved.is_file():
    raise FileNotFoundError(f"Student file does not exist: {resolved}")
  return _load_cached(str(resolved), resolved.stat().st_mtime_ns)


def load_student_function(path: str | Path, name: str) -> Callable:
  function = getattr(load_student_module(path), name, None)
  if not callable(function):
    raise AttributeError(f"student.py must define callable {name}()")
  return function
