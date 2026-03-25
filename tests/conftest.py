from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
_MISSING = object()

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@contextmanager
def patched_modules(stub_modules: dict[str, ModuleType]) -> Iterator[None]:
    original_modules: dict[str, object] = {}

    try:
        for module_name, module in stub_modules.items():
            original_modules[module_name] = sys.modules.get(module_name, _MISSING)
            sys.modules[module_name] = module
        yield
    finally:
        for module_name, original_module in original_modules.items():
            if original_module is _MISSING:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module


@pytest.fixture
def load_backend_module():
    def _load(module_basename: str, stub_modules: dict[str, ModuleType] | None = None):
        module_name = f"test_{module_basename}_{uuid4().hex}"
        module_path = BACKEND_DIR / f"{module_basename}.py"
        with patched_modules(stub_modules or {}):
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load backend module: {module_basename}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

    return _load
