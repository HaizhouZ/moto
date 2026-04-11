from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_local_extension():
    root = repo_root()
    lib_path = root / "build" / "libmoto.so"
    ext_candidates = sorted((root / "build" / "bindings").glob("moto_pywrap*.so"))
    if not lib_path.exists() or not ext_candidates:
        raise ImportError(
            "local moto build artifacts not found; expected build/libmoto.so and "
            "build/bindings/moto_pywrap*.so"
        )

    # Resolve the extension against the local libmoto instead of an installed copy.
    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

    ext_path = ext_candidates[0]
    spec = importlib.util.spec_from_file_location("moto.moto_pywrap", ext_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to create import spec for {ext_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["moto.moto_pywrap"] = module
    sys.modules["bindings.moto_pywrap"] = module
    spec.loader.exec_module(module)
    return module

