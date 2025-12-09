"""
Compatibility shim for the missing redis_support source module.

The original file is unavailable, but the compiled bytecode survives in
__pycache__/redis_support_legacy.cpython-311.pyc. Importing this module
loads that bytecode so existing code keeps working until the source is
recovered.
"""

from __future__ import annotations

import marshal
from pathlib import Path
from types import CodeType
from typing import Dict, Any

_LEGACY_BYTECODE = Path(__file__).parent / "__pycache__/redis_support_legacy.cpython-311.pyc"
_PYC_HEADER_SIZE = 16
_BOOTSTRAPPED = False


def _load_legacy_code() -> CodeType:
    if not _LEGACY_BYTECODE.exists():
        raise FileNotFoundError(
            f"Missing redis_support legacy bytecode at {_LEGACY_BYTECODE}. "
            "Cannot bootstrap the module."
        )
    with _LEGACY_BYTECODE.open("rb") as fh:
        fh.read(_PYC_HEADER_SIZE)
        return marshal.load(fh)


def _bootstrap(namespace: Dict[str, Any]) -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    code = _load_legacy_code()
    exec(code, namespace)
    _BOOTSTRAPPED = True


_bootstrap(globals())
