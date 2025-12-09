"""
Shim loader for scheduler_watchdog_v2.

This auto-generated file loads the preserved bytecode so imports keep
working until the original source is restored.
"""

from __future__ import annotations

import marshal
from pathlib import Path
from types import CodeType
from typing import Any, Dict

_LEGACY_BYTECODE = Path(__file__).parent / "__pycache__/scheduler_watchdog_v2_legacy.cpython-311.pyc"
_PYC_HEADER_SIZE = 16
_BOOTSTRAPPED = False


def _load_legacy_code() -> CodeType:
    if not _LEGACY_BYTECODE.exists():
        raise FileNotFoundError(
            f"Missing legacy bytecode for scheduler_watchdog_v2 at {_LEGACY_BYTECODE}."
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
