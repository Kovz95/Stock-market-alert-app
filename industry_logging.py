"""
Shim loader for the missing industry_logging module.

The original Python source is absent, but the compiled bytecode still
exists in __pycache__/industry_logging_legacy.cpython-311.pyc. Importing
this module executes the preserved bytecode so callers continue to work
until the source file is restored.
"""

from __future__ import annotations

import marshal
from pathlib import Path
from types import CodeType
from typing import Any, Dict

_LEGACY_BYTECODE = Path(__file__).parent / "__pycache__/industry_logging_legacy.cpython-311.pyc"
_PYC_HEADER_SIZE = 16
_BOOTSTRAPPED = False


def _load_legacy_code() -> CodeType:
    if not _LEGACY_BYTECODE.exists():
        raise FileNotFoundError(
            f"Missing industry_logging legacy bytecode at {_LEGACY_BYTECODE}. "
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
