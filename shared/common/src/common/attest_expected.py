from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    from ._attest_expected import EXPECTED_SELF_SHA256 as _EXPECTED_SELF_SHA256
except Exception:  # pragma: no cover - generated file may be missing in dev
    _EXPECTED_SELF_SHA256: dict[str, str] = {}


@lru_cache(maxsize=1)
def _resolve_expected_entry() -> Optional[tuple[Path, str]]:
    try:
        import common.attest  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - native module absent
        return None

    module_path = Path(common.attest.__file__).resolve()  # type: ignore[attr-defined]
    expected = _EXPECTED_SELF_SHA256.get(module_path.name)
    if expected:
        return module_path, expected
    return None


def get_expected_self_sha256() -> Optional[str]:
    resolved = _resolve_expected_entry()
    if resolved is None:
        return None
    return resolved[1]
