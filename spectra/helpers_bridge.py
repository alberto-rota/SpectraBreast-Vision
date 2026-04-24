"""Bridge to ``helpers.py`` at the repo root so the package keeps working
regardless of how it's launched (``python -m spectra`` or via the legacy
pipeline scripts).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from helpers import xyzeuler_to_hmat  # noqa: E402

__all__ = ["xyzeuler_to_hmat"]
