"""PyQt6 GUI for the voice dubbing workflow."""

from __future__ import annotations

import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"

if CODE_ROOT.exists():
    code_root_text = str(CODE_ROOT)
    if code_root_text not in sys.path:
        sys.path.insert(0, code_root_text)
