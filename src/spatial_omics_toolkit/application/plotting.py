from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def ensure_arial() -> str:
    """Register the shared Arial files and reject silent font substitution."""
    from matplotlib import font_manager

    family = font_manager.FontProperties(family="Arial", style="normal", weight="normal")
    try:
        return str(font_manager.findfont(family, fallback_to_default=False))
    except ValueError:
        software_root = Path(os.environ.get("PAULI_FAST_SOFTWARES", "/home/groups/EmiliLab/pauli/00_environments"))
        font_dir = software_root / "share" / "fonts" / "arial"
        if font_dir.is_dir():
            for font in sorted(font_dir.iterdir()):
                if font.is_file() and font.suffix.lower() == ".ttf":
                    font_manager.fontManager.addfont(str(font))
        try:
            return str(font_manager.findfont(family, fallback_to_default=False))
        except ValueError as error:
            raise RuntimeError(f"Arial is required for scientific plots; install/register the approved font under {font_dir}.") from error
