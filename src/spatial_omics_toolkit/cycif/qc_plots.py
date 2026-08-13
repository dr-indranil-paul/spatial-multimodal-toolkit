"""Shared matplotlib helpers for the per-step QC artifacts written by steps 06-12.

Every step in this pipeline that takes an ``output_dir`` argument writes a
human-inspectable PNG (or CSV) alongside its returned data, per this lab's
requirement that a person can look at real output at each stage and confirm
it looks right before moving to the next one — never just trust an in-memory
object silently handed to the next function. These are notebook diagnostic
plots (headless-safe, ``Agg`` backend), not publication figures — they do not
follow the manuscript figure-style rules (5pt fonts, exact canvas size),
which apply only to figures destined for a paper.
"""

from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PathLike = Union[str, Path]


def ensure_dir(output_dir: PathLike) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(fig, path: Path, verbose: bool = True) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"  QC plot -> {path}")
    return path
