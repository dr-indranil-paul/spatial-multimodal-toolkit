"""Plot styling helper providing Matplotlib and Plotnine publication styles.

Enforces the project plot contract: genuine Arial typography, 5 pt normal text,
0.5 pt strokes, bounded spines, subtle major grid behind data, editable PDF/PS
fonts (Type 42), fixed canvas (no tight bbox), un-interpolated images, and a
colorblind-safe qualitative color cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from plotnine.themes.theme import theme as PlotnineTheme


COLORBLIND_SAFE_CYCLE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
)


def register_project_arial() -> Path:
    """Find, register, and verify the project Arial TTF with Matplotlib.

    Searches upward from this file for `00_environments/share/fonts/arial/Arial.TTF`,
    registers the exact TTF file in Matplotlib's font manager, and verifies that
    requesting 'Arial' resolves directly to this file.

    Raises:
        FileNotFoundError: If the Arial.TTF file is not found in parent directories.
        RuntimeError: If Matplotlib resolves 'Arial' to any different font path.
    """
    from matplotlib import font_manager

    relative = Path("00_environments/share/fonts/arial/Arial.TTF")
    root = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / relative).is_file()
        ),
        None,
    )
    if root is None:
        raise FileNotFoundError(f"Project Arial font not found: {relative}")
    font_path = (root / relative).resolve()
    font_manager.fontManager.addfont(str(font_path))
    resolved = Path(
        font_manager.findfont("Arial", fallback_to_default=False)
    ).resolve()
    if resolved != font_path:
        raise RuntimeError(
            f"Arial resolved to {resolved}, expected {font_path}"
        )
    return font_path


def matplotlib_rc_params() -> dict[str, object]:
    """Return standard Matplotlib rcParams dictionary adhering to plot contract.

    Typography: Arial, 5 pt normal text across axes, labels, ticks, and legends.
    Strokes: 0.5 pt for spines, lines, patches, ticks, and grids.
    Spines & Grid: All four spines visible/bounded; major grid #E5E7EB behind data.
    Output: PDF/PS fonttype 42 (TrueType / editable), savefig.bbox None (no tight bbox).
    Images: Image interpolation disabled.
    Palette: Colorblind-safe qualitative cycler (Wong palette).
    """
    from cycler import cycler

    return {
        # Typography
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": 5.0,
        "font.weight": "normal",
        "axes.titlesize": 5.0,
        "axes.titleweight": "normal",
        "axes.labelsize": 5.0,
        "axes.labelweight": "normal",
        "xtick.labelsize": 5.0,
        "ytick.labelsize": 5.0,
        "legend.fontsize": 5.0,
        "legend.title_fontsize": 5.0,
        "figure.titlesize": 5.0,
        "figure.titleweight": "normal",
        # Stroke widths (0.5 pt)
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.5,
        "patch.linewidth": 0.5,
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        # Bounded spines
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        # Major grid: subtle gray behind data
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.color": "#E5E7EB",
        "grid.linestyle": "-",
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        # Canvas and export fidelity (no tight bbox; editable Type 42 fonts)
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Image interpolation disabled
        "image.interpolation": "none",
        "image.resample": False,
        # Color palette: colorblind-safe qualitative cycle
        "axes.prop_cycle": cycler(color=list(COLORBLIND_SAFE_CYCLE)),
    }


def apply_matplotlib_style() -> None:
    """Register project Arial font and apply shared plot rcParams globally."""
    import matplotlib as mpl

    register_project_arial()
    mpl.rcParams.update(matplotlib_rc_params())


def base_pytheme(
    base_size: float = 5.0,
    base_family: str = "Arial",
    show_legend: bool = False,
    x_angle: float = 0.0,
    x_vjust: float = 0.5,
    x_hjust: float = 0.5,
) -> PlotnineTheme:
    """Construct a Plotnine theme matching the project publication plot contract.

    Uses lazy imports so Plotnine is only required when invoking theme builders.
    Matches 5 pt normal text, 0.5 pt lines/spines, #E5E7EB major grid, blank minor grid,
    and non-bold strip text with no in-axis plot title.
    """
    from plotnine import (
        element_blank,
        element_line,
        element_rect,
        element_text,
        theme,
        theme_bw,
    )

    return theme_bw(base_size=base_size, base_family=base_family) + theme(
        text=element_text(size=base_size, family=base_family),
        axis_title=element_text(size=base_size, weight="normal"),
        axis_text=element_text(size=base_size, color="black"),
        axis_text_x=element_text(angle=x_angle, vjust=x_vjust, hjust=x_hjust),
        axis_ticks_length=2.5,
        axis_line=element_line(color="black", size=0.5),
        panel_border=element_rect(color="black", size=0.5, fill="none"),
        panel_grid_major=element_line(color="#E5E7EB", size=0.5),
        panel_grid_minor=element_blank(),
        strip_text=element_text(size=base_size, weight="normal"),
        strip_background=element_blank(),
        panel_background=element_rect(fill="white", color="black", size=0.5),
        plot_title=element_blank(),
        legend_key=element_blank(),
        legend_title=element_text(size=base_size, weight="normal"),
        legend_text=element_text(size=base_size),
        legend_position="right" if show_legend else "none",
    )


def apply_plotnine_style() -> None:
    """Set base_pytheme() as Plotnine's persistent default theme."""
    from plotnine import theme_set

    theme_set(base_pytheme())


def project_color_scale(**kwargs: Any) -> Any:
    """Return a Plotnine discrete color scale using the shared safe palette."""
    from plotnine import scale_color_manual

    return scale_color_manual(values=list(COLORBLIND_SAFE_CYCLE), **kwargs)


def project_fill_scale(**kwargs: Any) -> Any:
    """Return a Plotnine discrete fill scale using the shared safe palette."""
    from plotnine import scale_fill_manual

    return scale_fill_manual(values=list(COLORBLIND_SAFE_CYCLE), **kwargs)


def my_pytheme_no_legend(
    base_size: float = 5.0, base_family: str = "Arial"
) -> PlotnineTheme:
    """Plotnine theme without legend and horizontal x-axis text."""
    return base_pytheme(base_size, base_family, show_legend=False, x_angle=0.0)


def my_pytheme_no_legend_x90deg(
    base_size: float = 5.0, base_family: str = "Arial"
) -> PlotnineTheme:
    """Plotnine theme without legend and 90-degree rotated x-axis text."""
    return base_pytheme(
        base_size,
        base_family,
        show_legend=False,
        x_angle=90.0,
        x_vjust=1.0,
        x_hjust=0.5,
    )


def my_pytheme_with_legend(
    base_size: float = 5.0, base_family: str = "Arial"
) -> PlotnineTheme:
    """Plotnine theme with right legend and horizontal x-axis text."""
    return base_pytheme(base_size, base_family, show_legend=True, x_angle=0.0)


def my_pytheme_with_legend_x90deg(
    base_size: float = 5.0, base_family: str = "Arial"
) -> PlotnineTheme:
    """Plotnine theme with right legend and 90-degree rotated x-axis text."""
    return base_pytheme(
        base_size,
        base_family,
        show_legend=True,
        x_angle=90.0,
        x_vjust=1.0,
        x_hjust=0.5,
    )
