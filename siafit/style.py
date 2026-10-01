"""Matplotlib defaults used by the Si II fitting diagnostics."""

from __future__ import annotations

PLOT_STYLE = {
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.minor.width": 2,
    "ytick.minor.width": 2,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 2,
    "axes.labelsize": 20,
    "legend.framealpha": 0.75,
    "legend.fontsize": 10,
    "legend.borderaxespad": 1,
    # Keep the requested font first, with portable serif fallbacks for systems
    # where Times New Roman is not installed.
    "font.family": ["Times New Roman", "Times", "DejaVu Serif"],
    "savefig.bbox": "tight",
    "figure.figsize": (9.7, 6),
}


def apply_plot_style() -> None:
    """Apply the package's self-contained plotting style."""
    import matplotlib as mpl
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    families = [name for name in PLOT_STYLE["font.family"] if name in available]
    mpl.rcParams.update({**PLOT_STYLE, "font.family": families or ["DejaVu Serif"]})
