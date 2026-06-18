"""
chi_chi_plot.py
===============

Shared plotting for the chi-RCR vs chi-BL handoff. ONE function, used by both:
  * ``export_chi_chi_datasets.py`` -- renders a check-plot at export time (proves the five
    categories landed in the right regions before anything is handed off), and
  * ``example_replot.py``          -- the colleague's reusable example.

The figure reproduces the thesis ``chi_vs_chi`` panel: BL-$\\chi$ (x) vs RCR-$\\chi$ (y),
the five labelled data categories, the diagonal, and the shaded RCR/BL cut regions drawn
from the same cut definitions used to select the events (``_chi_chi_core.CUTS``).

Only chi values are needed to plot (no trace loading), so this works off the in-memory
export dict on the cluster with no raw-data access.
"""

import matplotlib
matplotlib.use("Agg")          # headless-safe for batch/cluster runs
import matplotlib.pyplot as plt
import numpy as np

from HRAStationDataAnalysis.ChiChiHandoff import _chi_chi_core as core
from HRAStationDataAnalysis.ChiChiHandoff import chi_chi_loader as L


# Per-category point styling: (color, marker, size, zorder, label). Mirrors the thesis panel.
STYLE = {
    "data":           ("gray",       ".", 6,   1, "Data"),
    "pass_bl":        ("blue",       "*", 70,  3, "Pass BL Cuts"),
    "pass_rcr":       ("purple",     "*", 70,  3, "Pass RCR Cuts"),
    "identified_bl":  ("gold",       "o", 60,  4, "Identified BL"),
    "identified_rcr": ("darkorange", "*", 150, 5, "Identified RCR"),
}


def _draw_cut_regions(ax, cuts=core.CUTS):
    """Shade the RCR (green) and BL (orange) pass regions exactly as S01's chi_vs_chi panel.

    RCR region: RCR-chi in (BL-chi + threshold, BL-chi + max_diff), clipped to RCR-chi > 0.75.
    BL region : the mirror below the diagonal. Boundaries match the event-selection cuts.
    """
    rcr_chi_cut = cuts["chi_rcr_line_chi"][0]
    bl_chi_cut = cuts["chi_2016_line_chi"][0]
    thr = cuts["chi_diff_threshold"]
    dmax = cuts.get("chi_diff_max", 0.2)

    x = np.linspace(0, 1, 200)

    # RCR pass region (above diagonal).
    y_lo = np.maximum(rcr_chi_cut, x + thr)
    y_hi = np.minimum(1.0, x + dmax)
    m = y_lo < y_hi
    if np.any(m):
        ax.fill_between(x[m], y_lo[m], y_hi[m], color="green", alpha=0.10,
                        label="RCR cut region", zorder=0)
    ax.plot([0, rcr_chi_cut], [rcr_chi_cut, rcr_chi_cut], color="purple",
            linestyle="--", linewidth=1.2, zorder=2)

    # BL pass region (below diagonal).
    yb_hi = np.minimum(1.0, x - thr)
    yb_lo = np.maximum(rcr_chi_cut, x - dmax)
    mb = yb_lo < yb_hi
    if np.any(mb):
        ax.fill_between(x[mb], yb_lo[mb], yb_hi[mb], color="orange", alpha=0.10,
                        label="BL cut region", zorder=0)
    ax.plot([rcr_chi_cut + thr, rcr_chi_cut + dmax], [rcr_chi_cut, rcr_chi_cut],
            color="darkorange", linestyle="--", linewidth=1.2, zorder=2)


def make_chi_chi_plot(export, ax=None, draw_cuts=True, title=None):
    """Reproduce the thesis BL-chi vs RCR-chi panel from an export dict.

    Returns the Axes. Scatters the five categories (with N counts in the legend), the
    diagonal RCR-chi == BL-chi, and -- if ``draw_cuts`` -- the shaded cut regions.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if draw_cuts:
        _draw_cut_regions(ax)

    ax.plot([0, 1], [0, 1], "--", color="red", linewidth=1, zorder=2)  # RCR-chi == BL-chi

    for name, (color, marker, size, z, label) in STYLE.items():
        bl, rcr = L.category_points(export, name)
        if bl.size == 0:
            continue
        alpha = 0.4 if name == "data" else 0.9
        edge = "none" if name == "data" else "black"
        ax.scatter(bl, rcr, c=color, marker=marker, s=size, alpha=alpha,
                   edgecolors=edge, linewidths=0.4, label=f"{label} (N={bl.size})", zorder=z)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"BL-$\chi$")
    ax.set_ylabel(r"RCR-$\chi$")
    ax.set_title(title or "Reflected-CR search: BL-$\\chi$ vs RCR-$\\chi$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    return ax


def save_chi_chi_plot(export, out_path, title=None):
    """Render the panel and write it to ``out_path``. Returns the path."""
    ax = make_chi_chi_plot(export, title=title)
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return out_path
