"""
example_replot.py
=================

Worked example for the colleague: rebuild the chi-RCR vs chi-BL 2D scatter from the
handoff export, then demonstrate pulling a waveform on demand for one identified event.

Run from the ReflectionAnalysis repo root, on the cluster (so the raw traces are reachable):

    python -m HRAStationDataAnalysis.ChiChiHandoff.example_replot output/chi_chi_export_3.21.26n3.pkl

The chi values come straight from the export (no recomputation). Traces are only loaded
for the handful of points you actually ask for.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from HRAStationDataAnalysis.ChiChiHandoff import chi_chi_loader as L


# Plot styling per category: (color, marker, size, zorder, label).
STYLE = {
    "data":           ("gray",       ".", 6,   1, "Data"),
    "pass_bl":        ("blue",       "*", 70,  3, "Pass BL Cuts"),
    "pass_rcr":       ("purple",     "*", 70,  3, "Pass RCR Cuts"),
    "identified_bl":  ("gold",       "o", 60,  4, "Identified BL"),
    "identified_rcr": ("darkorange", "*", 150, 5, "Identified RCR"),
}


def make_chi_chi_plot(export, ax=None):
    """Scatter all five categories on the BL-chi (x) vs RCR-chi (y) plane."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    for name, (color, marker, size, z, label) in STYLE.items():
        bl, rcr = L.category_points(export, name)
        if bl.size == 0:
            continue
        alpha = 0.4 if name == "data" else 0.9
        edge = "none" if name == "data" else "black"
        ax.scatter(bl, rcr, c=color, marker=marker, s=size, alpha=alpha,
                   edgecolors=edge, linewidths=0.4, label=f"{label} (N={bl.size})", zorder=z)

    ax.plot([0, 1], [0, 1], "--", color="red", linewidth=1, zorder=2)  # chi_rcr == chi_bl diagonal
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"BL-$\chi$")
    ax.set_ylabel(r"RCR-$\chi$")
    ax.set_title("Reflected-CR search: BL-$\\chi$ vs RCR-$\\chi$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    return ax


def demo_trace_load(export):
    """Pull one waveform per identified set to confirm trace access works end-to-end."""
    for name in ("identified_rcr", "identified_bl"):
        recs = L.category_records(export, name)
        if not recs:
            print(f"[example] {name}: no records.")
            continue
        r = recs[0]
        try:
            trace = L.load_trace(export, r)
            shape = np.asarray(trace).shape
            print(f"[example] {name}: loaded trace for St{r['station_id']} "
                  f"(source={r['trace_source']}) -> shape {shape}, "
                  f"chi_bl={r['chi_bl']:.3f}, chi_rcr={r['chi_rcr']:.3f}")
        except Exception as exc:
            print(f"[example] {name}: trace load failed ({exc}). "
                  f"Check that the raw data path is reachable from here.")


def main():
    export_path = sys.argv[1] if len(sys.argv) > 1 else "output/chi_chi_export_3.21.26n3.pkl"
    export = L.load_export(export_path)
    L.print_summary(export)

    ax = make_chi_chi_plot(export)
    out_png = "chi_chi_replot_example.png"
    ax.figure.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[example] saved {out_png}")

    demo_trace_load(export)


if __name__ == "__main__":
    main()
