"""
example_replot.py
=================

Worked example for the colleague: rebuild the chi-RCR vs chi-BL 2D scatter from the
handoff export, then demonstrate pulling a waveform on demand for one identified event.

Run from the ReflectionAnalysis repo root, on the cluster (so the raw traces are reachable):

    python -m HRAStationDataAnalysis.ChiChiHandoff.example_replot output/chi_chi_export_3.21.26n3.pkl

The chi values come straight from the export (no recomputation). Traces are only loaded
for the handful of points you actually ask for.

The chi-chi figure itself comes from the shared ``chi_chi_plot.make_chi_chi_plot`` -- the
very same function ``export_chi_chi_datasets.py`` uses to write its check-plot, so what you
reproduce here is identical to what was emitted at export time.
"""

import sys

import numpy as np

from HRAStationDataAnalysis.ChiChiHandoff import chi_chi_loader as L
from HRAStationDataAnalysis.ChiChiHandoff.chi_chi_plot import make_chi_chi_plot


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
