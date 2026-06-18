# chi-RCR vs chi-BL data handoff

Standalone bundle for sharing the five categories of the thesis **BL-χ vs RCR-χ**
2D scatter (the `chi_vs_chi` panel of `S01_plotSNRChiComparisons.py`) so a colleague
can run their own analysis on each sub-dataset.

- **chi_bl** = `Chi2016` (backlobe self-template correlation)
- **chi_rcr** = `ChiRCR` (reflected-cosmic-ray template correlation)
- Scope: **summed over stations 13, 14, 15, 17, 18, 19, 30** (a `station_id` column is kept).

## The five categories

| Category | Definition | Trace source |
|---|---|---|
| `data` | every event surviving the C00 quality cuts (the full table) | nurFiles |
| `pass_rcr` | `data` rows passing the RCR analysis cuts (+ day-uniqueness, − excluded) | nurFiles |
| `pass_bl` | `data` rows passing the mirrored BL analysis cuts | nurFiles |
| `identified_bl` | 2016-found backlobe (time-matched) **∪** coincidence-tagged backlobe | nurFiles + pickle |
| `identified_rcr` | coincidence-tagged RCR events (11230, 11243) | pickle |

## Design

The export stores **chi values + ids + a trace *reference*** — never the traces
themselves. Two reference kinds:

- **nurFiles**: a raw concatenated index into the per-station `*_Traces*.npy` shards.
  The loader re-globs + concatenates those shards (fully deterministic) and indexes in.
- **pickle**: an `(event_id, station_id, slot)` handle into the coincidence pickle,
  which already carries its own traces (see `C02` `parameters_to_add`).

So nothing is recomputed and no large waveform arrays are duplicated. The colleague
loads each waveform on demand, only for the points they care about.

## Files

| File | Who runs it | Purpose |
|---|---|---|
| `_chi_chi_core.py` | — | Frozen copy of the thesis loading/cut/category logic (numpy only) |
| `export_chi_chi_datasets.py` | **you, once, on the cluster** | Writes `output/chi_chi_export_<date>.pkl` |
| `chi_chi_loader.py` | colleague | `load_export`, `category_records/points`, `load_trace(s)` |
| `example_replot.py` | colleague | Rebuilds the 2D scatter + demos a trace load |

## How to use

**1. Produce the export (you, on the cluster, from the repo root):**

```bash
python -m HRAStationDataAnalysis.ChiChiHandoff.export_chi_chi_datasets
```

Hand the colleague: the resulting `output/chi_chi_export_*.pkl`, plus
`_chi_chi_core.py`, `chi_chi_loader.py`, and `example_replot.py`.

**2. Colleague side (needs read access to the same raw data):**

```python
from HRAStationDataAnalysis.ChiChiHandoff import chi_chi_loader as L

export = L.load_export("output/chi_chi_export_3.21.26n3.pkl")
L.print_summary(export)

bl, rcr   = L.category_points(export, "identified_rcr")   # chi points, no data load
recs      = L.category_records(export, "identified_rcr")  # full records
trace     = L.load_trace(export, recs[0])                 # waveform on demand
```

If the raw data lives somewhere other than the default config paths, pass
`nurfiles_folder=` / `coincidence_pickle_path=` to `load_trace`, or edit
`core.CONFIG` in `_chi_chi_core.py`.

## VERIFY on first real run (could not be tested off-cluster)

The raw data is not synced to the dev machine, so this was written correct-by-construction
against the known schema. On the first cluster run, sanity-check:

1. **2016-found JSON path** — `_chi_chi_core.CONFIG["found_2016_json_path"]` is
   `StationDataAnalysis/2016FoundEvents.json` (the path used in `S01`, *not* under
   `HRAStationDataAnalysis/`). Confirm it resolves; otherwise `identified_bl` loses its 2016 half.
2. **Category counts** — compare the printed `Pass RCR` / `Pass BL` / `Identified` counts
   against the `N=` labels on the thesis figure; they should match exactly.
3. **Coincidence pickle keys** — station keys may be `int` or `str`; the loader tries both.
4. **Trace shape** — `example_replot.py` prints one waveform shape per identified set;
   confirm it is what your analysis expects (channels × samples).
