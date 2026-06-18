"""
chi_chi_loader.py
=================

Colleague-facing loader for the chi-RCR vs chi-BL handoff.

You are given:
    * ``chi_chi_export_*.pkl``   -- small: chi values, SNR, time, ids, and trace *references*.
    * this file + ``_chi_chi_core.py``.

You provide (once): read access to the raw data this export points at --
    * the per-station nurFiles shards   (for Data / Pass / 2016-found-BL traces), and/or
    * the coincidence pickle             (for the coincidence Identified traces).
Nothing is recomputed and no traces were duplicated into the export; this module pulls
each waveform on demand.

TYPICAL USAGE
-------------
    from HRAStationDataAnalysis.ChiChiHandoff import chi_chi_loader as L

    export = L.load_export("output/chi_chi_export_3.21.26n3.pkl")
    L.print_summary(export)

    # Points for plotting (chi_bl, chi_rcr) for any of the five categories:
    bl, rcr = L.category_points(export, "pass_rcr")

    # Records (each has chi_bl/chi_rcr/snr/time/station_id + a trace handle):
    recs = L.category_records(export, "identified_rcr")

    # Waveform for one record (auto-routes nurfiles vs pickle):
    trace = L.load_trace(export, recs[0])

Point the loader at your data locations either by editing ``core.CONFIG`` paths or by
passing ``nurfiles_folder=`` / ``coincidence_pickle_path=`` to the load functions.
"""

import os
import pickle
import functools

import numpy as np

from HRAStationDataAnalysis.ChiChiHandoff import _chi_chi_core as core


# ---------------------------------------------------------------------------
# Loading the export + listing categories.
# ---------------------------------------------------------------------------

def load_export(path):
    """Load the export pickle written by ``export_chi_chi_datasets.py``."""
    with open(path, "rb") as f:
        return pickle.load(f)


CATEGORIES = ("data", "pass_rcr", "pass_bl", "identified_bl", "identified_rcr")


def category_records(export, name):
    """Return a uniform list of record dicts for one of the five categories.

    Each record has at least: station_id, time, snr, chi_bl, chi_rcr, trace_source,
    plus the source-specific handle (``raw_index`` for nurfiles; ``coinc_event_id`` +
    ``slot`` for pickle).
    """
    if name == "data":
        return _table_records(export, np.arange(len(export["table"]["snr"])))
    if name == "pass_rcr":
        return _table_records(export, export["category_indices"]["pass_rcr"])
    if name == "pass_bl":
        return _table_records(export, export["category_indices"]["pass_bl"])
    if name == "identified_bl":
        return list(export["identified_bl"]["from_2016"]) + list(export["identified_bl"]["from_coincidence"])
    if name == "identified_rcr":
        return list(export["identified_rcr"]["from_coincidence"])
    raise ValueError(f"Unknown category '{name}'. Choose from {CATEGORIES}.")


def _table_records(export, indices):
    """Turn rows of the summed 'table' (a nurfiles category) into record dicts."""
    t = export["table"]
    out = []
    for i in np.asarray(indices, dtype=int):
        out.append({
            "station_id": int(t["station_id"][i]),
            "event_id": int(t["event_id"][i]),
            "time": float(t["time"][i]),
            "snr": float(t["snr"][i]),
            "chi_bl": float(t["chi_bl"][i]),
            "chi_rcr": float(t["chi_rcr"][i]),
            "trace_source": "nurfiles",
            "raw_index": int(t["raw_index"][i]),
        })
    return out


def category_points(export, name):
    """Convenience: ``(chi_bl_array, chi_rcr_array)`` for a category, ready to scatter."""
    recs = category_records(export, name)
    bl = np.array([r["chi_bl"] for r in recs], dtype=float)
    rcr = np.array([r["chi_rcr"] for r in recs], dtype=float)
    return bl, rcr


def print_summary(export):
    print("chi-chi export summary")
    print("  chi_bl  =", export["meta"]["chi_bl_is"])
    print("  chi_rcr =", export["meta"]["chi_rcr_is"])
    for name in CATEGORIES:
        print(f"  {name:16s}: {len(category_records(export, name))} points")


# ---------------------------------------------------------------------------
# Trace loading -- the part you run yourself (raw data must be reachable).
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _load_station_traces(folder, date, station_id):
    """Load + concatenate one station's raw Traces shards (cached for the session)."""
    return core.load_station_data(folder, date, int(station_id), "Traces")


@functools.lru_cache(maxsize=None)
def _load_coincidence_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_trace(export, record, nurfiles_folder=None, coincidence_pickle_path=None):
    """Return the waveform (numpy array, channels x samples) for a single record.

    Routes automatically by ``record['trace_source']``:
      * 'nurfiles' -> raw Traces shard indexed by ``raw_index``.
      * 'pickle'   -> coincidence pickle ``[event][stations][sid]['Traces'][slot]``.

    Paths default to ``core.CONFIG``; override per call if your layout differs.
    """
    src = record.get("trace_source")

    if src == "nurfiles":
        folder = nurfiles_folder or core.station_data_folder(core.CONFIG)
        date = core.CONFIG["date"]
        traces = _load_station_traces(folder, date, record["station_id"])
        if traces.size == 0:
            raise FileNotFoundError(
                f"No Traces shards for Station {record['station_id']} under {folder}. "
                f"Point nurfiles_folder= at your raw data."
            )
        return traces[record["raw_index"]]

    if src == "pickle":
        path = coincidence_pickle_path or core.CONFIG["coincidence_pickle_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Coincidence pickle not found at {path}. Pass coincidence_pickle_path=."
            )
        events = _load_coincidence_pickle(path)
        event = events.get(record["coinc_event_id"], events.get(str(record["coinc_event_id"])))
        station_payload = event["stations"].get(record["station_id"], event["stations"].get(str(record["station_id"])))
        return np.asarray(station_payload["Traces"][record["slot"]])

    raise ValueError(f"Unknown trace_source: {src!r}")


def load_traces(export, records, **paths):
    """Vectorized ``load_trace`` over a list of records -> list of waveform arrays."""
    return [load_trace(export, r, **paths) for r in records]
