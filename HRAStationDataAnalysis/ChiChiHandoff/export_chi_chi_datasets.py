"""
export_chi_chi_datasets.py
==========================

RUN THIS ONCE, ON THE CLUSTER (where the raw nurFiles ``.npy`` shards and the
coincidence pickle live), from the ReflectionAnalysis repo root:

    python -m HRAStationDataAnalysis.ChiChiHandoff.export_chi_chi_datasets

It reproduces the summed chi-RCR vs chi-BL dataset behind the thesis ``chi_vs_chi``
panel and writes a single SMALL pickle (``chi_chi_export_{date_processing}.pkl``)
containing, for the five plot categories:

    * the chi values (chi_bl == Chi2016, chi_rcr == ChiRCR), SNR, time, station id,
      event id, and
    * a *reference* for re-loading the full waveform (trace) on demand --
      NOT the traces themselves.

Trace references come in two flavours:
    * ``nurfiles``  : a raw concatenated index into the per-station ``*_Traces*.npy``
                      shards (Data, Pass-RCR, Pass-BL, and the 2016 half of Identified-BL).
    * ``pickle``    : an (event id, station id, trigger slot) handle into the coincidence
                      pickle, which stores its own traces (the coincidence halves of
                      Identified-BL / Identified-RCR).

The companion ``chi_chi_loader.py`` turns either reference back into a waveform.

NOTE: this script cannot be exercised off-cluster because the raw data is not synced
locally. It is written to be correct-by-construction against the known data schema; the
inline ``VERIFY`` comments flag the few things worth a sanity check on the first real run.
"""

import os
import json
import pickle

import numpy as np

from HRAStationDataAnalysis.ChiChiHandoff import _chi_chi_core as core


# ---------------------------------------------------------------------------
# Step 1: build the summed nurFiles table (the "Data" category) + Pass subsets.
# ---------------------------------------------------------------------------

def build_summed_table(config=core.CONFIG):
    """Reproduce the summed per-station data exactly as S01 does, but also record,
    for every surviving row, the *raw concatenated index* needed to reload its trace.

    Returns a dict of equal-length arrays (the "Data" category table):
        station_id, event_id, time, snr, chi_bl, chi_rcr, raw_index
    where ``raw_index`` indexes into ``load_station_data(..., 'Traces')`` for that station.
    """
    folder = core.station_data_folder(config)
    cuts_folder = core.cuts_data_folder(config)
    date = config["date"]

    cols = {k: [] for k in ("station_id", "event_id", "time", "snr", "chi_bl", "chi_rcr", "raw_index")}

    for sid in config["station_ids"]:
        snr_raw = core.load_station_data(folder, date, sid, "SNR")
        chi2016_raw = core.load_station_data(folder, date, sid, "Chi2016")
        chircr_raw = core.load_station_data(folder, date, sid, "ChiRCR")
        time_raw = core.load_station_data(folder, date, sid, "Time")
        evid_raw = core.load_station_data(folder, date, sid, "EventIDs")

        if chi2016_raw.size == 0 or chircr_raw.size == 0:
            print(f"[export] Station {sid}: missing Chi data, skipping.")
            continue

        # Reproduce S01's mask pipeline: initial validity -> unique (time,evid) -> C00 cuts.
        initial_mask, unique_indices = core.get_time_event_masks(time_raw, evid_raw)

        cuts_mask = core.load_cuts_mask(date, sid, cuts_folder)
        if cuts_mask is None:
            # S01 hard-quits here; we refuse to silently include uncut data.
            raise RuntimeError(f"No C00 cuts file for Station {sid} (date {date}). Aborting.")

        # Guard against the length mismatch S01 also guards (truncate cuts to data length).
        n_unique = len(unique_indices)
        if len(cuts_mask) != n_unique:
            print(f"[export] Station {sid}: cuts mask len {len(cuts_mask)} != unique len {n_unique}; truncating.")
            cuts_mask = cuts_mask[:n_unique]

        final_indices = unique_indices[cuts_mask]            # indices into the initial-masked subset
        abs_raw = np.where(initial_mask)[0]                  # raw positions of initial-masked rows
        raw_index = abs_raw[final_indices]                   # raw position of each surviving row

        cols["station_id"].append(np.full(len(final_indices), sid, dtype=np.int64))
        cols["event_id"].append(evid_raw[raw_index].astype(np.int64))
        cols["time"].append(time_raw[raw_index].astype(np.float64))
        cols["snr"].append(snr_raw[initial_mask][final_indices].astype(np.float64))
        cols["chi_bl"].append(chi2016_raw[initial_mask][final_indices].astype(np.float64))
        cols["chi_rcr"].append(chircr_raw[initial_mask][final_indices].astype(np.float64))
        cols["raw_index"].append(raw_index.astype(np.int64))

        print(f"[export] Station {sid}: {len(final_indices)} events after cuts.")

    if not cols["station_id"]:
        raise RuntimeError("No station data was loaded; nothing to export.")

    table = {k: np.concatenate(v) for k, v in cols.items()}
    return table


def compute_pass_indices(table, config=core.CONFIG):
    """Return ``{'pass_rcr': idx, 'pass_bl': idx}`` as integer index arrays into ``table``.

    Faithfully reproduces S01.run_analysis_for_station: the raw ``all_cuts`` mask, then
    the per-(station, day) uniqueness cull on the passing events, then removal of the
    hand-excluded events (only when they would otherwise pass).
    """
    data_dict = {"snr": table["snr"], "ChiRCR": table["chi_rcr"], "Chi2016": table["chi_bl"]}

    masks_rcr = core.get_all_cut_masks(data_dict, core.CUTS, cut_type="rcr")
    masks_bl = core.get_all_cut_masks(data_dict, core.CUTS, cut_type="backlobe")

    excluded_set = set(config.get("excluded_events", core.EXCLUDED_EVENTS))

    def finalize(all_cuts_mask):
        all_cuts = all_cuts_mask.copy()

        # Day-uniqueness on the passing subset (S01 removes later-in-day duplicates).
        passing_idx = np.where(all_cuts)[0]
        if len(passing_idx) > 0:
            keep = core.filter_unique_events_by_day(table["time"][passing_idx], table["station_id"][passing_idx])
            all_cuts[passing_idx[~keep]] = False

        # Hand-excluded events: only drop if currently passing (matches S01's excluded_mask).
        for i in np.where(all_cuts)[0]:
            if (int(table["station_id"][i]), int(table["event_id"][i])) in excluded_set:
                all_cuts[i] = False

        return np.where(all_cuts)[0].astype(np.int64)

    return {"pass_rcr": finalize(masks_rcr["all_cuts"]), "pass_bl": finalize(masks_bl["all_cuts"])}


# ---------------------------------------------------------------------------
# Step 2: the 2016-found backlobe half of "Identified BL" (nurFiles, time-matched).
# ---------------------------------------------------------------------------

def build_identified_bl_2016(config=core.CONFIG):
    """Time-match the 2016-found backlobe events into the raw nurFiles arrays.

    Returns a list of per-event record dicts (nurfiles trace source). Mirrors the
    ``Station{id}Found`` -> target-times -> ``time_map`` logic in S01.__main__.
    Empty if the JSON is absent.
    """
    json_path = config["found_2016_json_path"]
    if not os.path.exists(json_path):
        print(f"[export] 2016-found JSON not at {json_path}; Identified-BL(2016) will be empty. VERIFY path.")
        return []

    with open(json_path, "r") as f:
        found_json = json.load(f)

    folder = core.station_data_folder(config)
    date = config["date"]
    records = []

    for sid in config["station_ids"]:
        key = f"Station{sid}Found"
        if key not in found_json:
            continue

        time_raw = core.load_station_data(folder, date, sid, "Time")
        snr_raw = core.load_station_data(folder, date, sid, "SNR")
        chi2016_raw = core.load_station_data(folder, date, sid, "Chi2016")
        chircr_raw = core.load_station_data(folder, date, sid, "ChiRCR")
        evid_raw = core.load_station_data(folder, date, sid, "EventIDs")
        if time_raw.size == 0:
            continue

        time_map = {t: i for i, t in enumerate(time_raw)}     # raw-space, exactly like S01
        for t in found_json[key]:
            if t in time_map:
                ri = time_map[t]
                records.append({
                    "station_id": int(sid),
                    "event_id": int(evid_raw[ri]) if evid_raw.size else -1,
                    "time": float(time_raw[ri]),
                    "snr": float(snr_raw[ri]) if snr_raw.size else np.nan,
                    "chi_bl": float(chi2016_raw[ri]),
                    "chi_rcr": float(chircr_raw[ri]),
                    "trace_source": "nurfiles",
                    "raw_index": int(ri),
                })

    print(f"[export] Identified-BL (2016-found): {len(records)} events.")
    return records


# ---------------------------------------------------------------------------
# Step 3: the coincidence halves of Identified-BL / Identified-RCR (pickle source).
# ---------------------------------------------------------------------------

def _coincidence_category(event_id, station_int):
    """Port of S01.build_coincidence_station_overlays category logic."""
    default = "RCR" if event_id in core.COINCIDENCE_RCR_EVENT_IDS else "Backlobe"
    override = core.COINCIDENCE_STATION_CATEGORY_OVERRIDES.get(event_id, {})
    return override.get(station_int, default)


def build_identified_coincidence(config=core.CONFIG):
    """Extract per-(event, station, trigger) identified records from the coincidence pickle.

    Returns ``(identified_rcr, identified_bl_coinc)`` lists of record dicts. Each record
    carries chi/snr/time plus a pickle handle (event_id, station_id, slot) so the loader
    can fetch the matching trace from the pickle. Empty lists if the pickle is absent.
    """
    path = config["coincidence_pickle_path"]
    if not os.path.exists(path):
        print(f"[export] Coincidence pickle not at {path}; Identified coincidence sets empty. VERIFY path.")
        return [], []

    with open(path, "rb") as f:
        events_dict = pickle.load(f)

    station_set = set(config["station_ids"])
    rcr_records, bl_records = [], []

    for event_id in core.REQUESTED_COINCIDENCE_EVENT_IDS:
        event = events_dict.get(event_id, events_dict.get(str(event_id)))
        if event is None or not isinstance(event, dict):
            continue
        stations_info = event.get("stations", {})

        for station_key, payload in stations_info.items():
            try:
                sid = int(station_key)
            except (TypeError, ValueError):
                continue
            if sid not in station_set:
                continue

            snr = np.asarray(payload.get("SNR", []), dtype=float)
            chi_bl = np.asarray(payload.get("Chi2016", []), dtype=float)
            chi_rcr = np.asarray(payload.get("ChiRCR", []), dtype=float)
            times = np.asarray(payload.get("Times", []), dtype=float) if payload.get("Times") is not None else None

            n = min(snr.size, chi_bl.size, chi_rcr.size)
            if n == 0:
                continue

            category = _coincidence_category(event_id, sid)
            for slot in range(n):
                if not (np.isfinite(snr[slot]) and np.isfinite(chi_bl[slot]) and np.isfinite(chi_rcr[slot])):
                    continue
                rec = {
                    "coinc_event_id": int(event_id),
                    "station_id": sid,
                    "slot": int(slot),                 # index into the station payload's Traces list
                    "time": float(times[slot]) if times is not None and slot < times.size else np.nan,
                    "snr": float(snr[slot]),
                    "chi_bl": float(chi_bl[slot]),
                    "chi_rcr": float(chi_rcr[slot]),
                    "category": category,
                    "trace_source": "pickle",
                }
                (rcr_records if category == "RCR" else bl_records).append(rec)

    print(f"[export] Identified-RCR (coincidence): {len(rcr_records)} triggers; "
          f"Identified-BL (coincidence): {len(bl_records)} triggers.")
    return rcr_records, bl_records


# ---------------------------------------------------------------------------
# Step 4: assemble + save.
# ---------------------------------------------------------------------------

def build_export(config=core.CONFIG):
    table = build_summed_table(config)
    pass_idx = compute_pass_indices(table, config)
    ident_bl_2016 = build_identified_bl_2016(config)
    ident_rcr_coinc, ident_bl_coinc = build_identified_coincidence(config)

    export = {
        "meta": {
            "description": "chi-RCR vs chi-BL handoff dataset (summed over stations).",
            "chi_bl_is": "Chi2016 (backlobe self-template correlation)",
            "chi_rcr_is": "ChiRCR (reflected-cosmic-ray template correlation)",
            "config": {k: config[k] for k in ("date", "date_cuts", "date_coincidence", "station_ids")},
            "cuts": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in core.CUTS.items()},
            "category_definitions": {
                "data": "every event surviving C00 cuts (the full 'table'); nurfiles traces.",
                "pass_rcr": "table rows passing RCR analysis cuts (+day-unique, -excluded); nurfiles traces.",
                "pass_bl": "table rows passing mirrored BL analysis cuts (+day-unique, -excluded); nurfiles traces.",
                "identified_bl": "union of 2016-found BL (nurfiles) and coincidence BL (pickle traces).",
                "identified_rcr": "coincidence-tagged RCR events 11230/11243 (pickle traces).",
            },
        },
        # Full summed dataset = the "Data" category. Every nurfiles trace is reachable
        # via (station_id, raw_index).
        "table": table,
        # Integer indices INTO 'table' for the cut-passing subsets.
        "category_indices": {
            "pass_rcr": pass_idx["pass_rcr"],
            "pass_bl": pass_idx["pass_bl"],
        },
        # Identified sets (own records; trace_source tells you where the waveform lives).
        "identified_bl": {
            "from_2016": ident_bl_2016,          # nurfiles
            "from_coincidence": ident_bl_coinc,  # pickle
        },
        "identified_rcr": {
            "from_coincidence": ident_rcr_coinc,  # pickle
        },
    }
    return export


def main():
    # Plot-folder date tag, used only to name the output file. Falls back if config.ini absent.
    date_processing = "3.21.26n3"
    try:
        import configparser
        cp = configparser.ConfigParser()
        cp.read("HRAStationDataAnalysis/config.ini")
        date_processing = cp["PARAMETERS"]["date_processing"]
    except Exception:
        pass

    export = build_export(core.CONFIG)

    out_dir = os.path.join("HRAStationDataAnalysis", "ChiChiHandoff", "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"chi_chi_export_{date_processing}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(export, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_data = len(export["table"]["snr"])
    print("\n[export] DONE.")
    print(f"  Data (all):       {n_data}")
    print(f"  Pass RCR:         {len(export['category_indices']['pass_rcr'])}")
    print(f"  Pass BL:          {len(export['category_indices']['pass_bl'])}")
    print(f"  Identified BL:    {len(export['identified_bl']['from_2016'])} (2016) "
          f"+ {len(export['identified_bl']['from_coincidence'])} (coinc)")
    print(f"  Identified RCR:   {len(export['identified_rcr']['from_coincidence'])} (coinc)")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
