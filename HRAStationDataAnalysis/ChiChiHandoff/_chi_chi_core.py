"""
_chi_chi_core.py
================

Frozen, dependency-light core for the chi-RCR vs chi-BL (BL-$\\chi$ vs RCR-$\\chi$)
data handoff.

This module is a *self-contained copy* of the exact data-loading, masking, cut, and
event-categorization logic used to produce the chi-chi 2D scatter plot in the thesis
(the ``chi_vs_chi`` panel of
``HRAStationDataAnalysis/S01_plotSNRChiComparisons.py``).

WHY A COPY (rather than importing S01):
  - The thesis pipeline is frozen, so this logic will not drift.
  - Importing S01 pulls in NuRadioReco, HRAEventObject, the C03 plotting stack, etc.
    A handoff bundle for a colleague should depend on nothing but ``numpy`` so it
    runs anywhere the raw data files live.

NAMING CONVENTION (matches the thesis code, restated here so the mapping is explicit):
  - ``Chi2016``  == BL-$\\chi$  (backlobe self-template correlation)  -> exported as ``chi_bl``
  - ``ChiRCR``   == RCR-$\\chi$ (reflected-cosmic-ray template correlation) -> exported as ``chi_rcr``

PROVENANCE OF EACH OF THE FIVE PLOT CATEGORIES:
  1. ``Data``            : every event surviving the C00 quality cuts (nurFiles source).
  2. ``Pass RCR Cuts``   : Data events passing the RCR analysis cuts (+ day-uniqueness, - excluded).
  3. ``Pass BL Cuts``    : Data events passing the mirrored backlobe analysis cuts.
  4. ``Identified BL``   : union of  (a) 2016-found backlobe events (time-matched into nurFiles)
                                 and (b) coincidence-tagged backlobe events (from the coincidence pickle).
  5. ``Identified RCR``  : coincidence-tagged RCR events (event ids 11230, 11243) from the coincidence pickle.

Categories 1-3 and the 2016 half of 4 are nurFiles-sourced (traces reloaded from the
raw ``*_Traces*.npy`` files). The coincidence halves of 4 and 5 are pickle-sourced
(traces live inside the coincidence pickle itself; see C02 ``parameters_to_add``).
"""

import os
import glob
import datetime

import numpy as np


# ---------------------------------------------------------------------------
# Configuration (mirrors HRAStationDataAnalysis/config.ini at handoff time).
# Override any of these from the calling script if the run dates change.
# ---------------------------------------------------------------------------

CONFIG = {
    # Date tags identifying which processed dataset / cuts / coincidence set to use.
    "date": "9.1.25",                # nurFiles data tag
    "date_cuts": "9.18.25",          # C00 cuts tag
    "date_coincidence": "9.24.25",   # coincidence pickle tag

    # Stations entering the summed chi-chi plot.
    "station_ids": [13, 14, 15, 17, 18, 19, 30],

    # Folders (relative to the ReflectionAnalysis repo root, as used by the thesis code).
    "station_data_folder": "HRAStationDataAnalysis/StationData/nurFiles/{date}/",
    "cuts_data_folder": "HRAStationDataAnalysis/StationData/cuts/{date_cuts}/",

    # Coincidence pickle (absolute cluster path used in S01). Each event carries, per
    # station, the full parameter set INCLUDING 'Traces' and 'Times' (see C02 line ~282).
    "coincidence_pickle_path": (
        "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/"
        "9.24.25_CoincidenceDatetimes_passing_cuts_with_all_params_recalcZenAzi_calcPol.pkl"
    ),

    # 2016-found backlobe events, keyed "Station{id}Found" -> list of unix times.
    # NOTE: S01 reads this from 'StationDataAnalysis/2016FoundEvents.json' (note: NOT the
    # HRAStationDataAnalysis subfolder). VERIFY this path on the cluster before the first run.
    "found_2016_json_path": "StationDataAnalysis/2016FoundEvents.json",
}

# Coincidence event ids that make up the "Identified" sets, copied verbatim from S01.
REQUESTED_COINCIDENCE_EVENT_IDS = [
    3047, 3432, 10195, 10231, 10273, 10284, 10444, 10449,
    10466, 10471, 10554, 11197, 11220, 11230, 11236, 11243,
]

# The two coincidence events whose default category is RCR (all others default to Backlobe).
COINCIDENCE_RCR_EVENT_IDS = {11230, 11243}

# Per-(event, station) category overrides inside the two RCR events
# (mirrors special_station_map in S01.build_coincidence_station_overlays).
COINCIDENCE_STATION_CATEGORY_OVERRIDES = {
    11230: {13: "RCR", 17: "Backlobe"},
    11243: {30: "RCR", 17: "Backlobe"},
}

# Analysis cut definitions, copied verbatim from S01.__main__.
CUTS = {
    "snr_max": 50,
    "chi_rcr_line_snr": np.array([0, 7, 8.5, 15, 20, 30, 100]),
    "chi_rcr_line_chi": np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]),  # flat at 0.75
    "chi_diff_threshold": 0.0,
    "chi_diff_max": 0.2,
    "chi_2016_line_snr": np.array([0, 7, 8.5, 15, 20, 30, 100]),
    "chi_2016_line_chi": np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]),  # flat at 0.75
}

# Events hand-excluded in S01 (only removed from a "Pass" set if they would otherwise pass).
EXCLUDED_EVENTS = [
    (18, 82), (18, 520), (18, 681),
    (15, 1472768),
    (19, 3621320), (19, 4599318), (19, 4599919),
]


# ---------------------------------------------------------------------------
# Frozen primitives copied from C_utils / S01 (numpy-only).
# ---------------------------------------------------------------------------

def get_time_event_masks(times_raw, event_ids):
    """Copy of ``C_utils.getTimeEventMasks``.

    Returns ``(initial_mask, unique_indices)`` such that the canonical per-station
    array is ``raw_array[initial_mask][unique_indices]`` (before the C00 cut is applied).

    - ``initial_mask`` drops zero / pre-2013 timestamps (junk).
    - ``unique_indices`` selects the first occurrence of each unique (time, event_id) pair,
      sorted, indexing INTO the initial-masked subset.
    """
    zerotime_mask = times_raw != 0
    min_datetime_threshold = datetime.datetime(2013, 1, 1).timestamp()
    pretime_mask = times_raw >= min_datetime_threshold
    initial_mask = zerotime_mask & pretime_mask

    times_m = times_raw[initial_mask]
    event_ids_m = event_ids[initial_mask]

    time_eventid_pairs = np.stack((times_m, event_ids_m), axis=-1)
    _, unique_indices = np.unique(time_eventid_pairs, axis=0, return_index=True)
    unique_indices.sort()

    return initial_mask, unique_indices


def load_station_data(folder, date, station_id, data_name):
    """Copy of ``S01.load_station_data``: glob + concatenate the per-station ``.npy`` shards.

    Files are named ``{date}_Station{station_id}_{data_name}*`` and concatenated in
    sorted order along axis 0. Returns an empty array if nothing matches.
    """
    file_pattern = os.path.join(folder, f"{date}_Station{station_id}_{data_name}*")
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        return np.array([])

    data_arrays = [np.load(f, allow_pickle=True) for f in file_list]
    data_arrays = [arr for arr in data_arrays if arr.size > 0]
    if not data_arrays:
        return np.array([])

    return np.concatenate(data_arrays, axis=0)


def load_cuts_mask(date, station_id, cuts_data_folder):
    """Copy of ``S01.load_cuts_for_station``: AND together every boolean array in the C00 cuts file.

    Returns the combined keep-mask, or ``None`` if the cuts file is absent.
    """
    cuts_file = os.path.join(cuts_data_folder, f"{date}_Station{station_id}_Cuts.npy")
    if not os.path.exists(cuts_file):
        return None

    cuts_data = np.load(cuts_file, allow_pickle=True)[()]
    final_cuts_mask = np.ones(len(cuts_data["L1_mask"]), dtype=bool)
    for cut_key in cuts_data.keys():
        final_cuts_mask &= cuts_data[cut_key]
    return final_cuts_mask


def filter_unique_events_by_day(times, station_ids):
    """Copy of ``S01.filter_unique_events_by_day``.

    Keep only the first event seen per (station, UTC-day). Returns a boolean keep-mask.
    """
    seen_combinations = set()
    keep_mask = np.zeros(len(times), dtype=bool)
    for i, (t, sid) in enumerate(zip(times, station_ids)):
        date_str = datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")
        combo = (sid, date_str)
        if combo not in seen_combinations:
            seen_combinations.add(combo)
            keep_mask[i] = True
    return keep_mask


def get_all_cut_masks(data_dict, cuts=CUTS, cut_type="rcr"):
    """Thesis ``get_all_cut_masks`` (the ``all_cuts`` key is what defines a "Pass" set).

    Both selections require SNR < snr_max, a chi floor, and a chi-difference band:

    ``cut_type='rcr'``      -> RCR-chi > 0.75  AND  RCR-chi - BL-chi in (0, chi_diff_max).
    ``cut_type='backlobe'`` -> BL-chi  > 0.75  AND  RCR-chi - BL-chi in (-chi_diff_max, 0).

    NOTE: the two differ in *which* chi carries the 0.75 floor -- RCR-chi for the RCR cut,
    BL-chi for the backlobe cut. This matches the thesis figure (vertical BL-chi=0.75 boundary
    on the backlobe region), and corrects an earlier copy that floored both on RCR-chi.
    """
    snr = data_dict["snr"]
    chircr = data_dict["ChiRCR"]
    chi2016 = data_dict["Chi2016"]

    chi_diff = chircr - chi2016

    masks = {}
    masks["snr_cut"] = snr < cuts["snr_max"]

    if cut_type == "rcr":
        floor = np.interp(snr, cuts["chi_rcr_line_snr"], cuts["chi_rcr_line_chi"])
        masks["snr_line_cut"] = chircr > floor          # RCR-chi floor
        masks["chi_diff_cut"] = (chi_diff > cuts["chi_diff_threshold"]) & (chi_diff < cuts.get("chi_diff_max", 999))
    elif cut_type == "backlobe":
        floor = np.interp(snr, cuts["chi_2016_line_snr"], cuts["chi_2016_line_chi"])
        masks["snr_line_cut"] = chi2016 > floor         # BL-chi floor (thesis version)
        masks["chi_diff_cut"] = (chi_diff < -cuts["chi_diff_threshold"]) & (chi_diff > -cuts.get("chi_diff_max", 999))
    else:
        raise ValueError(f"Unknown cut_type: {cut_type}")

    masks["snr_and_snr_line"] = masks["snr_cut"] & masks["snr_line_cut"]
    masks["all_cuts"] = masks["snr_cut"] & masks["snr_line_cut"] & masks["chi_diff_cut"]
    return masks


# ---------------------------------------------------------------------------
# Path helpers.
# ---------------------------------------------------------------------------

def station_data_folder(config=CONFIG):
    return config["station_data_folder"].format(date=config["date"])


def cuts_data_folder(config=CONFIG):
    return config["cuts_data_folder"].format(date_cuts=config["date_cuts"])
