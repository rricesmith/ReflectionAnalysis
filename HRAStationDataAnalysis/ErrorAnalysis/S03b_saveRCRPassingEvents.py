"""
S03b_saveRCRPassingEvents.py
============================
Applies the nominal RCR cuts to station data, identifies passing events,
loads all available parameters (traces, SNR, chi values, angles, etc.),
and saves them into a single compressed dictionary file (.npz).

Also provides load_rcr_events() and iterate_rcr_events() for easy access
by collaborators.

Usage (save):
    python -m HRAStationDataAnalysis.ErrorAnalysis.S03b_saveRCRPassingEvents

Usage (load, from any script):
    from HRAStationDataAnalysis.ErrorAnalysis.S03b_saveRCRPassingEvents import (
        load_rcr_events, iterate_rcr_events
    )
    events = load_rcr_events('path/to/rcr_passing_events.npz')
    for evt in iterate_rcr_events('path/to/rcr_passing_events.npz'):
        print(evt['station_id'], evt['event_id'], evt['snr'], evt['traces'].shape)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import configparser
import glob
from datetime import datetime
from icecream import ic

from HRAStationDataAnalysis.C_utils import getTimeEventMasks


# ============================================================================
# Constants — must match S03b_selectedPlots.py
# ============================================================================
SNR_PREFILTER = 50

NOMINAL_CUTS = {
    'snr_max': 50,
    'chi_rcr_line_snr': np.array([0, 7, 8.5, 15, 20, 30, 100]),
    'chi_rcr_line_chi': np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]),
    'chi_diff_threshold': 0.0,
    'chi_diff_max': 0.2,
}

# ---------------------------------------------------------------------------
# Cut variants for error-band categorization
# ---------------------------------------------------------------------------
# chi_rcr variation (from the chi_diff_threshold scan cross-cut: 0.74–0.76)
# These produce the 9(+3/-2) event count seen in the thesis.
UPPER_CHIRCR = {**NOMINAL_CUTS, 'chi_rcr_line_chi': np.full_like(NOMINAL_CUTS['chi_rcr_line_chi'], 0.74)}
LOWER_CHIRCR = {**NOMINAL_CUTS, 'chi_rcr_line_chi': np.full_like(NOMINAL_CUTS['chi_rcr_line_chi'], 0.76)}

# chi_diff variation (from the chi_rcr_flat scan cross-cut: ±0.05)
UPPER_CHIDIFF = {**NOMINAL_CUTS, 'chi_diff_threshold': NOMINAL_CUTS['chi_diff_threshold'] - 0.05}
LOWER_CHIDIFF = {**NOMINAL_CUTS, 'chi_diff_threshold': NOMINAL_CUTS['chi_diff_threshold'] + 0.05}

# Integer category labels (same values used for all three categorization systems)
CATEGORY_ALWAYS     = 0  # passes even the tighter (LOWER) cut
CATEGORY_NOMINAL    = 1  # passes nominal but not tighter — could fail
CATEGORY_ADDITIONAL = 2  # passes looser (UPPER) but not nominal — could additionally pass

# Events excluded from analysis (double-counted)
EXCLUDED_EVENTS = [
    (18, 82), (18, 520), (18, 681),
    (15, 1472768),
    (19, 3621320), (19, 4599318), (19, 4599919),
]

STATION_IDS = [13, 14, 15, 17, 18, 19, 30]

# All parameter file prefixes saved by HRADataConvertToNpy.py
# Traces shape: (N, 4, 256), all others: (N,)
DATA_NAMES = ['Traces', 'SNR', 'Chi2016', 'ChiRCR', 'ChiBad', 'Azi', 'Zen']


# ============================================================================
# Data loading helpers (from S03b_selectedPlots.py)
# ============================================================================

def load_station_data(folder, date, station_id, data_name):
    """Loads and concatenates data files for a specific station and data type."""
    file_pattern = os.path.join(folder, f'{date}_Station{station_id}_{data_name}*')
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        return np.array([])
    data_arrays = [np.load(f, allow_pickle=True) for f in file_list]
    data_arrays = [arr for arr in data_arrays if arr.size > 0]
    if not data_arrays:
        return np.array([])
    return np.concatenate(data_arrays, axis=0)


def load_cuts_for_station(date, station_id, cuts_data_folder):
    """Load the cuts from C00 for a specific station and date."""
    cuts_file = os.path.join(cuts_data_folder, f'{date}_Station{station_id}_Cuts.npy')
    if not os.path.exists(cuts_file):
        ic(f"Warning: Cuts file not found for station {station_id} on date {date}.")
        return None
    try:
        cuts_data = np.load(cuts_file, allow_pickle=True)[()]
        final_cuts_mask = np.ones(len(cuts_data['L1_mask']), dtype=bool)
        for cut_key in cuts_data.keys():
            final_cuts_mask &= cuts_data[cut_key]
        return final_cuts_mask
    except Exception as e:
        ic(f"Error loading cuts file {cuts_file}: {e}")
        return None


# ============================================================================
# RCR cut application (from S03b_selectedPlots.py apply_cuts, rcr mode only)
# ============================================================================

def apply_rcr_cuts(snr, chi_rcr, chi_2016, cuts):
    """
    Apply nominal RCR cuts. Returns boolean mask of passing events.

    Cuts applied:
      - SNR < snr_max
      - ChiRCR > chi_rcr_line (interpolated on SNR)
      - chi_diff (ChiRCR - Chi2016) > chi_diff_threshold
      - chi_diff < chi_diff_max
    """
    chi_diff = chi_rcr - chi_2016
    mask = np.ones(len(snr), dtype=bool)

    # SNR upper bound
    mask &= snr < cuts['snr_max']

    # ChiRCR floor (interpolated line)
    chi_line = np.interp(snr, cuts['chi_rcr_line_snr'], cuts['chi_rcr_line_chi'])
    mask &= chi_rcr > chi_line

    # Chi difference window
    mask &= chi_diff > cuts['chi_diff_threshold']
    mask &= chi_diff < cuts['chi_diff_max']

    return mask


# ============================================================================
# Load & iterate functions (for colleague use)
# ============================================================================

def load_rcr_events(filepath):
    """
    Load the saved RCR-passing events dictionary from an .npz file.

    Returns a dict with keys:
        'station_ids'  : int array (N,) — station ID per event
        'event_ids'    : int array (N,) — event ID per event
        'times'        : float array (N,) — Unix timestamp per event
        'traces'       : float array (N, 4, 256) — waveform traces (4 channels, 256 samples)
        'snr'          : float array (N,) — signal-to-noise ratio
        'chi_rcr'      : float array (N,) — RCR chi value
        'chi_2016'     : float array (N,) — 2016 (backlobe) chi value
        'chi_bad'      : float array (N,) — bad-template chi value
        'azi'          : float array (N,) — reconstructed azimuth (rad)
        'zen'          : float array (N,) — reconstructed zenith (rad)
    """
    data = np.load(filepath, allow_pickle=False)
    return {key: data[key] for key in data.files}


def iterate_rcr_events(filepath):
    """
    Generator that yields one event at a time as a dict.

    Each yielded dict contains:
        'station_id'      : int
        'event_id'        : int
        'time'            : float (Unix timestamp)
        'traces'          : ndarray (4, 256)
        'snr'             : float
        'chi_rcr'         : float
        'chi_2016'        : float
        'chi_bad'         : float
        'azi'             : float
        'zen'             : float
        'category_chircr' : int (0=always, 1=nominal, 2=additional) — present only if
                            the npz was built with chi_rcr-axis labelling (S04b v3+)
        'category_chidiff': int (0=always, 1=nominal, 2=additional) — present only if
                            the npz was built with chi_diff-axis labelling (S04b v3+)
        'category_combined': int (0=always, 1=nominal, 2=additional) — present only if
                            the npz was built with combined labelling (S04b v3+)

    Example:
        for evt in iterate_rcr_events('rcr_passing_events.npz'):
            print(f"Station {evt['station_id']}, SNR={evt['snr']:.1f}")
            traces = evt['traces']  # shape (4, 256)
    """
    events = load_rcr_events(filepath)
    n = len(events['station_ids'])
    has_cat_chircr   = 'category_chircr'   in events
    has_cat_chidiff  = 'category_chidiff'  in events
    has_cat_combined = 'category_combined' in events
    has_in_nominal   = 'in_nominal' in events
    has_in_chircr    = 'in_chircr'  in events
    has_in_chidiff   = 'in_chidiff' in events
    for i in range(n):
        evt = {
            'station_id': int(events['station_ids'][i]),
            'event_id':   int(events['event_ids'][i]),
            'time':       float(events['times'][i]),
            'traces':     events['traces'][i],
            'snr':        float(events['snr'][i]),
            'chi_rcr':    float(events['chi_rcr'][i]),
            'chi_2016':   float(events['chi_2016'][i]),
            'chi_bad':    float(events['chi_bad'][i]),
            'azi':        float(events['azi'][i]),
            'zen':        float(events['zen'][i]),
        }
        if has_cat_chircr:
            evt['category_chircr']   = int(events['category_chircr'][i])
        if has_cat_chidiff:
            evt['category_chidiff']  = int(events['category_chidiff'][i])
        if has_cat_combined:
            evt['category_combined'] = int(events['category_combined'][i])
        if has_in_nominal:
            evt['in_nominal'] = bool(events['in_nominal'][i])
        if has_in_chircr:
            evt['in_chircr']  = bool(events['in_chircr'][i])
        if has_in_chidiff:
            evt['in_chidiff'] = bool(events['in_chidiff'][i])
        yield evt


# ============================================================================
# Main: find RCR-passing data events and save
# ============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    config = configparser.ConfigParser()
    config.read('HRAStationDataAnalysis/config.ini')
    date = config['PARAMETERS']['date']
    date_cuts = config['PARAMETERS']['date_cuts']
    date_processing = config['PARAMETERS']['date_processing']

    station_data_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'
    cuts_data_folder = f'HRAStationDataAnalysis/StationData/cuts/{date_cuts}/'
    output_folder = f'HRAStationDataAnalysis/ErrorAnalysis/output/{date_processing}/'
    os.makedirs(output_folder, exist_ok=True)

    ic.configureOutput(prefix='SaveRCR | ')

    # --- Collect events passing RCR cuts, per station ---
    # Accumulate all parameters for passing events
    out_station_ids = []
    out_event_ids = []
    out_times = []
    out_traces = []
    out_snr = []
    out_chi_rcr = []
    out_chi_2016 = []
    out_chi_bad = []
    out_azi = []
    out_zen = []

    excluded_set = set(EXCLUDED_EVENTS)

    for station_id in STATION_IDS:
        ic(f"Processing station {station_id}...")

        # Load core parameters needed for masking
        snr_array = load_station_data(station_data_folder, date, station_id, 'SNR')
        chi2016_array = load_station_data(station_data_folder, date, station_id, 'Chi2016')
        chircr_array = load_station_data(station_data_folder, date, station_id, 'ChiRCR')
        times = load_station_data(station_data_folder, date, station_id, 'Time')
        event_ids_raw = load_station_data(station_data_folder, date, station_id, 'EventIDs')

        if chi2016_array.size == 0 or chircr_array.size == 0:
            ic(f"  No data for station {station_id}, skipping.")
            continue

        # Apply time/event masks (zero-time, pre-2013, unique events)
        initial_mask, unique_indices = getTimeEventMasks(times, event_ids_raw)

        # Apply C00 quality cuts
        cuts_mask = load_cuts_for_station(date, station_id, cuts_data_folder)
        if cuts_mask is None:
            ic(f"  Error: No cuts found for Station {station_id}.")
            sys.exit(1)

        temp_times = times[initial_mask][unique_indices]
        if len(cuts_mask) != len(temp_times):
            cuts_mask = cuts_mask[:len(temp_times)]
        final_indices = unique_indices[cuts_mask]

        # Extract quality-cut data
        snr_cut = snr_array[initial_mask][final_indices]
        chi2016_cut = chi2016_array[initial_mask][final_indices]
        chircr_cut = chircr_array[initial_mask][final_indices]
        times_cut = times[initial_mask][final_indices]
        evtids_cut = event_ids_raw[initial_mask][final_indices]

        # SNR prefilter
        snr_prefilt = snr_cut < SNR_PREFILTER
        snr_cut = snr_cut[snr_prefilt]
        chi2016_cut = chi2016_cut[snr_prefilt]
        chircr_cut = chircr_cut[snr_prefilt]
        times_cut = times_cut[snr_prefilt]
        evtids_cut = evtids_cut[snr_prefilt]
        final_indices_filt = final_indices[snr_prefilt]

        # Exclude double-counted events
        keep = np.ones(len(snr_cut), dtype=bool)
        for idx in range(len(snr_cut)):
            if (station_id, evtids_cut[idx]) in excluded_set:
                keep[idx] = False
        snr_cut = snr_cut[keep]
        chi2016_cut = chi2016_cut[keep]
        chircr_cut = chircr_cut[keep]
        times_cut = times_cut[keep]
        evtids_cut = evtids_cut[keep]
        final_indices_filt = final_indices_filt[keep]

        # Apply RCR cuts
        rcr_mask = apply_rcr_cuts(snr_cut, chircr_cut, chi2016_cut, NOMINAL_CUTS)
        n_passing = np.sum(rcr_mask)
        ic(f"  Station {station_id}: {n_passing} events pass RCR cuts (out of {len(snr_cut)})")

        if n_passing == 0:
            continue

        # Now load all remaining parameters for passing events only
        # We need the indices into the raw (initial_mask-applied) arrays
        passing_raw_indices = final_indices_filt[rcr_mask]

        # Load traces and other parameters using the same indexing
        traces_array = load_station_data(station_data_folder, date, station_id, 'Traces')
        chi_bad_array = load_station_data(station_data_folder, date, station_id, 'ChiBad')
        azi_array = load_station_data(station_data_folder, date, station_id, 'Azi')
        zen_array = load_station_data(station_data_folder, date, station_id, 'Zen')

        # Apply initial_mask then select passing indices
        traces_pass = traces_array[initial_mask][passing_raw_indices]
        chi_bad_pass = chi_bad_array[initial_mask][passing_raw_indices] if chi_bad_array.size > 0 else np.full(n_passing, np.nan)
        azi_pass = azi_array[initial_mask][passing_raw_indices] if azi_array.size > 0 else np.full(n_passing, np.nan)
        zen_pass = zen_array[initial_mask][passing_raw_indices] if zen_array.size > 0 else np.full(n_passing, np.nan)

        # Accumulate
        out_station_ids.append(np.full(n_passing, station_id, dtype=int))
        out_event_ids.append(evtids_cut[rcr_mask].astype(int))
        out_times.append(times_cut[rcr_mask])
        out_traces.append(traces_pass)
        out_snr.append(snr_cut[rcr_mask])
        out_chi_rcr.append(chircr_cut[rcr_mask])
        out_chi_2016.append(chi2016_cut[rcr_mask])
        out_chi_bad.append(chi_bad_pass)
        out_azi.append(azi_pass)
        out_zen.append(zen_pass)

    # --- Concatenate all stations ---
    if not out_snr:
        ic("No events passed RCR cuts across any station.")
        sys.exit(0)

    result = {
        'station_ids': np.concatenate(out_station_ids),
        'event_ids':   np.concatenate(out_event_ids),
        'times':       np.concatenate(out_times),
        'traces':      np.concatenate(out_traces),       # (N, 4, 256)
        'snr':         np.concatenate(out_snr),
        'chi_rcr':     np.concatenate(out_chi_rcr),
        'chi_2016':    np.concatenate(out_chi_2016),
        'chi_bad':     np.concatenate(out_chi_bad),
        'azi':         np.concatenate(out_azi),
        'zen':         np.concatenate(out_zen),
    }

    n_total = len(result['snr'])
    ic(f"Total RCR-passing events: {n_total}")
    for sid in STATION_IDS:
        n_st = np.sum(result['station_ids'] == sid)
        if n_st > 0:
            ic(f"  Station {sid}: {n_st} events")

    # --- Save ---
    output_path = os.path.join(output_folder, 'rcr_passing_events.npz')
    np.savez_compressed(output_path, **result)
    filesize_mb = os.path.getsize(output_path) / 1e6
    ic(f"Saved to {output_path} ({filesize_mb:.1f} MB)")
    ic(f"Keys: {list(result.keys())}")
    ic(f"Traces shape: {result['traces'].shape}")

    # --- Quick verification printout ---
    print(f"\n{'='*60}")
    print(f"RCR-Passing Events Summary")
    print(f"{'='*60}")
    print(f"Total events: {n_total}")
    print(f"Output file:  {output_path}")
    print(f"File size:    {filesize_mb:.1f} MB")
    print(f"\nPer-event fields:")
    for key, val in result.items():
        print(f"  {key:15s}  shape={str(val.shape):20s}  dtype={val.dtype}")
    print(f"\nTo load:")
    print(f"  from HRAStationDataAnalysis.ErrorAnalysis.S03b_saveRCRPassingEvents import load_rcr_events, iterate_rcr_events")
    print(f"  events = load_rcr_events('{output_path}')")
    print(f"  for evt in iterate_rcr_events('{output_path}'):")
    print(f"      print(evt['station_id'], evt['snr'], evt['traces'].shape)")
