"""
S03_cutScanAnalysis.py
======================
Cut-scan error analysis for RCR event identification.

For each of the 4 cut parameters (chi_rcr_flat, chi_diff_threshold, chi_diff_max, snr_max),
scans over a range of values while holding the others fixed at nominal. At each scan point,
computes:
  1. RCR sim weighted count in the RCR region (+ statistical error)
  2. BL sim weighted count in the RCR region (+ statistical error)
  3. Combined (RCR + BL) sim weighted count (predicted total data)
  4. Number of 2016 Backlobe events passing cuts
  5. Number of real data events passing cuts (after day-uniqueness + exclusion filters)

Uses the 0.15 chi-diff pre-filter on BL sim (matching S01).

Simulation weights are rescaled so that total sum matches the known livetime-scaled
expected event counts from HRAAnalysis (RCR: 22.058, BL: 560.107). Each event's weight
is proportional to its original internal weight, preserving the relative distribution.

Output:
  - Scan plots (single y-axis: expected vs observed events)
  - Additional plot variants for chi_rcr_flat (log-scale)
  - Additional extended-range scans for chi_diff_threshold
  - Summary table with expected event counts
  - All saved to ErrorAnalysis/plots/
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
import configparser
import json
import glob
import h5py
import pickle
from datetime import datetime
from icecream import ic

from HRAStationDataAnalysis.C_utils import getTimeEventMasks
from HRASimulation.HRAEventObject import HRAevent


# ============================================================================
# Known Expected Event Totals (livetime-scaled, from HRAAnalysis)
# ============================================================================
# These are the trusted total expected events across all stations and livetimes.
# Sim weights are rescaled so that sum(weights) matches these values exactly,
# preserving the relative weight distribution across events.
KNOWN_RCR_TOTAL = 22.05815658
KNOWN_RCR_ERROR = 2.320078714
KNOWN_BL_TOTAL = 560.1066252
KNOWN_BL_ERROR = 35.21137122


def rescale_weights(sim_data, known_total):
    """
    Rescale event weights so their sum equals the known total expected events.
    new_w_i = (w_i / sum(w)) * known_total
    Preserves relative weight distribution across events.
    """
    raw_sum = np.sum(sim_data['weights'])
    if raw_sum > 0:
        scale = known_total / raw_sum
        sim_data['weights'] = sim_data['weights'] * scale
        ic(f"Rescaled weights: raw_sum={raw_sum:.4f} -> new_sum={np.sum(sim_data['weights']):.4f} (scale={scale:.4f})")
    else:
        ic("Warning: raw weight sum is 0, cannot rescale.")


# ============================================================================
# Data Loading Functions (reused from S01)
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


def loadHRAfromH5(filename):
    """Loads a list of HRAevent objects from an HDF5 file."""
    ic(f"Loading HRA event list from: {filename}")
    eventList = []
    with h5py.File(filename, 'r') as hf:
        for key in hf.keys():
            obj_bytes = hf[key][0]
            event = pickle.loads(obj_bytes)
            eventList.append(event)
    ic(f"Successfully loaded {len(eventList)} events.")
    return eventList


def get_sim_data(HRAeventList, direct_weight_name, reflected_weight_name,
                 direct_stations, reflected_stations, sigma=4.5,
                 apply_chi_diff_prefilter=True):
    """
    Extracts SNR, Chi, and weights from the HRAevent list.

    Parameters
    ----------
    apply_chi_diff_prefilter : bool
        If True, removes BL sim events where (ChiRCR - Chi2016) > 0.15 (matching S01).
        If False, keeps all events for worst-case background estimate.
    """
    direct_data = {'snr': [], 'Chi2016': [], 'ChiRCR': [], 'weights': []}
    reflected_data = {'snr': [], 'Chi2016': [], 'ChiRCR': [], 'weights': []}

    for event in HRAeventList:
        direct_weight = event.getWeight(direct_weight_name, primary=True, sigma=sigma)
        if not np.isnan(direct_weight) and direct_weight > 0:
            triggered_direct = [st_id for st_id in direct_stations if event.hasTriggered(st_id, sigma)]
            if triggered_direct:
                split_weight = direct_weight / len(triggered_direct)
                for st_id in triggered_direct:
                    snr, chi_dict = event.getSNR(st_id), event.getChi(st_id)
                    if snr is not None and chi_dict:
                        direct_data['snr'].append(snr)
                        direct_data['Chi2016'].append(chi_dict.get('Chi2016', np.nan))
                        direct_data['ChiRCR'].append(chi_dict.get('ChiRCR', np.nan))
                        direct_data['weights'].append(split_weight)

        reflected_weight = event.getWeight(reflected_weight_name, primary=True, sigma=sigma)
        if not np.isnan(reflected_weight) and reflected_weight > 0:
            triggered_reflected = [st_id for st_id in reflected_stations if event.hasTriggered(st_id, sigma)]
            if triggered_reflected:
                split_weight = reflected_weight / len(triggered_reflected)
                for st_id in triggered_reflected:
                    snr, chi_dict = event.getSNR(st_id), event.getChi(st_id)
                    if snr is not None and chi_dict:
                        reflected_data['snr'].append(snr)
                        reflected_data['Chi2016'].append(chi_dict.get('Chi2016', np.nan))
                        reflected_data['ChiRCR'].append(chi_dict.get('ChiRCR', np.nan))
                        reflected_data['weights'].append(split_weight)

    for data_dict in [direct_data, reflected_data]:
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key])

        if apply_chi_diff_prefilter and len(data_dict['ChiRCR']) > 0:
            mask = (data_dict['ChiRCR'] - data_dict['Chi2016']) <= 0.15
            ic(f"Pre-filter: keeping {np.sum(mask)}/{len(mask)} events where (ChiRCR - Chi2016) <= 0.15")
            for key in data_dict:
                data_dict[key] = data_dict[key][mask]

    return direct_data, reflected_data


def filter_unique_events_by_day(times, station_ids):
    """Returns a boolean mask keeping only the first event per station per day."""
    seen_combinations = set()
    keep_mask = np.zeros(len(times), dtype=bool)
    for i, (t, sid) in enumerate(zip(times, station_ids)):
        date_str = datetime.utcfromtimestamp(t).strftime('%Y-%m-%d')
        combo = (sid, date_str)
        if combo not in seen_combinations:
            seen_combinations.add(combo)
            keep_mask[i] = True
    return keep_mask


# ============================================================================
# Cut Analysis Functions
# ============================================================================

def apply_cuts(data_dict, cuts, cut_type='rcr'):
    """
    Apply all cuts and return a boolean mask of events passing.
    Mirrors get_all_cut_masks from S01.
    """
    snr = data_dict['snr']
    chircr = data_dict['ChiRCR']
    chi2016 = data_dict['Chi2016']

    chi_rcr_snr_cut_values = np.interp(snr, cuts['chi_rcr_line_snr'], cuts['chi_rcr_line_chi'])
    chi_diff = chircr - chi2016

    snr_cut = snr < cuts['snr_max']
    snr_line_cut = chircr > chi_rcr_snr_cut_values

    if cut_type == 'rcr':
        chi_diff_cut = (chi_diff > cuts['chi_diff_threshold']) & (chi_diff < cuts.get('chi_diff_max', 999))
    elif cut_type == 'backlobe':
        chi_diff_cut = (chi_diff < -cuts['chi_diff_threshold']) & (chi_diff > -cuts.get('chi_diff_max', 999))
    else:
        chi_diff_cut = np.ones_like(snr, dtype=bool)

    all_cuts = snr_cut & snr_line_cut & chi_diff_cut
    return all_cuts


def compute_weighted_count_and_error(weights, mask):
    """
    For simulation data with weights, compute expected count and statistical error.

    N_expected = sum(w_i) for events passing the mask
    sigma = sqrt(sum(w_i^2)) for events passing the mask
    """
    w = weights[mask]
    if len(w) == 0:
        return 0.0, 0.0
    n_expected = np.sum(w)
    sigma = np.sqrt(np.sum(w**2))
    return n_expected, sigma


def scan_cut_parameter(param_name, scan_values, nominal_cuts, sim_direct, sim_reflected,
                       data_dict, data_station_ids, bl_2016_data, excluded_events_mask):
    """
    Scan one cut parameter over a range of values, holding others at nominal.

    Returns arrays of (rcr_counts, rcr_errors, bl_counts, bl_errors,
                        bl_2016_counts, data_counts) for each scan value.
    """
    n = len(scan_values)
    rcr_counts = np.zeros(n)
    rcr_errors = np.zeros(n)
    bl_counts = np.zeros(n)
    bl_errors = np.zeros(n)
    bl_2016_counts = np.zeros(n)
    data_counts = np.zeros(n)

    for i, val in enumerate(scan_values):
        # Build modified cuts dictionary
        cuts = dict(nominal_cuts)
        # Need to deep copy arrays
        cuts['chi_rcr_line_snr'] = nominal_cuts['chi_rcr_line_snr'].copy()
        cuts['chi_rcr_line_chi'] = nominal_cuts['chi_rcr_line_chi'].copy()
        cuts['chi_2016_line_snr'] = nominal_cuts['chi_2016_line_snr'].copy()
        cuts['chi_2016_line_chi'] = nominal_cuts['chi_2016_line_chi'].copy()

        if param_name == 'chi_rcr_flat':
            cuts['chi_rcr_line_chi'] = np.full_like(cuts['chi_rcr_line_chi'], val)
            cuts['chi_2016_line_chi'] = np.full_like(cuts['chi_2016_line_chi'], val)
        elif param_name == 'chi_diff_threshold':
            cuts['chi_diff_threshold'] = val
        elif param_name == 'chi_diff_max':
            cuts['chi_diff_max'] = val
        elif param_name == 'snr_max':
            cuts['snr_max'] = val

        # RCR sim
        rcr_mask = apply_cuts(sim_reflected, cuts, cut_type='rcr')
        rcr_counts[i], rcr_errors[i] = compute_weighted_count_and_error(
            sim_reflected['weights'], rcr_mask)

        # BL sim
        bl_mask = apply_cuts(sim_direct, cuts, cut_type='rcr')
        bl_counts[i], bl_errors[i] = compute_weighted_count_and_error(
            sim_direct['weights'], bl_mask)

        # 2016 BL events
        if bl_2016_data is not None and len(bl_2016_data['snr']) > 0:
            bl_2016_mask = apply_cuts(bl_2016_data, cuts, cut_type='rcr')
            bl_2016_counts[i] = np.sum(bl_2016_mask)
        else:
            bl_2016_counts[i] = 0

        # Real data (with day-uniqueness and exclusion)
        data_mask = apply_cuts(data_dict, cuts, cut_type='rcr')
        # Apply exclusion mask
        data_mask &= ~excluded_events_mask
        # Apply day-uniqueness
        passing_indices = np.where(data_mask)[0]
        if len(passing_indices) > 0:
            times_pass = data_dict['Time'][passing_indices]
            sids_pass = data_station_ids[passing_indices]
            unique_mask = filter_unique_events_by_day(times_pass, sids_pass)
            data_counts[i] = np.sum(unique_mask)
        else:
            data_counts[i] = 0

    return rcr_counts, rcr_errors, bl_counts, bl_errors, bl_2016_counts, data_counts


# ============================================================================
# Plotting Functions
# ============================================================================

# Mapping from internal parameter names to professional axis labels
PARAM_LABELS = {
    'chi_rcr_flat': r'$\chi_{\mathrm{RCR}}$ Flat Cut',
    'chi_diff_threshold': r'$\Delta\chi$ Threshold ($\chi_{\mathrm{RCR}} - \chi_{2016}$)',
    'chi_diff_max': r'$\Delta\chi$ Maximum ($\chi_{\mathrm{RCR}} - \chi_{2016}$)',
    'snr_max': 'SNR Maximum',
}

# Mapping from internal parameter names to professional title labels
PARAM_TITLES = {
    'chi_rcr_flat': r'$\chi_{\mathrm{RCR}}$ Flat',
    'chi_diff_threshold': r'$\Delta\chi$ Threshold',
    'chi_diff_max': r'$\Delta\chi$ Maximum',
    'snr_max': 'SNR Maximum',
}


def plot_cut_scan(param_name, scan_values, nominal_value,
                  rcr_counts, rcr_errors, bl_counts, bl_errors,
                  bl_2016_counts, data_counts,
                  output_path, data_yscale='linear'):
    """
    Create a scan plot for one cut parameter.
    Weights are already rescaled to expected events, so plots directly.

    Parameters
    ----------
    data_yscale : str
        Scale for the y-axis ('linear' or 'log').
    """
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # Combined sim (RCR + BL) with propagated error
    both_counts = rcr_counts + bl_counts
    both_errors = np.sqrt(rcr_errors**2 + bl_errors**2)

    # Simulation lines
    ax1.plot(scan_values, rcr_counts, 'g-', linewidth=2, label='RCR Sim')
    ax1.fill_between(scan_values, rcr_counts - rcr_errors, rcr_counts + rcr_errors,
                     color='green', alpha=0.2)

    ax1.plot(scan_values, bl_counts, color='orange', linestyle='-', linewidth=2,
             label='BL Sim')
    ax1.fill_between(scan_values, bl_counts - bl_errors, bl_counts + bl_errors,
                     color='orange', alpha=0.2)

    ax1.plot(scan_values, both_counts, color='purple', linestyle='--', linewidth=2,
             label='Both Sim')
    ax1.fill_between(scan_values, both_counts - both_errors, both_counts + both_errors,
                     color='purple', alpha=0.15)

    # Data lines on same axis
    ax1.plot(scan_values, data_counts, 'k-o', markersize=4, linewidth=1.5,
             label='Data Events')
    ax1.plot(scan_values, bl_2016_counts, 'c-s', markersize=4, linewidth=1.5,
             label='2016 BL Events')

    xlabel = PARAM_LABELS.get(param_name, param_name)
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel('Simulated Events for HRA', fontsize=12)
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)

    if data_yscale == 'log':
        ax1.set_yscale('log')

    # Cut value line
    ax1.axvline(x=nominal_value, color='red', linestyle='--', linewidth=1.5,
                label=f'Cut at {nominal_value}', alpha=0.7)

    ax1.legend(loc='best', fontsize=9)

    title_label = PARAM_TITLES.get(param_name, param_name)
    plt.title(f'Cut Scan: {title_label}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    ic(f"Saved scan plot: {output_path}")


def print_summary_table(nominal_cuts, sim_direct, sim_reflected,
                        data_dict, data_station_ids, bl_2016_data,
                        excluded_events_mask, label, output_path=None):
    """
    Print and optionally save a summary table at nominal cut values.
    Weights are already rescaled to expected events.
    """
    # Compute nominal counts (weights are already in expected events)
    rcr_mask = apply_cuts(sim_reflected, nominal_cuts, cut_type='rcr')
    rcr_count, rcr_err = compute_weighted_count_and_error(sim_reflected['weights'], rcr_mask)

    bl_mask = apply_cuts(sim_direct, nominal_cuts, cut_type='rcr')
    bl_count, bl_err = compute_weighted_count_and_error(sim_direct['weights'], bl_mask)

    both_count = rcr_count + bl_count
    both_err = np.sqrt(rcr_err**2 + bl_err**2)

    if bl_2016_data is not None and len(bl_2016_data['snr']) > 0:
        bl_2016_mask = apply_cuts(bl_2016_data, nominal_cuts, cut_type='rcr')
        bl_2016_count = int(np.sum(bl_2016_mask))
    else:
        bl_2016_count = 0

    data_mask = apply_cuts(data_dict, nominal_cuts, cut_type='rcr')
    data_mask &= ~excluded_events_mask
    passing_indices = np.where(data_mask)[0]
    if len(passing_indices) > 0:
        times_pass = data_dict['Time'][passing_indices]
        sids_pass = data_station_ids[passing_indices]
        unique_mask = filter_unique_events_by_day(times_pass, sids_pass)
        data_count = int(np.sum(unique_mask))
        data_count_no_daycut = int(len(passing_indices))
    else:
        data_count = 0
        data_count_no_daycut = 0

    # Significance
    if bl_count > 0:
        sig_s_over_sqrt_b = rcr_count / np.sqrt(bl_count)
        sig_s_over_sqrt_sb = rcr_count / np.sqrt(rcr_count + bl_count)
    else:
        sig_s_over_sqrt_b = float('inf') if rcr_count > 0 else 0
        sig_s_over_sqrt_sb = float('inf') if rcr_count > 0 else 0

    lines = [
        f"",
        f"{'='*85}",
        f"  Summary Table: {label}",
        f"{'='*85}",
        f"",
        f"  Known Totals (from HRAAnalysis):",
        f"    RCR: {KNOWN_RCR_TOTAL:.4f} +/- {KNOWN_RCR_ERROR:.4f}",
        f"    BL:  {KNOWN_BL_TOTAL:.4f} +/- {KNOWN_BL_ERROR:.4f}",
        f"",
        f"  Nominal Cuts:",
        f"    ChiRCR > {nominal_cuts['chi_rcr_line_chi'][0]:.2f} (flat)",
        f"    {nominal_cuts['chi_diff_threshold']:.2f} < (ChiRCR - Chi2016) < {nominal_cuts['chi_diff_max']:.2f}",
        f"    SNR < {nominal_cuts['snr_max']}",
        f"",
        f"  {'Category':<25} {'Expected Evts':>14} {'Error':>12}",
        f"  {'-'*53}",
        f"  {'RCR Sim':<25} {rcr_count:>14.4f} {rcr_err:>12.4f}",
        f"  {'BL Sim':<25} {bl_count:>14.4f} {bl_err:>12.4f}",
        f"  {'Both Sim (RCR+BL)':<25} {both_count:>14.4f} {both_err:>12.4f}",
        f"  {'2016 BL Events':<25} {bl_2016_count:>12d}     sqrt(N)={np.sqrt(bl_2016_count):.2f}",
        f"  {'Data Events (w/ day cut)':<25} {data_count:>12d}     sqrt(N)={np.sqrt(data_count):.2f}",
        f"  {'Data Events (no day cut)':<25} {data_count_no_daycut:>12d}",
        f"",
        f"  Significance Estimates:",
        f"    S / sqrt(B)     = {sig_s_over_sqrt_b:.3f}",
        f"    S / sqrt(S + B) = {sig_s_over_sqrt_sb:.3f}",
        f"{'='*85}",
        f"",
    ]

    output = '\n'.join(lines)
    print(output)

    if output_path:
        with open(output_path, 'a') as f:
            f.write(output + '\n')
        ic(f"Appended summary to: {output_path}")

    return {
        'rcr_count': rcr_count, 'rcr_err': rcr_err,
        'bl_count': bl_count, 'bl_err': bl_err,
        'both_count': both_count, 'both_err': both_err,
        'bl_2016_count': bl_2016_count,
        'data_count': data_count, 'data_count_no_daycut': data_count_no_daycut,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    config = configparser.ConfigParser()
    config.read('HRAStationDataAnalysis/config.ini')
    date = config['PARAMETERS']['date']
    date_cuts = config['PARAMETERS']['date_cuts']
    date_processing = config['PARAMETERS']['date_processing']

    sim_file = config['SIMULATION']['sim_file']
    direct_weight_name = config['SIMULATION']['direct_weight_name']
    reflected_weight_name = config['SIMULATION']['reflected_weight_name']
    sim_sigma = float(config['SIMULATION']['sigma'])

    station_data_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'
    cuts_data_folder = f'HRAStationDataAnalysis/StationData/cuts/{date_cuts}/'
    plot_folder = f'HRAStationDataAnalysis/ErrorAnalysis/plots/{date_processing}/'
    os.makedirs(plot_folder, exist_ok=True)

    ic.configureOutput(prefix='S03 CutScan | ')

    # --- Define Stations ---
    station_ids = [13, 14, 15, 17, 18, 19, 30]
    direct_stations = [13, 14, 15, 17, 18, 19, 30]
    reflected_stations = [113, 114, 115, 117, 118, 119, 130]

    # --- Define Nominal Cuts (must match S01 exactly) ---
    nominal_cuts = {
        'snr_max': 50,
        'chi_rcr_line_snr': np.array([0, 7, 8.5, 15, 20, 30, 100]),
        'chi_rcr_line_chi': np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]),
        'chi_diff_threshold': 0.0,
        'chi_diff_max': 0.2,
        'chi_2016_line_snr': np.array([0, 7, 8.5, 15, 20, 30, 100]),
        'chi_2016_line_chi': np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]),
    }

    # --- Define Scan Ranges ---
    scan_params = {
        'chi_rcr_flat': {
            'values': np.linspace(0.5, 0.95, 20),
            'nominal': 0.75,
        },
        'chi_diff_threshold': {
            'values': np.linspace(-0.1, 0.1, 20),
            'nominal': 0.0,
        },
        'chi_diff_max': {
            'values': np.linspace(0.05, 0.4, 20),
            'nominal': 0.2,
        },
        'snr_max': {
            'values': np.linspace(10, 100, 20),
            'nominal': 50,
        },
    }

    # Extended chi_diff_threshold scans (lower bounds of -0.3 and -0.2)
    chi_diff_extended_scans = {
        'chi_diff_threshold_ext03': {
            'values': np.linspace(-0.3, 0.1, 30),
            'nominal': 0.0,
        },
        'chi_diff_threshold_ext02': {
            'values': np.linspace(-0.2, 0.1, 25),
            'nominal': 0.0,
        },
    }

    # --- Excluded events (double-counted with 2016 BL dataset) ---
    excluded_events_list = [
        (18, 82), (18, 520), (18, 681),
        (15, 1472768),
        (19, 3621320), (19, 4599318), (19, 4599919)
    ]

    # --- Load Simulation Data ---
    HRAeventList = loadHRAfromH5(sim_file)

    # --- Load 2016 BL Events ---
    json_path = 'StationDataAnalysis/2016FoundEvents.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            found_events_json = json.load(f)
        ic(f"Loaded 2016 Found Events from {json_path}")
    else:
        ic(f"Warning: JSON file not found at {json_path}")
        found_events_json = {}

    # --- Load and Assemble Summed Station Data ---
    ic("Loading and assembling station data...")
    all_snr, all_chi2016, all_chircr, all_times, all_station_ids_arr, all_event_ids = (
        [], [], [], [], [], [])

    for station_id in station_ids:
        snr_array = load_station_data(station_data_folder, date, station_id, 'SNR')
        Chi2016_array = load_station_data(station_data_folder, date, station_id, 'Chi2016')
        ChiRCR_array = load_station_data(station_data_folder, date, station_id, 'ChiRCR')
        times = load_station_data(station_data_folder, date, station_id, 'Time')
        event_ids_raw = load_station_data(station_data_folder, date, station_id, 'EventIDs')

        if Chi2016_array.size == 0 or ChiRCR_array.size == 0:
            ic(f"Skipping Station {station_id} due to missing Chi data.")
            continue

        # Apply initial time and uniqueness cuts (matching S01)
        initial_mask, unique_indices = getTimeEventMasks(times, event_ids_raw)

        # Apply C00 cuts
        cuts_mask = load_cuts_for_station(date, station_id, cuts_data_folder)
        if cuts_mask is not None:
            temp_times = times[initial_mask][unique_indices]
            if len(cuts_mask) != len(temp_times):
                cuts_mask = cuts_mask[:len(temp_times)]
            final_indices = unique_indices[cuts_mask]
        else:
            ic(f"Error: No cuts found for Station {station_id}.")
            sys.exit(1)

        all_snr.append(snr_array[initial_mask][final_indices])
        all_chi2016.append(Chi2016_array[initial_mask][final_indices])
        all_chircr.append(ChiRCR_array[initial_mask][final_indices])
        all_times.append(times[initial_mask][final_indices])
        all_station_ids_arr.append(np.full(len(final_indices), station_id, dtype=int))
        all_event_ids.append(event_ids_raw[initial_mask][final_indices])

        ic(f"Station {station_id}: {len(final_indices)} events after cuts.")

    # Concatenate all stations
    data_dict = {
        'snr': np.concatenate(all_snr),
        'Chi2016': np.concatenate(all_chi2016),
        'ChiRCR': np.concatenate(all_chircr),
        'Time': np.concatenate(all_times),
    }
    data_station_ids = np.concatenate(all_station_ids_arr)
    data_event_ids = np.concatenate(all_event_ids)

    ic(f"Total data events: {len(data_dict['snr'])}")

    # Build excluded events mask
    excluded_set = set(excluded_events_list)
    excluded_mask = np.zeros(len(data_dict['snr']), dtype=bool)
    for idx in range(len(data_dict['snr'])):
        if (data_station_ids[idx], data_event_ids[idx]) in excluded_set:
            excluded_mask[idx] = True
    ic(f"Excluded {np.sum(excluded_mask)} double-counted events.")

    # --- Load 2016 BL Data (from JSON times matched to loaded station data) ---
    bl_2016_snr, bl_2016_chi2016, bl_2016_chircr = [], [], []
    for station_id in station_ids:
        station_key = f"Station{station_id}Found"
        if station_key not in found_events_json:
            continue
        target_times = found_events_json[station_key]

        snr_array = load_station_data(station_data_folder, date, station_id, 'SNR')
        Chi2016_array = load_station_data(station_data_folder, date, station_id, 'Chi2016')
        ChiRCR_array = load_station_data(station_data_folder, date, station_id, 'ChiRCR')
        times_raw = load_station_data(station_data_folder, date, station_id, 'Time')

        if snr_array.size == 0:
            continue

        time_map = {t: i for i, t in enumerate(times_raw)}
        found_indices = [time_map[t] for t in target_times if t in time_map]
        found_indices = np.unique(found_indices)

        if len(found_indices) > 0:
            bl_2016_snr.append(snr_array[found_indices])
            bl_2016_chi2016.append(Chi2016_array[found_indices])
            bl_2016_chircr.append(ChiRCR_array[found_indices])

    if bl_2016_snr:
        bl_2016_data = {
            'snr': np.concatenate(bl_2016_snr),
            'Chi2016': np.concatenate(bl_2016_chi2016),
            'ChiRCR': np.concatenate(bl_2016_chircr),
        }
        ic(f"Loaded {len(bl_2016_data['snr'])} 2016 BL events.")
    else:
        bl_2016_data = None
        ic("No 2016 BL events found.")

    # --- Run Analysis (with pre-filter, matching S01) ---
    summary_file = os.path.join(plot_folder, 'summary_table.txt')
    with open(summary_file, 'w') as f:
        f.write(f"S03 Cut Scan Error Analysis — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load sim data with pre-filter (matching S01 behavior)
    sim_direct, sim_reflected = get_sim_data(
        HRAeventList, direct_weight_name, reflected_weight_name,
        direct_stations, reflected_stations, sigma=sim_sigma,
        apply_chi_diff_prefilter=True
    )

    ic(f"Sim direct: {len(sim_direct['snr'])} entries, raw weight sum = {np.sum(sim_direct['weights']):.4f}")
    ic(f"Sim reflected: {len(sim_reflected['snr'])} entries, raw weight sum = {np.sum(sim_reflected['weights']):.4f}")

    # Rescale weights to match known expected event totals from HRAAnalysis
    rescale_weights(sim_reflected, KNOWN_RCR_TOTAL)
    rescale_weights(sim_direct, KNOWN_BL_TOTAL)

    # Summary table at nominal
    print_summary_table(
        nominal_cuts, sim_direct, sim_reflected,
        data_dict, data_station_ids, bl_2016_data,
        excluded_mask, "Nominal Cuts",
        output_path=summary_file
    )

    # --- Run standard scans ---
    for param_name, param_info in scan_params.items():
        ic(f"Scanning {param_name}...")
        rcr_c, rcr_e, bl_c, bl_e, bl16_c, data_c = scan_cut_parameter(
            param_name, param_info['values'], nominal_cuts,
            sim_direct, sim_reflected,
            data_dict, data_station_ids, bl_2016_data,
            excluded_mask
        )

        output_path = os.path.join(plot_folder, f'scan_{param_name}.png')
        plot_cut_scan(
            param_name, param_info['values'], param_info['nominal'],
            rcr_c, rcr_e, bl_c, bl_e, bl16_c, data_c,
            output_path
        )

        # Extra variants for chi_rcr_flat
        if param_name == 'chi_rcr_flat':
            output_log = os.path.join(plot_folder, f'scan_{param_name}_logscale.png')
            plot_cut_scan(
                param_name, param_info['values'], param_info['nominal'],
                rcr_c, rcr_e, bl_c, bl_e, bl16_c, data_c,
                output_log, data_yscale='log'
            )

    # --- Run extended chi_diff_threshold scans ---
    for ext_name, ext_info in chi_diff_extended_scans.items():
        ic(f"Scanning {ext_name}...")
        rcr_c, rcr_e, bl_c, bl_e, bl16_c, data_c = scan_cut_parameter(
            'chi_diff_threshold', ext_info['values'], nominal_cuts,
            sim_direct, sim_reflected,
            data_dict, data_station_ids, bl_2016_data,
            excluded_mask
        )

        output_path = os.path.join(plot_folder, f'scan_{ext_name}.png')
        plot_cut_scan(
            'chi_diff_threshold', ext_info['values'], ext_info['nominal'],
            rcr_c, rcr_e, bl_c, bl_e, bl16_c, data_c,
            output_path
        )

    ic("\nDone. All outputs saved to: " + plot_folder)
