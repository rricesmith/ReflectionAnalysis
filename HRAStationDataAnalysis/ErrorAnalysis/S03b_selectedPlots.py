"""
S03b_selectedPlots.py
=====================
Streamlined cut-analysis plots, built on S03's data loading and weighting.

Produces only:
  1. Coincidence cut test (3-subplot figure with BL after coinc, RCR sim, general sims, errors)
  2. dist_errorbar (no-cuts, full-range, log-scale, errorbar data)
  3. Parameter width assessment (sim distributions after RCR/BL cuts, with errors)
  4. Cut interaction cumulative (delta-chi [-0.15,0.15] normal; chi-RCR full log)
  5. Scale factor fit (same style as cut interaction, with/without original sims)
  6. Text output (events passing, errors, scale factors, predictions)

Global changes vs S03:
  - SNR < 50 prefilter applied to ALL data immediately after loading
  - Data legend positioned in upper right, below Simulation legend
  - Marker convention: Identified RCR = red triangles, Identified BL = cyan squares,
    Pass RCR Cuts = green stars (black outline), Pass BL Cuts = yellow circles (black outline)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import copy
import numpy as np
import matplotlib.pyplot as plt
import configparser
import json
import glob
import h5py
import pickle
from datetime import datetime
from icecream import ic

from NuRadioReco.utilities import units
from HRAStationDataAnalysis.C_utils import getTimeEventMasks
from HRASimulation.HRAEventObject import HRAevent
from RCRSimulation.RCRAnalysis import (
    getEnergyZenithBins, getEventRate,
    getBinnedTriggerRate, get_all_r_triggers,
)
from RCRSimulation.S04_RCRChapter4Plots import (
    load_combined_events, MB_REFLECTED_SIMS, find_direct_trigger,
)


# ============================================================================
# Constants
# ============================================================================
LIVETIME_YEARS = 11.7
SNR_PREFILTER = 50  # applied to ALL data before any analysis

PARAM_LABELS = {
    'chi_rcr_flat': r'$\chi_{\mathrm{RCR}}$',
    'chi_diff_threshold': r'$\Delta\chi$ ($\chi_{\mathrm{RCR}} - \chi_{BL}$)',
    'chi_diff_max': r'$\Delta\chi$ ($\chi_{\mathrm{RCR}} - \chi_{BL}$)',
    'snr_max': 'SNR',
    'chi_bl': r'$\chi_{BL}$',
}

PARAM_TITLES = {
    'chi_rcr_flat': r'$\chi_{\mathrm{RCR}}$',
    'chi_diff_threshold': r'$\Delta\chi$',
    'chi_diff_max': r'$\Delta\chi$ Maximum',
    'snr_max': 'SNR Maximum',
    'chi_bl': r'$\chi_{BL}$',
}

# Font sizes
FONTSIZE_LABEL = 14
FONTSIZE_TITLE = 16
FONTSIZE_TICK = 12
FONTSIZE_LEGEND = 10
FONTSIZE_LEGEND_TITLE = 11

# Complementary-log scale: focuses on values near 1.0
# forward: x -> -log10(1.001 - x)   (spreads 0.8-1.0 region)
# inverse: y -> 1.001 - 10^(-y)
_COMP_LOG_EPS = 0.001  # small offset to avoid log(0) at x=1
def _comp_log_forward(x):
    return -np.log10(_COMP_LOG_EPS + 1.0 - np.clip(x, 0, 1.0))
def _comp_log_inverse(y):
    return 1.0 + _COMP_LOG_EPS - 10.0**(-y)

def set_comp_log_scale(ax, axis='both'):
    """Apply complementary-log scale to an axis to zoom into 0.8–1.0 region."""
    funcs = (_comp_log_forward, _comp_log_inverse)
    if axis in ('x', 'both'):
        ax.set_xscale('function', functions=funcs)
    if axis in ('y', 'both'):
        ax.set_yscale('function', functions=funcs)


# ============================================================================
# Data Loading (reused from S03)
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
    """Extracts SNR, Chi, and weights from the HRAevent list."""
    direct_data = {'snr': [], 'Chi2016': [], 'ChiRCR': [], 'weights': [], 'energy': [], 'zenith': []}
    reflected_data = {'snr': [], 'Chi2016': [], 'ChiRCR': [], 'weights': [], 'energy': [], 'zenith': []}

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
                        direct_data['energy'].append(event.energy)
                        direct_data['zenith'].append(event.zenith)

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
                        reflected_data['energy'].append(event.energy)
                        reflected_data['zenith'].append(event.zenith)

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


def load_s04_event_rates(numpy_folder, max_distance):
    """Load S04 event rates for bin-by-bin weighting."""
    sim_name = MB_REFLECTED_SIMS["HRA"]
    events = load_combined_events(numpy_folder, sim_name)
    if events is None:
        raise FileNotFoundError(f"Could not load RCR simulation: {sim_name} from {numpy_folder}")

    e_bins, z_bins = getEnergyZenithBins()
    events_list = list(events)
    r_triggers = get_all_r_triggers(events_list)
    if not r_triggers:
        raise RuntimeError(f"No R-based triggers found in {sim_name}")

    direct_trigger = find_direct_trigger(events_list)
    if direct_trigger is None:
        raise RuntimeError(f"No untagged (direct) trigger found in {sim_name}")
    dir_trig_rate, _, _ = getBinnedTriggerRate(events_list, direct_trigger)
    direct_rate = getEventRate(dir_trig_rate, e_bins, z_bins, max_distance)

    reflected_rates = {}
    for r_val, trig_name in sorted(r_triggers.items()):
        _, ref_rate, _ = getBinnedTriggerRate(events_list, trig_name)
        reflected_event_rate = getEventRate(ref_rate, e_bins, z_bins, max_distance)
        reflected_rates[r_val] = reflected_event_rate

    return reflected_rates, direct_rate, e_bins, z_bins


def assign_binned_weights(sim_data, events_per_bin, e_bins, z_bins, label=""):
    """Assign weights to sim events based on their energy-zenith bin."""
    energies = sim_data['energy']
    zeniths = sim_data['zenith']
    n_events = len(energies)
    n_e = len(e_bins) - 1
    n_z = len(z_bins) - 1

    e_indices = np.digitize(energies, e_bins) - 1
    z_indices = np.digitize(zeniths, z_bins) - 1

    counts_per_bin = np.zeros((n_e, n_z), dtype=int)
    for k in range(n_events):
        ei, zi = int(e_indices[k]), int(z_indices[k])
        if 0 <= ei < n_e and 0 <= zi < n_z:
            counts_per_bin[ei, zi] += 1

    new_weights = np.zeros(n_events)
    for k in range(n_events):
        ei, zi = int(e_indices[k]), int(z_indices[k])
        if 0 <= ei < n_e and 0 <= zi < n_z and counts_per_bin[ei, zi] > 0:
            new_weights[k] = events_per_bin[ei, zi] / counts_per_bin[ei, zi]

    sim_data['weights'] = new_weights
    n_empty = np.sum((events_per_bin > 0) & (counts_per_bin == 0))
    ic(f"{label}: Total weighted events = {np.sum(new_weights):.4f}, "
       f"bins with expected>0 but 0 sim events: {n_empty}")


# ============================================================================
# Cut Analysis Functions
# ============================================================================

def apply_cuts(data_dict, cuts, cut_type='rcr', exclude_param=None):
    """Apply cuts and return a boolean mask of events passing."""
    snr = data_dict['snr']
    chircr = data_dict['ChiRCR']
    chi2016 = data_dict['Chi2016']
    chi_diff = chircr - chi2016
    mask = np.ones(len(snr), dtype=bool)

    if exclude_param != 'snr_max':
        mask &= snr < cuts['snr_max']

    if exclude_param != 'chi_rcr_flat':
        chi_line_values = np.interp(snr, cuts['chi_rcr_line_snr'], cuts['chi_rcr_line_chi'])
        if cut_type == 'backlobe':
            mask &= chi2016 > chi_line_values
        else:
            mask &= chircr > chi_line_values

    if cut_type == 'rcr':
        if exclude_param != 'chi_diff_threshold':
            mask &= chi_diff > cuts['chi_diff_threshold']
        if exclude_param != 'chi_diff_max':
            mask &= chi_diff < cuts.get('chi_diff_max', 999)
    elif cut_type == 'backlobe':
        if exclude_param != 'chi_diff_threshold':
            mask &= chi_diff < -cuts['chi_diff_threshold']
        if exclude_param != 'chi_diff_max':
            mask &= chi_diff > -cuts.get('chi_diff_max', 999)

    return mask


def compute_weighted_count_and_error(weights, mask):
    """Compute expected count and statistical error for weighted sim events."""
    w = weights[mask]
    if len(w) == 0:
        return 0.0, 0.0
    return np.sum(w), np.sqrt(np.sum(w**2))


def get_param_values(param_name, data_dict):
    """Extract parameter values from data dictionary."""
    if param_name == 'chi_rcr_flat':
        return data_dict['ChiRCR']
    elif param_name in ('chi_diff_threshold', 'chi_diff_max'):
        return data_dict['ChiRCR'] - data_dict['Chi2016']
    elif param_name == 'snr_max':
        return data_dict['snr']
    elif param_name == 'chi_bl':
        return data_dict['Chi2016']
    else:
        raise ValueError(f"Unknown param_name: {param_name}")


# ============================================================================
# Legend helper
# ============================================================================

def add_split_legend(ax, sim_handles, data_handles):
    """Add Simulation legend in upper right, Data legend directly below it."""
    sim_legend = ax.legend(
        handles=sim_handles,
        labels=[h.get_label() for h in sim_handles],
        loc='upper right', fontsize=FONTSIZE_LEGEND,
        title='Simulation', title_fontsize=FONTSIZE_LEGEND_TITLE)
    ax.add_artist(sim_legend)

    # Position Data legend below the Simulation legend using bbox_to_anchor
    # Use upper right with a vertical offset
    sim_box = sim_legend.get_frame()
    n_sim = len(sim_handles)
    # Approximate: each legend entry ~0.04 of axes height, plus title ~0.04
    vertical_offset = (n_sim + 1) * 0.045
    data_loc = (1.0, 1.0 - vertical_offset)
    ax.legend(
        handles=data_handles,
        labels=[h.get_label() for h in data_handles],
        loc='upper right', fontsize=FONTSIZE_LEGEND,
        title='Data', title_fontsize=FONTSIZE_LEGEND_TITLE,
        bbox_to_anchor=data_loc)


# ============================================================================
# Plot 1: Coincidence Cut Test (3 subplots)
# ============================================================================

def plot_coincidence_cut_test(coinc_bl_data, coinc_rcr_data,
                               coinc_bl_rate, coinc_bl_rate_err,
                               coinc_rcr_rate, coinc_rcr_rate_err,
                               sim_direct, sim_reflected,
                               sim_direct_high, sim_reflected_high,
                               sim_direct_low, sim_reflected_low,
                               output_path, n_bins=50):
    """
    3-subplot figure: chi-RCR, chi-BL, SNR.
    Shows: Coincidence BL events, Coincidence RCR events (weighted by total rate),
           general RCR Sim (no cut), general BL Sim (no cut).
    All with error bands.
    SNR subplot has log x-scale.
    """
    # Assign weights to coinc events: total_rate / n_events
    n_coinc_bl = len(coinc_bl_data['snr'])
    n_coinc_rcr = len(coinc_rcr_data['snr'])
    if n_coinc_bl == 0 and n_coinc_rcr == 0:
        ic("Coincidence cut test: no coincidence events, skipping")
        return

    coinc_bl_weights = np.full(n_coinc_bl, coinc_bl_rate / n_coinc_bl) if n_coinc_bl > 0 else np.array([])
    coinc_bl_weights_hi = np.full(n_coinc_bl, (coinc_bl_rate + coinc_bl_rate_err) / n_coinc_bl) if n_coinc_bl > 0 else np.array([])
    coinc_bl_weights_lo = np.full(n_coinc_bl, (coinc_bl_rate - coinc_bl_rate_err) / n_coinc_bl) if n_coinc_bl > 0 else np.array([])

    coinc_rcr_weights = np.full(n_coinc_rcr, coinc_rcr_rate / n_coinc_rcr) if n_coinc_rcr > 0 else np.array([])
    coinc_rcr_weights_hi = np.full(n_coinc_rcr, (coinc_rcr_rate + coinc_rcr_rate_err) / n_coinc_rcr) if n_coinc_rcr > 0 else np.array([])
    coinc_rcr_weights_lo = np.full(n_coinc_rcr, (coinc_rcr_rate - coinc_rcr_rate_err) / n_coinc_rcr) if n_coinc_rcr > 0 else np.array([])

    params_to_plot = [
        ('chi_rcr_flat', (0.2, 0.9), r'$\chi_{\mathrm{RCR}}$', False),
        ('chi_bl', (0.0, 1.0), r'$\chi_{BL}$', False),
        ('snr_max', (3, 100), 'SNR', True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    for ax_idx, (pname, prange, plabel, log_x) in enumerate(params_to_plot):
        ax = axes[ax_idx]
        vals_bl_sim = get_param_values(pname, sim_direct)
        vals_rcr_sim = get_param_values(pname, sim_reflected)

        if log_x:
            bin_edges = np.logspace(np.log10(prange[0]), np.log10(prange[1]), n_bins + 1)
        else:
            bin_edges = np.linspace(prange[0], prange[1], n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1:] - bin_edges[:-1]

        # General sims (no cut) — background
        bl_hist_all, _ = np.histogram(vals_bl_sim, bins=bin_edges, weights=sim_direct['weights'])
        rcr_hist_all, _ = np.histogram(vals_rcr_sim, bins=bin_edges, weights=sim_reflected['weights'])

        # Statistical errors for general sims
        bl_err_all = np.sqrt(np.histogram(vals_bl_sim, bins=bin_edges,
                                           weights=sim_direct['weights']**2)[0])
        rcr_stat_err_all = np.sqrt(np.histogram(vals_rcr_sim, bins=bin_edges,
                                                  weights=sim_reflected['weights']**2)[0])

        # RCR R-sweep error
        rcr_hist_hi, _ = np.histogram(vals_rcr_sim, bins=bin_edges, weights=sim_reflected_high['weights'])
        rcr_hist_lo, _ = np.histogram(vals_rcr_sim, bins=bin_edges, weights=sim_reflected_low['weights'])
        rcr_band_lo_all = np.minimum(rcr_hist_lo, rcr_hist_hi) - rcr_stat_err_all
        rcr_band_hi_all = np.maximum(rcr_hist_lo, rcr_hist_hi) + rcr_stat_err_all

        # Coincidence BL events (weighted)
        if n_coinc_bl > 0:
            coinc_bl_vals = get_param_values(pname, coinc_bl_data)
            coinc_bl_hist, _ = np.histogram(coinc_bl_vals, bins=bin_edges, weights=coinc_bl_weights)
            coinc_bl_hist_hi, _ = np.histogram(coinc_bl_vals, bins=bin_edges, weights=coinc_bl_weights_hi)
            coinc_bl_hist_lo, _ = np.histogram(coinc_bl_vals, bins=bin_edges, weights=coinc_bl_weights_lo)
        else:
            coinc_bl_hist = np.zeros(n_bins)
            coinc_bl_hist_hi = np.zeros(n_bins)
            coinc_bl_hist_lo = np.zeros(n_bins)

        # Coincidence RCR events (weighted)
        if n_coinc_rcr > 0:
            coinc_rcr_vals = get_param_values(pname, coinc_rcr_data)
            coinc_rcr_hist, _ = np.histogram(coinc_rcr_vals, bins=bin_edges, weights=coinc_rcr_weights)
            coinc_rcr_hist_hi, _ = np.histogram(coinc_rcr_vals, bins=bin_edges, weights=coinc_rcr_weights_hi)
            coinc_rcr_hist_lo, _ = np.histogram(coinc_rcr_vals, bins=bin_edges, weights=coinc_rcr_weights_lo)
        else:
            coinc_rcr_hist = np.zeros(n_bins)
            coinc_rcr_hist_hi = np.zeros(n_bins)
            coinc_rcr_hist_lo = np.zeros(n_bins)

        # Plot general sims (lighter, background)
        ax.step(bin_edges[:-1], rcr_hist_all, where='post', color='green', linewidth=1.5,
                linestyle=':', alpha=0.6, label='RCR Sim (no cut)')
        ax.fill_between(bin_edges[:-1], rcr_band_lo_all, rcr_band_hi_all,
                         step='post', color='green', alpha=0.08)

        ax.step(bin_edges[:-1], bl_hist_all, where='post', color='orange', linewidth=1.5,
                linestyle=':', alpha=0.6, label='BL Sim (no cut)')
        ax.bar(bin_centers, 2 * bl_err_all, bottom=bl_hist_all - bl_err_all, width=bin_width,
               color='orange', alpha=0.08, edgecolor='none')

        # Plot coincidence event distributions (foreground)
        ax.step(bin_edges[:-1], coinc_bl_hist, where='post', color='orange', linewidth=2,
                label=f'Coinc BL ({coinc_bl_rate:.1f}$\\pm${coinc_bl_rate_err:.1f} evts/yr)')
        ax.fill_between(bin_edges[:-1],
                         np.minimum(coinc_bl_hist_lo, coinc_bl_hist_hi),
                         np.maximum(coinc_bl_hist_lo, coinc_bl_hist_hi),
                         step='post', color='orange', alpha=0.2)

        if n_coinc_rcr > 0:
            ax.step(bin_edges[:-1], coinc_rcr_hist, where='post', color='green', linewidth=2,
                    label=f'Coinc RCR ({coinc_rcr_rate:.1f}$\\pm${coinc_rcr_rate_err:.2f} evts/yr)')
            ax.fill_between(bin_edges[:-1],
                             np.minimum(coinc_rcr_hist_lo, coinc_rcr_hist_hi),
                             np.maximum(coinc_rcr_hist_lo, coinc_rcr_hist_hi),
                             step='post', color='green', alpha=0.2)

        ax.set_xlabel(plabel, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel('Events per Bin (evts/yr)', fontsize=FONTSIZE_LABEL)
        ax.set_title(f'Coincidence Events: {plabel}', fontsize=FONTSIZE_TITLE)
        ax.tick_params(axis='both', labelsize=FONTSIZE_TICK)
        ax.legend(fontsize=9, loc='upper left')
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.001)
        ax.grid(True, alpha=0.3)
        if log_x:
            ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    ic(f"Coincidence cut test: {n_coinc_bl} BL events, {n_coinc_rcr} RCR events")


# ============================================================================
# Plot 2: dist_errorbar (no-cuts debug distributions)
# ============================================================================

def plot_dist_errorbar(param_name, nominal_value, sim_direct, sim_reflected,
                       sim_direct_high, sim_reflected_high,
                       sim_direct_low, sim_reflected_low,
                       data_dict, identified_bl_data, identified_rcr_data,
                       excluded_events_mask, output_path,
                       n_bins=50, param_range=None, show_cut_line=True):
    """
    No-cuts, full-range, log-scale distribution with errorbar data.
    RCR sim: statistical + R-sweep error band.
    BL sim: statistical error only.
    """
    param_label = PARAM_LABELS.get(param_name, param_name)
    title_label = PARAM_TITLES.get(param_name, param_name)

    rcr_vals = get_param_values(param_name, sim_reflected)
    bl_vals = get_param_values(param_name, sim_direct)
    data_vals = get_param_values(param_name, data_dict)
    data_not_excl = ~excluded_events_mask

    log_x = (param_name == 'snr_max')

    if param_range is None:
        all_v = np.concatenate([rcr_vals, bl_vals, data_vals[data_not_excl]])
        param_range = (np.nanmin(all_v), np.nanmax(all_v))

    if log_x:
        bin_edges = np.logspace(np.log10(max(param_range[0], 1e-3)),
                                np.log10(param_range[1]), n_bins + 1)
    else:
        bin_edges = np.linspace(param_range[0], param_range[1], n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1:] - bin_edges[:-1]

    # Weighted histograms
    rcr_hist, _ = np.histogram(rcr_vals, bins=bin_edges, weights=sim_reflected['weights'])
    bl_hist, _ = np.histogram(bl_vals, bins=bin_edges, weights=sim_direct['weights'])
    both_hist = rcr_hist + bl_hist

    # Statistical errors
    rcr_stat_err = np.sqrt(np.histogram(rcr_vals, bins=bin_edges,
                                         weights=sim_reflected['weights']**2)[0])
    bl_stat_err = np.sqrt(np.histogram(bl_vals, bins=bin_edges,
                                        weights=sim_direct['weights']**2)[0])
    both_stat_err = np.sqrt(rcr_stat_err**2 + bl_stat_err**2)

    # RCR R-sweep bands
    rcr_hist_hi, _ = np.histogram(rcr_vals, bins=bin_edges, weights=sim_reflected_high['weights'])
    rcr_hist_lo, _ = np.histogram(rcr_vals, bins=bin_edges, weights=sim_reflected_low['weights'])
    rcr_band_lo = np.minimum(rcr_hist_lo, rcr_hist_hi) - rcr_stat_err
    rcr_band_hi = np.maximum(rcr_hist_lo, rcr_hist_hi) + rcr_stat_err

    # BL: stat error only
    bl_band_lo = bl_hist - bl_stat_err
    bl_band_hi = bl_hist + bl_stat_err

    # Both: combine
    both_band_lo = np.minimum(rcr_hist_lo + bl_hist, rcr_hist_hi + bl_hist) - both_stat_err
    both_band_hi = np.maximum(rcr_hist_lo + bl_hist, rcr_hist_hi + bl_hist) + both_stat_err

    # Data histogram
    data_hist, _ = np.histogram(data_vals[data_not_excl], bins=bin_edges)

    # Identified sets
    if identified_bl_data is not None and len(identified_bl_data['snr']) > 0:
        ibl_hist, _ = np.histogram(get_param_values(param_name, identified_bl_data), bins=bin_edges)
    else:
        ibl_hist = np.zeros(n_bins)
    if identified_rcr_data is not None and len(identified_rcr_data['snr']) > 0:
        ircr_hist, _ = np.histogram(get_param_values(param_name, identified_rcr_data), bins=bin_edges)
    else:
        ircr_hist = np.zeros(n_bins)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    sim_handles = []
    ax.fill_between(bin_centers, rcr_band_lo, rcr_band_hi, color='green', alpha=0.15, step='mid')
    h = ax.step(bin_edges[:-1], rcr_hist, where='post', color='green', linewidth=2, label='RCR Sim')
    sim_handles.append(h[0])

    ax.bar(bin_centers, bl_band_hi - bl_band_lo, bottom=bl_band_lo, width=bin_width,
           color='orange', alpha=0.15, edgecolor='none')
    h = ax.step(bin_edges[:-1], bl_hist, where='post', color='orange', linewidth=2, label='BL Sim')
    sim_handles.append(h[0])

    ax.fill_between(bin_centers, both_band_lo, both_band_hi, color='purple', alpha=0.08, step='mid')
    h = ax.step(bin_edges[:-1], both_hist, where='post', color='purple', linewidth=2,
                linestyle='--', label='Both Sim')
    sim_handles.append(h[0])

    if show_cut_line:
        cut_line = ax.axvline(x=nominal_value, color='red', linestyle='--', linewidth=1.5,
                               label=f'Cut at {nominal_value}', alpha=0.7)
        sim_handles.append(cut_line)

    data_handles = []
    data_err_bars = np.sqrt(np.maximum(data_hist, 0))
    h = ax.errorbar(bin_centers, data_hist, yerr=data_err_bars, fmt='ko',
                     markersize=7, markeredgecolor='white', markeredgewidth=1.5,
                     capsize=3, elinewidth=1.2, label='Data Events', zorder=6)
    data_handles.append(h)

    if np.any(ibl_hist > 0):
        h, = ax.plot(bin_centers, ibl_hist, 'cs', markersize=6,
                     markeredgecolor='white', markeredgewidth=1.0,
                     label='Identified BL', zorder=6)
        data_handles.append(h)
    if np.any(ircr_hist > 0):
        h, = ax.plot(bin_centers, ircr_hist, 'r^', markersize=7,
                     markeredgecolor='white', markeredgewidth=1.0,
                     label='Identified RCR', zorder=6)
        data_handles.append(h)

    ax.set_xlabel(param_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Events per Bin', fontsize=FONTSIZE_LABEL)
    ax.set_title(f'Distribution: {title_label}', fontsize=FONTSIZE_TITLE)
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.1)
    if log_x:
        ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    add_split_legend(ax, sim_handles, data_handles)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    ic(f"Saved dist_errorbar plot: {output_path}")


# ============================================================================
# Plot 3: Parameter Width Assessment (with cuts applied, with errors)
# ============================================================================

def _draw_param_width_on_ax(ax, param_name, sim_direct, sim_reflected,
                             sim_direct_high, sim_reflected_high,
                             sim_direct_low, sim_reflected_low,
                             data_dict, excluded_events_mask,
                             nominal_cuts, n_bins=50, param_range=None,
                             show_snr10_subset=True, compact=False):
    """
    Core drawing logic for parameter width histogram on a given axes.
    compact=True reduces font sizes and legend for subplot use.
    """
    param_label = PARAM_LABELS.get(param_name, param_name)
    title_label = PARAM_TITLES.get(param_name, param_name)

    fs_label = FONTSIZE_LABEL - 2 if compact else FONTSIZE_LABEL
    fs_title = FONTSIZE_TITLE - 2 if compact else FONTSIZE_TITLE
    fs_tick = FONTSIZE_TICK - 1 if compact else FONTSIZE_TICK
    fs_legend = 8 if compact else FONTSIZE_LEGEND

    # Apply ALL RCR cuts to RCR sim, ALL BL cuts to BL sim (no exclusion)
    rcr_mask = apply_cuts(sim_reflected, nominal_cuts, cut_type='rcr')
    bl_mask = apply_cuts(sim_direct, nominal_cuts, cut_type='backlobe')

    rcr_vals = get_param_values(param_name, sim_reflected)
    bl_vals = get_param_values(param_name, sim_direct)

    # SNR > 10 subsets (within already-cut events)
    rcr_snr10 = rcr_mask & (sim_reflected['snr'] > 10)
    bl_snr10 = bl_mask & (sim_direct['snr'] > 10)

    # Data subsets
    data_vals = get_param_values(param_name, data_dict)
    data_not_excl = ~excluded_events_mask

    data_rcr_mask = apply_cuts(data_dict, nominal_cuts, cut_type='rcr') & data_not_excl
    data_bl_mask = apply_cuts(data_dict, nominal_cuts, cut_type='backlobe') & data_not_excl

    if param_range is None:
        all_v = np.concatenate([rcr_vals[rcr_mask], bl_vals[bl_mask], data_vals[data_not_excl]])
        param_range = (np.nanmin(all_v), np.nanmax(all_v))

    bin_edges = np.linspace(param_range[0], param_range[1], n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # RCR sim histograms (after RCR cuts)
    rcr_hist_all, _ = np.histogram(rcr_vals[rcr_mask], bins=bin_edges,
                                    weights=sim_reflected['weights'][rcr_mask])

    # RCR R-sweep error for all (after RCR cuts)
    rcr_hist_hi, _ = np.histogram(rcr_vals[rcr_mask], bins=bin_edges,
                                   weights=sim_reflected_high['weights'][rcr_mask])
    rcr_hist_lo, _ = np.histogram(rcr_vals[rcr_mask], bins=bin_edges,
                                   weights=sim_reflected_low['weights'][rcr_mask])
    rcr_stat_err = np.sqrt(np.histogram(rcr_vals[rcr_mask], bins=bin_edges,
                                         weights=sim_reflected['weights'][rcr_mask]**2)[0])
    rcr_band_lo = np.minimum(rcr_hist_lo, rcr_hist_hi) - rcr_stat_err
    rcr_band_hi = np.maximum(rcr_hist_lo, rcr_hist_hi) + rcr_stat_err

    # BL sim histograms (after BL cuts)
    bl_hist_all, _ = np.histogram(bl_vals[bl_mask], bins=bin_edges,
                                   weights=sim_direct['weights'][bl_mask])

    bl_stat_err = np.sqrt(np.histogram(bl_vals[bl_mask], bins=bin_edges,
                                        weights=sim_direct['weights'][bl_mask]**2)[0])

    # Plot sim distributions
    sim_handles = []
    ax.fill_between(bin_centers, rcr_band_lo, rcr_band_hi, color='green', alpha=0.12, step='mid')
    h = ax.step(bin_edges[:-1], rcr_hist_all, where='post', color='green', linewidth=2,
                label='RCR Sim (RCR cuts)')
    sim_handles.append(h[0])

    if show_snr10_subset:
        rcr_hist_snr10, _ = np.histogram(rcr_vals[rcr_snr10], bins=bin_edges,
                                          weights=sim_reflected['weights'][rcr_snr10])
        h = ax.step(bin_edges[:-1], rcr_hist_snr10, where='post', color='green', linewidth=2,
                    linestyle='--', label='RCR Sim (SNR>10)')
        sim_handles.append(h[0])

    ax.bar(bin_centers, 2 * bl_stat_err, bottom=bl_hist_all - bl_stat_err, width=bin_width,
           color='orange', alpha=0.12, edgecolor='none')
    h = ax.step(bin_edges[:-1], bl_hist_all, where='post', color='orange', linewidth=2,
                label='BL Sim (BL cuts)')
    sim_handles.append(h[0])

    if show_snr10_subset:
        bl_hist_snr10, _ = np.histogram(bl_vals[bl_snr10], bins=bin_edges,
                                         weights=sim_direct['weights'][bl_snr10])
        h = ax.step(bin_edges[:-1], bl_hist_snr10, where='post', color='orange', linewidth=2,
                    linestyle='--', label='BL Sim (SNR>10)')
        sim_handles.append(h[0])

    # Data histograms
    data_handles = []
    data_rcr_hist, _ = np.histogram(data_vals[data_rcr_mask], bins=bin_edges)
    data_bl_hist, _ = np.histogram(data_vals[data_bl_mask], bins=bin_edges)
    h, = ax.plot(bin_centers, data_rcr_hist, marker='*', color='green', markersize=8,
                 markeredgecolor='black', markeredgewidth=0.5,
                 linestyle='none', label='Data Pass RCR Cuts', zorder=6)
    data_handles.append(h)
    h, = ax.plot(bin_centers, data_bl_hist, marker='o', color='yellow', markersize=6,
                 markeredgecolor='black', markeredgewidth=0.5,
                 linestyle='none', label='Data Pass BL Cuts', zorder=6)
    data_handles.append(h)

    ax.set_xlabel(param_label, fontsize=fs_label)
    ax.set_ylabel('Events per Bin', fontsize=fs_label)
    ax.set_title(f'{title_label}', fontsize=fs_title)
    ax.tick_params(axis='both', labelsize=fs_tick)
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.1)
    ax.grid(True, alpha=0.3)

    if compact:
        ax.legend(fontsize=fs_legend, loc='upper left')
    else:
        add_split_legend(ax, sim_handles, data_handles)


def plot_parameter_width_assessment(param_name, sim_direct, sim_reflected,
                                    sim_direct_high, sim_reflected_high,
                                    sim_direct_low, sim_reflected_low,
                                    data_dict, excluded_events_mask,
                                    nominal_cuts, output_path,
                                    n_bins=50, param_range=None,
                                    show_snr10_subset=True):
    """Single-parameter standalone figure."""
    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_param_width_on_ax(ax, param_name, sim_direct, sim_reflected,
                             sim_direct_high, sim_reflected_high,
                             sim_direct_low, sim_reflected_low,
                             data_dict, excluded_events_mask, nominal_cuts,
                             n_bins=n_bins, param_range=param_range,
                             show_snr10_subset=show_snr10_subset, compact=False)
    fig.suptitle('Parameter Width (Sim: all RCR cuts on RCR, all BL cuts on BL)',
                 fontsize=FONTSIZE_TITLE, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    ic(f"Saved parameter width plot: {output_path}")


def plot_parameter_width_combined(width_params_list, sim_direct, sim_reflected,
                                   sim_direct_high, sim_reflected_high,
                                   sim_direct_low, sim_reflected_low,
                                   data_dict, excluded_events_mask,
                                   nominal_cuts, output_path,
                                   n_bins=50, show_snr10_subset=True):
    """
    Combined 1x3 subplot figure for all parameter widths.
    width_params_list: list of (param_name, param_range) tuples.
    """
    n_params = len(width_params_list)
    fig, axes = plt.subplots(1, n_params, figsize=(7 * n_params, 6))
    if n_params == 1:
        axes = [axes]

    for ax, (wp_name, wp_range) in zip(axes, width_params_list):
        _draw_param_width_on_ax(ax, wp_name, sim_direct, sim_reflected,
                                 sim_direct_high, sim_reflected_high,
                                 sim_direct_low, sim_reflected_low,
                                 data_dict, excluded_events_mask, nominal_cuts,
                                 n_bins=n_bins, param_range=wp_range,
                                 show_snr10_subset=show_snr10_subset, compact=True)

    snr10_str = " (incl. SNR>10 subset)" if show_snr10_subset else ""
    fig.suptitle(f'Parameter Width (Sim: all RCR cuts on RCR, all BL cuts on BL){snr10_str}',
                 fontsize=FONTSIZE_TITLE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    ic(f"Saved combined parameter width plot: {output_path}")


# ============================================================================
# Plot 4: Cut Interaction Cumulative
# ============================================================================

def plot_cut_interaction(param_name, nominal_cuts, nominal_value,
                         cross_cut_param, cross_cut_range,
                         sim_direct, sim_reflected,
                         sim_direct_high, sim_reflected_high,
                         sim_direct_low, sim_reflected_low,
                         data_dict, data_station_ids,
                         identified_bl_data, identified_rcr_data,
                         excluded_events_mask, output_path,
                         x_range=None, yscale='linear'):
    """
    Cumulative distribution with cross-cut band.
    Data = bar-style points (vertical whisker: min/max from cross-cut range, box at center).
    Identified RCR = single points at y=1 at their actual x-value.
    """
    # Build three cut configs: nominal, cross-low, cross-high
    def _make_cross_cuts(cross_val):
        cc = dict(nominal_cuts)
        if param_name == 'chi_rcr_flat':
            cc['chi_rcr_line_chi'] = np.zeros_like(nominal_cuts['chi_rcr_line_chi'])
        elif param_name == 'chi_diff_threshold':
            cc['chi_diff_threshold'] = -999
        elif param_name == 'chi_diff_max':
            cc['chi_diff_max'] = 999
        elif param_name == 'snr_max':
            cc['snr_max'] = 9999
        if cross_cut_param == 'chi_rcr_flat':
            cc['chi_rcr_line_chi'] = np.full_like(nominal_cuts['chi_rcr_line_chi'], cross_val)
        elif cross_cut_param == 'chi_diff_threshold':
            cc['chi_diff_threshold'] = cross_val
        elif cross_cut_param == 'chi_diff_max':
            cc['chi_diff_max'] = cross_val
        return cc

    cross_lo, cross_hi = cross_cut_range
    if cross_cut_param == 'chi_rcr_flat':
        cross_nominal = nominal_cuts['chi_rcr_line_chi'][0]
    elif cross_cut_param == 'chi_diff_threshold':
        cross_nominal = nominal_cuts['chi_diff_threshold']
    elif cross_cut_param == 'chi_diff_max':
        cross_nominal = nominal_cuts['chi_diff_max']
    else:
        cross_nominal = (cross_lo + cross_hi) / 2

    cuts_nom = _make_cross_cuts(cross_nominal)
    cuts_lo = _make_cross_cuts(cross_lo)
    cuts_hi = _make_cross_cuts(cross_hi)

    # Apply cross-cut masks
    rcr_mask_nom = apply_cuts(sim_reflected, cuts_nom, cut_type='rcr')
    rcr_mask_lo = apply_cuts(sim_reflected, cuts_lo, cut_type='rcr')
    rcr_mask_hi = apply_cuts(sim_reflected, cuts_hi, cut_type='rcr')
    bl_mask_nom = apply_cuts(sim_direct, cuts_nom, cut_type='rcr')
    bl_mask_lo = apply_cuts(sim_direct, cuts_lo, cut_type='rcr')
    bl_mask_hi = apply_cuts(sim_direct, cuts_hi, cut_type='rcr')

    rcr_all_vals = get_param_values(param_name, sim_reflected)
    bl_all_vals = get_param_values(param_name, sim_direct)

    if x_range is not None:
        x_scan = np.linspace(x_range[0], x_range[1], 200)
    else:
        all_vals = np.concatenate([rcr_all_vals[rcr_mask_nom], bl_all_vals[bl_mask_nom]])
        if len(all_vals) == 0:
            ic(f"Cut-interaction {param_name}: no events, skipping")
            return
        x_scan = np.linspace(np.nanmin(all_vals), np.nanmax(all_vals), 200)

    if param_name in ('chi_rcr_flat', 'chi_diff_threshold'):
        pass_fn = lambda vals, x: vals > x
    elif param_name in ('snr_max', 'chi_diff_max'):
        pass_fn = lambda vals, x: vals < x
    else:
        raise ValueError(f"Unknown param_name: {param_name}")

    def cumulative(vals, weights, mask, x_arr):
        v, w = vals[mask], weights[mask]
        return np.array([np.sum(w[pass_fn(v, x)]) for x in x_arr])

    def cumulative_stat_err(vals, weights, mask, x_arr):
        v, w = vals[mask], weights[mask]
        return np.array([np.sqrt(np.sum(w[pass_fn(v, x)]**2)) for x in x_arr])

    # Central estimates (nominal weights)
    rcr_cum_nom = cumulative(rcr_all_vals, sim_reflected['weights'], rcr_mask_nom, x_scan)
    bl_cum_nom = cumulative(bl_all_vals, sim_direct['weights'], bl_mask_nom, x_scan)

    # Cross-cut band
    rcr_cum_lo = cumulative(rcr_all_vals, sim_reflected['weights'], rcr_mask_lo, x_scan)
    rcr_cum_hi = cumulative(rcr_all_vals, sim_reflected['weights'], rcr_mask_hi, x_scan)
    bl_cum_lo = cumulative(bl_all_vals, sim_direct['weights'], bl_mask_lo, x_scan)
    bl_cum_hi = cumulative(bl_all_vals, sim_direct['weights'], bl_mask_hi, x_scan)

    # R-sweep band (for RCR)
    rcr_cum_rhi = cumulative(rcr_all_vals, sim_reflected_high['weights'], rcr_mask_nom, x_scan)
    rcr_cum_rlo = cumulative(rcr_all_vals, sim_reflected_low['weights'], rcr_mask_nom, x_scan)

    # Statistical errors
    rcr_stat = cumulative_stat_err(rcr_all_vals, sim_reflected['weights'], rcr_mask_nom, x_scan)
    bl_stat = cumulative_stat_err(bl_all_vals, sim_direct['weights'], bl_mask_nom, x_scan)

    # Combine errors for RCR: R-sweep + cross-cut + stat
    rcr_all_lo = np.minimum(rcr_cum_lo, np.minimum(rcr_cum_hi, rcr_cum_rlo)) - rcr_stat
    rcr_all_hi = np.maximum(rcr_cum_lo, np.maximum(rcr_cum_hi, rcr_cum_rhi)) + rcr_stat

    # BL: cross-cut + stat
    bl_all_lo = np.minimum(bl_cum_lo, bl_cum_hi) - bl_stat
    bl_all_hi = np.maximum(bl_cum_lo, bl_cum_hi) + bl_stat

    both_cum_nom = rcr_cum_nom + bl_cum_nom
    both_all_lo = rcr_all_lo + bl_all_lo
    both_all_hi = rcr_all_hi + bl_all_hi

    # Data cumulative: nominal, low, high cross-cuts for bar style
    data_vals = get_param_values(param_name, data_dict)
    data_cross_mask_nom = apply_cuts(data_dict, cuts_nom, cut_type='rcr') & ~excluded_events_mask
    data_cross_mask_lo = apply_cuts(data_dict, cuts_lo, cut_type='rcr') & ~excluded_events_mask
    data_cross_mask_hi = apply_cuts(data_dict, cuts_hi, cut_type='rcr') & ~excluded_events_mask

    data_cum_nom = np.zeros(len(x_scan), dtype=int)
    data_cum_lo = np.zeros(len(x_scan), dtype=int)
    data_cum_hi = np.zeros(len(x_scan), dtype=int)
    for ix, x in enumerate(x_scan):
        for dm, dc in [(data_cross_mask_nom, data_cum_nom),
                       (data_cross_mask_lo, data_cum_lo),
                       (data_cross_mask_hi, data_cum_hi)]:
            pidx = np.where(dm & pass_fn(data_vals, x))[0]
            if len(pidx) > 0:
                umask = filter_unique_events_by_day(
                    data_dict['Time'][pidx], data_station_ids[pidx])
                dc[ix] = int(np.sum(umask))

    # Identified BL cumulative
    if identified_bl_data is not None and len(identified_bl_data['snr']) > 0:
        ibl_cross_mask = apply_cuts(identified_bl_data, cuts_nom, cut_type='rcr')
        ibl_vals = get_param_values(param_name, identified_bl_data)[ibl_cross_mask]
        ibl_cum = np.array([int(np.sum(pass_fn(ibl_vals, x))) for x in x_scan])
    else:
        ibl_cum = np.zeros_like(x_scan, dtype=int)

    # Identified RCR: just plot as individual points at y=1
    if identified_rcr_data is not None and len(identified_rcr_data['snr']) > 0:
        ircr_cross_mask = apply_cuts(identified_rcr_data, cuts_nom, cut_type='rcr')
        ircr_x_vals = get_param_values(param_name, identified_rcr_data)[ircr_cross_mask]
    else:
        ircr_x_vals = np.array([])

    # Plot
    cross_label = PARAM_TITLES.get(cross_cut_param, cross_cut_param)

    fig, ax = plt.subplots(figsize=(10, 7))

    sim_handles = []
    ax.fill_between(x_scan, rcr_all_lo, rcr_all_hi, color='green', alpha=0.15)
    h, = ax.plot(x_scan, rcr_cum_nom, 'g-', linewidth=2, label='RCR Sim')
    sim_handles.append(h)
    ax.fill_between(x_scan, bl_all_lo, bl_all_hi, color='orange', alpha=0.15)
    h, = ax.plot(x_scan, bl_cum_nom, color='orange', linewidth=2, label='BL Sim')
    sim_handles.append(h)
    ax.fill_between(x_scan, both_all_lo, both_all_hi, color='purple', alpha=0.1)
    h, = ax.plot(x_scan, both_cum_nom, 'purple', linewidth=2, linestyle='--', label='Both Sim')
    sim_handles.append(h)
    cut_line = ax.axvline(x=nominal_value, color='red', linestyle='--', linewidth=1.5,
                           label=f'Cut at {nominal_value}', alpha=0.7)
    sim_handles.append(cut_line)

    # Data as bar-style: vertical whisker (min/max) with box at center
    data_handles = []
    step = max(1, len(x_scan) // 40)
    idx_pts = np.arange(0, len(x_scan), step)
    data_min = np.minimum(data_cum_lo[idx_pts], data_cum_hi[idx_pts]).astype(float)
    data_max = np.maximum(data_cum_lo[idx_pts], data_cum_hi[idx_pts]).astype(float)
    data_center = data_cum_nom[idx_pts].astype(float)
    # Whisker from min to max, marker at center
    err_lo = data_center - data_min
    err_hi = data_max - data_center
    h = ax.errorbar(x_scan[idx_pts], data_center,
                     yerr=[err_lo, err_hi],
                     fmt='ks', markersize=5, capsize=3, elinewidth=1.2, capthick=1.2,
                     label='Data Events', zorder=6)
    data_handles.append(h)

    if np.any(ibl_cum > 0):
        h, = ax.plot(x_scan, ibl_cum, 'c-', linewidth=1.5, label='Identified BL')
        data_handles.append(h)

    # Identified RCR as individual points at y=1
    if len(ircr_x_vals) > 0:
        h, = ax.plot(ircr_x_vals, np.ones_like(ircr_x_vals), 'r^', markersize=8,
                     markeredgecolor='white', markeredgewidth=0.8,
                     label='Identified RCR', zorder=7)
        data_handles.append(h)

    xlabel = PARAM_LABELS.get(param_name, param_name)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Events Passing Cut', fontsize=FONTSIZE_LABEL)
    ax.set_title(f'Cumulative: {PARAM_TITLES.get(param_name, param_name)} '
                 f'(band: {cross_label} {cross_lo}–{cross_hi})', fontsize=FONTSIZE_TITLE)
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    ax.grid(True, alpha=0.3)

    if yscale == 'log':
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.1)
    if x_range is not None:
        ax.set_xlim(x_range)

    add_split_legend(ax, sim_handles, data_handles)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    ic(f"Saved cut-interaction plot: {output_path}")

    # Return cumulatives at nominal cut value for text output
    nom_idx = np.argmin(np.abs(x_scan - nominal_value))
    return {
        'rcr_nom': rcr_cum_nom[nom_idx], 'rcr_lo': rcr_all_lo[nom_idx], 'rcr_hi': rcr_all_hi[nom_idx],
        'bl_nom': bl_cum_nom[nom_idx], 'bl_lo': bl_all_lo[nom_idx], 'bl_hi': bl_all_hi[nom_idx],
        'both_nom': both_cum_nom[nom_idx], 'data_nom': data_cum_nom[nom_idx],
        'data_lo': data_cum_lo[nom_idx], 'data_hi': data_cum_hi[nom_idx],
        'rcr_stat': rcr_stat[nom_idx], 'bl_stat': bl_stat[nom_idx],
    }


# ============================================================================
# Plot 5: Scale Factor Fit (matching cut interaction style)
# ============================================================================

def compute_scale_factor(param_name, nominal_value, nominal_cuts,
                         cross_cut_param, cross_cut_range,
                         sim_reflected, sim_reflected_high, sim_reflected_low,
                         data_dict, data_station_ids,
                         excluded_events_mask, x_range=None):
    """
    Compute the scale factor that best fits RCR sim to data at/above the cut line.
    Returns scale_factor, scale_error, and the cumulative arrays.
    """
    # Build cross-cut configs
    cross_lo, cross_hi = cross_cut_range

    def _make_cross_cuts(cross_val):
        cc = dict(nominal_cuts)
        if param_name == 'chi_rcr_flat':
            cc['chi_rcr_line_chi'] = np.zeros_like(nominal_cuts['chi_rcr_line_chi'])
        elif param_name == 'chi_diff_threshold':
            cc['chi_diff_threshold'] = -999
        if cross_cut_param == 'chi_rcr_flat':
            cc['chi_rcr_line_chi'] = np.full_like(nominal_cuts['chi_rcr_line_chi'], cross_val)
        elif cross_cut_param == 'chi_diff_threshold':
            cc['chi_diff_threshold'] = cross_val
        return cc

    if cross_cut_param == 'chi_rcr_flat':
        cross_nominal = nominal_cuts['chi_rcr_line_chi'][0]
    elif cross_cut_param == 'chi_diff_threshold':
        cross_nominal = nominal_cuts['chi_diff_threshold']
    else:
        cross_nominal = (cross_lo + cross_hi) / 2

    cuts_nom = _make_cross_cuts(cross_nominal)

    rcr_mask = apply_cuts(sim_reflected, cuts_nom, cut_type='rcr')
    rcr_all_vals = get_param_values(param_name, sim_reflected)

    if x_range is not None:
        x_scan = np.linspace(x_range[0], x_range[1], 200)
    else:
        v = rcr_all_vals[rcr_mask]
        if len(v) == 0:
            return 1.0, 0.0, None
        x_scan = np.linspace(np.nanmin(v), np.nanmax(v), 200)

    if param_name in ('chi_rcr_flat', 'chi_diff_threshold'):
        pass_fn = lambda vals, x: vals > x
        fit_mask = x_scan >= nominal_value
    elif param_name in ('snr_max', 'chi_diff_max'):
        pass_fn = lambda vals, x: vals < x
        fit_mask = x_scan <= nominal_value
    else:
        raise ValueError(f"Unknown param_name: {param_name}")

    def cumulative(vals, weights, mask, x_arr):
        v, w = vals[mask], weights[mask]
        return np.array([np.sum(w[pass_fn(v, x)]) for x in x_arr])

    rcr_cum = cumulative(rcr_all_vals, sim_reflected['weights'], rcr_mask, x_scan)
    rcr_cum_hi = cumulative(rcr_all_vals, sim_reflected_high['weights'], rcr_mask, x_scan)
    rcr_cum_lo = cumulative(rcr_all_vals, sim_reflected_low['weights'], rcr_mask, x_scan)

    # Data cumulative (with all cross cuts applied)
    data_vals = get_param_values(param_name, data_dict)
    data_cross_mask = apply_cuts(data_dict, cuts_nom, cut_type='rcr') & ~excluded_events_mask
    data_cum = np.zeros(len(x_scan), dtype=int)
    for ix, x in enumerate(x_scan):
        pidx = np.where(data_cross_mask & pass_fn(data_vals, x))[0]
        if len(pidx) > 0:
            umask = filter_unique_events_by_day(
                data_dict['Time'][pidx], data_station_ids[pidx])
            data_cum[ix] = int(np.sum(umask))

    # Fit scale factor at/above cut line
    fit_rcr = rcr_cum[fit_mask]
    fit_data = data_cum[fit_mask].astype(float)

    if np.sum(fit_rcr**2) > 0:
        scale_factor = np.sum(fit_data * fit_rcr) / np.sum(fit_rcr**2)
    else:
        scale_factor = 1.0

    # Scale factor error from R-sweep
    fit_rcr_hi = rcr_cum_hi[fit_mask]
    fit_rcr_lo = rcr_cum_lo[fit_mask]
    if np.sum(fit_rcr_hi**2) > 0:
        scale_hi = np.sum(fit_data * fit_rcr_hi) / np.sum(fit_rcr_hi**2)
    else:
        scale_hi = scale_factor
    if np.sum(fit_rcr_lo**2) > 0:
        scale_lo = np.sum(fit_data * fit_rcr_lo) / np.sum(fit_rcr_lo**2)
    else:
        scale_lo = scale_factor

    scale_error = abs(scale_hi - scale_lo) / 2

    ic(f"Scale factor fit for {param_name}: {scale_factor:.4f} +/- {scale_error:.4f}")

    return scale_factor, scale_error, {
        'x_scan': x_scan, 'rcr_cum': rcr_cum, 'rcr_cum_hi': rcr_cum_hi,
        'rcr_cum_lo': rcr_cum_lo, 'data_cum': data_cum, 'fit_mask': fit_mask,
    }


def plot_scale_factor(param_name, nominal_value, nominal_cuts,
                      cross_cut_param, cross_cut_range,
                      sim_direct, sim_reflected,
                      sim_direct_high, sim_reflected_high,
                      sim_direct_low, sim_reflected_low,
                      data_dict, data_station_ids,
                      identified_bl_data, identified_rcr_data,
                      excluded_events_mask, output_folder,
                      x_range=None, yscale='linear'):
    """
    Two versions:
    1. Scale factor only (no sims, just data + corrected RCR sim)
    2. Full plot (sims + corrected RCR sim overlay)
    Both match the cut-interaction style.
    """
    scale_factor, scale_error, arrays = compute_scale_factor(
        param_name, nominal_value, nominal_cuts,
        cross_cut_param, cross_cut_range,
        sim_reflected, sim_reflected_high, sim_reflected_low,
        data_dict, data_station_ids, excluded_events_mask,
        x_range=x_range)

    if arrays is None:
        return scale_factor, scale_error

    x_scan = arrays['x_scan']
    rcr_cum = arrays['rcr_cum']
    rcr_cum_hi = arrays['rcr_cum_hi']
    rcr_cum_lo = arrays['rcr_cum_lo']
    data_cum = arrays['data_cum']

    # Identified RCR x-values
    if identified_rcr_data is not None and len(identified_rcr_data['snr']) > 0:
        cross_lo, cross_hi = cross_cut_range
        if cross_cut_param == 'chi_rcr_flat':
            cross_nominal = nominal_cuts['chi_rcr_line_chi'][0]
        elif cross_cut_param == 'chi_diff_threshold':
            cross_nominal = nominal_cuts['chi_diff_threshold']
        else:
            cross_nominal = (cross_lo + cross_hi) / 2

        def _mc(cv):
            cc = dict(nominal_cuts)
            if param_name == 'chi_rcr_flat':
                cc['chi_rcr_line_chi'] = np.zeros_like(nominal_cuts['chi_rcr_line_chi'])
            elif param_name == 'chi_diff_threshold':
                cc['chi_diff_threshold'] = -999
            if cross_cut_param == 'chi_rcr_flat':
                cc['chi_rcr_line_chi'] = np.full_like(nominal_cuts['chi_rcr_line_chi'], cv)
            elif cross_cut_param == 'chi_diff_threshold':
                cc['chi_diff_threshold'] = cv
            return cc

        cuts_nom = _mc(cross_nominal)
        ircr_cross_mask = apply_cuts(identified_rcr_data, cuts_nom, cut_type='rcr')
        ircr_x_vals = get_param_values(param_name, identified_rcr_data)[ircr_cross_mask]
    else:
        ircr_x_vals = np.array([])

    cross_label = PARAM_TITLES.get(cross_cut_param, cross_cut_param)

    # Version 1: Scale factor only
    fig, ax = plt.subplots(figsize=(10, 7))

    sim_handles = []
    corrected_cum = rcr_cum * scale_factor
    # Error band: multiply R-sweep high/low cumulatives by scale factor range
    corrected_lo = np.minimum(rcr_cum_lo * (scale_factor - scale_error),
                              rcr_cum_hi * (scale_factor - scale_error))
    corrected_hi = np.maximum(rcr_cum_lo * (scale_factor + scale_error),
                              rcr_cum_hi * (scale_factor + scale_error))
    ax.fill_between(x_scan, corrected_lo, corrected_hi, color='gray', alpha=0.15)
    h, = ax.plot(x_scan, corrected_cum, '--', color='gray', linewidth=2,
                 label=f'Corrected RCR Sim (x{scale_factor:.2f})')
    sim_handles.append(h)
    cut_line = ax.axvline(x=nominal_value, color='red', linestyle='--', linewidth=1.5,
                           label=f'Cut at {nominal_value}', alpha=0.7)
    sim_handles.append(cut_line)

    data_handles = []
    step = max(1, len(x_scan) // 40)
    idx_pts = np.arange(0, len(x_scan), step)
    data_err = np.sqrt(np.maximum(data_cum[idx_pts].astype(float), 0))
    h = ax.errorbar(x_scan[idx_pts], data_cum[idx_pts], yerr=data_err,
                     fmt='ks', markersize=5, capsize=3, elinewidth=1.2, capthick=1.2,
                     label='Data Events', zorder=6)
    data_handles.append(h)

    if len(ircr_x_vals) > 0:
        h, = ax.plot(ircr_x_vals, np.ones_like(ircr_x_vals), 'r^', markersize=8,
                     markeredgecolor='white', markeredgewidth=0.8,
                     label='Identified RCR', zorder=7)
        data_handles.append(h)

    ax.set_xlabel(PARAM_LABELS.get(param_name, param_name), fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Events Passing Cut', fontsize=FONTSIZE_LABEL)
    ax.set_title(f'Scale Factor: {PARAM_TITLES.get(param_name, param_name)} '
                 f'(factor = {scale_factor:.3f} +/- {scale_error:.3f})', fontsize=FONTSIZE_TITLE)
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    ax.grid(True, alpha=0.3)
    if yscale == 'log':
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.1)
    if x_range is not None:
        ax.set_xlim(x_range)
    add_split_legend(ax, sim_handles, data_handles)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'scale_factor_{param_name}.png'), dpi=150)
    plt.close(fig)

    # Version 2: Full plot with sims + corrected
    ci_results = plot_cut_interaction(
        param_name, nominal_cuts, nominal_value,
        cross_cut_param, cross_cut_range,
        sim_direct, sim_reflected,
        sim_direct_high, sim_reflected_high,
        sim_direct_low, sim_reflected_low,
        data_dict, data_station_ids,
        identified_bl_data, identified_rcr_data,
        excluded_events_mask,
        os.path.join(output_folder, f'scale_factor_{param_name}_with_sims_TEMP.png'),
        x_range=x_range, yscale=yscale)

    # Re-do the full plot but add the corrected line
    # We need to rebuild this. Call the full version then overlay.
    # Actually, let's just build the combined plot directly.
    _plot_scale_factor_with_sims(
        param_name, nominal_value, nominal_cuts,
        cross_cut_param, cross_cut_range,
        sim_direct, sim_reflected,
        sim_direct_high, sim_reflected_high,
        sim_direct_low, sim_reflected_low,
        data_dict, data_station_ids,
        identified_bl_data, identified_rcr_data,
        excluded_events_mask,
        scale_factor, scale_error,
        os.path.join(output_folder, f'scale_factor_{param_name}_with_sims.png'),
        x_range=x_range, yscale=yscale)

    # Clean up temp file
    temp_path = os.path.join(output_folder, f'scale_factor_{param_name}_with_sims_TEMP.png')
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return scale_factor, scale_error


def _plot_scale_factor_with_sims(param_name, nominal_value, nominal_cuts,
                                  cross_cut_param, cross_cut_range,
                                  sim_direct, sim_reflected,
                                  sim_direct_high, sim_reflected_high,
                                  sim_direct_low, sim_reflected_low,
                                  data_dict, data_station_ids,
                                  identified_bl_data, identified_rcr_data,
                                  excluded_events_mask,
                                  scale_factor, scale_error,
                                  output_path, x_range=None, yscale='linear'):
    """Full cut-interaction plot with corrected RCR sim added."""
    # Same setup as plot_cut_interaction
    cross_lo, cross_hi = cross_cut_range

    def _make_cross_cuts(cross_val):
        cc = dict(nominal_cuts)
        if param_name == 'chi_rcr_flat':
            cc['chi_rcr_line_chi'] = np.zeros_like(nominal_cuts['chi_rcr_line_chi'])
        elif param_name == 'chi_diff_threshold':
            cc['chi_diff_threshold'] = -999
        if cross_cut_param == 'chi_rcr_flat':
            cc['chi_rcr_line_chi'] = np.full_like(nominal_cuts['chi_rcr_line_chi'], cross_val)
        elif cross_cut_param == 'chi_diff_threshold':
            cc['chi_diff_threshold'] = cross_val
        return cc

    if cross_cut_param == 'chi_rcr_flat':
        cross_nominal = nominal_cuts['chi_rcr_line_chi'][0]
    elif cross_cut_param == 'chi_diff_threshold':
        cross_nominal = nominal_cuts['chi_diff_threshold']
    else:
        cross_nominal = (cross_lo + cross_hi) / 2

    cuts_nom = _make_cross_cuts(cross_nominal)
    cuts_lo = _make_cross_cuts(cross_lo)
    cuts_hi = _make_cross_cuts(cross_hi)

    rcr_mask_nom = apply_cuts(sim_reflected, cuts_nom, cut_type='rcr')
    rcr_mask_lo = apply_cuts(sim_reflected, cuts_lo, cut_type='rcr')
    rcr_mask_hi = apply_cuts(sim_reflected, cuts_hi, cut_type='rcr')
    bl_mask_nom = apply_cuts(sim_direct, cuts_nom, cut_type='rcr')
    bl_mask_lo = apply_cuts(sim_direct, cuts_lo, cut_type='rcr')
    bl_mask_hi = apply_cuts(sim_direct, cuts_hi, cut_type='rcr')

    rcr_all_vals = get_param_values(param_name, sim_reflected)
    bl_all_vals = get_param_values(param_name, sim_direct)

    if x_range is not None:
        x_scan = np.linspace(x_range[0], x_range[1], 200)
    else:
        all_vals = np.concatenate([rcr_all_vals[rcr_mask_nom], bl_all_vals[bl_mask_nom]])
        x_scan = np.linspace(np.nanmin(all_vals), np.nanmax(all_vals), 200)

    if param_name in ('chi_rcr_flat', 'chi_diff_threshold'):
        pass_fn = lambda vals, x: vals > x
    else:
        pass_fn = lambda vals, x: vals < x

    def cumulative(vals, weights, mask, x_arr):
        v, w = vals[mask], weights[mask]
        return np.array([np.sum(w[pass_fn(v, x)]) for x in x_arr])

    def cumulative_stat_err(vals, weights, mask, x_arr):
        v, w = vals[mask], weights[mask]
        return np.array([np.sqrt(np.sum(w[pass_fn(v, x)]**2)) for x in x_arr])

    rcr_cum_nom = cumulative(rcr_all_vals, sim_reflected['weights'], rcr_mask_nom, x_scan)
    bl_cum_nom = cumulative(bl_all_vals, sim_direct['weights'], bl_mask_nom, x_scan)

    rcr_cum_lo = cumulative(rcr_all_vals, sim_reflected['weights'], rcr_mask_lo, x_scan)
    rcr_cum_hi = cumulative(rcr_all_vals, sim_reflected['weights'], rcr_mask_hi, x_scan)
    bl_cum_lo = cumulative(bl_all_vals, sim_direct['weights'], bl_mask_lo, x_scan)
    bl_cum_hi = cumulative(bl_all_vals, sim_direct['weights'], bl_mask_hi, x_scan)

    rcr_cum_rhi = cumulative(rcr_all_vals, sim_reflected_high['weights'], rcr_mask_nom, x_scan)
    rcr_cum_rlo = cumulative(rcr_all_vals, sim_reflected_low['weights'], rcr_mask_nom, x_scan)

    rcr_stat = cumulative_stat_err(rcr_all_vals, sim_reflected['weights'], rcr_mask_nom, x_scan)
    bl_stat = cumulative_stat_err(bl_all_vals, sim_direct['weights'], bl_mask_nom, x_scan)

    rcr_all_lo = np.minimum(rcr_cum_lo, np.minimum(rcr_cum_hi, rcr_cum_rlo)) - rcr_stat
    rcr_all_hi = np.maximum(rcr_cum_lo, np.maximum(rcr_cum_hi, rcr_cum_rhi)) + rcr_stat
    bl_all_lo = np.minimum(bl_cum_lo, bl_cum_hi) - bl_stat
    bl_all_hi = np.maximum(bl_cum_lo, bl_cum_hi) + bl_stat
    both_cum_nom = rcr_cum_nom + bl_cum_nom
    both_all_lo = rcr_all_lo + bl_all_lo
    both_all_hi = rcr_all_hi + bl_all_hi

    # Data
    data_vals = get_param_values(param_name, data_dict)
    data_cross_mask_nom = apply_cuts(data_dict, cuts_nom, cut_type='rcr') & ~excluded_events_mask
    data_cross_mask_lo = apply_cuts(data_dict, cuts_lo, cut_type='rcr') & ~excluded_events_mask
    data_cross_mask_hi = apply_cuts(data_dict, cuts_hi, cut_type='rcr') & ~excluded_events_mask

    data_cum_nom = np.zeros(len(x_scan), dtype=int)
    data_cum_lo = np.zeros(len(x_scan), dtype=int)
    data_cum_hi = np.zeros(len(x_scan), dtype=int)
    for ix, x in enumerate(x_scan):
        for dm, dc in [(data_cross_mask_nom, data_cum_nom),
                       (data_cross_mask_lo, data_cum_lo),
                       (data_cross_mask_hi, data_cum_hi)]:
            pidx = np.where(dm & pass_fn(data_vals, x))[0]
            if len(pidx) > 0:
                umask = filter_unique_events_by_day(
                    data_dict['Time'][pidx], data_station_ids[pidx])
                dc[ix] = int(np.sum(umask))

    # Identified
    if identified_bl_data is not None and len(identified_bl_data['snr']) > 0:
        ibl_cross_mask = apply_cuts(identified_bl_data, cuts_nom, cut_type='rcr')
        ibl_vals = get_param_values(param_name, identified_bl_data)[ibl_cross_mask]
        ibl_cum = np.array([int(np.sum(pass_fn(ibl_vals, x))) for x in x_scan])
    else:
        ibl_cum = np.zeros_like(x_scan, dtype=int)

    if identified_rcr_data is not None and len(identified_rcr_data['snr']) > 0:
        ircr_cross_mask = apply_cuts(identified_rcr_data, cuts_nom, cut_type='rcr')
        ircr_x_vals = get_param_values(param_name, identified_rcr_data)[ircr_cross_mask]
    else:
        ircr_x_vals = np.array([])

    cross_label = PARAM_TITLES.get(cross_cut_param, cross_cut_param)

    fig, ax = plt.subplots(figsize=(10, 7))

    sim_handles = []
    ax.fill_between(x_scan, rcr_all_lo, rcr_all_hi, color='green', alpha=0.15)
    h, = ax.plot(x_scan, rcr_cum_nom, 'g-', linewidth=2, label='RCR Sim')
    sim_handles.append(h)
    ax.fill_between(x_scan, bl_all_lo, bl_all_hi, color='orange', alpha=0.15)
    h, = ax.plot(x_scan, bl_cum_nom, color='orange', linewidth=2, label='BL Sim')
    sim_handles.append(h)
    ax.fill_between(x_scan, both_all_lo, both_all_hi, color='purple', alpha=0.1)
    h, = ax.plot(x_scan, both_cum_nom, 'purple', linewidth=2, linestyle='--', label='Both Sim')
    sim_handles.append(h)

    # Corrected RCR Sim with error band from R-sweep * scale factor range
    corrected_cum = rcr_cum_nom * scale_factor
    corrected_band_lo = np.minimum(rcr_cum_rlo * (scale_factor - scale_error),
                                    rcr_cum_rhi * (scale_factor - scale_error))
    corrected_band_hi = np.maximum(rcr_cum_rlo * (scale_factor + scale_error),
                                    rcr_cum_rhi * (scale_factor + scale_error))
    ax.fill_between(x_scan, corrected_band_lo, corrected_band_hi, color='gray', alpha=0.15)
    h, = ax.plot(x_scan, corrected_cum, '--', color='gray', linewidth=2.5,
                 label=f'Corrected RCR (x{scale_factor:.2f})')
    sim_handles.append(h)

    cut_line = ax.axvline(x=nominal_value, color='red', linestyle='--', linewidth=1.5,
                           label=f'Cut at {nominal_value}', alpha=0.7)
    sim_handles.append(cut_line)

    data_handles = []
    step = max(1, len(x_scan) // 40)
    idx_pts = np.arange(0, len(x_scan), step)
    data_min = np.minimum(data_cum_lo[idx_pts], data_cum_hi[idx_pts]).astype(float)
    data_max = np.maximum(data_cum_lo[idx_pts], data_cum_hi[idx_pts]).astype(float)
    data_center = data_cum_nom[idx_pts].astype(float)
    err_lo = data_center - data_min
    err_hi = data_max - data_center
    h = ax.errorbar(x_scan[idx_pts], data_center,
                     yerr=[err_lo, err_hi],
                     fmt='ks', markersize=5, capsize=3, elinewidth=1.2, capthick=1.2,
                     label='Data Events', zorder=6)
    data_handles.append(h)

    if np.any(ibl_cum > 0):
        h, = ax.plot(x_scan, ibl_cum, 'c-', linewidth=1.5, label='Identified BL')
        data_handles.append(h)

    if len(ircr_x_vals) > 0:
        h, = ax.plot(ircr_x_vals, np.ones_like(ircr_x_vals), 'r^', markersize=8,
                     markeredgecolor='white', markeredgewidth=0.8,
                     label='Identified RCR', zorder=7)
        data_handles.append(h)

    ax.set_xlabel(PARAM_LABELS.get(param_name, param_name), fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Events Passing Cut', fontsize=FONTSIZE_LABEL)
    ax.set_title(f'Cumulative + Corrected RCR: {PARAM_TITLES.get(param_name, param_name)}',
                 fontsize=FONTSIZE_TITLE)
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    ax.grid(True, alpha=0.3)
    if yscale == 'log':
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.1)
    if x_range is not None:
        ax.set_xlim(x_range)

    add_split_legend(ax, sim_handles, data_handles)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    ic(f"Saved scale factor with sims: {output_path}")


# ============================================================================
# Plot 6: 2D Chi-Chi Histogram (Coincidence Events)
# ============================================================================

def plot_chi_chi_coinc_2d(coinc_bl_data, coinc_rcr_data,
                           coinc_bl_rate, coinc_rcr_rate,
                           sim_direct, sim_reflected,
                           identified_bl_data, identified_rcr_data,
                           nominal_cuts, output_path, n_bins=50,
                           use_comp_log=False):
    """
    Two subplots of chi-BL vs chi-RCR 2D histograms.
    Left: Coinc BL Sim (histogram), overlaid with Identified BL and RCR.
          Red dashed diagonal (0,0)->(1,1).
    Right: Coinc RCR Sim (histogram under data), overlaid with Identified events.
           Dashed lines showing RCR cut in RCR region (above diagonal) and
           BL cut in BL region (below diagonal).
    """
    import matplotlib.colors as mcolors

    chi_rcr_cut = nominal_cuts['chi_rcr_line_chi'][0]  # flat cut value

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left subplot: BL sim background + coinc BL events + identified ---
    # Use BL simulation as background
    bl_chi2016 = sim_direct['Chi2016']
    bl_chircr = sim_direct['ChiRCR']
    bl_weights = sim_direct['weights']

    h_bl = ax_left.hist2d(bl_chi2016, bl_chircr, bins=n_bins,
                           range=[[0, 1], [0, 1]], weights=bl_weights,
                           norm=mcolors.LogNorm(), cmap='Oranges', zorder=1)
    plt.colorbar(h_bl[3], ax=ax_left, label='BL Sim (evts/yr)')

    # Diagonal
    ax_left.plot([0, 1], [0, 1], 'r--', linewidth=1.5, alpha=0.7, label='Diagonal', zorder=2)

    # Cut lines (same on both subplots)
    # RCR region (above diagonal): horizontal line at chi-RCR = cut
    ax_left.plot([0, chi_rcr_cut], [chi_rcr_cut, chi_rcr_cut], 'b--', linewidth=2,
                  alpha=0.8, zorder=2)
    # BL region (below diagonal): vertical line at chi-BL = cut
    ax_left.plot([chi_rcr_cut, chi_rcr_cut], [0, chi_rcr_cut], 'b--', linewidth=2,
                  alpha=0.8, label=f'Cut ($\\chi$>{chi_rcr_cut})', zorder=2)

    # Overlay Coinc BL events
    if len(coinc_bl_data['snr']) > 0:
        ax_left.scatter(coinc_bl_data['Chi2016'], coinc_bl_data['ChiRCR'],
                         marker='s', s=40, c='cyan', edgecolors='black', linewidths=0.5,
                         label=f'Coinc BL ({len(coinc_bl_data["snr"])} evts)', zorder=4)

    # Overlay Coinc RCR events
    if len(coinc_rcr_data['snr']) > 0:
        ax_left.scatter(coinc_rcr_data['Chi2016'], coinc_rcr_data['ChiRCR'],
                         marker='^', s=60, c='red', edgecolors='black', linewidths=0.5,
                         label=f'Coinc RCR ({len(coinc_rcr_data["snr"])} evts)', zorder=5)

    # Overlay Identified BL
    if identified_bl_data is not None and len(identified_bl_data['snr']) > 0:
        ax_left.scatter(identified_bl_data['Chi2016'], identified_bl_data['ChiRCR'],
                         marker='s', s=30, c='cyan', edgecolors='white', linewidths=0.8,
                         alpha=0.8, label='Identified BL', zorder=3)

    # Overlay Identified RCR
    if identified_rcr_data is not None and len(identified_rcr_data['snr']) > 0:
        ax_left.scatter(identified_rcr_data['Chi2016'], identified_rcr_data['ChiRCR'],
                         marker='^', s=50, c='red', edgecolors='white', linewidths=0.8,
                         alpha=0.8, label='Identified RCR', zorder=3)

    ax_left.set_xlabel(r'$\chi_{BL}$', fontsize=FONTSIZE_LABEL)
    ax_left.set_ylabel(r'$\chi_{\mathrm{RCR}}$', fontsize=FONTSIZE_LABEL)
    ax_left.set_title('Coincidence: BL Sim Background', fontsize=FONTSIZE_TITLE)
    ax_left.legend(fontsize=8, loc='lower right')
    ax_left.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    if use_comp_log:
        set_comp_log_scale(ax_left, axis='both')
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)

    # --- Right subplot: RCR sim background + identified events + cut lines ---
    rcr_chi2016 = sim_reflected['Chi2016']
    rcr_chircr = sim_reflected['ChiRCR']
    rcr_weights = sim_reflected['weights']

    h_rcr = ax_right.hist2d(rcr_chi2016, rcr_chircr, bins=n_bins,
                              range=[[0, 1], [0, 1]], weights=rcr_weights,
                              norm=mcolors.LogNorm(), cmap='Greens', zorder=1)
    plt.colorbar(h_rcr[3], ax=ax_right, label='RCR Sim (evts/yr)')

    # Diagonal
    ax_right.plot([0, 1], [0, 1], 'r--', linewidth=1.5, alpha=0.7, label='Diagonal', zorder=2)

    # Cut lines: RCR cut in RCR region (above diagonal), BL cut in BL region (below diagonal)
    # RCR region: chi-RCR > chi-BL, so flat chi-RCR cut at chi_rcr_cut
    ax_right.plot([0, chi_rcr_cut], [chi_rcr_cut, chi_rcr_cut], 'b--', linewidth=2,
                   alpha=0.8, label=f'RCR cut ($\\chi_{{RCR}}$>{chi_rcr_cut})', zorder=3)
    # BL region: chi-BL > chi-RCR, so flat chi-BL cut at chi_rcr_cut (same value)
    ax_right.plot([chi_rcr_cut, chi_rcr_cut], [0, chi_rcr_cut], 'b--', linewidth=2,
                   alpha=0.8, label=f'BL cut ($\\chi_{{BL}}$>{chi_rcr_cut})', zorder=3)

    # Overlay Coinc BL events
    if len(coinc_bl_data['snr']) > 0:
        ax_right.scatter(coinc_bl_data['Chi2016'], coinc_bl_data['ChiRCR'],
                          marker='s', s=40, c='cyan', edgecolors='black', linewidths=0.5,
                          label='Coinc BL', zorder=5)

    # Overlay Coinc RCR events
    if len(coinc_rcr_data['snr']) > 0:
        ax_right.scatter(coinc_rcr_data['Chi2016'], coinc_rcr_data['ChiRCR'],
                          marker='^', s=60, c='red', edgecolors='black', linewidths=0.5,
                          label='Coinc RCR', zorder=5)

    # Overlay Identified events
    if identified_bl_data is not None and len(identified_bl_data['snr']) > 0:
        ax_right.scatter(identified_bl_data['Chi2016'], identified_bl_data['ChiRCR'],
                          marker='s', s=30, c='cyan', edgecolors='white', linewidths=0.8,
                          alpha=0.8, label='Identified BL', zorder=4)
    if identified_rcr_data is not None and len(identified_rcr_data['snr']) > 0:
        ax_right.scatter(identified_rcr_data['Chi2016'], identified_rcr_data['ChiRCR'],
                          marker='^', s=50, c='red', edgecolors='white', linewidths=0.8,
                          alpha=0.8, label='Identified RCR', zorder=4)

    ax_right.set_xlabel(r'$\chi_{BL}$', fontsize=FONTSIZE_LABEL)
    ax_right.set_ylabel(r'$\chi_{\mathrm{RCR}}$', fontsize=FONTSIZE_LABEL)
    ax_right.set_title('Coincidence: RCR Sim Background + Cuts', fontsize=FONTSIZE_TITLE)
    ax_right.legend(fontsize=8, loc='lower right')
    ax_right.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    if use_comp_log:
        set_comp_log_scale(ax_right, axis='both')
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    ic(f"Saved chi-chi coinc 2D plot: {output_path}")


# ============================================================================
# Plot 7: Chi-Chi Cut Space (Station Data with pass/fail markers)
# ============================================================================

def plot_chi_chi_cut_space(data_dict, excluded_events_mask, nominal_cuts,
                            identified_bl_data, identified_rcr_data,
                            output_path, use_comp_log=False):
    """
    Chi-BL vs Chi-RCR scatter of station data.
    Gray dots: data not passing any cut.
    Purple stars: data passing RCR cuts.
    Green circles: data passing BL cuts (but failing RCR).
    Identified events layered above passing, but below non-passing data is gray.
    Identified RCR: red triangles. Identified BL: cyan squares.
    """
    chi_bl = data_dict['Chi2016']
    chi_rcr = data_dict['ChiRCR']
    not_excl = ~excluded_events_mask

    # Apply cuts
    rcr_pass = apply_cuts(data_dict, nominal_cuts, cut_type='rcr') & not_excl
    bl_pass = apply_cuts(data_dict, nominal_cuts, cut_type='backlobe') & not_excl

    # Categories: pass RCR, pass BL (but not RCR), neither
    neither = not_excl & ~rcr_pass & ~bl_pass

    chi_rcr_cut = nominal_cuts['chi_rcr_line_chi'][0]

    fig, ax = plt.subplots(figsize=(9, 8))

    # Layer 1: gray dots (neither pass)
    ax.scatter(chi_bl[neither], chi_rcr[neither],
               s=4, c='gray', alpha=0.3, label='Data (no cut passed)', zorder=1)

    # Layer 2: Identified events (under passing data)
    if identified_bl_data is not None and len(identified_bl_data['snr']) > 0:
        ax.scatter(identified_bl_data['Chi2016'], identified_bl_data['ChiRCR'],
                   marker='s', s=50, c='cyan', edgecolors='black', linewidths=0.5,
                   label='Identified BL', zorder=3)
    if identified_rcr_data is not None and len(identified_rcr_data['snr']) > 0:
        ax.scatter(identified_rcr_data['Chi2016'], identified_rcr_data['ChiRCR'],
                   marker='^', s=60, c='red', edgecolors='black', linewidths=0.5,
                   label='Identified RCR', zorder=3)

    # Layer 3: passing events on top
    ax.scatter(chi_bl[bl_pass], chi_rcr[bl_pass],
               marker='o', s=30, c='yellow', edgecolors='black', linewidths=0.5,
               label=f'Pass BL Cuts ({int(np.sum(bl_pass))})', zorder=4)
    ax.scatter(chi_bl[rcr_pass], chi_rcr[rcr_pass],
               marker='*', s=60, c='green', edgecolors='black', linewidths=0.5,
               label=f'Pass RCR Cuts ({int(np.sum(rcr_pass))})', zorder=5)

    # Diagonal
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, alpha=0.5, zorder=2)

    # Cut lines
    # RCR region (above diagonal): horizontal line at chi-RCR = cut
    ax.plot([0, chi_rcr_cut], [chi_rcr_cut, chi_rcr_cut], 'b--', linewidth=1.5,
            alpha=0.7, zorder=2)
    # BL region (below diagonal): vertical line at chi-BL = cut
    ax.plot([chi_rcr_cut, chi_rcr_cut], [0, chi_rcr_cut], 'b--', linewidth=1.5,
            alpha=0.7, zorder=2)

    ax.set_xlabel(r'$\chi_{BL}$', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r'$\chi_{\mathrm{RCR}}$', fontsize=FONTSIZE_LABEL)
    ax.set_title(r'$\chi$-$\chi$ Cut Space: Station Data', fontsize=FONTSIZE_TITLE)
    ax.legend(fontsize=FONTSIZE_LEGEND, loc='lower right')
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    if use_comp_log:
        set_comp_log_scale(ax, axis='both')
    else:
        ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    ic(f"Saved chi-chi cut space plot: {output_path}")


# ============================================================================
# Text Output
# ============================================================================

def print_text_output(nominal_cuts, sim_direct, sim_reflected,
                      sim_reflected_high, sim_reflected_low,
                      data_dict, data_station_ids,
                      identified_bl_data, identified_rcr_data,
                      excluded_events_mask,
                      scale_factors, cut_interaction_results,
                      output_path):
    """
    Print comprehensive text output with event counts, errors, scale factors.
    """
    lines = []
    lines.append(f"S03b Selected Plots — Text Output")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"{'='*85}")
    lines.append(f"")

    # Events passing nominal cuts
    rcr_mask = apply_cuts(sim_reflected, nominal_cuts, cut_type='rcr')
    rcr_count, rcr_stat_err = compute_weighted_count_and_error(sim_reflected['weights'], rcr_mask)
    bl_mask = apply_cuts(sim_direct, nominal_cuts, cut_type='rcr')
    bl_count, bl_stat_err = compute_weighted_count_and_error(sim_direct['weights'], bl_mask)

    # R-sweep variation for RCR
    rcr_mask_hi = apply_cuts(sim_reflected, nominal_cuts, cut_type='rcr')
    rcr_count_hi, _ = compute_weighted_count_and_error(sim_reflected_high['weights'], rcr_mask_hi)
    rcr_count_lo, _ = compute_weighted_count_and_error(sim_reflected_low['weights'], rcr_mask)
    rcr_r_variation = abs(rcr_count_hi - rcr_count_lo) / 2
    rcr_total_err = np.sqrt(rcr_stat_err**2 + rcr_r_variation**2)

    both_count = rcr_count + bl_count
    both_stat_err = np.sqrt(rcr_stat_err**2 + bl_stat_err**2)
    both_total_err = np.sqrt(rcr_total_err**2 + bl_stat_err**2)

    # Data
    data_mask = apply_cuts(data_dict, nominal_cuts, cut_type='rcr')
    data_mask &= ~excluded_events_mask
    pidx = np.where(data_mask)[0]
    if len(pidx) > 0:
        umask = filter_unique_events_by_day(
            data_dict['Time'][pidx], data_station_ids[pidx])
        data_count = int(np.sum(umask))
    else:
        data_count = 0

    # Identified
    if identified_bl_data is not None and len(identified_bl_data['snr']) > 0:
        ibl_mask = apply_cuts(identified_bl_data, nominal_cuts, cut_type='rcr')
        ibl_count = int(np.sum(ibl_mask))
    else:
        ibl_count = 0
    if identified_rcr_data is not None and len(identified_rcr_data['snr']) > 0:
        ircr_mask = apply_cuts(identified_rcr_data, nominal_cuts, cut_type='rcr')
        ircr_count = int(np.sum(ircr_mask))
    else:
        ircr_count = 0

    lines.append(f"  Events Passing Nominal Cuts:")
    lines.append(f"    ChiRCR > {nominal_cuts['chi_rcr_line_chi'][0]:.2f}, "
                 f"{nominal_cuts['chi_diff_threshold']:.2f} < dChi < {nominal_cuts['chi_diff_max']:.2f}, "
                 f"SNR < {nominal_cuts['snr_max']}")
    lines.append(f"")
    lines.append(f"    {'Source':<30} {'Count':>10} {'Stat Err':>10} {'R-sweep':>10} {'Total Err':>10}")
    lines.append(f"    {'-'*72}")
    lines.append(f"    {'RCR Sim':<30} {rcr_count:>10.4f} {rcr_stat_err:>10.4f} "
                 f"{rcr_r_variation:>10.4f} {rcr_total_err:>10.4f}")
    lines.append(f"    {'BL Sim':<30} {bl_count:>10.4f} {bl_stat_err:>10.4f} "
                 f"{'N/A':>10} {bl_stat_err:>10.4f}")
    lines.append(f"    {'Both Sim (RCR+BL)':<30} {both_count:>10.4f} {both_stat_err:>10.4f} "
                 f"{'—':>10} {both_total_err:>10.4f}")
    lines.append(f"    {'Data (day-unique)':<30} {data_count:>10d} "
                 f"{np.sqrt(data_count):>10.2f} {'':>10} {np.sqrt(data_count):>10.2f}")
    lines.append(f"    {'Identified BL':<30} {ibl_count:>10d}")
    lines.append(f"    {'Identified RCR':<30} {ircr_count:>10d}")
    lines.append(f"")

    # Scale factors
    lines.append(f"  Scale Factors (RCR Sim -> Data at/above cut):")
    lines.append(f"    {'Parameter':<25} {'Scale':>10} {'Error':>10}")
    lines.append(f"    {'-'*47}")
    for param_name, (sf, sf_err) in scale_factors.items():
        lines.append(f"    {param_name:<25} {sf:>10.4f} {sf_err:>10.4f}")
    lines.append(f"")

    # Predicted events using scale factor
    lines.append(f"  Predicted Events (Scale Factor x RCR Sim):")
    lines.append(f"    {'Parameter':<25} {'Pred Above Cut':>15} {'Pred Total':>15}")
    lines.append(f"    {'-'*57}")
    for param_name, (sf, sf_err) in scale_factors.items():
        pred_above = rcr_count * sf
        pred_above_err = np.sqrt((rcr_total_err * sf)**2 + (rcr_count * sf_err)**2)
        total_rcr = np.sum(sim_reflected['weights'])
        pred_total = total_rcr * sf
        pred_total_err = total_rcr * sf_err
        lines.append(f"    {param_name:<25} {pred_above:>10.3f}+/-{pred_above_err:<5.3f}"
                     f" {pred_total:>10.3f}+/-{pred_total_err:<5.3f}")
    lines.append(f"")

    # Cut interaction results
    if cut_interaction_results:
        lines.append(f"  Cut Interaction Results at Nominal:")
        lines.append(f"    {'Parameter':<25} {'RCR':>8} {'BL':>8} {'Both':>8} "
                     f"{'Data':>8} {'Data range':>15}")
        lines.append(f"    {'-'*75}")
        for param_name, res in cut_interaction_results.items():
            if res is not None:
                lines.append(f"    {param_name:<25} {res['rcr_nom']:>8.3f} "
                             f"{res['bl_nom']:>8.3f} {res['both_nom']:>8.3f} "
                             f"{res['data_nom']:>8d} "
                             f"{res['data_lo']:>6d}–{res['data_hi']:<6d}")
        lines.append(f"")

    lines.append(f"{'='*85}")

    output = '\n'.join(lines)
    print(output)
    with open(output_path, 'w') as f:
        f.write(output + '\n')
    ic(f"Saved text output: {output_path}")


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
    plot_folder = f'HRAStationDataAnalysis/ErrorAnalysis/plots/{date_processing}/selected/'
    os.makedirs(plot_folder, exist_ok=True)

    ic.configureOutput(prefix='S03b | ')

    # --- Stations ---
    station_ids = [13, 14, 15, 17, 18, 19, 30]
    direct_stations = [13, 14, 15, 17, 18, 19, 30]
    reflected_stations = [113, 114, 115, 117, 118, 119, 130]

    # --- Nominal Cuts ---
    nominal_cuts = {
        'snr_max': 50,
        'chi_rcr_line_snr': np.array([0, 7, 8.5, 15, 20, 30, 100]),
        'chi_rcr_line_chi': np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]),
        'chi_diff_threshold': 0.0,
        'chi_diff_max': 0.2,
        'chi_2016_line_snr': np.array([0, 7, 8.5, 15, 20, 30, 100]),
        'chi_2016_line_chi': np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]),
    }

    # --- Excluded events ---
    excluded_events_list = [
        (18, 82), (18, 520), (18, 681),
        (15, 1472768),
        (19, 3621320), (19, 4599318), (19, 4599919)
    ]

    # --- Load Simulation Data ---
    HRAeventList = loadHRAfromH5(sim_file)
    sim_direct, sim_reflected = get_sim_data(
        HRAeventList, direct_weight_name, reflected_weight_name,
        direct_stations, reflected_stations, sigma=sim_sigma,
        apply_chi_diff_prefilter=True)

    ic(f"Sim direct: {len(sim_direct['snr'])} entries")
    ic(f"Sim reflected: {len(sim_reflected['snr'])} entries")

    # --- Load S04 event rates ---
    rcr_config = configparser.ConfigParser()
    rcr_config.read('RCRSimulation/config.ini')
    numpy_folder = rcr_config.get("FOLDERS", "numpy_folder",
                                   fallback="RCRSimulation/output/numpy")
    max_distance_rcr = float(rcr_config.get("SIMULATION", "distance_km",
                                             fallback="5")) / 2 * units.km

    reflected_rates, direct_rate, e_bins, z_bins = load_s04_event_rates(
        numpy_folder, max_distance_rcr)

    r_vals = sorted(reflected_rates.keys())
    stacked_ref = np.array([reflected_rates[r] for r in r_vals])
    high_ref_rate = np.nanmax(stacked_ref, axis=0)
    low_ref_rate = np.nanmin(stacked_ref, axis=0)
    central_ref_rate = (high_ref_rate + low_ref_rate) / 2

    high_ref_events = high_ref_rate * LIVETIME_YEARS
    low_ref_events = low_ref_rate * LIVETIME_YEARS
    central_ref_events = central_ref_rate * LIVETIME_YEARS
    direct_events = direct_rate * LIVETIME_YEARS

    # Assign weights
    assign_binned_weights(sim_reflected, central_ref_events, e_bins, z_bins, label="RCR reflected")
    assign_binned_weights(sim_direct, direct_events, e_bins, z_bins, label="BL direct")

    sim_reflected_high = copy.deepcopy(sim_reflected)
    sim_reflected_low = copy.deepcopy(sim_reflected)
    sim_direct_high = copy.deepcopy(sim_direct)
    sim_direct_low = copy.deepcopy(sim_direct)
    assign_binned_weights(sim_reflected_high, high_ref_events, e_bins, z_bins, label="RCR high")
    assign_binned_weights(sim_reflected_low, low_ref_events, e_bins, z_bins, label="RCR low")
    assign_binned_weights(sim_direct_high, direct_events, e_bins, z_bins, label="BL high")
    assign_binned_weights(sim_direct_low, direct_events, e_bins, z_bins, label="BL low")

    # --- Load 2016 BL Events ---
    json_path = 'StationDataAnalysis/2016FoundEvents.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            found_events_json = json.load(f)
    else:
        ic(f"Warning: JSON file not found at {json_path}")
        found_events_json = {}

    # --- Load Station Data ---
    ic("Loading station data...")
    all_snr, all_chi2016, all_chircr, all_times, all_station_ids_arr, all_event_ids = (
        [], [], [], [], [], [])

    for station_id in station_ids:
        snr_array = load_station_data(station_data_folder, date, station_id, 'SNR')
        Chi2016_array = load_station_data(station_data_folder, date, station_id, 'Chi2016')
        ChiRCR_array = load_station_data(station_data_folder, date, station_id, 'ChiRCR')
        times = load_station_data(station_data_folder, date, station_id, 'Time')
        event_ids_raw = load_station_data(station_data_folder, date, station_id, 'EventIDs')

        if Chi2016_array.size == 0 or ChiRCR_array.size == 0:
            continue

        initial_mask, unique_indices = getTimeEventMasks(times, event_ids_raw)
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

    data_dict = {
        'snr': np.concatenate(all_snr),
        'Chi2016': np.concatenate(all_chi2016),
        'ChiRCR': np.concatenate(all_chircr),
        'Time': np.concatenate(all_times),
    }
    data_station_ids = np.concatenate(all_station_ids_arr)
    data_event_ids = np.concatenate(all_event_ids)

    ic(f"Total data events before SNR prefilter: {len(data_dict['snr'])}")

    # === SNR < 50 PREFILTER ===
    snr_prefilter = data_dict['snr'] < SNR_PREFILTER
    for key in data_dict:
        data_dict[key] = data_dict[key][snr_prefilter]
    data_station_ids = data_station_ids[snr_prefilter]
    data_event_ids = data_event_ids[snr_prefilter]
    ic(f"Total data events after SNR < {SNR_PREFILTER} prefilter: {len(data_dict['snr'])}")

    # Build excluded events mask (after prefilter)
    excluded_set = set(excluded_events_list)
    excluded_mask = np.zeros(len(data_dict['snr']), dtype=bool)
    for idx in range(len(data_dict['snr'])):
        if (data_station_ids[idx], data_event_ids[idx]) in excluded_set:
            excluded_mask[idx] = True
    ic(f"Excluded {np.sum(excluded_mask)} double-counted events.")

    # --- Load 2016 BL Data ---
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
    else:
        bl_2016_data = None

    # --- Load Coincidence Events ---
    date_coincidence = config['PARAMETERS']['date_coincidence']
    coincidence_pickle_path = (
        "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/"
        f"{date_coincidence}_CoincidenceDatetimes_passing_cuts_with_all_params"
        "_recalcZenAzi_calcPol.pkl"
    )
    requested_coincidence_event_ids = [
        3047, 3432, 10195, 10231, 10273, 10284, 10444, 10449,
        10466, 10471, 10554, 11197, 11220, 11230, 11236, 11243,
    ]
    special_coinc_ids = {11230, 11243}
    special_coinc_station_map = {
        11230: {13: "RCR", 17: "Backlobe"},
        11243: {30: "RCR", 17: "Backlobe"},
    }

    coinc_bl_snr, coinc_bl_chi2016, coinc_bl_chircr = [], [], []
    coinc_rcr_snr, coinc_rcr_chi2016, coinc_rcr_chircr = [], [], []

    if os.path.exists(coincidence_pickle_path):
        with open(coincidence_pickle_path, 'rb') as handle:
            coinc_events_dict = pickle.load(handle)

        for event_id in requested_coincidence_event_ids:
            event_obj = coinc_events_dict.get(event_id) or coinc_events_dict.get(str(event_id))
            if event_obj is None:
                continue
            stations_info = event_obj.get('stations', {}) if isinstance(event_obj, dict) else {}
            default_category = "RCR" if event_id in special_coinc_ids else "Backlobe"
            for station_key, station_payload in stations_info.items():
                try:
                    station_int = int(station_key)
                except (TypeError, ValueError):
                    continue
                if station_int not in station_ids:
                    continue
                category = default_category
                if event_id in special_coinc_station_map:
                    category = special_coinc_station_map[event_id].get(station_int, category)

                snr_v = np.asarray(station_payload.get('SNR', []), dtype=float)
                chi2016_v = np.asarray(station_payload.get('Chi2016', []), dtype=float)
                chircr_v = np.asarray(station_payload.get('ChiRCR', []), dtype=float)
                min_len = min(snr_v.size, chi2016_v.size, chircr_v.size)
                if min_len == 0:
                    continue
                snr_v, chi2016_v, chircr_v = snr_v[:min_len], chi2016_v[:min_len], chircr_v[:min_len]
                valid = np.isfinite(snr_v) & np.isfinite(chi2016_v) & np.isfinite(chircr_v)
                if not np.any(valid):
                    continue
                if category == "Backlobe":
                    coinc_bl_snr.extend(snr_v[valid].tolist())
                    coinc_bl_chi2016.extend(chi2016_v[valid].tolist())
                    coinc_bl_chircr.extend(chircr_v[valid].tolist())
                else:
                    coinc_rcr_snr.extend(snr_v[valid].tolist())
                    coinc_rcr_chi2016.extend(chi2016_v[valid].tolist())
                    coinc_rcr_chircr.extend(chircr_v[valid].tolist())

    coinc_bl_raw = {
        'snr': np.array(coinc_bl_snr), 'Chi2016': np.array(coinc_bl_chi2016),
        'ChiRCR': np.array(coinc_bl_chircr),
    }
    coinc_rcr_raw = {
        'snr': np.array(coinc_rcr_snr), 'Chi2016': np.array(coinc_rcr_chi2016),
        'ChiRCR': np.array(coinc_rcr_chircr),
    }

    # Save full coinc sets BEFORE dedup (for coinc cut test plot)
    coinc_bl_full = {k: v.copy() for k, v in coinc_bl_raw.items()}
    coinc_rcr_full = {k: v.copy() for k, v in coinc_rcr_raw.items()}

    # Deduplicate coinc BL against 2016 BL
    if bl_2016_data is not None and len(coinc_bl_raw['snr']) > 0:
        bl16_tuples = set(zip(
            np.round(bl_2016_data['snr'], 6),
            np.round(bl_2016_data['ChiRCR'], 6),
            np.round(bl_2016_data['Chi2016'], 6),
        ))
        keep = np.ones(len(coinc_bl_raw['snr']), dtype=bool)
        for i in range(len(coinc_bl_raw['snr'])):
            t = (round(coinc_bl_raw['snr'][i], 6),
                 round(coinc_bl_raw['ChiRCR'][i], 6),
                 round(coinc_bl_raw['Chi2016'][i], 6))
            if t in bl16_tuples:
                keep[i] = False
        for key in coinc_bl_raw:
            coinc_bl_raw[key] = coinc_bl_raw[key][keep]

    # Identified BL = 2016 BL + deduplicated Coinc BL
    if bl_2016_data is not None:
        identified_bl_data = {
            'snr': np.concatenate([bl_2016_data['snr'], coinc_bl_raw['snr']]),
            'Chi2016': np.concatenate([bl_2016_data['Chi2016'], coinc_bl_raw['Chi2016']]),
            'ChiRCR': np.concatenate([bl_2016_data['ChiRCR'], coinc_bl_raw['ChiRCR']]),
        }
    elif len(coinc_bl_raw['snr']) > 0:
        identified_bl_data = coinc_bl_raw
    else:
        identified_bl_data = None

    identified_rcr_data = coinc_rcr_raw if len(coinc_rcr_raw['snr']) > 0 else None

    ic(f"Identified BL: {len(identified_bl_data['snr']) if identified_bl_data else 0} events")
    ic(f"Identified RCR: {len(identified_rcr_data['snr']) if identified_rcr_data else 0} events")

    # No-cuts config
    no_cuts = dict(nominal_cuts)
    no_cuts['snr_max'] = 9999
    no_cuts['chi_rcr_line_chi'] = np.zeros_like(nominal_cuts['chi_rcr_line_chi'])
    no_cuts['chi_diff_threshold'] = -999
    no_cuts['chi_diff_max'] = 999

    # =========================================================================
    # PLOT 1: Coincidence Cut Test (3-subplot)
    # =========================================================================
    ic("Generating coincidence cut test plot...")
    # Coincidence event rates (evts/yr)
    COINC_RCR_RATE = 1.7
    COINC_RCR_RATE_ERR = 0.25
    COINC_BL_RATE = 31.7
    COINC_BL_RATE_ERR = 3.5
    plot_coincidence_cut_test(
        coinc_bl_full, coinc_rcr_full,
        COINC_BL_RATE, COINC_BL_RATE_ERR,
        COINC_RCR_RATE, COINC_RCR_RATE_ERR,
        sim_direct, sim_reflected,
        sim_direct_high, sim_reflected_high,
        sim_direct_low, sim_reflected_low,
        os.path.join(plot_folder, 'coinc_cut_test.png'))

    # =========================================================================
    # PLOT 2: dist_errorbar
    # =========================================================================
    ic("Generating dist_errorbar plots...")
    dist_eb_folder = os.path.join(plot_folder, 'dist_errorbar')
    os.makedirs(dist_eb_folder, exist_ok=True)
    dist_eb_params = [
        ('chi_rcr_flat', nominal_cuts['chi_rcr_line_chi'][0], (0.2, 0.9), 50),
        ('chi_diff_threshold', nominal_cuts['chi_diff_threshold'], (-0.3, 0.3), 50),
        ('snr_max', nominal_cuts['snr_max'], (3, 50), 50),
        ('chi_bl', 0.0, (0.0, 1.0), 50),
    ]
    for dbg_param, dbg_nominal, dbg_range, dbg_bins in dist_eb_params:
        for show_cut, suffix in [(True, ''), (False, '_nocut')]:
            plot_dist_errorbar(
                dbg_param, dbg_nominal, sim_direct, sim_reflected,
                sim_direct_high, sim_reflected_high,
                sim_direct_low, sim_reflected_low,
                data_dict, identified_bl_data, identified_rcr_data,
                excluded_mask,
                os.path.join(dist_eb_folder, f'dist_{dbg_param}{suffix}.png'),
                n_bins=dbg_bins, param_range=dbg_range, show_cut_line=show_cut)

    # =========================================================================
    # PLOT 3: Parameter Width Assessment
    # =========================================================================
    ic("Generating parameter width plots...")
    width_folder = os.path.join(plot_folder, 'param_width')
    os.makedirs(width_folder, exist_ok=True)
    width_params = [
        ('chi_rcr_flat', (0.2, 0.9)),
        ('chi_bl', (0.0, 1.0)),
        ('snr_max', (3, 50)),
    ]
    for wp_name, wp_range in width_params:
        # Version with SNR>10 subset
        plot_parameter_width_assessment(
            wp_name, sim_direct, sim_reflected,
            sim_direct_high, sim_reflected_high,
            sim_direct_low, sim_reflected_low,
            data_dict, excluded_mask, nominal_cuts,
            os.path.join(width_folder, f'param_width_{wp_name}.png'),
            n_bins=50, param_range=wp_range, show_snr10_subset=True)
        # Version without SNR>10 subset
        plot_parameter_width_assessment(
            wp_name, sim_direct, sim_reflected,
            sim_direct_high, sim_reflected_high,
            sim_direct_low, sim_reflected_low,
            data_dict, excluded_mask, nominal_cuts,
            os.path.join(width_folder, f'param_width_{wp_name}_nosnr10.png'),
            n_bins=50, param_range=wp_range, show_snr10_subset=False)

    # Combined 1x3 subplot versions
    plot_parameter_width_combined(
        width_params, sim_direct, sim_reflected,
        sim_direct_high, sim_reflected_high,
        sim_direct_low, sim_reflected_low,
        data_dict, excluded_mask, nominal_cuts,
        os.path.join(width_folder, 'param_width_combined.png'),
        n_bins=50, show_snr10_subset=True)
    plot_parameter_width_combined(
        width_params, sim_direct, sim_reflected,
        sim_direct_high, sim_reflected_high,
        sim_direct_low, sim_reflected_low,
        data_dict, excluded_mask, nominal_cuts,
        os.path.join(width_folder, 'param_width_combined_nosnr10.png'),
        n_bins=50, show_snr10_subset=False)

    # =========================================================================
    # PLOT 4: Cut Interaction Cumulative
    # =========================================================================
    ic("Generating cut interaction plots...")
    ci_folder = os.path.join(plot_folder, 'cut_interaction')
    os.makedirs(ci_folder, exist_ok=True)

    cut_interaction_results = {}

    # delta-chi: normal y-scale, range -0.15 to 0.15
    ci_results_dchi = plot_cut_interaction(
        'chi_diff_threshold', nominal_cuts, nominal_cuts['chi_diff_threshold'],
        'chi_rcr_flat', (0.74, 0.76),
        sim_direct, sim_reflected,
        sim_direct_high, sim_reflected_high,
        sim_direct_low, sim_reflected_low,
        data_dict, data_station_ids,
        identified_bl_data, identified_rcr_data,
        excluded_mask,
        os.path.join(ci_folder, 'cumulative_chi_diff.png'),
        x_range=(-0.15, 0.15), yscale='linear')
    cut_interaction_results['chi_diff_threshold'] = ci_results_dchi

    # chi-RCR: logarithmic y-scale, full range
    ci_results_chircr = plot_cut_interaction(
        'chi_rcr_flat', nominal_cuts, nominal_cuts['chi_rcr_line_chi'][0],
        'chi_diff_threshold', (-0.05, 0.05),
        sim_direct, sim_reflected,
        sim_direct_high, sim_reflected_high,
        sim_direct_low, sim_reflected_low,
        data_dict, data_station_ids,
        identified_bl_data, identified_rcr_data,
        excluded_mask,
        os.path.join(ci_folder, 'cumulative_chi_rcr.png'),
        x_range=None, yscale='log')
    cut_interaction_results['chi_rcr_flat'] = ci_results_chircr

    # =========================================================================
    # PLOT 5: Scale Factor Fit
    # =========================================================================
    ic("Generating scale factor plots...")
    sf_folder = os.path.join(plot_folder, 'scale_factor')
    os.makedirs(sf_folder, exist_ok=True)

    scale_factors = {}

    sf_chi, sf_chi_err = plot_scale_factor(
        'chi_rcr_flat', 0.75, nominal_cuts,
        'chi_diff_threshold', (-0.05, 0.05),
        sim_direct, sim_reflected,
        sim_direct_high, sim_reflected_high,
        sim_direct_low, sim_reflected_low,
        data_dict, data_station_ids,
        identified_bl_data, identified_rcr_data,
        excluded_mask, sf_folder,
        x_range=(0.5, 0.9), yscale='log')
    scale_factors['chi_rcr_flat'] = (sf_chi, sf_chi_err)

    sf_dchi, sf_dchi_err = plot_scale_factor(
        'chi_diff_threshold', 0.0, nominal_cuts,
        'chi_rcr_flat', (0.74, 0.76),
        sim_direct, sim_reflected,
        sim_direct_high, sim_reflected_high,
        sim_direct_low, sim_reflected_low,
        data_dict, data_station_ids,
        identified_bl_data, identified_rcr_data,
        excluded_mask, sf_folder,
        x_range=(-0.15, 0.15), yscale='linear')
    scale_factors['chi_diff_threshold'] = (sf_dchi, sf_dchi_err)

    # =========================================================================
    # PLOT 6: 2D Chi-Chi Coincidence Histograms
    # =========================================================================
    ic("Generating 2D chi-chi coincidence plot...")
    plot_chi_chi_coinc_2d(
        coinc_bl_full, coinc_rcr_full,
        COINC_BL_RATE, COINC_RCR_RATE,
        sim_direct, sim_reflected,
        identified_bl_data, identified_rcr_data,
        nominal_cuts,
        os.path.join(plot_folder, 'chi_chi_coinc_2d.png'))
    # Zoomed version (complementary-log scale focusing on 0.8-1.0)
    plot_chi_chi_coinc_2d(
        coinc_bl_full, coinc_rcr_full,
        COINC_BL_RATE, COINC_RCR_RATE,
        sim_direct, sim_reflected,
        identified_bl_data, identified_rcr_data,
        nominal_cuts,
        os.path.join(plot_folder, 'chi_chi_coinc_2d_zoomed.png'),
        use_comp_log=True)

    # =========================================================================
    # PLOT 7: Chi-Chi Cut Space (Station Data)
    # =========================================================================
    ic("Generating chi-chi cut space plot...")
    plot_chi_chi_cut_space(
        data_dict, excluded_mask, nominal_cuts,
        identified_bl_data, identified_rcr_data,
        os.path.join(plot_folder, 'chi_chi_cut_space.png'))
    # Zoomed version
    plot_chi_chi_cut_space(
        data_dict, excluded_mask, nominal_cuts,
        identified_bl_data, identified_rcr_data,
        os.path.join(plot_folder, 'chi_chi_cut_space_zoomed.png'),
        use_comp_log=True)

    # =========================================================================
    # TEXT OUTPUT
    # =========================================================================
    ic("Generating text output...")
    print_text_output(
        nominal_cuts, sim_direct, sim_reflected,
        sim_reflected_high, sim_reflected_low,
        data_dict, data_station_ids,
        identified_bl_data, identified_rcr_data,
        excluded_mask,
        scale_factors, cut_interaction_results,
        os.path.join(plot_folder, 'summary.txt'))

    ic(f"\nDone. All outputs saved to: {plot_folder}")
