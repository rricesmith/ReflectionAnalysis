"""
S04_distributionComparison.py
=============================
Additional justification for RCR cut values through distribution comparisons
and figure-of-merit optimization.

This script is supplementary to S03 (the primary cut-scan analysis). It provides:

1. Chi distribution comparison: ChiRCR and Chi2016 distributions of the 9 found
   events vs RCR sim, BL sim, and all data. Weighted KS tests included.

2. Chi-diff distribution: Histogram of (ChiRCR - Chi2016) for the 9 events
   overlaid on RCR sim and BL sim distributions.

3. SNR distribution check: Same comparison for SNR.

4. Figure of Merit scan: S/sqrt(B) and S/sqrt(S+B) vs each cut parameter value.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
import configparser
import json
from datetime import datetime
from icecream import ic
from scipy.stats import ks_2samp

# Reuse loading and cut functions from S03
from HRAStationDataAnalysis.ErrorAnalysis.S03_cutScanAnalysis import (
    load_station_data, load_cuts_for_station, loadHRAfromH5,
    get_sim_data, filter_unique_events_by_day,
    apply_cuts, compute_weighted_count_and_error, scan_cut_parameter,
)
from HRAStationDataAnalysis.C_utils import getTimeEventMasks


# ============================================================================
# Distribution Comparison Functions
# ============================================================================

def weighted_ks_test(sample, reference, ref_weights):
    """
    Approximate weighted KS test.

    Computes the KS statistic between an unweighted sample and a weighted
    reference distribution by creating the weighted empirical CDF.

    Returns (ks_stat, interpretation_string).
    Note: p-value from ks_2samp is not strictly valid for weighted data,
    so we report the statistic and interpret qualitatively.
    """
    if len(sample) == 0 or len(reference) == 0:
        return np.nan, "insufficient data"

    # Build weighted CDF of reference
    sort_idx = np.argsort(reference)
    ref_sorted = reference[sort_idx]
    w_sorted = ref_weights[sort_idx]
    w_cumsum = np.cumsum(w_sorted)
    w_cdf = w_cumsum / w_cumsum[-1]

    # Build CDF of sample
    sample_sorted = np.sort(sample)
    sample_cdf = np.arange(1, len(sample_sorted) + 1) / len(sample_sorted)

    # Interpolate reference CDF at sample points
    ref_cdf_at_sample = np.interp(sample_sorted, ref_sorted, w_cdf, left=0, right=1)

    ks_stat = np.max(np.abs(sample_cdf - ref_cdf_at_sample))

    if ks_stat < 0.2:
        interp = "good agreement"
    elif ks_stat < 0.4:
        interp = "moderate agreement"
    else:
        interp = "poor agreement"

    return ks_stat, interp


def plot_distribution_comparison(found_events_data, sim_rcr, sim_bl,
                                 all_data, param_name, xlabel,
                                 output_path, bins=30):
    """
    Plot distribution of a parameter for found events vs simulations and data.
    Includes weighted KS test results.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # All data (normalized)
    if len(all_data) > 0:
        ax.hist(all_data, bins=bins, density=True, alpha=0.3, color='gray',
                label=f'All Data (N={len(all_data)})', histtype='stepfilled')

    # BL sim (weighted, normalized)
    if len(sim_bl['weights']) > 0:
        ax.hist(sim_bl[param_name], bins=bins, weights=sim_bl['weights'],
                density=True, alpha=0.4, color='orange',
                label='BL Sim (weighted)', histtype='stepfilled')

    # RCR sim (weighted, normalized)
    if len(sim_rcr['weights']) > 0:
        ax.hist(sim_rcr[param_name], bins=bins, weights=sim_rcr['weights'],
                density=True, alpha=0.4, color='green',
                label='RCR Sim (weighted)', histtype='stepfilled')

    # Found events
    if len(found_events_data) > 0:
        ax.hist(found_events_data, bins=bins, density=True, alpha=0.8,
                color='red', histtype='step', linewidth=2,
                label=f'Found Events (N={len(found_events_data)})')

        # Weighted KS tests
        ks_rcr, interp_rcr = weighted_ks_test(
            found_events_data, sim_rcr[param_name], sim_rcr['weights'])
        ks_bl, interp_bl = weighted_ks_test(
            found_events_data, sim_bl[param_name], sim_bl['weights'])

        textstr = (f'KS vs RCR Sim: {ks_rcr:.3f} ({interp_rcr})\n'
                   f'KS vs BL Sim:  {ks_bl:.3f} ({interp_bl})')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Normalized Density', fontsize=12)
    ax.set_title(f'Distribution Comparison: {xlabel}', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    ic(f"Saved: {output_path}")


def plot_figure_of_merit(param_name, scan_values, nominal_value,
                         rcr_counts, bl_counts, output_path):
    """
    Plot S/sqrt(B) and S/sqrt(S+B) vs cut parameter value.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # S / sqrt(B)
    with np.errstate(divide='ignore', invalid='ignore'):
        s_over_sqrt_b = np.where(bl_counts > 0, rcr_counts / np.sqrt(bl_counts), 0)
        s_over_sqrt_sb = np.where(
            (rcr_counts + bl_counts) > 0,
            rcr_counts / np.sqrt(rcr_counts + bl_counts), 0)

    ax.plot(scan_values, s_over_sqrt_b, 'b-o', markersize=4, linewidth=1.5,
            label=r'S / $\sqrt{B}$')
    ax.plot(scan_values, s_over_sqrt_sb, 'r-s', markersize=4, linewidth=1.5,
            label=r'S / $\sqrt{S + B}$')

    ax.axvline(x=nominal_value, color='gray', linestyle='--', linewidth=1.5,
               label=f'Nominal = {nominal_value}', alpha=0.7)

    # Find and mark optimal
    if np.any(s_over_sqrt_b > 0):
        opt_idx = np.argmax(s_over_sqrt_b)
        ax.axvline(x=scan_values[opt_idx], color='blue', linestyle=':', linewidth=1,
                   alpha=0.5)
        ax.annotate(f'Opt S/√B = {scan_values[opt_idx]:.3f}',
                    xy=(scan_values[opt_idx], s_over_sqrt_b[opt_idx]),
                    xytext=(10, 10), textcoords='offset points', fontsize=9,
                    color='blue')

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Figure of Merit', fontsize=12)
    ax.set_title(f'Figure of Merit: {param_name}', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    ic(f"Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # --- Configuration (same as S03) ---
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
    plot_folder = f'HRAStationDataAnalysis/ErrorAnalysis/AdditionalJustification/plots/{date_processing}/'
    os.makedirs(plot_folder, exist_ok=True)

    ic.configureOutput(prefix='S04 DistComp | ')

    # --- Define Stations and Cuts ---
    station_ids = [13, 14, 15, 17, 18, 19, 30]
    direct_stations = [13, 14, 15, 17, 18, 19, 30]
    reflected_stations = [113, 114, 115, 117, 118, 119, 130]

    nominal_cuts = {
        'snr_max': 50,
        'chi_rcr_line_snr': np.array([0, 7, 8.5, 15, 20, 30, 100]),
        'chi_rcr_line_chi': np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]),
        'chi_diff_threshold': 0.0,
        'chi_diff_max': 0.2,
        'chi_2016_line_snr': np.array([0, 7, 8.5, 15, 20, 30, 100]),
        'chi_2016_line_chi': np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]),
    }

    excluded_events_list = [
        (18, 82), (18, 520), (18, 681),
        (15, 1472768),
        (19, 3621320), (19, 4599318), (19, 4599919)
    ]

    scan_params = {
        'chi_rcr_flat': {'values': np.linspace(0.5, 0.95, 20), 'nominal': 0.75},
        'chi_diff_threshold': {'values': np.linspace(-0.1, 0.1, 20), 'nominal': 0.0},
        'chi_diff_max': {'values': np.linspace(0.05, 0.4, 20), 'nominal': 0.2},
        'snr_max': {'values': np.linspace(10, 100, 20), 'nominal': 50},
    }

    # --- Load Sim Data (with pre-filter, matching S01) ---
    HRAeventList = loadHRAfromH5(sim_file)
    sim_direct, sim_reflected = get_sim_data(
        HRAeventList, direct_weight_name, reflected_weight_name,
        direct_stations, reflected_stations, sigma=sim_sigma,
        apply_chi_diff_prefilter=True
    )

    # --- Load and Assemble Summed Station Data ---
    all_snr, all_chi2016, all_chircr, all_times = [], [], [], []
    all_station_ids_arr, all_event_ids = [], []

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
            sys.exit(f"No cuts found for Station {station_id}.")

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

    # Build excluded mask
    excluded_set = set(excluded_events_list)
    excluded_mask = np.zeros(len(data_dict['snr']), dtype=bool)
    for idx in range(len(data_dict['snr'])):
        if (data_station_ids[idx], data_event_ids[idx]) in excluded_set:
            excluded_mask[idx] = True

    # --- Identify the Found Events (passing all nominal cuts) ---
    data_mask = apply_cuts(data_dict, nominal_cuts, cut_type='rcr')
    data_mask &= ~excluded_mask

    passing_indices = np.where(data_mask)[0]
    if len(passing_indices) > 0:
        times_pass = data_dict['Time'][passing_indices]
        sids_pass = data_station_ids[passing_indices]
        unique_mask = filter_unique_events_by_day(times_pass, sids_pass)
        found_indices = passing_indices[unique_mask]
    else:
        found_indices = np.array([], dtype=int)

    found_events = {
        'snr': data_dict['snr'][found_indices],
        'Chi2016': data_dict['Chi2016'][found_indices],
        'ChiRCR': data_dict['ChiRCR'][found_indices],
    }
    n_found = len(found_indices)
    ic(f"Found {n_found} events passing all nominal cuts (expected: 9)")

    # =============================================
    # 1. Distribution Comparisons
    # =============================================
    ic("Generating distribution comparison plots...")

    # ChiRCR distribution
    plot_distribution_comparison(
        found_events['ChiRCR'], sim_reflected, sim_direct,
        data_dict['ChiRCR'], 'ChiRCR', r'RCR-$\chi$',
        os.path.join(plot_folder, 'dist_ChiRCR.png'),
        bins=np.linspace(0, 1, 40)
    )

    # Chi2016 distribution
    plot_distribution_comparison(
        found_events['Chi2016'], sim_reflected, sim_direct,
        data_dict['Chi2016'], 'Chi2016', r'BL-$\chi$',
        os.path.join(plot_folder, 'dist_Chi2016.png'),
        bins=np.linspace(0, 1, 40)
    )

    # Chi difference distribution
    found_chi_diff = found_events['ChiRCR'] - found_events['Chi2016']
    sim_rcr_diff = {'ChiDiff': sim_reflected['ChiRCR'] - sim_reflected['Chi2016'],
                    'weights': sim_reflected['weights']}
    sim_bl_diff = {'ChiDiff': sim_direct['ChiRCR'] - sim_direct['Chi2016'],
                   'weights': sim_direct['weights']}
    all_data_diff = data_dict['ChiRCR'] - data_dict['Chi2016']

    # Custom chi-diff plot (can't directly reuse plot_distribution_comparison for derived quantity)
    fig, ax = plt.subplots(figsize=(10, 7))
    diff_bins = np.linspace(-0.4, 0.4, 40)

    ax.hist(all_data_diff, bins=diff_bins, density=True, alpha=0.3, color='gray',
            label=f'All Data (N={len(all_data_diff)})', histtype='stepfilled')
    ax.hist(sim_bl_diff['ChiDiff'], bins=diff_bins, weights=sim_bl_diff['weights'],
            density=True, alpha=0.4, color='orange',
            label='BL Sim (weighted)', histtype='stepfilled')
    ax.hist(sim_rcr_diff['ChiDiff'], bins=diff_bins, weights=sim_rcr_diff['weights'],
            density=True, alpha=0.4, color='green',
            label='RCR Sim (weighted)', histtype='stepfilled')
    if len(found_chi_diff) > 0:
        ax.hist(found_chi_diff, bins=diff_bins, density=True, alpha=0.8,
                color='red', histtype='step', linewidth=2,
                label=f'Found Events (N={len(found_chi_diff)})')

        # KS tests
        ks_rcr, interp_rcr = weighted_ks_test(
            found_chi_diff, sim_rcr_diff['ChiDiff'], sim_rcr_diff['weights'])
        ks_bl, interp_bl = weighted_ks_test(
            found_chi_diff, sim_bl_diff['ChiDiff'], sim_bl_diff['weights'])
        textstr = (f'KS vs RCR Sim: {ks_rcr:.3f} ({interp_rcr})\n'
                   f'KS vs BL Sim:  {ks_bl:.3f} ({interp_bl})')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Mark the cut boundaries
    ax.axvline(x=0.0, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.7,
               label='chi_diff_threshold (0.0)')
    ax.axvline(x=0.2, color='darkgreen', linestyle=':', linewidth=1.5, alpha=0.7,
               label='chi_diff_max (0.2)')

    ax.set_xlabel(r'RCR-$\chi$ - BL-$\chi$', fontsize=12)
    ax.set_ylabel('Normalized Density', fontsize=12)
    ax.set_title(r'Distribution Comparison: RCR-$\chi$ - BL-$\chi$', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, 'dist_ChiDiff.png'), dpi=150)
    plt.close(fig)

    # SNR distribution
    plot_distribution_comparison(
        found_events['snr'], sim_reflected, sim_direct,
        data_dict['snr'], 'snr', 'SNR',
        os.path.join(plot_folder, 'dist_SNR.png'),
        bins=np.logspace(np.log10(3), np.log10(100), 30)
    )

    # =============================================
    # 2. Figure of Merit Scans
    # =============================================
    ic("Generating figure of merit scans...")

    # Load 2016 BL events for scan (not strictly needed for FoM but reused)
    json_path = 'StationDataAnalysis/2016FoundEvents.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            found_events_json = json.load(f)
    else:
        found_events_json = {}

    bl_2016_snr, bl_2016_chi2016, bl_2016_chircr = [], [], []
    for station_id in station_ids:
        station_key = f"Station{station_id}Found"
        if station_key not in found_events_json:
            continue
        target_times = found_events_json[station_key]
        snr_arr = load_station_data(station_data_folder, date, station_id, 'SNR')
        chi2016_arr = load_station_data(station_data_folder, date, station_id, 'Chi2016')
        chircr_arr = load_station_data(station_data_folder, date, station_id, 'ChiRCR')
        times_raw = load_station_data(station_data_folder, date, station_id, 'Time')
        if snr_arr.size == 0:
            continue
        time_map = {t: i for i, t in enumerate(times_raw)}
        found_idx = np.unique([time_map[t] for t in target_times if t in time_map])
        if len(found_idx) > 0:
            bl_2016_snr.append(snr_arr[found_idx])
            bl_2016_chi2016.append(chi2016_arr[found_idx])
            bl_2016_chircr.append(chircr_arr[found_idx])

    bl_2016_data = None
    if bl_2016_snr:
        bl_2016_data = {
            'snr': np.concatenate(bl_2016_snr),
            'Chi2016': np.concatenate(bl_2016_chi2016),
            'ChiRCR': np.concatenate(bl_2016_chircr),
        }

    for param_name, param_info in scan_params.items():
        rcr_c, rcr_e, bl_c, bl_e, bl16_c, data_c = scan_cut_parameter(
            param_name, param_info['values'], nominal_cuts,
            sim_direct, sim_reflected,
            data_dict, data_station_ids, bl_2016_data,
            excluded_mask
        )

        plot_figure_of_merit(
            param_name, param_info['values'], param_info['nominal'],
            rcr_c, bl_c,
            os.path.join(plot_folder, f'fom_{param_name}.png')
        )

    # =============================================
    # 3. Summary of KS Tests
    # =============================================
    ic("Generating KS test summary...")
    summary_lines = [
        f"S04 Distribution Comparison Summary — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Found Events: N = {n_found}",
        "",
    ]

    for param, label in [('ChiRCR', 'RCR-chi'), ('Chi2016', 'BL-chi'), ('snr', 'SNR')]:
        ks_rcr, interp_rcr = weighted_ks_test(
            found_events[param], sim_reflected[param], sim_reflected['weights'])
        ks_bl, interp_bl = weighted_ks_test(
            found_events[param], sim_direct[param], sim_direct['weights'])
        summary_lines.append(f"{label}:")
        summary_lines.append(f"  vs RCR Sim: KS = {ks_rcr:.4f} ({interp_rcr})")
        summary_lines.append(f"  vs BL Sim:  KS = {ks_bl:.4f} ({interp_bl})")
        summary_lines.append("")

    # Chi-diff
    ks_rcr, interp_rcr = weighted_ks_test(
        found_chi_diff,
        sim_reflected['ChiRCR'] - sim_reflected['Chi2016'],
        sim_reflected['weights'])
    ks_bl, interp_bl = weighted_ks_test(
        found_chi_diff,
        sim_direct['ChiRCR'] - sim_direct['Chi2016'],
        sim_direct['weights'])
    summary_lines.append("Chi Diff (RCR-chi - BL-chi):")
    summary_lines.append(f"  vs RCR Sim: KS = {ks_rcr:.4f} ({interp_rcr})")
    summary_lines.append(f"  vs BL Sim:  KS = {ks_bl:.4f} ({interp_bl})")

    summary_text = '\n'.join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(plot_folder, 'ks_test_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text + '\n')
    ic(f"Saved KS summary to: {summary_path}")

    ic("Done.")
