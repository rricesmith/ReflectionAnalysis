"""
S04b_singleStationRCRPlots.py
=============================
Identifies single-station RCR-passing events by applying the nominal RCR cuts
to all station data (mirrors the pipeline in S03b_saveRCRPassingEvents), saves
the passing events as an .npz file so S04c can use them, then produces a
two-panel figure for each event showing:

  - Left panel:  waveform trace of the loudest channel (voltage vs. time in ns)
  - Right panel: frequency spectrum of the same channel (amplitude vs. frequency in GHz)

A textbox in the upper-right of the spectrum panel displays SNR, chi_RCR, and chi_BL.
One .png file is saved per event.

Usage:
    python -m HRAStationDataAnalysis.ErrorAnalysis.S04b_singleStationRCRPlots
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import configparser
import matplotlib.pyplot as plt
from datetime import datetime
from icecream import ic

from NuRadioReco.utilities import fft
from HRAStationDataAnalysis.C_utils import getTimeEventMasks
from HRAStationDataAnalysis.ErrorAnalysis.S03b_saveRCRPassingEvents import (
    load_station_data,
    load_cuts_for_station,
    apply_rcr_cuts,
    iterate_rcr_events,
    NOMINAL_CUTS,
    SNR_PREFILTER,
    EXCLUDED_EVENTS,
    STATION_IDS,
)

# Sampling parameters
SAMPLING_RATE_HZ = 2e9   # 2 GHz — ARIANNA station sampling rate


# ============================================================================
# Data collection
# ============================================================================

def collect_and_save_rcr_events(date, date_cuts, output_path):
    """
    Apply nominal RCR cuts to all station data and save passing events as .npz.

    Uses the same pipeline as S03b_saveRCRPassingEvents so that S04c can load
    the resulting file directly.

    Parameters
    ----------
    date : str
        Data date string (e.g. '9.1.25'), used to locate nurFiles.
    date_cuts : str
        Cuts date string (e.g. '9.18.25'), used to locate cut files.
    output_path : str
        Full path for the output .npz file.

    Returns
    -------
    bool
        True if at least one event was saved, False otherwise.
    """
    station_data_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'
    cuts_data_folder    = f'HRAStationDataAnalysis/StationData/cuts/{date_cuts}/'

    out_station_ids, out_event_ids, out_times  = [], [], []
    out_traces, out_snr, out_chi_rcr           = [], [], []
    out_chi_2016, out_chi_bad                  = [], []
    out_azi, out_zen                            = [], []

    excluded_set = set(EXCLUDED_EVENTS)

    for station_id in STATION_IDS:
        ic(f"Processing station {station_id}...")

        snr_array      = load_station_data(station_data_folder, date, station_id, 'SNR')
        chi2016_array  = load_station_data(station_data_folder, date, station_id, 'Chi2016')
        chircr_array   = load_station_data(station_data_folder, date, station_id, 'ChiRCR')
        times          = load_station_data(station_data_folder, date, station_id, 'Time')
        event_ids_raw  = load_station_data(station_data_folder, date, station_id, 'EventIDs')

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
        snr_cut      = snr_array[initial_mask][final_indices]
        chi2016_cut  = chi2016_array[initial_mask][final_indices]
        chircr_cut   = chircr_array[initial_mask][final_indices]
        times_cut    = times[initial_mask][final_indices]
        evtids_cut   = event_ids_raw[initial_mask][final_indices]

        # SNR prefilter
        snr_prefilt        = snr_cut < SNR_PREFILTER
        snr_cut            = snr_cut[snr_prefilt]
        chi2016_cut        = chi2016_cut[snr_prefilt]
        chircr_cut         = chircr_cut[snr_prefilt]
        times_cut          = times_cut[snr_prefilt]
        evtids_cut         = evtids_cut[snr_prefilt]
        final_indices_filt = final_indices[snr_prefilt]

        # Exclude double-counted events
        keep = np.ones(len(snr_cut), dtype=bool)
        for idx in range(len(snr_cut)):
            if (station_id, evtids_cut[idx]) in excluded_set:
                keep[idx] = False
        snr_cut            = snr_cut[keep]
        chi2016_cut        = chi2016_cut[keep]
        chircr_cut         = chircr_cut[keep]
        times_cut          = times_cut[keep]
        evtids_cut         = evtids_cut[keep]
        final_indices_filt = final_indices_filt[keep]

        # Apply RCR cuts
        rcr_mask  = apply_rcr_cuts(snr_cut, chircr_cut, chi2016_cut, NOMINAL_CUTS)
        n_passing = int(np.sum(rcr_mask))
        ic(f"  Station {station_id}: {n_passing} events pass RCR cuts (out of {len(snr_cut)})")

        if n_passing == 0:
            continue

        # Indices into the initial_mask-applied arrays for passing events
        passing_raw_indices = final_indices_filt[rcr_mask]

        # Load remaining parameters for passing events only
        traces_array   = load_station_data(station_data_folder, date, station_id, 'Traces')
        chi_bad_array  = load_station_data(station_data_folder, date, station_id, 'ChiBad')
        azi_array      = load_station_data(station_data_folder, date, station_id, 'Azi')
        zen_array      = load_station_data(station_data_folder, date, station_id, 'Zen')

        traces_pass   = traces_array[initial_mask][passing_raw_indices]
        chi_bad_pass  = (chi_bad_array[initial_mask][passing_raw_indices]
                         if chi_bad_array.size > 0 else np.full(n_passing, np.nan))
        azi_pass      = (azi_array[initial_mask][passing_raw_indices]
                         if azi_array.size > 0 else np.full(n_passing, np.nan))
        zen_pass      = (zen_array[initial_mask][passing_raw_indices]
                         if zen_array.size > 0 else np.full(n_passing, np.nan))

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

    if not out_snr:
        ic("No events passed RCR cuts across any station.")
        return False

    result = {
        'station_ids': np.concatenate(out_station_ids),
        'event_ids':   np.concatenate(out_event_ids),
        'times':       np.concatenate(out_times),
        'traces':      np.concatenate(out_traces),
        'snr':         np.concatenate(out_snr),
        'chi_rcr':     np.concatenate(out_chi_rcr),
        'chi_2016':    np.concatenate(out_chi_2016),
        'chi_bad':     np.concatenate(out_chi_bad),
        'azi':         np.concatenate(out_azi),
        'zen':         np.concatenate(out_zen),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **result)
    n_total = len(result['snr'])
    ic(f"Saved {n_total} RCR-passing events to {output_path}")
    return True


# ============================================================================
# Plotting helper
# ============================================================================

def plot_event(evt, output_dir):
    """
    Create and save a two-panel waveform + spectrum figure for a single event.

    Parameters
    ----------
    evt : dict
        Single event dict from iterate_rcr_events(), containing keys:
        station_id, event_id, time, traces (4,256), snr, chi_rcr, chi_2016.
    output_dir : str
        Directory into which the .png is saved.
    """
    traces = evt['traces']           # (4, 256)

    # --- Select loudest channel by peak absolute amplitude ---
    amplitudes = [np.max(np.abs(traces[ch])) for ch in range(4)]
    selected_ch = int(np.argmax(amplitudes))
    trace = traces[selected_ch]      # (256,)

    # --- Time axis (nanoseconds) ---
    # Sampling interval = 1 / 2e9 Hz = 0.5 ns
    time_ax_ns = np.linspace(0, (len(trace) - 1) * 0.5, len(trace))

    # --- Frequency axis and spectrum ---
    freq_ax_ghz = np.fft.rfftfreq(len(trace), d=1.0 / SAMPLING_RATE_HZ) / 1e9
    spectrum = np.abs(fft.time2freq(trace, SAMPLING_RATE_HZ))
    spectrum[0] = 0   # zero DC component

    # --- Figure layout ---
    fig, (ax_trace, ax_spectrum) = plt.subplots(1, 2, figsize=(11, 4))

    # Figure-level title: human-readable UTC timestamp
    title_time = datetime.utcfromtimestamp(evt['time']).strftime('%Y-%m-%d %H:%M:%S UTC')
    fig.suptitle(
        f"Station {evt['station_id']} | {title_time}",
        fontsize=13
    )

    # --- Left panel: waveform trace ---
    ax_trace.plot(time_ax_ns, trace, lw=1.5)
    ax_trace.set_xlabel('Time (ns)')
    ax_trace.set_ylabel('Voltage (V)')
    ax_trace.grid(True)

    # --- Right panel: frequency spectrum ---
    ax_spectrum.plot(freq_ax_ghz, spectrum, lw=1.5)
    ax_spectrum.set_xlabel('Frequency (GHz)')
    ax_spectrum.set_ylabel('Amplitude')
    ax_spectrum.set_xlim(0, 1)
    ax_spectrum.grid(True)

    # --- Textbox: SNR, chi values, and arrival angles in upper-right of spectrum panel ---
    # Use LaTeX for Greek letters with subscripts; convert angles from radians to degrees
    azi_val = evt.get('azi', float('nan'))
    zen_val = evt.get('zen', float('nan'))
    if (np.isfinite(azi_val) and np.isfinite(zen_val)
            and not (azi_val == 0.0 and zen_val == 0.0)):
        azi_str = f"{np.degrees(azi_val) % 360:.1f}"
        zen_str = f"{np.degrees(zen_val):.1f}"
    else:
        azi_str = "N/A"
        zen_str = "N/A"

    textbox_str = (
        f"SNR = {evt['snr']:.1f}\n"
        r"$\chi_\mathrm{RCR}$" + f" = {evt['chi_rcr']:.2f}\n"
        r"$\chi_\mathrm{BL}$"  + f" = {evt['chi_2016']:.2f}\n"
        f"Zen = {zen_str} deg\n"
        f"Azi = {azi_str} deg"
    )
    ax_spectrum.text(
        0.97, 0.95, textbox_str,
        transform=ax_spectrum.transAxes,
        ha='right', va='top',
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', fc='wheat', alpha=0.7)
    )

    # --- Save ---
    plt.tight_layout()
    filename = f"event_{evt['station_id']}_{evt['event_id']}.png"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return save_path


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    ic.configureOutput(prefix='RCRPlots | ')

    # --- Configuration ---
    config = configparser.ConfigParser()
    config_path = os.path.join('HRAStationDataAnalysis', 'config.ini')
    if not os.path.exists(config_path):
        config_path = 'config.ini'
    config.read(config_path)

    date            = config['PARAMETERS']['date']
    date_cuts       = config['PARAMETERS']['date_cuts']
    date_processing = config['PARAMETERS']['date_processing']

    input_file = os.path.join(
        'HRAStationDataAnalysis', 'ErrorAnalysis', 'output',
        date_processing, 'rcr_passing_events.npz'
    )
    # Prefer the angles-updated file written by S04c when available
    input_file_with_angles = os.path.join(
        'HRAStationDataAnalysis', 'ErrorAnalysis', 'output',
        date_processing, 'rcr_passing_events_with_angles.npz'
    )
    if os.path.exists(input_file_with_angles):
        ic(f"Using angles-updated file from S04c: {input_file_with_angles}")
        input_file = input_file_with_angles

    output_dir  = os.path.join(
        'HRAStationDataAnalysis', 'ErrorAnalysis', 'plots',
        date_processing, 'rcr_single_station_plots'
    )

    ic(f"date={date}, date_cuts={date_cuts}, date_processing={date_processing}")
    ic(f"Data file: {input_file}")
    ic(f"Output directory: {output_dir}")

    # --- Collect and save if the npz does not already exist ---
    if not os.path.exists(input_file):
        ic(f"'{input_file}' not found — running data collection...")
        ok = collect_and_save_rcr_events(date, date_cuts, input_file)
        if not ok:
            ic("Data collection produced no events. Exiting.")
            sys.exit(0)
    else:
        ic(f"Found existing data file, skipping collection.")

    # --- Plot ---
    os.makedirs(output_dir, exist_ok=True)

    n_saved = 0
    for evt in iterate_rcr_events(input_file):
        save_path = plot_event(evt, output_dir)
        n_saved += 1
        ic(f"[{n_saved}] Station {evt['station_id']}  event {evt['event_id']}"
           f"  SNR={evt['snr']:.1f}  -> {save_path}")

    ic(f"Done. {n_saved} plots saved to {output_dir}")
