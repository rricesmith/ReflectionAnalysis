"""
S04b_singleStationRCRPlots.py
=============================
For each RCR-passing event saved in rcr_passing_events.npz, produce a two-panel
figure showing:
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
import matplotlib.pyplot as plt
from datetime import datetime
from icecream import ic

from NuRadioReco.utilities import fft
from HRAStationDataAnalysis.ErrorAnalysis.S03b_saveRCRPassingEvents import iterate_rcr_events


# ============================================================================
# Paths
# ============================================================================
INPUT_FILE  = 'HRAStationDataAnalysis/ErrorAnalysis/output/3.9.26/rcr_passing_events.npz'
OUTPUT_DIR  = 'HRAStationDataAnalysis/ErrorAnalysis/plots/3.9.26/rcr_single_station_plots'

# Sampling parameters
SAMPLING_RATE_HZ = 2e9   # 2 GHz — ARIANNA station sampling rate


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
        station_id, event_id, time, traces (4,256), snr, chi_rcr, chi_2016, chi_bad,
        azi, zen.
    output_dir : str
        Directory into which the .png is saved.
    """
    traces = evt['traces']           # (4, 256)

    # --- Select loudest channel by peak absolute amplitude ---
    amplitudes = [np.max(np.abs(traces[ch])) for ch in range(4)]
    selected_ch = np.argmax(amplitudes)
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
    ax_trace.plot(time_ax_ns, trace, lw=1.0)
    ax_trace.set_xlabel('Time (ns)')
    ax_trace.set_ylabel('Voltage (V)')
    ax_trace.set_title(f'Channel {selected_ch} (loudest)')
    ax_trace.grid(True)

    # --- Right panel: frequency spectrum ---
    ax_spectrum.plot(freq_ax_ghz, spectrum, lw=1.0)
    ax_spectrum.set_xlabel('Frequency (GHz)')
    ax_spectrum.set_ylabel('Amplitude')
    ax_spectrum.set_title('Frequency Spectrum')
    ax_spectrum.set_xlim(0, 1)
    ax_spectrum.grid(True)

    # --- Textbox: SNR and chi values in upper-right of spectrum panel ---
    # Use LaTeX for Greek letters with subscripts
    textbox_str = (
        f"SNR = {evt['snr']:.1f}\n"
        r"$\chi_\mathrm{RCR}$" + f" = {evt['chi_rcr']:.2f}\n"
        r"$\chi_\mathrm{BL}$"  + f" = {evt['chi_2016']:.2f}"
    )
    ax_spectrum.text(
        0.97, 0.95, textbox_str,
        transform=ax_spectrum.transAxes,
        ha='right', va='top',
        fontsize=14,
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ic.configureOutput(prefix='RCRPlots | ')

    ic(f"Input: {INPUT_FILE}")
    ic(f"Output directory: {OUTPUT_DIR}")

    n_saved = 0
    for evt in iterate_rcr_events(INPUT_FILE):
        save_path = plot_event(evt, OUTPUT_DIR)
        n_saved += 1
        ic(f"[{n_saved}] Station {evt['station_id']}  event {evt['event_id']}  SNR={evt['snr']:.1f}  -> {save_path}")

    ic(f"Done. {n_saved} plots saved to {OUTPUT_DIR}")
