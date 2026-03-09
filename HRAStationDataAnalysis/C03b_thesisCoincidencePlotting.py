"""
C03b_thesisCoincidencePlotting.py
---------------------------------
Thesis-ready master plots for coincidence events.
Based on C03_coincidenceEventPlotting.py but with:
  - Only master event plots (pass and fail)
  - No subplot titles
  - Single loudest-channel trace + spectrum (not all 4 channels)
  - chi_{2016} renamed to chi_{BL}
  - Chi legend in bottom-right, station legend in top-right
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from icecream import ic
import datetime
import gc
from collections import defaultdict
import matplotlib.gridspec as gridspec
from NuRadioReco.utilities import fft, units
import itertools
import time
from typing import Dict, List, Optional, Set

from templateCrossCorr import DEFAULT_TRACE_SAMPLING_HZ, get_xcorr_for_channel

# --- Reuse helpers and cut functions from C03 ---
from C03_coincidenceEventPlotting import (
    SectionTimer,
    _progress,
    PROGRESS_EVERY,
    NUM_TRACE_CHANNELS,
    _prepare_trace_array,
    _iter_channel_traces,
    _extract_snr_value,
    _find_loudest_trace,
    _load_pickle,
    check_chi_cut,
    check_angle_cut,
    check_fft_cut,
    check_time_cut,
    check_coincidence_cuts,
    compute_event_self_similarity,
)


# --- Thesis Master Plot: single loudest channel, no titles, chi_BL labelling ---
def plot_single_master_event_thesis(event_id, event_details, output_dir, dataset_name, title_suffix=""):
    """
    Thesis-ready master event plot.

    Differences from the original plot_single_master_event:
      - Only one trace row (loudest channel) instead of all 4 channels
      - No subplot titles
      - chi_{2016} relabelled to chi_{BL}
      - Chi-type legend placed in bottom-right of the chi-SNR plot
      - Station legend remains in top-right

    Returns:
        str: Path to the saved plot file, or None if failed
    """
    with SectionTimer(f"Thesis master plot for event {event_id}"):
        if not isinstance(event_details, dict):
            ic(f"Warning: Event {event_id} data is not a dictionary. Skipping master plot.")
            return None

    os.makedirs(output_dir, exist_ok=True)

    # Compute self-match metrics if not already present
    derived_metrics = event_details.get("derived_metrics")
    if not isinstance(derived_metrics, dict):
        derived_metrics = {}
        event_details["derived_metrics"] = derived_metrics
    self_match_summary = derived_metrics.get("self_match")
    if self_match_summary is None:
        compute_event_self_similarity({event_id: event_details})
        self_match_summary = derived_metrics.get("self_match")

    # Station colours and markers
    color_map = {13: 'tab:blue', 14: 'tab:orange', 15: 'tab:green',
                 17: 'tab:red', 18: 'tab:purple', 19: 'sienna', 30: 'tab:brown'}
    default_color = 'grey'
    marker_list = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'X', '+']

    cut_results = event_details.get('cut_results', {})
    passes_overall_analysis = event_details.get('passes_analysis_cuts', False)

    # --- Figure layout: scatter + polar on top, one trace + one spectrum, text box ---
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3,
                           height_ratios=[4, 2, 2.5])

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_polar = fig.add_subplot(gs[0, 1], polar=True)
    ax_trace = fig.add_subplot(gs[1, 0])
    ax_spectrum = fig.add_subplot(gs[1, 1])
    ax_text_box = fig.add_subplot(gs[2, :])

    # Event timestamp
    event_time_str = "Unknown Time"
    if "datetime" in event_details and event_details["datetime"] is not None:
        try:
            event_time_dt = datetime.datetime.fromtimestamp(event_details["datetime"])
            event_time_str = event_time_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        except Exception as e:
            ic(f"Error formatting datetime for event {event_id}: {e}")

    # Main figure title
    cut_status = "PASSES ALL CUTS" if passes_overall_analysis else "FAILS CUTS"
    main_title = (f"Master Plot: Event {event_id} ({dataset_name}) - {cut_status}"
                  f"{title_suffix}\nTime: {event_time_str}")
    fig.suptitle(main_title, fontsize=16, y=0.98)

    # --- Build text info lines ---
    status_text = "PASS" if passes_overall_analysis else "FAIL"
    text_info_lines = [f"Event ID: {event_id} -- Overall: {status_text} (Analysis Cuts)"]
    text_info_lines.append(
        f"Cut Status -> Time: {'Passed' if cut_results.get('time_cut_passed') else 'Failed'}, "
        f"Chi: {'Passed' if cut_results.get('chi_cut_passed') else 'Failed'}, "
        f"Angle: {'Passed' if cut_results.get('angle_cut_passed') else 'Failed'}, "
        f"FFT: {'Passed' if cut_results.get('fft_cut_passed') else 'Failed'}"
    )

    # Self-match info
    def _fmt_idx(idx):
        if isinstance(idx, int):
            return idx + 1
        try:
            return int(idx) + 1
        except (TypeError, ValueError):
            return None

    def _fmt_float(value, precision=2):
        try:
            if value is None or not np.isfinite(value):
                return "N/A"
            return f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            return "N/A"

    if isinstance(self_match_summary, dict):
        template_station = self_match_summary.get("template_station")
        template_trigger_idx = self_match_summary.get("template_trigger_idx")
        template_channel_idx = self_match_summary.get("template_channel_idx")
        template_snr_val = self_match_summary.get("template_snr")
        best_station = self_match_summary.get("match_station")
        best_trigger_idx = self_match_summary.get("match_trigger_idx")
        best_channel_idx = self_match_summary.get("match_channel_idx")
        best_chi = self_match_summary.get("best_match_abs_chi")
        best_snr = self_match_summary.get("match_snr")
        avg_pair_snr = self_match_summary.get("avg_pair_snr")
        stations_compared = self_match_summary.get("stations_compared", [])
        station_matches = self_match_summary.get("station_matches", {})

        text_info_lines.append("--- Template Self-Match ---")
        text_info_lines.append(
            "  Template: St{} T{} Ch{} SNR={}".format(
                template_station if template_station is not None else "?",
                _fmt_idx(template_trigger_idx) or "?",
                _fmt_idx(template_channel_idx) or "?",
                _fmt_float(template_snr_val, precision=1),
            )
        )
        text_info_lines.append(
            "  Best match: St{} T{} Ch{} chi={} SNR={} (avg pair SNR={})".format(
                best_station if best_station is not None else "?",
                _fmt_idx(best_trigger_idx) or "?",
                _fmt_idx(best_channel_idx) or "?",
                _fmt_float(best_chi, precision=2),
                _fmt_float(best_snr, precision=1),
                _fmt_float(avg_pair_snr, precision=1),
            )
        )
        if stations_compared:
            text_info_lines.append(
                "  Stations compared: {}".format(", ".join(str(s) for s in stations_compared))
            )
        if isinstance(station_matches, dict):
            sorted_keys = sorted(
                station_matches.keys(),
                key=lambda key: (
                    0 if str(key) == str(template_station) else 1,
                    int(key) if str(key).isdigit() else str(key),
                ),
            )
            text_info_lines.append("  Station chi overview:")
            for station_key in sorted_keys:
                station_info = station_matches.get(station_key) or {}
                role = "template" if str(station_key) == str(template_station) else "partner"
                text_info_lines.append(
                    "    St{} ({}): chi={} SNR={} T{} Ch{}".format(
                        station_key, role,
                        _fmt_float(station_info.get("chi"), precision=2),
                        _fmt_float(station_info.get("snr"), precision=1),
                        _fmt_idx(station_info.get("trigger_idx")) or "?",
                        _fmt_idx(station_info.get("channel_idx")) or "?",
                    )
                )

    text_info_lines.append("--- Station Triggers ---")

    # --- Find the loudest channel across all stations/triggers ---
    loudest_info = _find_loudest_trace(event_details)
    loudest_key = None  # (station_id_str, trigger_idx, channel_idx)
    if loudest_info is not None:
        loudest_key = (
            str(loudest_info["station_id"]),
            loudest_info["trigger_idx"],
            loudest_info["channel_idx"],
        )

    # --- Loop over stations to populate scatter, polar, and text ---
    legend_handles_for_fig = {}
    loudest_trace_plotted = False

    station_items = list(event_details.get("stations", {}).items())
    for s_idx, (station_id_str, station_data) in enumerate(station_items):
        _progress(s_idx, len(station_items), f"Thesis master event {event_id} stations")
        try:
            station_id_int = int(station_id_str)
        except ValueError:
            continue
        color = color_map.get(station_id_int, default_color)
        snr_values = station_data.get("SNR", [])
        num_triggers = len(snr_values)
        if num_triggers == 0:
            continue

        zen_values_rad = (station_data.get("Zen", []) + [np.nan] * num_triggers)[:num_triggers]
        azi_values_rad = (station_data.get("Azi", []) + [np.nan] * num_triggers)[:num_triggers]
        pol_angle_values_rad = (station_data.get("PolAngle", []) + [np.nan] * num_triggers)[:num_triggers]
        pol_angle_err_values_rad = (station_data.get("PolAngleErr", []) + [np.nan] * num_triggers)[:num_triggers]
        time_values = station_data.get("Time", [])
        chi_rcr_values = (station_data.get("ChiRCR", []) + [np.nan] * num_triggers)[:num_triggers]
        chi_bl_values = (station_data.get("Chi2016", []) + [np.nan] * num_triggers)[:num_triggers]
        event_ids_for_station = (station_data.get("event_ids", []) + ["N/A"] * num_triggers)[:num_triggers]
        all_traces_for_station = station_data.get("Traces", [])

        station_points = []

        for trigger_idx in range(num_triggers):
            marker = marker_list[trigger_idx % len(marker_list)]
            snr_val = snr_values[trigger_idx]
            chi_rcr_val = chi_rcr_values[trigger_idx]
            chi_bl_val = chi_bl_values[trigger_idx]
            zen_rad = zen_values_rad[trigger_idx]
            azi_rad = azi_values_rad[trigger_idx]
            pol_rad = pol_angle_values_rad[trigger_idx]
            pol_err_rad = pol_angle_err_values_rad[trigger_idx]
            current_event_id_val = event_ids_for_station[trigger_idx]
            traces_this_trigger = (all_traces_for_station[trigger_idx]
                                   if trigger_idx < len(all_traces_for_station) else [])

            # --- Scatter plot: SNR vs chi ---
            if snr_val is not None and not np.isnan(snr_val):
                if chi_bl_val is not None and not np.isnan(chi_bl_val):
                    ax_scatter.scatter(snr_val, chi_bl_val, c=color, marker=marker,
                                       s=60, alpha=0.9, zorder=3)
                    station_points.append((snr_val, chi_bl_val))
                if chi_rcr_val is not None and not np.isnan(chi_rcr_val):
                    ax_scatter.scatter(snr_val, chi_rcr_val, marker=marker, s=60, alpha=0.9,
                                       facecolors='none', edgecolors=color, linewidths=1.5, zorder=3)
                    station_points.append((snr_val, chi_rcr_val))

            # --- Polar plot ---
            if (zen_rad is not None and not np.isnan(zen_rad) and
                    azi_rad is not None and not np.isnan(azi_rad)):
                ax_polar.scatter(azi_rad, np.degrees(zen_rad), c=color, marker=marker,
                                 s=60, alpha=0.9)

            # --- Plot only the loudest channel trace + spectrum ---
            if (loudest_key is not None and not loudest_trace_plotted and
                    station_id_str == loudest_key[0] and trigger_idx == loudest_key[1]):
                padded_traces = (list(traces_this_trigger) + [None] * NUM_TRACE_CHANNELS)[:NUM_TRACE_CHANNELS]
                ch_idx = loudest_key[2]
                trace_ch_data = padded_traces[ch_idx] if ch_idx < len(padded_traces) else None
                if trace_ch_data is not None and hasattr(trace_ch_data, "__len__") and len(trace_ch_data) > 0:
                    trace_arr = np.asarray(trace_ch_data)
                    time_ax_ns = np.linspace(0, (len(trace_arr) - 1) * 0.5, len(trace_arr))
                    ax_trace.plot(time_ax_ns, trace_arr, c=color, ls='-', alpha=0.7)
                    ax_trace.set_ylabel("Voltage (V)", fontsize=8)
                    ax_trace.set_xlabel("Time (ns)", fontsize=8)
                    ax_trace.grid(True, ls=':', alpha=0.5)

                    # Frequency spectrum
                    sampling_rate_hz = 2e9
                    if len(trace_arr) > 1:
                        freq_ax_ghz = np.fft.rfftfreq(len(trace_arr), d=1 / sampling_rate_hz) / 1e9
                        spectrum = np.abs(fft.time2freq(trace_arr, sampling_rate_hz))
                        if len(spectrum) > 0:
                            spectrum[0] = 0
                        ax_spectrum.plot(freq_ax_ghz, spectrum, c=color, ls='-', alpha=0.6)
                        ax_spectrum.set_ylabel("Amplitude", fontsize=8)
                        ax_spectrum.set_xlabel("Frequency (GHz)", fontsize=8)
                        ax_spectrum.grid(True, ls=':', alpha=0.5)
                        ax_spectrum.set_xlim(0, 1)
                    loudest_trace_plotted = True

            # Station legend handle
            if station_id_int not in legend_handles_for_fig:
                legend_handles_for_fig[station_id_int] = Line2D(
                    [0], [0], color=color, linestyle='-', linewidth=4,
                    label=f"St {station_id_int}")

            # Text info
            chi_rcr_text = f"{chi_rcr_val:.2f}" if chi_rcr_val is not None and not np.isnan(chi_rcr_val) else "N/A"
            chi_bl_text = f"{chi_bl_val:.2f}" if chi_bl_val is not None and not np.isnan(chi_bl_val) else "N/A"
            zen_d_text = f"{np.degrees(zen_rad):.1f}deg" if zen_rad is not None and not np.isnan(zen_rad) else "N/A"
            azi_d_text = f"{(np.degrees(azi_rad) % 360):.1f}deg" if azi_rad is not None and not np.isnan(azi_rad) else "N/A"
            snr_fstr = f"{snr_val:.1f}" if snr_val is not None and not np.isnan(snr_val) else "N/A"
            ev_id_fstr = f"{int(current_event_id_val)}" if current_event_id_val not in ["N/A", np.nan, None] else "N/A"

            pol_angle_full_text = "N/A"
            if pol_rad is not None and not np.isnan(pol_rad):
                pol_angle_full_text = f"{np.degrees(pol_rad):.1f}"
                if pol_err_rad is not None and not np.isnan(pol_err_rad) and isinstance(pol_err_rad, (float, int)):
                    pol_angle_full_text += f" +/- {np.degrees(pol_err_rad):.1f}deg"
                else:
                    pol_angle_full_text += "deg"

            text_info_lines.append(
                f"  St{station_id_int} T{trigger_idx+1} Unix={time_values}: "
                f"ID={ev_id_fstr}, SNR={snr_fstr}, ChiRCR={chi_rcr_text}, "
                f"ChiBL={chi_bl_text}, Zen={zen_d_text}, Azi={azi_d_text}, Pol={pol_angle_full_text}"
            )

        # Connecting line for points within this station
        if len(station_points) > 1:
            station_points.sort()
            snrs, chis = zip(*station_points)
            ax_scatter.plot(snrs, chis, color=color, linestyle='--', marker=None,
                            alpha=0.8, zorder=2)

    # === SNR vs Chi scatter axis setup (no title) ===
    ax_scatter.set_xlabel("SNR")
    ax_scatter.set_ylabel(r"$\chi$")
    ax_scatter.set_xscale('log')
    ax_scatter.set_xlim(3, 100)
    ax_scatter.set_ylim(0, 1)
    ax_scatter.grid(True, linestyle='--', alpha=0.6)

    # Chi-type legend handles (chi_BL instead of chi_2016)
    chi_bl_handle = Line2D([0], [0], marker='o', color='k',
                           label=r'$\chi_{BL}$ (Filled)',
                           linestyle='None', markersize=8, markerfacecolor='k')
    chi_RCR_handle = Line2D([0], [0], marker='o', color='k',
                            label=r'$\chi_{RCR}$ (Outline)',
                            linestyle='None', markersize=8,
                            markerfacecolor='none', markeredgecolor='k')

    station_handles = list(legend_handles_for_fig.values())

    # Station legend in top-right
    if station_handles:
        leg_stations = ax_scatter.legend(handles=station_handles, loc='upper right',
                                         title="Stations")
        ax_scatter.add_artist(leg_stations)

    # Chi legend in bottom-right
    ax_scatter.legend(handles=[chi_bl_handle, chi_RCR_handle], loc='lower right',
                      title=r"$\chi$ Type")

    # === Polar axis (no title) ===
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_rlabel_position(22.5)
    ax_polar.set_rlim(0, 90)
    ax_polar.set_rticks(np.arange(0, 91, 30))
    ax_polar.grid(True, linestyle='--', alpha=0.5)

    # === Text box ===
    ax_text_box.axis('off')
    ax_text_box.text(0.01, 0.95, "\n".join(text_info_lines), ha='left', va='top',
                     fontsize=9, family='monospace', linespacing=1.4,
                     bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.6))

    # --- Save ---
    master_filename = os.path.join(output_dir, f'master_event_{event_id}.png')
    try:
        with SectionTimer("Saving thesis master plot"):
            plt.savefig(master_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()
        return master_filename
    except Exception as e:
        ic(f"Error saving master plot {master_filename}: {e}")
        plt.close(fig)
        gc.collect()
    return None


# --- Batch master-plot driver ---
def plot_master_events_thesis(events_dict, base_output_dir, dataset_name):
    """
    Generate thesis master event plots for both passing and failing events,
    saved into separate subfolders.
    """
    with SectionTimer(f"Thesis master event batch for {dataset_name}"):
        master_folder_base = os.path.join(base_output_dir, f"{dataset_name}_thesis_master_plots")
        pass_folder = os.path.join(master_folder_base, "pass_cuts")
        fail_folder = os.path.join(master_folder_base, "fail_cuts")
        os.makedirs(pass_folder, exist_ok=True)
        os.makedirs(fail_folder, exist_ok=True)

        passing_items = [(eid, ed) for eid, ed in events_dict.items()
                         if isinstance(ed, dict) and ed.get('passes_analysis_cuts', False)]
        failing_items = [(eid, ed) for eid, ed in events_dict.items()
                         if isinstance(ed, dict) and not ed.get('passes_analysis_cuts', False)]

        ic(f"Thesis master plots: {len(passing_items)} passing, {len(failing_items)} failing")

        for idx, (event_id, event_details) in enumerate(passing_items):
            _progress(idx, len(passing_items), "Thesis master (pass)")
            plot_single_master_event_thesis(event_id, event_details, pass_folder, dataset_name)

        for idx, (event_id, event_details) in enumerate(failing_items):
            _progress(idx, len(failing_items), "Thesis master (fail)")
            plot_single_master_event_thesis(event_id, event_details, fail_folder, dataset_name)

        ic(f"Thesis master plots complete for {dataset_name}.")


# --- Main Script ---
if __name__ == '__main__':
    ic.enable()
    import configparser

    config = configparser.ConfigParser()
    config_path = os.path.join('HRAStationDataAnalysis', 'config.ini')
    if not os.path.exists(config_path):
        config_path = 'config.ini'
    if not os.path.exists(config_path):
        ic(f"CRITICAL: config.ini not found.")
        exit()

    config.read(config_path)
    date_of_data = config['PARAMETERS']['date']
    date_of_coincidence = config['PARAMETERS']['date_coincidence']
    date_of_process = config['PARAMETERS']['date_processing']
    base_processed_data_dir = os.path.join("HRAStationDataAnalysis", "StationData", "processedNumpyData")
    processed_data_dir_for_date = os.path.join(base_processed_data_dir, date_of_data)

    # --- Locate data file (highest priority first) ---
    prefixes = [
        f"{date_of_coincidence}_CoincidenceDatetimes_passing_cuts_passing_cuts",
        f"{date_of_coincidence}_CoincidenceDatetimes_passing_cuts",
        f"{date_of_coincidence}_CoincidenceDatetimes",
    ]
    suffixes = [
        "with_all_params_recalcZenAzi_calcPol.pkl",
        "with_all_params_recalcZenAzi.pkl",
        "with_all_params.pkl",
    ]
    candidates = [os.path.join(processed_data_dir_for_date, f"{p}_{s}")
                  for p in prefixes for s in suffixes]

    ic("File search order:")
    for c in candidates:
        ic(f"  -> {os.path.basename(c)}")

    chosen_path = None
    for c in candidates:
        if os.path.exists(c):
            chosen_path = c
            break

    if chosen_path is None:
        ic("Error: No candidate data file found. Cannot proceed.")
        exit()

    ic(f"Using data file: {chosen_path}")
    events_data_dict = _load_pickle(chosen_path)
    if events_data_dict is None:
        ic(f"Could not load data from: {chosen_path}.")
        exit()

    dataset_name = "CoincidenceEvents"
    dataset_plot_suffix = f"CoincidenceEvents_{date_of_process}"
    output_plot_basedir = os.path.join("HRAStationDataAnalysis", "plots")

    is_passing_cuts_dataset = isinstance(chosen_path, str) and "passing_cuts" in chosen_path
    if is_passing_cuts_dataset:
        output_plot_basedir = os.path.join(output_plot_basedir, "passing_cuts")
    os.makedirs(output_plot_basedir, exist_ok=True)

    specific_dataset_plot_dir = os.path.join(output_plot_basedir, dataset_plot_suffix)
    os.makedirs(specific_dataset_plot_dir, exist_ok=True)

    # --- Apply analysis cuts (same logic as C03) ---
    if is_passing_cuts_dataset:
        with SectionTimer("Apply time cut to passing_cuts events"):
            time_cut_results = check_time_cut(events_data_dict, time_threshold_hours=24.0)

        num_passing_overall = 0
        num_failing_overall = 0
        for event_id, event_details in events_data_dict.items():
            if isinstance(event_details, dict):
                cut_results = event_details.get('cut_results', {})
                chi_cut_passed = cut_results.get('chi_cut_passed', True)
                angle_cut_passed = cut_results.get('angle_cut_passed', True)
                time_cut_passed = time_cut_results.get(event_id, False)
                fft_cut_passed = check_fft_cut(event_details, event_id)
                event_details['cut_results'] = {
                    'chi_cut_passed': chi_cut_passed,
                    'angle_cut_passed': angle_cut_passed,
                    'time_cut_passed': time_cut_passed,
                    'fft_cut_passed': fft_cut_passed,
                }
                event_details['passes_analysis_cuts'] = all(event_details['cut_results'].values())
                if event_details['passes_analysis_cuts']:
                    num_passing_overall += 1
                else:
                    num_failing_overall += 1
            else:
                num_failing_overall += 1
    else:
        # Apply chi/angle, then time, then FFT
        num_passing_overall = 0
        num_failing_overall = 0
        total_events = len(events_data_dict)

        with SectionTimer("Apply chi/angle cuts"):
            for loop_idx, (event_id, event_details) in enumerate(events_data_dict.items()):
                _progress(loop_idx, total_events, "Event cuts")
                if isinstance(event_details, dict):
                    chi_cut_passed = check_chi_cut(event_details)
                    angle_cut_passed = True  # Angle cut disabled (matching C03)
                    event_details['cut_results'] = {
                        'chi_cut_passed': chi_cut_passed,
                        'angle_cut_passed': angle_cut_passed,
                        'time_cut_passed': True,
                        'fft_cut_passed': True,
                    }
                    event_details['passes_analysis_cuts'] = chi_cut_passed and angle_cut_passed

        events_passing_chi_angle = {
            eid: ed for eid, ed in events_data_dict.items()
            if isinstance(ed, dict) and
               ed.get('cut_results', {}).get('chi_cut_passed', False) and
               ed.get('cut_results', {}).get('angle_cut_passed', False)
        }

        if events_passing_chi_angle:
            with SectionTimer("Apply time cut"):
                time_cut_results = check_time_cut(events_passing_chi_angle, time_threshold_hours=24.0)
        else:
            time_cut_results = {}

        for event_id, event_details in events_data_dict.items():
            if isinstance(event_details, dict) and 'cut_results' in event_details:
                event_details['cut_results']['time_cut_passed'] = time_cut_results.get(event_id, False)
                if (event_details['cut_results']['chi_cut_passed'] and
                        event_details['cut_results']['angle_cut_passed'] and
                        event_details['cut_results']['time_cut_passed']):
                    event_details['cut_results']['fft_cut_passed'] = check_fft_cut(event_details, event_id)
                else:
                    event_details['cut_results']['fft_cut_passed'] = False
                event_details['passes_analysis_cuts'] = all(event_details['cut_results'].values())
                if event_details['passes_analysis_cuts']:
                    num_passing_overall += 1
                else:
                    num_failing_overall += 1
            else:
                num_failing_overall += 1

    ic(f"Cuts applied: {num_passing_overall} passed, {num_failing_overall} failed")

    # --- Generate thesis master plots ---
    with SectionTimer("Thesis master plots"):
        plot_master_events_thesis(events_data_dict, specific_dataset_plot_dir, dataset_name)

    ic("Thesis coincidence plotting complete.")
