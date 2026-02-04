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
import itertools # For combinations in angle cut
import time
from typing import Dict, List, Optional, Set

from templateCrossCorr import DEFAULT_TRACE_SAMPLING_HZ, get_xcorr_for_channel

# --- Lightweight Timing + Progress Helpers ---
PROGRESS_EVERY = 50  # How often to print loop progress

class SectionTimer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        ic(f"⏳ {self.name} ...")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0 if self.t0 is not None else float('nan')
        ic(f"✅ {self.name} done in {dt:.2f}s")


def _progress(idx: int, total: int, label: str):
    """Periodic progress printing for long loops."""
    if total <= 0:
        return
    if idx == 0 or (idx + 1) % PROGRESS_EVERY == 0 or (idx + 1) == total:
        pct = 100.0 * (idx + 1) / total
        ic(f"[{label}] {idx + 1}/{total} ({pct:.1f}%)")

# --- Trace preparation helpers ---
NUM_TRACE_CHANNELS = 4  # Default number of waveform channels stored per trigger


def _prepare_trace_array(trace_obj) -> Optional[np.ndarray]:
    """Convert raw waveform data into a 1D float array, cleaning NaNs."""
    if trace_obj is None:
        return None
    arr = np.asarray(trace_obj, dtype=float)
    if arr.ndim == 0:
        return None
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        return None
    if not np.any(np.isfinite(arr)):
        return None
    if np.any(~np.isfinite(arr)):
        arr = np.nan_to_num(arr)
    return arr


def _iter_channel_traces(trace_channels, max_channels: int = NUM_TRACE_CHANNELS):
    """Yield (channel_idx, trace_array_like) pairs for a trigger."""
    if trace_channels is None:
        return
    try:
        sequence = list(trace_channels)
    except TypeError:
        return
    if max_channels is not None and max_channels > 0:
        sequence = sequence[:max_channels]
    for idx, channel_trace in enumerate(sequence):
        yield idx, channel_trace


def _extract_snr_value(station_data, trigger_idx: int) -> float:
    """Return SNR value for a given trigger index or NaN if unavailable."""
    snr_list = station_data.get("SNR", []) if isinstance(station_data, dict) else []
    if not isinstance(snr_list, (list, tuple, np.ndarray)):
        return float("nan")
    if trigger_idx is None or trigger_idx >= len(snr_list):
        return float("nan")
    snr_val = snr_list[trigger_idx]
    try:
        snr_float = float(snr_val)
    except (TypeError, ValueError):
        return float("nan")
    return snr_float if np.isfinite(snr_float) else float("nan")


def _find_loudest_trace(event_details) -> Optional[Dict[str, object]]:
    """Identify the single channel trace with the largest absolute amplitude."""
    if not isinstance(event_details, dict):
        return None
    stations = event_details.get("stations", {})
    if not isinstance(stations, dict):
        return None

    loudest = None
    for station_id_str, station_data in stations.items():
        if not isinstance(station_data, dict):
            continue
        traces_list = station_data.get("Traces", [])
        if not isinstance(traces_list, (list, tuple, np.ndarray)):
            continue
        for trigger_idx, trace_channels in enumerate(traces_list):
            for channel_idx, channel_trace in _iter_channel_traces(trace_channels):
                trace_arr = _prepare_trace_array(channel_trace)
                if trace_arr is None:
                    continue
                try:
                    amplitude = float(np.nanmax(np.abs(trace_arr)))
                except (ValueError, TypeError):
                    continue
                if not np.isfinite(amplitude):
                    continue
                if loudest is None or amplitude > loudest["amplitude"]:
                    loudest = {
                        "trace": trace_arr,
                        "amplitude": amplitude,
                        "station_id": station_id_str,
                        "trigger_idx": trigger_idx,
                        "channel_idx": channel_idx,
                        "snr": _extract_snr_value(station_data, trigger_idx),
                        "trace_length": int(trace_arr.size),
                    }
    return loudest

# --- Helper for Loading Data ---
def _load_pickle(filepath):
    """Loads data from a pickle file."""
    if os.path.exists(filepath):
        try:
            with SectionTimer(f"Loading pickle '{os.path.basename(filepath)}'"):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            # Basic size info if possible
            try:
                fsize_mb = os.path.getsize(filepath) / (1024 * 1024)
                ic(f"Loaded pickle size: {fsize_mb:.2f} MB")
            except Exception:
                pass
            return data
        except Exception as e:
            ic(f"Error loading pickle file {filepath}: {e}")
    return None

# --- Coincidence Event Cut Functions ---
def check_chi_cut(event_details, high_chi_threshold=0.6, low_chi_threshold=0.5, min_triggers_passing=2):
    """
    Checks if a coincidence event passes the Chi cut.
    A coincidence passes if at least 'min_triggers_passing' of its constituent
    station triggers have a ChiRCR or Chi2016 value above 'chi_threshold'.
    """
    high_thresh_passed = False
    low_thresh_passed = []
    if not isinstance(event_details, dict):
        return False

    for station_id_str, station_triggers_data in event_details.get("stations", {}).items():
        if not isinstance(station_triggers_data, dict):
            continue
            
        num_triggers_in_station = len(station_triggers_data.get('SNR', []))
        chi_rcr_list = station_triggers_data.get('ChiRCR', [])
        chi_2016_list = station_triggers_data.get('Chi2016', [])


        for i in range(num_triggers_in_station):
            if chi_rcr_list[i] >= high_chi_threshold or chi_2016_list[i] >= high_chi_threshold:
                if high_thresh_passed == True:
                    low_thresh_passed.append(True)
                else:
                    high_thresh_passed = True
            elif chi_rcr_list[i] >= low_chi_threshold or chi_2016_list[i] >= low_chi_threshold:
                low_thresh_passed.append(True)

    if high_thresh_passed and np.sum(low_thresh_passed) >= (min_triggers_passing - 1):
        return True

    return False

def check_angle_cut(event_details, zenith_margin_deg=20.0, azimuth_margin_deg=45.0):
    """
    Checks if a coincidence event passes the Angle (Zenith/Azimuth) agreement cut.
    Zenith and Azimuth values from event_details are assumed to be in RADIANS.
    Margins are in DEGREES.
    """
    if not isinstance(event_details, dict):
        return False

    valid_angles_deg = [] # List of (zenith_deg, azimuth_deg) tuples
    for station_id_str, station_triggers_data in event_details.get("stations", {}).items():
        if not isinstance(station_triggers_data, dict):
            continue
        
        num_triggers_in_station = len(station_triggers_data.get('SNR', []))
        zen_list_rad = station_triggers_data.get('Zen', []) 
        azi_list_rad = station_triggers_data.get('Azi', [])

        for i in range(num_triggers_in_station):
            zen_rad, azi_rad = None, None
            if i < len(zen_list_rad) and zen_list_rad[i] is not None and not np.isnan(zen_list_rad[i]):
                zen_rad = zen_list_rad[i]
            if i < len(azi_list_rad) and azi_list_rad[i] is not None and not np.isnan(azi_list_rad[i]):
                azi_rad = azi_list_rad[i]

            if zen_rad is not None and azi_rad is not None:
                # Convert to degrees for comparison logic
                valid_angles_deg.append((np.degrees(zen_rad), np.degrees(azi_rad)))

    if len(valid_angles_deg) < 2:
        return False # Cannot find a pair if less than 2 valid angle sets

    for pair in itertools.combinations(valid_angles_deg, 2):
        (zen1_deg, azi1_deg), (zen2_deg, azi2_deg) = pair # These are now in degrees
        
        delta_zen = abs(zen1_deg - zen2_deg)
        if delta_zen > zenith_margin_deg:
            continue

        delta_azi = abs(azi1_deg - azi2_deg)
        if delta_azi > 180.0: 
            delta_azi = 360.0 - delta_azi # Shortest angle
        
        if delta_azi <= azimuth_margin_deg:
            return True # Found a pair that agrees

    return False # No agreeing pair found

def check_fft_cut(event_details, event_id=None, max_fraction_threshold=0.1, min_failing_traces=2, debug_print=True):
    """
    Checks if a coincidence event passes the FFT cut.
    For station 18 triggers, calculates the fraction that the largest FFT value
    is of the total FFT for each channel, and fails the event if enough traces
    exceed the threshold.
    
    Args:
        event_details: Event data dictionary
        event_id: Event ID for logging purposes (optional)
        max_fraction_threshold: Maximum allowed fraction for largest FFT value (default 0.1)
        min_failing_traces: Minimum number of traces above threshold to fail event (default 2)
        debug_print: Whether to print debug information for each event (default True)
    
    Returns:
        bool: True if event passes FFT cut, False otherwise
    """
    if not isinstance(event_details, dict):
        return False
    
    if event_id is None:
        event_id = event_details.get('event_id', 'Unknown')
    
    traces_above_threshold = 0
    station_18_found = False
    
    # Check if station 18 is present (check both string and integer keys)
    stations_data = event_details.get("stations", {})
    station_18_data = None
    
    # Try both "18" (string) and 18 (integer) as keys
    if "18" in stations_data:
        station_18_data = stations_data["18"]
        station_18_found = True
    elif 18 in stations_data:
        station_18_data = stations_data[18]
        station_18_found = True
    else:
        if debug_print:
            print(f"Event {event_id}: Station 18 not found - FFT cut not applicable")
        return True  # Pass if station 18 not present
    
    # Get traces for station 18
    traces_list = station_18_data.get("Traces", [])
    if not traces_list:
        if debug_print:
            print(f"Event {event_id}: No traces found for station 18")
        return True  # Pass if no traces
    
    if debug_print:
        print(f"Event {event_id}: Station 18 FFT Analysis")
        print(f"  Found {len(traces_list)} triggers in station 18")
    
    # Analyze each trigger in station 18
    for trigger_idx, traces_for_trigger in enumerate(traces_list):
        if traces_for_trigger is None:
            continue
            
        # Ensure we have 4 channels, pad with None if needed
        padded_traces = (list(traces_for_trigger) + [None] * 4)[:4]
        
        channel_fractions = []
        
        for ch_idx, trace_data in enumerate(padded_traces):
            if trace_data is None or not hasattr(trace_data, "__len__") or len(trace_data) == 0:
                channel_fractions.append(np.nan)
                continue
                
            trace_array = np.asarray(trace_data)
            if len(trace_array) <= 1:
                channel_fractions.append(np.nan)
                continue
                
            # Calculate FFT
            sampling_rate_hz = 2e9  # 2 GHz sampling rate
            try:
                fft_spectrum = np.abs(fft.time2freq(trace_array, sampling_rate_hz))
                
                # Set DC component to 0
                if len(fft_spectrum) > 0:
                    fft_spectrum[0] = 0
                
                # Calculate total FFT power and max value
                total_fft = np.sum(fft_spectrum)
                max_fft = np.max(fft_spectrum)
                
                # Calculate fraction
                if total_fft > 0:
                    fraction = max_fft / total_fft
                else:
                    fraction = np.nan
                    
                channel_fractions.append(fraction)
                
            except Exception as e:
                if debug_print:
                    print(f"  Error calculating FFT for channel {ch_idx}: {e}")
                channel_fractions.append(np.nan)
        
        # Check if any channel in this trace exceeds threshold
        valid_fractions = [f for f in channel_fractions if not np.isnan(f)]
        for fraction in valid_fractions:
            if fraction > max_fraction_threshold:
                traces_above_threshold += 1
        
        if debug_print:
            fraction_str = ", ".join([f"{f:.4f}" if not np.isnan(f) else "NaN" for f in channel_fractions])
            print(f"  Trigger {trigger_idx + 1}: Channel fractions = [{fraction_str}]")
            exceeds_threshold = valid_fractions and any(f > max_fraction_threshold for f in valid_fractions)
            print(f"  Trigger {trigger_idx + 1}: Exceeds threshold ({max_fraction_threshold}): {exceeds_threshold}")
    
    # Determine if event passes FFT cut
    passes_fft_cut = traces_above_threshold < min_failing_traces
    
    if debug_print:
        print(f"Event {event_id}: Traces above threshold: {traces_above_threshold}, Passes FFT cut: {passes_fft_cut}")
    
    return passes_fft_cut

def check_time_cut(events_dict, time_threshold_hours=24.0):
    """
    Checks if coincidence events are isolated in time (not within time_threshold_hours of each other).
    Returns a dictionary with event IDs as keys and boolean values indicating if they pass the time cut.
    
    Args:
        events_dict: Dictionary of all events with event_id as keys
        time_threshold_hours: Time threshold in hours (default 1.0 hour)
    
    Returns:
        dict: {event_id: bool} indicating which events pass the time cut
    """
    with SectionTimer("Time cut computation"):
        time_threshold_seconds = time_threshold_hours * 3600  # Convert to seconds
        
        # Extract event times and sort by time
        event_times_list = []
        for event_id, event_details in events_dict.items():
            if isinstance(event_details, dict) and "datetime" in event_details:
                event_time = event_details["datetime"]
                if event_time is not None:
                    event_times_list.append((event_time, event_id))
        ic(f"Found {len(event_times_list)} events with valid times")
        
        # Sort by time (events are already time ordered, but this ensures it)
        event_times_list.sort()
        
        # Initialize all events with valid times as passing
        time_cut_results = {event_id: True for _, event_id in event_times_list}
        
        # Check each event only against subsequent events (more efficient)
        for i in range(len(event_times_list)):
            current_time, current_event_id = event_times_list[i]
            
            # Only check forward in time
            for j in range(i + 1, len(event_times_list)):
                other_time, other_event_id = event_times_list[j]
                
                time_diff = other_time - current_time  # Always positive since sorted
                
                if time_diff <= time_threshold_seconds:
                    # Both events fail the time cut
                    time_cut_results[current_event_id] = False
                    time_cut_results[other_event_id] = False
                else:
                    # Since events are sorted, no need to check further
                    break
        
        # Events without valid datetime are considered to fail the time cut
        for event_id in events_dict.keys():
            if event_id not in time_cut_results:
                time_cut_results[event_id] = False
        ic(f"Time cut map size: {len(time_cut_results)}")
        return time_cut_results

def check_coincidence_cuts(event_details, event_id=None, time_cut_result=True):
    """
    Main function to check if a coincidence event passes all defined analysis cuts.
    Returns a dictionary with the pass/fail status for each cut.
    
    Args:
        event_details: Details for a single event
        event_id: Event ID for logging purposes (optional)
        time_cut_result: Boolean result from the time cut (calculated externally)
    """
    results = {}
    results['time_cut_passed'] = time_cut_result
    results['chi_cut_passed'] = check_chi_cut(event_details)
    results['angle_cut_passed'] = check_angle_cut(event_details)
    results['fft_cut_passed'] = check_fft_cut(event_details, event_id)
    
    # Add more cut checks here in the future:
    # results['another_cut_passed'] = check_another_cut(event_details)
    
    return results


# --- Plotting Function 1: SNR vs Chi Parameters ---
def plot_snr_vs_chi(events_dict, output_dir, dataset_name):
    with SectionTimer(f"Plot SNR vs Chi for {dataset_name}"):
        ic(f"Generating SNR vs Chi plots for {dataset_name}")
        os.makedirs(output_dir, exist_ok=True)

    chi_params = ['ChiRCR', 'Chi2016', 'ChiBad']
    
    # Separate events by cut status
    passing_events = {k: v for k, v in events_dict.items() if v.get('passes_analysis_cuts', False)}
    failing_events = {k: v for k, v in events_dict.items() if not v.get('passes_analysis_cuts', False)}
    
    # Create plots for both passed and failed events
    for status, events_subset in [("passed", passing_events), ("failed", failing_events)]:
        if not events_subset:
            ic(f"No {status} events in {dataset_name} to plot for SNR vs Chi.")
            continue
        start_subset = time.perf_counter()
        fig, axs = plt.subplots(len(chi_params), 1, figsize=(12, 6 * len(chi_params)), sharex=True)
        if len(chi_params) == 1: axs = [axs]

        num_events = len(events_subset)
        colors_cmap = cm.get_cmap('jet', num_events if num_events > 1 else 2)

        for i, (event_id, event_data) in enumerate(events_subset.items()):
            _progress(i, num_events, f"SNR-chi {status}")
            event_color = colors_cmap(i)
            event_plot_data_collected = []
            
            for station_id_str, station_triggers_data in event_data.get("stations", {}).items():
                try: station_id_int = int(station_id_str)
                except ValueError: ic(f"Warning: Could not convert station ID '{station_id_str}' to int for event {event_id}. Skipping station."); continue

                snr_list = station_triggers_data.get('SNR', [])
                num_triggers_for_station = len(snr_list)
                if num_triggers_for_station == 0: continue

                chi_data_for_station = {}
                for cp in chi_params:
                    cp_list = station_triggers_data.get(cp, [])
                    chi_data_for_station[cp] = (cp_list + [np.nan] * num_triggers_for_station)[:num_triggers_for_station]

                for trigger_idx in range(num_triggers_for_station):
                    snr_val = snr_list[trigger_idx]
                    if snr_val is None or np.isnan(snr_val): continue

                    point_data = {'station_id': station_id_int, 'trigger_idx_orig': trigger_idx, 'SNR': snr_val}
                    for cp in chi_params: point_data[cp] = chi_data_for_station[cp][trigger_idx]
                    event_plot_data_collected.append(point_data)
            
            if not event_plot_data_collected: continue
            event_plot_data_collected.sort(key=lambda p: (p['station_id'], p['trigger_idx_orig']))
            snrs_event_sorted = np.array([p['SNR'] for p in event_plot_data_collected])
            
            for plot_idx, chi_param_name in enumerate(chi_params):
                chi_values_event_sorted = np.array([p[chi_param_name] for p in event_plot_data_collected])
                for point_d in event_plot_data_collected:
                    if point_d['SNR'] is not None and not np.isnan(point_d['SNR']) and \
                       point_d[chi_param_name] is not None and not np.isnan(point_d[chi_param_name]):
                        axs[plot_idx].scatter(point_d['SNR'], point_d[chi_param_name], color=event_color, s=30, alpha=0.6, zorder=5)

                valid_line_indices = ~np.isnan(snrs_event_sorted) & ~np.isnan(chi_values_event_sorted)
                if np.sum(valid_line_indices) > 1:
                        axs[plot_idx].plot(snrs_event_sorted[valid_line_indices], 
                                           chi_values_event_sorted[valid_line_indices], 
                                           linestyle='-', color=event_color, alpha=0.4, marker=None,
                                           label=f"Event {event_id}" if plot_idx == 0 and i < 15 else None)

        for plot_idx, chi_param_name in enumerate(chi_params):
            axs[plot_idx].set_ylabel(chi_param_name); axs[plot_idx].set_title(f'SNR vs {chi_param_name}')
            axs[plot_idx].grid(True, linestyle='--', alpha=0.6); axs[plot_idx].set_xscale('log')
            axs[plot_idx].set_xlim(3, 100); axs[plot_idx].set_ylim(0, 1)
        
        axs[-1].set_xlabel('SNR')
        handles, labels = axs[0].get_legend_handles_labels()
        if num_events > 0 and num_events <=15 and handles: 
            fig.legend(handles, labels, loc='center right', title="Events", bbox_to_anchor=(1.12, 0.5), fontsize='small')

        plt.suptitle(f'SNR vs Chi Parameters for {dataset_name} - {status.title()} Events', fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.88 if num_events <=15 and handles else 0.98 , 0.96])
        
        plot_filename = os.path.join(output_dir, f"{dataset_name}_snr_vs_chi_params_{status}.png")
        with SectionTimer(f"Saving SNR vs Chi plot ({status})"):
            plt.savefig(plot_filename, bbox_inches='tight')
        ic(f"Saved SNR vs Chi plot: {plot_filename} (subset took {(time.perf_counter()-start_subset):.2f}s)")
        plt.close(fig); gc.collect()


def compute_event_self_similarity(events_dict, sampling_rate_hz=DEFAULT_TRACE_SAMPLING_HZ) -> Dict:
    """Compute self-template match metrics for each event."""
    results: Dict = {}
    if not isinstance(events_dict, dict):
        return results

    try:
        sampling_rate = float(sampling_rate_hz)
    except (TypeError, ValueError):
        sampling_rate = float(DEFAULT_TRACE_SAMPLING_HZ)
    if sampling_rate <= 0:
        sampling_rate = float(DEFAULT_TRACE_SAMPLING_HZ)

    for event_id, event_details in events_dict.items():
        if not isinstance(event_details, dict):
            continue

        template_info = _find_loudest_trace(event_details)
        if template_info is None:
            continue

        template_trace = template_info["trace"]
        template_station_key = str(template_info["station_id"])

        best_match = None
        pairs_evaluated = 0
        stations_compared: Set[str] = set()
        station_best_map: Dict[str, Dict[str, object]] = {}

        stations = event_details.get("stations", {})
        if not isinstance(stations, dict):
            continue

        for station_key, station_data in stations.items():
            station_key_str = str(station_key)
            if station_key_str == template_station_key:
                # Skip re-matching the template station to itself
                continue
            if not isinstance(station_data, dict):
                continue

            traces_list = station_data.get("Traces", [])
            if not isinstance(traces_list, (list, tuple, np.ndarray)):
                continue

            for trigger_idx, trace_channels in enumerate(traces_list):
                for channel_idx, channel_trace in _iter_channel_traces(trace_channels):
                    trace_arr = _prepare_trace_array(channel_trace)
                    if trace_arr is None:
                        continue

                    xcorr_value = get_xcorr_for_channel(
                        trace_arr,
                        template_trace,
                        sampling_rate,
                        sampling_rate,
                    )
                    if isinstance(xcorr_value, dict):
                        xcorr_value = xcorr_value.get("xcorr", 0.0)
                    try:
                        xcorr_float = float(xcorr_value)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(xcorr_float):
                        continue

                    pairs_evaluated += 1
                    stations_compared.add(station_key_str)
                    abs_xcorr = abs(xcorr_float)
                    candidate = {
                        "station_id": station_key_str,
                        "trigger_idx": trigger_idx,
                        "channel_idx": channel_idx,
                        "xcorr": xcorr_float,
                        "abs_xcorr": abs_xcorr,
                        "snr": _extract_snr_value(station_data, trigger_idx),
                    }
                    if best_match is None or abs_xcorr > best_match["abs_xcorr"]:
                        best_match = candidate
                    existing = station_best_map.get(station_key_str)
                    if existing is None or abs_xcorr > existing["abs_xcorr"]:
                        station_best_map[station_key_str] = dict(candidate)

        if best_match is None or pairs_evaluated == 0:
            continue

        derived_bucket = event_details.get("derived_metrics")
        if not isinstance(derived_bucket, dict):
            derived_bucket = {}
            event_details["derived_metrics"] = derived_bucket

        station_matches: Dict[str, Dict[str, object]] = {}
        for station_key_str, candidate in station_best_map.items():
            station_matches[station_key_str] = {
                "chi": candidate["abs_xcorr"],
                "xcorr": candidate["xcorr"],
                "snr": candidate["snr"],
                "trigger_idx": candidate["trigger_idx"],
                "channel_idx": candidate["channel_idx"],
            }

        template_snr_val = template_info.get("snr")
        station_matches[template_station_key] = {
            "chi": 1.0,
            "xcorr": 1.0,
            "snr": template_snr_val,
            "trigger_idx": template_info["trigger_idx"],
            "channel_idx": template_info["channel_idx"],
        }

        snr_values: List[float] = []
        if template_snr_val is not None and np.isfinite(template_snr_val):
            snr_values.append(float(template_snr_val))
        match_snr_val = best_match.get("snr") if best_match is not None else None
        if match_snr_val is not None and np.isfinite(match_snr_val):
            snr_values.append(float(match_snr_val))
        avg_pair_snr = float(np.mean(snr_values)) if snr_values else float("nan")

        summary = {
            "sampling_rate_hz": sampling_rate,
            "template_station": template_station_key,
            "template_trigger_idx": template_info["trigger_idx"],
            "template_channel_idx": template_info["channel_idx"],
            "template_amplitude": template_info["amplitude"],
            "template_snr": template_info["snr"],
            "template_trace_length": template_info["trace_length"],
            "match_station": best_match["station_id"],
            "match_trigger_idx": best_match["trigger_idx"],
            "match_channel_idx": best_match["channel_idx"],
            "match_snr": best_match["snr"],
            "best_match_chi": best_match["xcorr"],
            "best_match_abs_chi": best_match["abs_xcorr"],
            "avg_pair_snr": avg_pair_snr,
            "pairs_evaluated": pairs_evaluated,
            "stations_compared": sorted(stations_compared),
            "station_matches": station_matches,
        }
        derived_bucket["self_match"] = summary
        results[event_id] = summary

    ic(f"Computed self-match metrics for {len(results)}/{len(events_dict)} events.")
    return results


def plot_self_similarity_snr_vs_chi(
    events_dict,
    output_dir,
    dataset_name,
    sampling_rate_hz=DEFAULT_TRACE_SAMPLING_HZ,
    only_passing=False,
    highlight_label="BL-RCR",
):
    """Plot event-level SNR vs chi using self-correlation metrics."""
    metrics = compute_event_self_similarity(events_dict, sampling_rate_hz=sampling_rate_hz)
    if not metrics:
        ic(f"No self-match metrics computed for {dataset_name}; skipping self-similarity plot.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    derived_metrics = event_details.get("derived_metrics")
    if not isinstance(derived_metrics, dict):
        derived_metrics = {}
        event_details["derived_metrics"] = derived_metrics
    self_match_summary = derived_metrics.get("self_match")
    if self_match_summary is None:
        compute_event_self_similarity({event_id: event_details})
        self_match_summary = derived_metrics.get("self_match")

    pass_snrs, pass_chis = [], []
    fail_snrs, fail_chis = [], []
    unknown_snrs, unknown_chis = [], []
    highlight_ids = {11230, 11243, "11230", "11243"}
    highlight_points = []

    for event_id, summary in metrics.items():
        event_entry = events_dict.get(event_id)
        if event_entry is None and isinstance(event_id, str) and event_id.isdigit():
            event_entry = events_dict.get(int(event_id))
        if event_entry is None and not isinstance(event_id, str):
            event_entry = events_dict.get(str(event_id))

        avg_pair_snr = summary.get("avg_pair_snr")
        match_snr = summary.get("match_snr")
        snr_value = avg_pair_snr
        if snr_value is None or not np.isfinite(snr_value) or snr_value <= 0:
            snr_value = match_snr
        chi_val = summary.get("best_match_abs_chi")
        if snr_value is None or not np.isfinite(snr_value) or snr_value <= 0:
            continue
        if chi_val is None or not np.isfinite(chi_val):
            continue

        passes_cuts = False
        if isinstance(event_entry, dict):
            passes_cuts = bool(event_entry.get('passes_analysis_cuts', False))
        elif event_entry is None:
            passes_cuts = None

        if only_passing and passes_cuts is not True:
            continue

        if passes_cuts is True:
            pass_snrs.append(snr_value)
            pass_chis.append(chi_val)
        elif passes_cuts is False:
            fail_snrs.append(snr_value)
            fail_chis.append(chi_val)
        else:
            unknown_snrs.append(snr_value)
            unknown_chis.append(chi_val)

        if event_id in highlight_ids or str(event_id) in highlight_ids:
            highlight_points.append((snr_value, chi_val))

    if not (pass_snrs or fail_snrs or unknown_snrs):
        ic(f"Self-match metrics for {dataset_name} contain no finite SNR/chi pairs.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    if pass_snrs:
        ax.scatter(pass_snrs, pass_chis, label='BL-BL', color='tab:green', alpha=0.75, s=55, edgecolors='k', linewidths=0.5)
    if fail_snrs:
        ax.scatter(fail_snrs, fail_chis, label='Failed cuts', color='tab:red', alpha=0.65, s=55, edgecolors='k', linewidths=0.5)
    if unknown_snrs:
        ax.scatter(unknown_snrs, unknown_chis, label='Unknown', color='tab:gray', alpha=0.6, s=50, edgecolors='k', linewidths=0.4)

    ax.set_xscale('log')
    ax.set_xlim(3, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Average SNR (template + partner)')
    ax.set_ylabel(r'Self-match $\chi$')
    ax.set_title(f'Self-match SNR vs $\chi$ for {dataset_name}')
    ax.grid(True, linestyle='--', alpha=0.6)

    if highlight_points:
        highlight_label_to_use = highlight_label
        if highlight_label_to_use in {"BL-BL", "BL-RCR", "Unknown", "Failed cuts"}:
            highlight_label_to_use = f"{highlight_label_to_use} candidate"
        hx, hy = zip(*highlight_points)
        ax.scatter(
            hx,
            hy,
            label=highlight_label_to_use,
            color='gold',
            edgecolors='black',
            marker='*',
            s=160,
            linewidths=1.1,
            zorder=6,
        )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='lower right')

    if only_passing:
        plot_filename = f"{dataset_name}_self_match_snr_vs_chi_passing_only.png"
    else:
        plot_filename = f"{dataset_name}_self_match_snr_vs_chi.png"
    plot_path = os.path.join(output_dir, plot_filename)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    ic(f"Saved self-match SNR vs Chi plot: {plot_path}")
    return plot_path


# --- Plotting Function 2: Parameter Histograms ---
def plot_parameter_histograms(events_dict, output_dir, dataset_name):
    with SectionTimer(f"Plot parameter histograms for {dataset_name}"):
        ic(f"Generating parameter histograms for {dataset_name} (with cut status)")
        os.makedirs(output_dir, exist_ok=True)
    params_to_histogram = ['SNR', 'ChiRCR', 'Chi2016', 'ChiBad', 'Zen', 'Azi', 'PolAngle'] # CHANGED: Added PolAngle
    param_values_all, param_values_passing_cuts, param_values_failing_cuts = defaultdict(list), defaultdict(list), defaultdict(list)
    for event_data in events_dict.values():
        passes_cuts = event_data.get('passes_analysis_cuts', False)
        for station_triggers in event_data.get("stations", {}).values():
            for param_name in params_to_histogram:
                for val in station_triggers.get(param_name, []):
                    if val is not None and not np.isnan(val):
                        # Convert angles to degrees for histogramming
                        val_to_plot = np.degrees(val) if param_name in ['Zen', 'Azi', 'PolAngle'] else val
                        param_values_all[param_name].append(val_to_plot)
                        if passes_cuts: param_values_passing_cuts[param_name].append(val_to_plot)
                        else: param_values_failing_cuts[param_name].append(val_to_plot)
    if not any(param_values_all.values()): ic(f"No valid data for histograms in {dataset_name}."); return
    num_hist_params = len(params_to_histogram); cols = 3 if num_hist_params > 4 else (2 if num_hist_params > 1 else 1)
    rows = (num_hist_params + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False); axs_flat = axs.flatten()
    for i, param_name in enumerate(params_to_histogram):
        ax = axs_flat[i]; ax.set_title(f'{param_name}'); ax.set_xlabel("Value"); ax.set_ylabel('Frequency')
        values_all, values_pass, values_fail = param_values_all.get(param_name,[]), param_values_passing_cuts.get(param_name,[]), param_values_failing_cuts.get(param_name,[])
        has_data = False
        
        # Set up bins - use log scale for SNR
        if param_name == 'SNR' and values_all:
            bins = np.logspace(np.log10(min(values_all)), np.log10(max(values_all)), 50)
        else:
            bins = 50
        
        # Plot Fail/Pass only (step histograms)
        if values_fail: ax.hist(values_fail, bins=bins, color='red', label='Fail Analysis Cuts', histtype='step', linewidth=2); has_data=True
        if values_pass: ax.hist(values_pass, bins=bins, color='black', linestyle='--', label='Pass Analysis Cuts', histtype='step', linewidth=2); has_data=True
        
        if has_data:
            if param_name in ['ChiRCR','Chi2016','ChiBad','SNR'] and any(v > 0 for v in values_all): ax.set_yscale('log'); ax.set_ylabel('Frequency (log scale)')
            if param_name in ['ChiRCR','Chi2016','ChiBad']: ax.set_xlim(0, 1)
            if param_name == 'SNR': ax.set_xlim(3, 100); ax.set_xscale('log')
            ax.grid(True, linestyle='--', alpha=0.6); ax.legend(fontsize='x-small')
        else: ax.text(0.5,0.5,f"No data for\n{param_name}",ha='center',va='center',transform=ax.transAxes); ax.set_title(f'{param_name}')
    for j in range(i + 1, len(axs_flat)): fig.delaxes(axs_flat[j])
    plt.suptitle(f'Parameter Histograms for {dataset_name} (by Cut Status)', fontsize=16); plt.tight_layout(rect=[0,0,1,0.96])
    plot_filename = os.path.join(output_dir, f"{dataset_name}_parameter_histograms_by_cut.png")
    with SectionTimer("Saving parameter histograms"):
        plt.savefig(plot_filename)
    ic(f"Saved: {plot_filename}")
    plt.close(fig); gc.collect()


# --- Plotting Function 3: Polar Plot (Zenith vs Azimuth) - Updated ---
def plot_polar_zen_azi(events_dict, output_dir, dataset_name, only_passing_cuts=False, specific_event_ids=None):
    """
    Generate polar Zenith vs Azimuth plot.
    
    Args:
        events_dict: Dictionary of all events
        output_dir: Directory to save plots
        dataset_name: Name of dataset for labeling
        only_passing_cuts: If True, plot only events that pass analysis cuts
        specific_event_ids: List of specific event IDs to plot. If provided, only these events will be plotted.
                           Example: [11230, 12345, 67890]
    """
    # Determine plot suffix based on filtering
    if specific_event_ids is not None:
        plot_suffix = f"_specific_events_{len(specific_event_ids)}"
        filter_description = f"Specific Events ({len(specific_event_ids)} events)"
        ic(f"Generating polar plot for specific events: {specific_event_ids}")
    elif only_passing_cuts:
        plot_suffix = "_passing_cuts"
        filter_description = "Events Passing All Cuts"
        ic(f"Generating polar plot for events passing all cuts")
    else:
        plot_suffix = ""
        filter_description = "All Events"
        ic(f"Generating polar plot for all events")

    with SectionTimer(f"Plot polar zen/azi for {dataset_name} - {filter_description}"):
        os.makedirs(output_dir, exist_ok=True)
        all_zen_rad_values, all_azi_rad_values = [], []
        plotted_event_count = 0

    # Filter events based on criteria
    events_to_plot = {}
    if specific_event_ids is not None:
        # Plot only specific events
        for event_id in specific_event_ids:
            if str(event_id) in events_dict:
                events_to_plot[str(event_id)] = events_dict[str(event_id)]
            elif event_id in events_dict:
                events_to_plot[event_id] = events_dict[event_id]
        ic(f"Found {len(events_to_plot)} out of {len(specific_event_ids)} requested specific events")
    elif only_passing_cuts:
        # Plot only events that pass cuts
        events_to_plot = {k: v for k, v in events_dict.items() 
                         if isinstance(v, dict) and v.get('passes_analysis_cuts', False)}
        ic(f"Found {len(events_to_plot)} events passing all cuts")
    else:
        # Plot all events
        events_to_plot = events_dict
        ic(f"Plotting all {len(events_to_plot)} events")

    # Extract angle data from filtered events and assign colors by event
    event_color_map = []  # List to store color index for each data point
    event_list = list(events_to_plot.keys())  # Get ordered list of event IDs
    
    for event_idx, (event_id, event_data) in enumerate(events_to_plot.items()):
        if not isinstance(event_data, dict):
            continue
            
        event_has_data = False
        for station_triggers in event_data.get("stations", {}).values():
            zen_list_rad = station_triggers.get('Zen', [])  # Assumed in RADIANS
            azi_list_rad = station_triggers.get('Azi', [])  # Assumed in RADIANS
            
            for k in range(len(zen_list_rad)):
                if k < len(azi_list_rad):
                    zen_r_val, azi_r_val = zen_list_rad[k], azi_list_rad[k]
                    if zen_r_val is not None and not np.isnan(zen_r_val) and \
                       azi_r_val is not None and not np.isnan(azi_r_val):
                        all_zen_rad_values.append(zen_r_val) 
                        all_azi_rad_values.append(azi_r_val)
                        event_color_map.append(event_idx)  # Assign same color index for all points from this event
                        event_has_data = True
        
        if event_has_data:
            plotted_event_count += 1

    if not all_azi_rad_values: 
        ic(f"No valid Zenith/Azimuth data to plot for {dataset_name} - {filter_description}.")
        return
    
    ic(f"Plotting {len(all_azi_rad_values)} data points from {plotted_event_count} events")
    
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    all_zen_deg_values = np.degrees(np.array(all_zen_rad_values))
    
    # Color points by event - each event gets a unique color
    scatter = ax.scatter(np.array(all_azi_rad_values),  # Azimuth (theta) in RADIANS
                         all_zen_deg_values,            # Zenith (r) in DEGREES
                         alpha=0.6, s=30, cmap='viridis', 
                         c=event_color_map)  # Color by event index
    
    # Set up polar plot
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)
    ax.set_rlim(0, 90)
    ax.set_rticks(np.arange(0, 91, 15))  # Zenith ticks in degrees
    
    # Create title with event count information
    title_text = f'Sky Plot: Zenith vs Azimuth for {dataset_name}\n'
    title_text += f'{filter_description} - {plotted_event_count} events, {len(all_azi_rad_values)} points\n'
    title_text += '(Zenith in degrees from center)'
    
    if specific_event_ids is not None:
        title_text += f'\nEvent IDs: {specific_event_ids}'
    
    ax.set_title(title_text, va='bottom', fontsize=12, pad=20)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add colorbar with event information
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Event Index', rotation=270, labelpad=15)
    
    # Set colorbar ticks to show event indices if not too many events
    if plotted_event_count <= 20:
        unique_event_indices = sorted(set(event_color_map))
        cbar.set_ticks(unique_event_indices)
        # Show event IDs on colorbar if specific events were requested
        if specific_event_ids is not None and len(specific_event_ids) <= 10:
            event_labels = [str(event_list[i]) for i in unique_event_indices if i < len(event_list)]
            if len(event_labels) == len(unique_event_indices):
                cbar.set_ticklabels(event_labels)
    
    # Generate filename with appropriate suffix
    plot_filename = os.path.join(output_dir, f"{dataset_name}_polar_zen_azi{plot_suffix}.png")
    
    with SectionTimer("Saving polar plot"):
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    
    ic(f"Saved polar Zenith vs Azimuth plot: {plot_filename}")
    plt.close(fig)
    gc.collect()


# --- Helper Function: Plot Single Master Event ---
def plot_single_master_event(event_id, event_details, output_dir, dataset_name, title_suffix=""):
    """
    Plots a single master event with all details.
    
    Args:
        event_id: ID of the event
        event_details: Event data dictionary
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset for labeling
        title_suffix: Additional text to add to the title
        
    Returns:
        str: Path to the saved plot file, or None if failed
    """
    with SectionTimer(f"Master plot for event {event_id}"):
        if not isinstance(event_details, dict):
            ic(f"Warning: Event {event_id} data is not a dictionary. Skipping master plot.")
            return None

    os.makedirs(output_dir, exist_ok=True)

    derived_metrics = event_details.get("derived_metrics")
    if not isinstance(derived_metrics, dict):
        derived_metrics = {}
        event_details["derived_metrics"] = derived_metrics
    self_match_summary = derived_metrics.get("self_match")
    if self_match_summary is None:
        compute_event_self_similarity({event_id: event_details})
        self_match_summary = derived_metrics.get("self_match")

    color_map = {13: 'tab:blue', 14: 'tab:orange', 15: 'tab:green',
                 17: 'tab:red', 18: 'tab:purple', 19: 'sienna', 30: 'tab:brown'}
    default_color = 'grey'; marker_list = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'X', '+']
    num_trace_channels = 4

    cut_results = event_details.get('cut_results', {})
    passes_overall_analysis = event_details.get('passes_analysis_cuts', False)
        
    # CHANGED: Increased figure height and adjusted GridSpec for better trace/spectrum layout
    fig = plt.figure(figsize=(18, 26))
    gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.9, wspace=0.3,
                           height_ratios=[4, 2, 2, 2, 2, 2.5])
    
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_polar = fig.add_subplot(gs[0, 1], polar=True)
    trace_axs = [fig.add_subplot(gs[1+i, 0]) for i in range(num_trace_channels)]
    spectrum_axs = [fig.add_subplot(gs[1+i, 1]) for i in range(num_trace_channels)]
    ax_text_box = fig.add_subplot(gs[5, :])
    
    event_time_str = "Unknown Time"
    if "datetime" in event_details and event_details["datetime"] is not None:
        try: event_time_dt = datetime.datetime.fromtimestamp(event_details["datetime"]); event_time_str = event_time_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        except Exception as e: ic(f"Error formatting datetime for event {event_id}: {e}. Timestamp: {event_details['datetime']}")
    
    # Determine title based on cut status
    cut_status = "PASSES ALL CUTS" if passes_overall_analysis else "FAILS CUTS"
    main_title = f"Master Plot: Event {event_id} ({dataset_name}) - {cut_status}{title_suffix}\nTime: {event_time_str}"
    fig.suptitle(main_title, fontsize=16, y=0.98)

    status_text = "PASS" if passes_overall_analysis else "FAIL"
    text_info_lines = [f"Event ID: {event_id} -- Overall: {status_text} (Analysis Cuts)"]
    text_info_lines.append(f"Cut Status -> Time: {'Passed' if cut_results.get('time_cut_passed') else 'Failed'}, Chi: {'Passed' if cut_results.get('chi_cut_passed') else 'Failed'}, Angle: {'Passed' if cut_results.get('angle_cut_passed') else 'Failed'}, FFT: {'Passed' if cut_results.get('fft_cut_passed') else 'Failed'}")

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

        text_info_lines.append("--- Template Self-Match ---")
        template_trigger_str = _fmt_idx(template_trigger_idx)
        template_channel_str = _fmt_idx(template_channel_idx)
        text_info_lines.append(
            "  Template: St{} T{} Ch{} SNR={}"
            .format(
                template_station if template_station is not None else "?",
                template_trigger_str if template_trigger_str is not None else "?",
                template_channel_str if template_channel_str is not None else "?",
                _fmt_float(template_snr_val, precision=1),
            )
        )
        text_info_lines.append(
            "  Best match: St{} T{} Ch{} χ={} SNR={} (avg pair SNR={})".format(
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
            text_info_lines.append("  Station χ overview:")
            for station_key in sorted_keys:
                station_info = station_matches.get(station_key) or {}
                role = "template" if str(station_key) == str(template_station) else "partner"
                text_info_lines.append(
                    "    St{} ({}): χ={} SNR={} T{} Ch{}".format(
                        station_key,
                        role,
                        _fmt_float(station_info.get("chi"), precision=2),
                        _fmt_float(station_info.get("snr"), precision=1),
                        _fmt_idx(station_info.get("trigger_idx")) or "?",
                        _fmt_idx(station_info.get("channel_idx")) or "?",
                    )
                )

    text_info_lines.append("--- Station Triggers ---")
    
    legend_handles_for_fig = {}; global_trace_min = float('inf'); global_trace_max = float('-inf')

    with SectionTimer("Compute global trace min/max"):
        for station_id_str_calc, station_data_calc in event_details.get("stations", {}).items():
            all_traces_for_st = station_data_calc.get("Traces", [])
            for traces_for_one_trig in all_traces_for_st:
                if traces_for_one_trig is not None and np.asarray(traces_for_one_trig).any(): 
                    padded_tr = (list(traces_for_one_trig) + [None]*num_trace_channels)[:num_trace_channels]
                    for tr_arr in padded_tr: 
                        if tr_arr is not None and hasattr(tr_arr, "__len__") and len(tr_arr) > 0:
                            c_min,c_max = np.nanmin(tr_arr), np.nanmax(tr_arr)
                            if not np.isnan(c_min): global_trace_min = min(global_trace_min, c_min)
                            if not np.isnan(c_max): global_trace_max = max(global_trace_max, c_max)
    if global_trace_min == float('inf'): global_trace_min = -0.1 
    if global_trace_max == float('-inf'): global_trace_max = 0.1
    if abs(global_trace_max - global_trace_min) < 1e-6 : global_trace_min -= 0.05; global_trace_max += 0.05 
    y_margin = (global_trace_max - global_trace_min) * 0.1; final_trace_ylim = (global_trace_min - y_margin, global_trace_max + y_margin)
    if final_trace_ylim[0] == final_trace_ylim[1]: final_trace_ylim = (final_trace_ylim[0]-0.1, final_trace_ylim[1]+0.1)

    station_items = list(event_details.get("stations", {}).items())
    for s_idx, (station_id_str, station_data) in enumerate(station_items):
        _progress(s_idx, len(station_items), f"Master event {event_id} stations")
        try: station_id_int = int(station_id_str)
        except ValueError: continue
        color = color_map.get(station_id_int, default_color)
        snr_values = station_data.get("SNR", []); num_triggers = len(snr_values)
        if num_triggers == 0: continue
        
        zen_values_rad = (station_data.get("Zen", []) + [np.nan] * num_triggers)[:num_triggers]
        azi_values_rad = (station_data.get("Azi", []) + [np.nan] * num_triggers)[:num_triggers]
        pol_angle_values_rad = (station_data.get("PolAngle", []) + [np.nan] * num_triggers)[:num_triggers]
        pol_angle_err_values_rad = (station_data.get("PolAngleErr", []) + [np.nan] * num_triggers)[:num_triggers]
        
        time_values = station_data.get("Time", [])
        chi_rcr_values = (station_data.get("ChiRCR", []) + [np.nan] * num_triggers)[:num_triggers]
        chi_2016_values = (station_data.get("Chi2016", []) + [np.nan] * num_triggers)[:num_triggers]
        event_ids_for_station = (station_data.get("event_ids", []) + ["N/A"] * num_triggers)[:num_triggers]
        all_traces_for_station = station_data.get("Traces", [])

        # NEW: Collect points for connecting lines within this station
        station_points = []

        for trigger_idx in range(num_triggers):
            marker = marker_list[trigger_idx % len(marker_list)]
            snr_val, chi_rcr_val, chi_2016_val = snr_values[trigger_idx], chi_rcr_values[trigger_idx], chi_2016_values[trigger_idx]
            zen_rad, azi_rad = zen_values_rad[trigger_idx], azi_values_rad[trigger_idx]
            pol_rad = pol_angle_values_rad[trigger_idx] 
            pol_err_rad = pol_angle_err_values_rad[trigger_idx]
            current_event_id_val = event_ids_for_station[trigger_idx]                
            traces_this_trigger = (all_traces_for_station[trigger_idx] if trigger_idx < len(all_traces_for_station) else [])
            padded_traces_this_trigger = (list(traces_this_trigger) + [None]*num_trace_channels)[:num_trace_channels]

            if snr_val is not None and not np.isnan(snr_val):
                if chi_2016_val is not None and not np.isnan(chi_2016_val):
                    ax_scatter.scatter(snr_val, chi_2016_val, c=color, marker=marker, s=60, alpha=0.9, zorder=3)
                    station_points.append((snr_val, chi_2016_val)) # Add point for station line
                if chi_rcr_val is not None and not np.isnan(chi_rcr_val):
                    ax_scatter.scatter(snr_val, chi_rcr_val, marker=marker, s=60, alpha=0.9, facecolors='none', edgecolors=color, linewidths=1.5, zorder=3)
                    station_points.append((snr_val, chi_rcr_val)) # Add point for station line
            
            if zen_rad is not None and not np.isnan(zen_rad) and azi_rad is not None and not np.isnan(azi_rad): 
                ax_polar.scatter(azi_rad, np.degrees(zen_rad), c=color, marker=marker, s=60, alpha=0.9)
            
            for ch_idx in range(num_trace_channels):
                trace_ch_data = padded_traces_this_trigger[ch_idx]
                if trace_ch_data is not None and hasattr(trace_ch_data, "__len__") and len(trace_ch_data) > 0:
                    trace_ch_data_arr = np.asarray(trace_ch_data)
                    time_ax_ns = np.linspace(0, (len(trace_ch_data_arr)-1)*0.5, len(trace_ch_data_arr)) 
                    trace_axs[ch_idx].plot(time_ax_ns, trace_ch_data_arr, c=color, ls='-' if trigger_idx % 2 == 0 else '--', alpha=0.7)
                    sampling_rate_hz = 2e9 
                    if len(trace_ch_data_arr) > 1: 
                        freq_ax_mhz = np.fft.rfftfreq(len(trace_ch_data_arr), d=1/sampling_rate_hz) / 1e6 
                        spectrum = np.abs(fft.time2freq(trace_ch_data_arr, sampling_rate_hz))
                        if len(spectrum)>0: spectrum[0]=0
                        # CHANGED: Spectrum line is now solid with slightly higher alpha
                        spectrum_axs[ch_idx].plot(freq_ax_mhz, spectrum, c=color, ls='-', alpha=0.6)
            
            if station_id_int not in legend_handles_for_fig:
                # CHANGED: Station legend now uses a line instead of a marker to avoid confusion
                legend_handles_for_fig[station_id_int] = Line2D([0], [0], color=color, linestyle='-', linewidth=4, label=f"St {station_id_int}")
            
            time_text = f"{time_values}"
            chi_rcr_text = f"{chi_rcr_val:.2f}" if chi_rcr_val is not None and not np.isnan(chi_rcr_val) else "N/A"
            chi_2016_text = f"{chi_2016_val:.2f}" if chi_2016_val is not None and not np.isnan(chi_2016_val) else "N/A"
            zen_d_text = f"{np.degrees(zen_rad):.1f}°" if zen_rad is not None and not np.isnan(zen_rad) else "N/A"
            azi_d_text = f"{(np.degrees(azi_rad) % 360):.1f}°" if azi_rad is not None and not np.isnan(azi_rad) else "N/A"
            snr_fstr = f"{snr_val:.1f}" if snr_val is not None and not np.isnan(snr_val) else "N/A"
            ev_id_fstr = f"{int(current_event_id_val)}" if current_event_id_val not in ["N/A", np.nan, None] else "N/A"

            pol_angle_full_text = "N/A"
            if pol_rad is not None and not np.isnan(pol_rad):
                pol_angle_full_text = f"{np.degrees(pol_rad):.1f}"
                if pol_err_rad is not None and not np.isnan(pol_err_rad) and isinstance(pol_err_rad, (float, int)):
                     pol_angle_full_text += f" ± {np.degrees(pol_err_rad):.1f}°"
                else:
                     pol_angle_full_text += "°"

            text_info_lines.append(f"  St{station_id_int} T{trigger_idx+1} Unix={time_text}: ID={ev_id_fstr}, SNR={snr_fstr}, ChiRCR={chi_rcr_text}, Chi2016={chi_2016_text}, Zen={zen_d_text}, Azi={azi_d_text}, Pol={pol_angle_full_text}")

        # NEW: Add connecting line for points within this station only
        if len(station_points) > 1:
            station_points.sort() # Sort by SNR
            snrs, chis = zip(*station_points)
            ax_scatter.plot(snrs, chis, color=color, linestyle='--', marker=None, alpha=0.8, zorder=2)

    # === START: SNR vs CHI PLOT SETUP ===
    ax_scatter.set_xlabel("SNR")
    ax_scatter.set_ylabel(r"$\chi$")
    ax_scatter.set_title(r"SNR vs. $\chi$")
    ax_scatter.set_xscale('log')
    ax_scatter.set_xlim(3, 100)
    ax_scatter.set_ylim(0, 1)
    ax_scatter.grid(True, linestyle='--', alpha=0.6)

    # Create handles for the two legends
    chi_2016_handle = Line2D([0], [0], marker='o', color='k', label=r'$\chi_{2016}$ (Filled)',
                             linestyle='None', markersize=8, markerfacecolor='k')
    chi_RCR_handle = Line2D([0], [0], marker='o', color='k', label=r'$\chi_{RCR}$ (Outline)',
                          linestyle='None', markersize=8, markerfacecolor='none', markeredgecolor='k')
    station_handles = list(legend_handles_for_fig.values())
    
    # Add the legends to the ax_scatter plot
    if station_handles:
        leg1 = ax_scatter.legend(handles=station_handles, loc='upper right', title="Stations")
        ax_scatter.add_artist(leg1)
    ax_scatter.legend(handles=[chi_2016_handle, chi_RCR_handle], loc='lower left', title=r"$\chi$ Type")
    # === END: SNR vs CHI PLOT SETUP ===
    
    ax_text_box.axis('off') 
    ax_text_box.text(0.01, 0.95, "\n".join(text_info_lines), ha='left', va='top', fontsize=9, 
                     family='monospace', linespacing=1.4, 
                     bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.6))

    ax_polar.set_theta_zero_location("N"); ax_polar.set_theta_direction(-1); ax_polar.set_rlabel_position(22.5)
    ax_polar.set_rlim(0, 90); ax_polar.set_rticks(np.arange(0, 91, 30)); ax_polar.set_title("Zenith (radius) vs Azimuth (angle)")
    ax_polar.grid(True, linestyle='--', alpha=0.5)
    
    for i in range(num_trace_channels):
        trace_axs[i].set_title(f"Trace - Ch {i+1}",fontsize=10); trace_axs[i].set_ylabel("Amp (mV)",fontsize=8); trace_axs[i].grid(True,ls=':',alpha=0.5); trace_axs[i].set_ylim(final_trace_ylim)
        spectrum_axs[i].set_title(f"Spectrum - Ch {i+1}",fontsize=10); spectrum_axs[i].set_ylabel("Mag",fontsize=8); spectrum_axs[i].grid(True,ls=':',alpha=0.5); spectrum_axs[i].set_xlim(0,1000)
        if i < num_trace_channels -1 : trace_axs[i].set_xticklabels([]); spectrum_axs[i].set_xticklabels([])
        else: trace_axs[i].set_xlabel("Time (ns)", fontsize=8); spectrum_axs[i].set_xlabel("Freq (MHz)", fontsize=8)

    master_filename = os.path.join(output_dir, f'master_event_{event_id}.png')
    try: 
        with SectionTimer("Saving master plot"):
            plt.savefig(master_filename, dpi=150, bbox_inches='tight')
        plt.close(fig); gc.collect()
        return master_filename
    except Exception as e: 
        ic(f"Error saving master plot {master_filename}: {e}")
        plt.close(fig); gc.collect()
    return None


# --- Plotting Function 4: Master Event Plot (Updated) ---
def plot_master_event_updated(events_dict, base_output_dir, dataset_name, include_failed_when_passing_source=False):
    """Generate master event plots.

    By default, plots only events that pass analysis cuts. If
    include_failed_when_passing_source is True, also plots all events that
    fail into a separate folder. This is intended for cases where the source
    data path contains 'passing_cuts' and we want to compare failed events
    after re-evaluating cuts.
    """
    with SectionTimer(f"Master event batch plots for {dataset_name}"):
        ic(
            f"Generating master event plots for {dataset_name}"
            + (", including failed events" if include_failed_when_passing_source else ", only for events that pass cuts.")
        )
        master_folder_base = os.path.join(base_output_dir, f"{dataset_name}_master_event_plots")
        pass_cuts_folder = os.path.join(master_folder_base, "pass_cuts")
        os.makedirs(pass_cuts_folder, exist_ok=True)

        passing_items = [(eid, ed) for eid, ed in events_dict.items() if isinstance(ed, dict) and ed.get('passes_analysis_cuts', False)]
        ic(f"Total passing events to plot: {len(passing_items)}")
        for idx, (event_id, event_details) in enumerate(passing_items):
            _progress(idx, len(passing_items), "Master plots")
            plot_single_master_event(event_id, event_details, pass_cuts_folder, dataset_name)
        
        # Optionally also plot failing events if requested
        if include_failed_when_passing_source:
            fail_cuts_folder = os.path.join(master_folder_base, "fail_cuts")
            os.makedirs(fail_cuts_folder, exist_ok=True)
            failing_items = [
                (eid, ed)
                for eid, ed in events_dict.items()
                if isinstance(ed, dict) and not ed.get('passes_analysis_cuts', False)
            ]
            ic(f"Total failing events to plot: {len(failing_items)}")
            for idx, (event_id, event_details) in enumerate(failing_items):
                _progress(idx, len(failing_items), "Master plots (fail)")
                plot_single_master_event(event_id, event_details, fail_cuts_folder, dataset_name)

        ic(
            f"Finished master event plots for {dataset_name}. "
            + ("Pass and fail events were plotted." if include_failed_when_passing_source else "Only events passing cuts were plotted.")
        )


# --- Function to Check for Key Events ---
def checkForKeyEvents(events_dict, target_timestamps, output_dir, dataset_name, time_tolerance_sec=1.0):
    """
    Searches for events that match within time_tolerance_sec of the provided target timestamps.
    Prints detailed information about found events and plots them to a 'specific_event_search/' subfolder.
    
    Args:
        events_dict: Dictionary of all events with event_id as keys
        target_timestamps: List of target datetime timestamps (as returned by datetime.timestamp())
        output_dir: Base output directory
        dataset_name: Name of the dataset for labeling
        time_tolerance_sec: Time tolerance in seconds for matching (default 1.0 second)
    
    Returns:
        dict: Dictionary of found events {event_id: event_details}
    """
    ic(f"Searching for key events in {dataset_name} with {len(target_timestamps)} target times")
    
    # Create output directory for specific event search
    search_output_dir = os.path.join(output_dir, "specific_event_search")
    os.makedirs(search_output_dir, exist_ok=True)
    
    found_events = {}
    
    # Convert target timestamps to a set for faster lookup
    target_set = set(target_timestamps)
    
    print(f"\n=== KEY EVENT SEARCH RESULTS for {dataset_name} ===")
    print(f"Searching for {len(target_timestamps)} target events within ±{time_tolerance_sec} seconds")
    print(f"Target timestamps: {[datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in target_timestamps]}")
    print("-" * 80)
    
    matches_found = 0
    
    for target_idx, target_timestamp in enumerate(target_timestamps):
        target_dt = datetime.datetime.fromtimestamp(target_timestamp)
        print(f"\nTarget {target_idx + 1}: {target_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        
        found_match = False
        
        # Search through all events for matches
        with SectionTimer(f"Search events for target #{target_idx + 1}"):
            for i_ev, (event_id, event_details) in enumerate(events_dict.items()):
                _progress(i_ev, len(events_dict), f"KeyEvent target {target_idx + 1}")
                if not isinstance(event_details, dict):
                    continue
                    
                event_timestamp = event_details.get("datetime")
                if event_timestamp is None:
                    continue
                    
                # Check if event time is within tolerance of target time
                time_diff = abs(event_timestamp - target_timestamp)
                
                if time_diff <= time_tolerance_sec:
                    found_match = True
                    matches_found += 1
                    found_events[event_id] = event_details
                    
                    # Format event timestamp for display
                    event_dt = datetime.datetime.fromtimestamp(event_timestamp)
                    
                    print(f"  ✓ FOUND MATCH: Event {event_id}")
                    print(f"    Event Time: {event_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                    print(f"    Time Difference: {time_diff:.3f} seconds")
                    
                    # Get cut results
                    cut_results = event_details.get('cut_results', {})
                    passes_overall = event_details.get('passes_analysis_cuts', False)
                    
                    print(f"    Overall Status: {'PASS' if passes_overall else 'FAIL'} (Analysis Cuts)")
                    print(f"    Cut Details:")
                    print(f"      - Time Cut: {'PASS' if cut_results.get('time_cut_passed', False) else 'FAIL'}")
                    print(f"      - Chi Cut: {'PASS' if cut_results.get('chi_cut_passed', False) else 'FAIL'}")
                    print(f"      - Angle Cut: {'PASS' if cut_results.get('angle_cut_passed', False) else 'FAIL'}")
                    print(f"      - FFT Cut: {'PASS' if cut_results.get('fft_cut_passed', False) else 'FAIL'}")
                    
                    # Print station information
                    stations_info = event_details.get("stations", {})
                    print(f"    Stations ({len(stations_info)} total):")
                    
                    for station_id_str, station_data in stations_info.items():
                        snr_values = station_data.get("SNR", [])
                        chi_rcr_values = station_data.get("ChiRCR", [])
                        chi_2016_values = station_data.get("Chi2016", [])
                        zen_values = station_data.get("Zen", [])
                        azi_values = station_data.get("Azi", [])
                        pol_values = station_data.get("PolAngle", [])
                        
                        if snr_values:  # Only print if there's data
                            print(f"      Station {station_id_str}: {len(snr_values)} triggers")
                            for i in range(len(snr_values)):
                                snr = snr_values[i] if i < len(snr_values) else None
                                chi_rcr = chi_rcr_values[i] if i < len(chi_rcr_values) else None
                                chi_2016 = chi_2016_values[i] if i < len(chi_2016_values) else None
                                zen = zen_values[i] if i < len(zen_values) else None
                                azi = azi_values[i] if i < len(azi_values) else None
                                pol = pol_values[i] if i < len(pol_values) else None
                                
                                snr_str = f"{snr:.1f}" if snr is not None and not np.isnan(snr) else "N/A"
                                chi_rcr_str = f"{chi_rcr:.2f}" if chi_rcr is not None and not np.isnan(chi_rcr) else "N/A"
                                chi_2016_str = f"{chi_2016:.2f}" if chi_2016 is not None and not np.isnan(chi_2016) else "N/A"
                                zen_str = f"{np.degrees(zen):.1f}°" if zen is not None and not np.isnan(zen) else "N/A"
                                azi_str = f"{np.degrees(azi):.1f}°" if azi is not None and not np.isnan(azi) else "N/A"
                                pol_str = f"{np.degrees(pol):.1f}°" if pol is not None and not np.isnan(pol) else "N/A"
                                
                                print(f"        Trigger {i+1}: SNR={snr_str}, ChiRCR={chi_rcr_str}, Chi2016={chi_2016_str}, Zen={zen_str}, Azi={azi_str}, Pol={pol_str}")
                    
                    # Plot the event
                    title_suffix = f" - TARGET MATCH #{target_idx + 1}"
                    plot_path = plot_single_master_event(event_id, event_details, search_output_dir, dataset_name, title_suffix)
                    
                    if plot_path:
                        print(f"    Plot saved: {plot_path}")
                    else:
                        print(f"    Warning: Failed to save plot for event {event_id}")
                    
                    print("-" * 40)
        
        if not found_match:
            print(f"  ✗ NO MATCH FOUND for target {target_idx + 1}")
            print("-" * 40)
    
    print(f"\n=== SEARCH SUMMARY ===")
    print(f"Total targets searched: {len(target_timestamps)}")
    print(f"Total matches found: {matches_found}")
    print(f"Found events: {list(found_events.keys()) if found_events else 'None'}")
    print(f"Plots saved to: {search_output_dir}")
    print("=" * 80)
    
    return found_events


# --- Function to Save Passing Events ---
def save_passing_events(events_dict, output_dir, dataset_name, date_of_process, chosen_path):
    """
    Saves only the events that pass the analysis cuts to a new pickle file.
    """
    ic(f"Saving events that pass cuts for {dataset_name}")
    
    # Filter events that pass cuts
    passing_events = {}
    for event_id, event_details in events_dict.items():
        if isinstance(event_details, dict) and event_details.get('passes_analysis_cuts', False):
            passing_events[event_id] = event_details
    
    if not passing_events:
        ic(f"No events pass the cuts for {dataset_name}. No file will be saved.")
        return None
    
    # Create output filename
    output_filepath = chosen_path.replace("CoincidenceDatetimes", "CoincidenceDatetimes_passing_cuts")
    # output_filename = f"{date_of_process}_CoincidenceDatetimes_passing_cuts.pkl"
    # output_filepath = os.path.join(output_dir, output_filename)
    
    # Save to pickle file
    try:
        with SectionTimer("Saving passing events pickle"):
            with open(output_filepath, 'wb') as f:
                pickle.dump(passing_events, f)
        ic(f"Saved {len(passing_events)} passing events to: {output_filepath}")
        return output_filepath
    except Exception as e:
        ic(f"Error saving passing events to {output_filepath}: {e}")
        return None


# --- Main Script ---
if __name__ == '__main__':
    ic.enable()
    import configparser
    config = configparser.ConfigParser()
    config_path = os.path.join('HRAStationDataAnalysis', 'config.ini') 
    if not os.path.exists(config_path): config_path = 'config.ini'
    if not os.path.exists(config_path): ic(f"CRITICAL: config.ini not found."); exit()
    config.read(config_path); date_of_data = config['PARAMETERS']['date']
    date_of_coincidence = config['PARAMETERS']['date_coincidence']
    date_of_process = config['PARAMETERS']['date_processing']
    base_processed_data_dir = os.path.join("HRAStationDataAnalysis", "StationData", "processedNumpyData")
    processed_data_dir_for_date = os.path.join(base_processed_data_dir, date_of_data)

    # List of times of specific events I believe are CRs    
    target_times = [
        datetime.datetime(2017, 2, 16, 19, 9, 51).timestamp(),
        datetime.datetime(2017, 10, 15, 4, 41, 20).timestamp(),
        datetime.datetime(2017, 1, 25, 22, 45, 58).timestamp()
    ]
    # Below are known CR from 2016 search
    target_times.extend([
        datetime.datetime(2015, 12, 11, 11, 20, 9).timestamp(),
        datetime.datetime(2015, 12, 21, 13, 46, 11).timestamp(),
        datetime.datetime(2016, 2, 11, 7, 52, 30).timestamp(),
        datetime.datetime(2016, 2, 14, 21, 21, 2).timestamp(),
        datetime.datetime(2016, 3, 18, 2, 42, 51).timestamp()
    ])

    # Plot specific events
    specific_events_to_plot = [3047, 3432, 10195, 10231, 10284, 10444,
                            10554, 11197, 11230]
    specific_events_with_eyeball = specific_events_to_plot + [10449, 10466, 11243, 10273, 10471, 11220, 11236] 



    # Build candidate list: try all passing_cuts variants first, then non-passing with calcPol first
    dataset_names = ["CoincidenceEvents"]
    dataset_plot_suffixes = [f"CoincidenceEvents_{date_of_process}"]
    output_plot_basedir = os.path.join("HRAStationDataAnalysis", "plots")
    os.makedirs(output_plot_basedir, exist_ok=True)
    datasets_to_plot_info = []

    prefixes = [
        f"{date_of_coincidence}_CoincidenceDatetimes_passing_cuts",
        f"{date_of_coincidence}_CoincidenceDatetimes",
    ]
    # Ensure calcPol is tried first within each prefix
    suffixes = [
        "with_all_params_recalcZenAzi_calcPol.pkl",
        "with_all_params_recalcZenAzi.pkl",
        "with_all_params.pkl",
    ]

    candidates = [os.path.join(processed_data_dir_for_date, f"{p}_{s}") for p in prefixes for s in suffixes]
    ic("File search order:")
    for c in candidates:
        ic(f"  -> {os.path.basename(c)}")

    chosen_path = None
    for c in candidates:
        if os.path.exists(c):
            chosen_path = c
            break

    if chosen_path is None:
        ic("Error: No candidate data file found. Cannot proceed for this dataset.")
    else:
        ic(f"Using data file: {chosen_path}")
        data = _load_pickle(chosen_path)
        if data is not None:
            datasets_to_plot_info.append({
                "name": dataset_names[0],
                "data": data,
                "plot_dir_suffix": dataset_plot_suffixes[0],
            })
            ic(f"Loaded: {chosen_path}")
        else:
            ic(f"Could not load data from: {chosen_path}.")

    if not datasets_to_plot_info: ic("No datasets loaded. Exiting."); exit()

    if isinstance(chosen_path, str) and ("passing_cuts" in os.path.basename(chosen_path) or "passing_cuts" in chosen_path):
        output_plot_basedir = os.path.join("HRAStationDataAnalysis", "plots", "passing_cuts")
        os.makedirs(output_plot_basedir, exist_ok=True)



    for dataset_info in datasets_to_plot_info:
        with SectionTimer("Per-dataset processing"):
            dataset_name_label = dataset_info["name"]
            events_data_dict = dataset_info["data"]
            specific_dataset_plot_dir = os.path.join(output_plot_basedir, dataset_info["plot_dir_suffix"])
            os.makedirs(specific_dataset_plot_dir, exist_ok=True)

            ic(f"\n--- Processing dataset for cuts and plots: {dataset_name_label} ---")
            if not isinstance(events_data_dict, dict) or not events_data_dict:
                ic(f"Dataset '{dataset_name_label}' is empty or not a dict. Skipping."); continue

            # Determine if we're working with a passing_cuts dataset
            is_passing_cuts_dataset = isinstance(chosen_path, str) and ("passing_cuts" in os.path.basename(chosen_path) or "passing_cuts" in chosen_path)
            ic(f"Dataset is passing_cuts type: {is_passing_cuts_dataset}")

            if is_passing_cuts_dataset:
                # For passing_cuts datasets, recalculate time and FFT cuts on the already-filtered events
                ic(f"Recalculating time and FFT cuts for passing_cuts dataset...")
                
                # Step 1: Apply time cut to all events in the passing_cuts dataset
                with SectionTimer("Apply time cut to passing_cuts events"):
                    time_cut_results = check_time_cut(events_data_dict, time_threshold_hours=24.0)
                    ic(f"Time cut applied to {len(time_cut_results)} events")
                
                # Step 2: Apply FFT cut to ALL events (since they already passed initial filtering)
                with SectionTimer("Apply FFT cuts to all passing_cuts events"):
                    ic(f"Applying FFT cuts to ALL {len(events_data_dict)} events in passing_cuts dataset")
                    
                    num_passing_overall = 0; num_failing_overall = 0
                    fft_applied_count = 0
                    for event_id, event_details in events_data_dict.items():
                        if isinstance(event_details, dict):
                            # Get existing cut results or create new ones
                            cut_results = event_details.get('cut_results', {})
                            
                            # Preserve existing chi and angle cuts (already calculated in original processing)
                            chi_cut_passed = cut_results.get('chi_cut_passed', True)  # Default True for passing_cuts
                            angle_cut_passed = cut_results.get('angle_cut_passed', True)  # Default True for passing_cuts
                            
                            # Apply time cut result
                            time_cut_passed = time_cut_results.get(event_id, False)
                            
                            # Apply FFT cut to ALL events in passing_cuts dataset
                            fft_cut_passed = check_fft_cut(event_details, event_id)
                            fft_applied_count += 1
                            
                            # Update cut results
                            event_details['cut_results'] = {
                                'chi_cut_passed': chi_cut_passed,
                                'angle_cut_passed': angle_cut_passed,
                                'time_cut_passed': time_cut_passed,
                                'fft_cut_passed': fft_cut_passed
                            }
                            
                            # Update overall pass status
                            event_details['passes_analysis_cuts'] = all(event_details['cut_results'].values())
                            
                            if event_details['passes_analysis_cuts']: 
                                num_passing_overall += 1
                            else: 
                                num_failing_overall += 1
                        else:
                            num_failing_overall += 1
                
                ic(f"After recalculating time and FFT cuts: {num_passing_overall} events passed, {num_failing_overall} events failed")
                ic(f"FFT cuts applied to {fft_applied_count} events")
                
                # Print detailed summary of all cut results for debugging
                chi_passing = sum(1 for v in events_data_dict.values() if isinstance(v, dict) and v.get('cut_results', {}).get('chi_cut_passed', False))
                angle_passing = sum(1 for v in events_data_dict.values() if isinstance(v, dict) and v.get('cut_results', {}).get('angle_cut_passed', False))
                time_passing = sum(1 for v in events_data_dict.values() if isinstance(v, dict) and v.get('cut_results', {}).get('time_cut_passed', False))
                fft_passing = sum(1 for v in events_data_dict.values() if isinstance(v, dict) and v.get('cut_results', {}).get('fft_cut_passed', False))
                total_events_valid = sum(1 for v in events_data_dict.values() if isinstance(v, dict))
                ic(f"Detailed cut summary for passing_cuts dataset:")
                ic(f"  Chi cuts: {chi_passing}/{total_events_valid} passed")
                ic(f"  Angle cuts: {angle_passing}/{total_events_valid} passed") 
                ic(f"  Time cuts: {time_passing}/{total_events_valid} passed")
                ic(f"  FFT cuts: {fft_passing}/{total_events_valid} passed")
                ic(f"  Overall: {num_passing_overall}/{total_events_valid} events pass ALL cuts")
            
            else:
                # For non-passing_cuts datasets, apply chi/angle cuts first, then time cut, then FFT cut
                num_passing_overall = 0; num_failing_overall = 0
                
                # Step 1: Apply chi and angle cuts
                with SectionTimer("Apply chi/angle cuts per event"):
                    total_events = len(events_data_dict)
                    for loop_idx, (event_id, event_details_loopvar) in enumerate(events_data_dict.items()):
                        _progress(loop_idx, total_events, "Event cuts")
                        if isinstance(event_details_loopvar, dict):
                            # Apply only chi and angle cuts initially (no time or FFT cut yet)
                            chi_cut_passed = check_chi_cut(event_details_loopvar)
                            # angle_cut_passed = check_angle_cut(event_details_loopvar)
                            angle_cut_passed = True  # Temporarily disable angle cut for testing

                            # Store initial cut results without time and FFT cuts
                            cut_results_dict = {
                                'chi_cut_passed': chi_cut_passed,
                                'angle_cut_passed': angle_cut_passed,
                                'time_cut_passed': True,  # Default to True, will be updated
                                'fft_cut_passed': True   # Default to True, will be updated
                            }
                            event_details_loopvar['cut_results'] = cut_results_dict
                            event_details_loopvar['passes_analysis_cuts'] = chi_cut_passed and angle_cut_passed  # Will be updated after other cuts
                            
                        elif event_details_loopvar is not None:
                            event_details_loopvar_placeholder = {'passes_analysis_cuts': False, 
                                                                  'cut_results': {'chi_cut_passed': False, 'angle_cut_passed': False, 'fft_cut_passed': False, 'time_cut_passed': False, 'error': 'Malformed event data'}}
                            if isinstance(events_data_dict, dict):
                                events_data_dict[event_id] = event_details_loopvar_placeholder
                
                # Step 2: Apply time cut to events that pass chi/angle cuts
                events_passing_chi_angle = {
                    event_id: event_details 
                    for event_id, event_details in events_data_dict.items() 
                    if isinstance(event_details, dict) and 
                       event_details.get('cut_results', {}).get('chi_cut_passed', False) and 
                       event_details.get('cut_results', {}).get('angle_cut_passed', False)
                }
                
                if events_passing_chi_angle:
                    with SectionTimer("Apply time cut"):
                        time_cut_results = check_time_cut(events_passing_chi_angle, time_threshold_hours=24.0)
                        ic(f"Time cut applied to {len(events_passing_chi_angle)} events that passed chi/angle cuts")
                else:
                    ic("No events passed chi/angle cuts, skipping time cut")
                    time_cut_results = {}
                
                # Step 3: Apply FFT cut to events that pass chi/angle/time cuts
                events_passing_chi_angle_time = {
                    event_id: event_details 
                    for event_id, event_details in events_data_dict.items() 
                    if isinstance(event_details, dict) and 
                       event_details.get('cut_results', {}).get('chi_cut_passed', False) and 
                       event_details.get('cut_results', {}).get('angle_cut_passed', False) and 
                       time_cut_results.get(event_id, False)
                }
                
                if events_passing_chi_angle_time:
                    with SectionTimer("Apply FFT cuts after time cut"):
                        ic(f"Applying FFT cuts to {len(events_passing_chi_angle_time)} events that passed chi/angle/time cuts")
                
                # Step 4: Update all events with final cut results
                fft_applied_count = 0
                for event_id, event_details in events_data_dict.items():
                    if isinstance(event_details, dict) and 'cut_results' in event_details:
                        # Update time cut result
                        event_details['cut_results']['time_cut_passed'] = time_cut_results.get(event_id, False)
                        
                        # Apply FFT cut only if chi, angle, and time cuts all passed
                        if (event_details['cut_results']['chi_cut_passed'] and 
                            event_details['cut_results']['angle_cut_passed'] and 
                            event_details['cut_results']['time_cut_passed']):
                            event_details['cut_results']['fft_cut_passed'] = check_fft_cut(event_details, event_id)
                            fft_applied_count += 1
                        else:
                            event_details['cut_results']['fft_cut_passed'] = False
                        
                        # Update overall pass status
                        event_details['passes_analysis_cuts'] = all(event_details['cut_results'].values())
                        
                        if event_details['passes_analysis_cuts']: 
                            num_passing_overall += 1
                        else: 
                            num_failing_overall += 1
                    else:
                        num_failing_overall += 1
                
                ic(f"After all cuts: {num_passing_overall} events passed, {num_failing_overall} events failed")
                ic(f"FFT cuts applied to {fft_applied_count} events")
                
                # Print summary of each cut type for debugging
                chi_passing = sum(1 for v in events_data_dict.values() if isinstance(v, dict) and v.get('cut_results', {}).get('chi_cut_passed', False))
                angle_passing = sum(1 for v in events_data_dict.values() if isinstance(v, dict) and v.get('cut_results', {}).get('angle_cut_passed', False))
                time_passing = sum(1 for v in events_data_dict.values() if isinstance(v, dict) and v.get('cut_results', {}).get('time_cut_passed', False))
                fft_passing = sum(1 for v in events_data_dict.values() if isinstance(v, dict) and v.get('cut_results', {}).get('fft_cut_passed', False))
                total_events = len(events_data_dict)
                ic(f"Cut summary - Chi: {chi_passing}/{total_events}, Angle: {angle_passing}/{total_events}, Time: {time_passing}/{total_events}, FFT: {fft_passing}/{total_events}")


            ic(f"Analysis cuts applied to '{dataset_name_label}': {num_passing_overall} events passed, {num_failing_overall} events failed overall.")

            with SectionTimer("Plot: SNR vs Chi"):
                plot_snr_vs_chi(events_data_dict, specific_dataset_plot_dir, dataset_name_label)
            with SectionTimer("Plot: Self-match SNR vs Chi (all)"):
                plot_self_similarity_snr_vs_chi(
                    events_data_dict,
                    specific_dataset_plot_dir,
                    dataset_name_label,
                    highlight_label="BL-RCR",
                )
            with SectionTimer("Plot: Self-match SNR vs Chi (passing only)"):
                plot_self_similarity_snr_vs_chi(
                    events_data_dict,
                    specific_dataset_plot_dir,
                    dataset_name_label,
                    only_passing=True,
                    highlight_label="BL-BL",
                )
            with SectionTimer("Plot: Histograms"):
                plot_parameter_histograms(events_data_dict, specific_dataset_plot_dir, dataset_name_label)
            with SectionTimer("Plot: Polar zen/azi"):
                # Plot all events
                plot_polar_zen_azi(events_data_dict, specific_dataset_plot_dir, dataset_name_label)
                
                # Plot only events passing cuts
                plot_polar_zen_azi(events_data_dict, specific_dataset_plot_dir, dataset_name_label, only_passing_cuts=True)
                
                # Plot specific events
                plot_polar_zen_azi(events_data_dict, specific_dataset_plot_dir, dataset_name_label, specific_event_ids=specific_events_to_plot)
                plot_polar_zen_azi(events_data_dict, specific_dataset_plot_dir, dataset_name_label, specific_event_ids=specific_events_with_eyeball)

            with SectionTimer("Plot: Master events batch"):
                include_failed = isinstance(chosen_path, str) and ("passing_cuts" in os.path.basename(chosen_path) or "passing_cuts" in chosen_path)
                plot_master_event_updated(
                    events_data_dict,
                    specific_dataset_plot_dir,
                    dataset_name_label,
                    include_failed_when_passing_source=include_failed,
                )
            
            with SectionTimer("Key event search"):
                found_events = checkForKeyEvents(
                    events_data_dict, 
                    target_times, 
                    specific_dataset_plot_dir, 
                    dataset_name_label,
                    time_tolerance_sec=1.0
                )

            # Save events that pass cuts to a new file
            # Always save when processing passing_cuts datasets (to update with new time/FFT cut results)
            if is_passing_cuts_dataset or any(v.get('passes_analysis_cuts', False) for v in events_data_dict.values() if isinstance(v, dict)):
                saved_filepath = save_passing_events(events_data_dict, processed_data_dir_for_date, dataset_name_label, date_of_process, chosen_path)
                if saved_filepath:
                    ic(f"Events passing cuts saved to: {saved_filepath}")
            else:
                ic(f"No events pass cuts, skipping save for {dataset_name_label}")

        # Printing all keys for structure of dictionary for convenience
        # And all subkeys
        ic(f"Dataset '{dataset_name_label}' structure:")
        for event_id, event_details in events_data_dict.items():
            ic(f"Event ID: {event_id}, Keys: {list(event_details.keys())}")
            if 'stations' in event_details:
                for station_id, station_data in event_details['stations'].items():
                    ic(f"  Station ID: {station_id}, Keys: {list(station_data.keys())}")
            break

        ic(f"--- Finished plots for: {dataset_name_label} ---")
    ic("\nAll plotting complete.")