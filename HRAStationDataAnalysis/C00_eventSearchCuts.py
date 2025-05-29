import configparser
import os
import datetime
import argparse
from icecream import ic
import numpy as np
import glob
import itertools # For plot markers and combinations
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gc
import collections # For OrderedDict and defaultdict
import pickle
import tempfile

# --- Pickle Load/Save Helpers (from previous response, ensure these are defined) ---
def _load_pickle(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f: return pickle.load(f)
        except Exception as e: ic(f"Error loading pickle file {filepath}: {e}")
    return None

def _save_pickle_atomic(data, filepath):
    temp_file_path = None 
    try:
        base_dir = os.path.dirname(filepath); os.makedirs(base_dir, exist_ok=True) if base_dir else (base_dir := '.')
        fd, temp_file_path = tempfile.mkstemp(suffix='.tmp', dir=base_dir)
        with os.fdopen(fd, 'wb') as tf:
            pickle.dump(data, tf, protocol=pickle.HIGHEST_PROTOCOL); tf.flush(); os.fsync(tf.fileno()) 
        os.replace(temp_file_path, filepath); ic(f"Atomically saved: {filepath}"); temp_file_path = None
    except Exception as e: ic(f"Error saving pickle atomically to {filepath}: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path); ic(f"Removed temp file {temp_file_path} after error.")
            except OSError as oe: ic(f"Error removing temp file {temp_file_path}: {oe}")

# --- Duration Formatting Functions (from previous response) ---
def format_duration(total_seconds_input):
    if not isinstance(total_seconds_input, (int, float)) or total_seconds_input < 0: return "Invalid duration"
    if abs(total_seconds_input) < 1e-9: return "0 Seconds"
    total_seconds = int(round(total_seconds_input))
    days = total_seconds // (24 * 3600); remaining_seconds = total_seconds % (24 * 3600)
    hours = remaining_seconds // 3600; remaining_seconds %= 3600
    minutes = remaining_seconds // 60; seconds = remaining_seconds % 60
    parts = []
    if days > 0: parts.append(f"{days} Day{'s' if days != 1 else ''}")
    if hours > 0: parts.append(f"{hours} Hour{'s' if hours != 1 else ''}")
    if minutes > 0: parts.append(f"{minutes} Minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts: parts.append(f"{seconds} Second{'s' if seconds != 1 else ''}")
    return " ".join(parts) if parts else "0 Seconds"

def format_duration_short(total_seconds):
    if not isinstance(total_seconds, (int, float)) or total_seconds < 0: return "N/A"
    if abs(total_seconds) < 1e-9 : return "0.0s"
    hours = total_seconds / 3600.0;
    if abs(hours) >= 0.1: return f"{hours:.1f}h"
    minutes = total_seconds / 60.0;
    if abs(minutes) >= 0.1: return f"{minutes:.1f}m"
    return f"{total_seconds:.1f}s"

# --- GTI Manipulation Functions (from previous response) ---
def merge_gti_list(gti_list):
    if not gti_list: return []
    gti_list.sort(key=lambda x: (x[0], x[1]))
    merged = []; current_start, current_end = gti_list[0]
    for i in range(1, len(gti_list)):
        next_start, next_end = gti_list[i]
        if next_start < current_end: current_end = max(current_end, next_end)
        else: merged.append([current_start, current_end]); current_start, current_end = next_start, next_end
    merged.append([current_start, current_end]); return merged

def intersect_gti_lists(gti_list1, gti_list2):
    if not gti_list1 or not gti_list2: return []
    m1 = merge_gti_list(list(gti_list1)); m2 = merge_gti_list(list(gti_list2))
    intersection = []; i = 0; j = 0
    while i < len(m1) and j < len(m2):
        s1, e1 = m1[i]; s2, e2 = m2[j]
        overlap_s = max(s1, s2); overlap_e = min(e1, e2)
        if overlap_s < overlap_e: intersection.append([overlap_s, overlap_e])
        if e1 < e2: i += 1
        else: j += 1
    return merge_gti_list(intersection)


# --- Livetime and Overlap Calculation Functions (from previous response) ---
def calculate_livetime(times_survived_input, threshold_seconds, start_bound=None, end_bound=None):
    # ... (implementation from previous response remains the same) ...
    if not isinstance(times_survived_input, np.ndarray): times_survived_input = np.array(times_survived_input)
    if len(times_survived_input) == 0: return 0.0, []
    times_survived_input.sort()
    times_survived = times_survived_input
    if start_bound is not None or end_bound is not None:
        temp_times = times_survived_input
        if start_bound is not None: temp_times = temp_times[temp_times >= start_bound]
        if end_bound is not None: temp_times = temp_times[temp_times <= end_bound]
        times_survived = temp_times
    n = len(times_survived); raw_active_periods = []
    if n == 0: return 0.0, []
    if n == 1:
        t_event = times_survived[0]; duration_singular = 0.5 * threshold_seconds
        raw_active_periods.append([t_event - duration_singular / 2.0, t_event + duration_singular / 2.0])
    else:
        for i in range(n - 1):
            t_current, t_next = times_survived[i], times_survived[i+1]
            if (t_next - t_current) < threshold_seconds: raw_active_periods.append([t_current, t_next])
        for i in range(n):
            t_event = times_survived[i]; is_first, is_last = (i == 0), (i == n - 1)
            conn_prev = not is_first and (t_event - times_survived[i-1]) < threshold_seconds
            conn_next = not is_last and (times_survived[i+1] - t_event) < threshold_seconds
            if not conn_prev and not conn_next:
                duration_singular = 0.5 * threshold_seconds
                raw_active_periods.append([t_event - duration_singular / 2.0, t_event + duration_singular / 2.0])
    if not raw_active_periods: return 0.0, []
    merged_periods = merge_gti_list(raw_active_periods)
    total_livetime_seconds = sum(end - start for start, end in merged_periods)
    return total_livetime_seconds, merged_periods


def calculate_stations_combination_overlap(dict_station_active_periods):
    # ... (implementation from previous response remains the same) ...
    if not dict_station_active_periods or len(dict_station_active_periods) < 1: return 0.0, []
    station_ids = list(dict_station_active_periods.keys())
    if not station_ids: return 0.0, []
    current_overlap_gtis = merge_gti_list(list(dict_station_active_periods[station_ids[0]]))
    for k in range(1, len(station_ids)):
        st_id = station_ids[k]
        current_overlap_gtis = intersect_gti_lists(current_overlap_gtis, dict_station_active_periods[st_id])
        if not current_overlap_gtis: break
    total_overlap_seconds = sum(end - start for start, end in current_overlap_gtis)
    return total_overlap_seconds, current_overlap_gtis

def calculate_N_or_more_stations_livetime(all_station_gti_lists, N_min_stations, relevant_station_ids=None):
    # ... (implementation from previous response remains the same) ...
    if N_min_stations <= 0: return 0.0, []
    filter_ids = relevant_station_ids if relevant_station_ids else all_station_gti_lists.keys()
    active_periods_to_consider = {st_id: merge_gti_list(list(gtis)) for st_id, gtis in all_station_gti_lists.items() if st_id in filter_ids and gtis}
    if not active_periods_to_consider or len(active_periods_to_consider) < N_min_stations : return 0.0, []
    time_points = set()
    for gti_list in active_periods_to_consider.values():
        for start, end in gti_list: time_points.add(start); time_points.add(end)
    if not time_points: return 0.0, []
    sorted_unique_times = sorted(list(time_points)); N_overlap_gtis = []
    for i in range(len(sorted_unique_times) - 1):
        t_start, t_end = sorted_unique_times[i], sorted_unique_times[i+1]
        if t_start >= t_end: continue
        test_point = t_start; num_active = 0
        for gti_list in active_periods_to_consider.values():
            for gti_s, gti_e in gti_list:
                if test_point >= gti_s and test_point < gti_e: num_active += 1; break
        if num_active >= N_min_stations: N_overlap_gtis.append([t_start, t_end])
    merged_N_overlap_gtis = merge_gti_list(N_overlap_gtis)
    total_N_overlap_seconds = sum(end - start for start, end in merged_N_overlap_gtis)
    return total_N_overlap_seconds, merged_N_overlap_gtis

# --- Cut Functions (Modified cluster_cut) ---
def cluster_cut(times, traces, event_ids, amplitude_threshold, time_period, cut_frequency):
    """
    Creates a mask to remove events that occur in bursts, considering unique (Time, EventID) pairs
    for triggering the cut frequency.
    """
    times = np.array(times)
    traces = np.array(traces)
    event_ids = np.array(event_ids) # Ensure event_ids is also a numpy array
    n = len(times)

    if n == 0: 
        return np.array([], dtype=bool)
    if len(event_ids) != n:
        ic("Error: times and event_ids arrays must have the same length in cluster_cut.")
        # Fallback: return a mask that passes all events or handle error as appropriate
        return np.ones(n, dtype=bool) 

    mask = np.ones(n, dtype=bool)

    # Determine which events have high amplitude
    if traces.ndim == 3 and traces.shape[1:3] == (4,256) : # Expected (N,4,256)
        high_amplitude_events = np.any(np.abs(traces) > amplitude_threshold, axis=(1, 2))
    elif traces.ndim == 1: # If traces is already a 1D max amplitude array
        high_amplitude_events = np.abs(traces) > amplitude_threshold
    else: 
        ic(f"Warning: Unexpected traces shape {traces.shape} in cluster_cut. Assuming no high_amplitude_events.")
        high_amplitude_events = np.zeros(n, dtype=bool)

    # Determine which high-amplitude events are "primary triggers" (not repeats of immediate predecessor)
    # An event is a primary trigger if it's high amplitude AND it's the first event,
    # OR its (Time, EventID) is different from the previous event's (Time, EventID).
    is_primary_trigger_event = np.zeros(n, dtype=bool)
    if n > 0:
        for i in range(n):
            if high_amplitude_events[i]:
                if i == 0:
                    is_primary_trigger_event[i] = True
                else:
                    # Check if it's a repeat of the immediate previous event
                    if not (times[i] == times[i-1] and event_ids[i] == event_ids[i-1]):
                        is_primary_trigger_event[i] = True
                    # Else, it's a repeat of the immediate previous (Time, EventID), so is_primary_trigger_event[i] remains False

    start_idx = 0 
    current_primary_trigger_count_in_window = 0
    time_period_seconds_val = time_period.total_seconds()

    for end_idx in range(n):
        # Add the event at 'end_idx' to the current window count IF it's a primary trigger
        if is_primary_trigger_event[end_idx]:
            current_primary_trigger_count_in_window += 1
        
        # Shrink the window from the 'start_idx'
        while (times[end_idx] - times[start_idx]) >= time_period_seconds_val:
            if is_primary_trigger_event[start_idx]: # Decrement count if a primary trigger is leaving window
                current_primary_trigger_count_in_window -= 1
            start_idx += 1
            if start_idx > end_idx: # Safety break, window is now empty or invalid
                 # Reset count if window becomes empty this way, though ideally start_idx <= end_idx
                if start_idx > end_idx : current_primary_trigger_count_in_window = 0
                break 
        
        # If the current count of *primary trigger* events in the window [start_idx, end_idx]
        # is greater than or equal to the cut frequency, mark ALL events (including repeats)
        # in this window for removal.
        if current_primary_trigger_count_in_window >= cut_frequency:
            mask[start_idx : end_idx+1] = False # Mask events from current window start to current window end
            
    return mask

# ... (L1_cut, approximate_bad_times, format_duration, calculate_livetime, GTI functions, plotting functions remain the same) ...
def L1_cut(traces, power_cut=0.3):
    # ... (implementation from previous response, ensure NuRadioReco.utilities.fft is importable) ...
    try: from NuRadioReco.utilities.fft import time2freq
    except ImportError: ic("NuRadioReco.utilities.fft not found for L1_cut. Returning all pass."); return np.ones(traces.shape[0], dtype=bool)
    if traces.ndim != 3 or traces.shape[1] != 4 or traces.shape[2] != 256: return np.ones(traces.shape[0], dtype=bool)
    n_events = traces.shape[0]; mask = np.ones(n_events, dtype=bool)
    for i in range(n_events):
        for channel_idx in range(traces.shape[1]):
            trace_channel_freq = np.abs(time2freq(traces[i, channel_idx, :], 2e9)) 
            total_power = np.sum(trace_channel_freq)
            if total_power == 0: continue
            if np.any(trace_channel_freq > power_cut * total_power): mask[i] = False; break 
    return mask
# approximate_bad_times can be included if used.

# --- Plotting Functions (plot_cuts_amplitudes, plot_cuts_rates, plot_concurrent_station_summary_strips - from previous response) ---
def plot_cuts_amplitudes(times_unix, values_data, amp_name, output_dir=".", livetime_threshold_seconds=3600.0, cuts_to_plot_dict=None):
    # ... (implementation from previous response) ...
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    dt_times = np.array([datetime.datetime.fromtimestamp(t) if t > 0 else datetime.datetime(1970,1,1) for t in times_unix])
    if values_data.ndim == 3 and values_data.shape[1:3] == (4, 256): max_amps = np.max(np.abs(values_data), axis=(1, 2))
    elif values_data.ndim == 1 and len(values_data) == len(times_unix): max_amps = np.array(values_data)
    else: return
    valid_data_mask = np.isfinite(times_unix) & np.isfinite(max_amps)
    for start_year in range(2013, 2020):
        markers = itertools.cycle(("v", "s", "*", "d", "P", "X"))
        season_start_dt = datetime.datetime(start_year, 10, 1); season_end_dt = datetime.datetime(start_year + 1, 4, 30, 23, 59, 59)
        season_start_unix, season_end_unix = season_start_dt.timestamp(), season_end_dt.timestamp()
        seasonal_dt_mask = (dt_times >= season_start_dt) & (dt_times <= season_end_dt)
        base_seasonal_mask_for_plot = seasonal_dt_mask & valid_data_mask
        if not np.any(base_seasonal_mask_for_plot): continue
        plt.figure(figsize=(12,7)); plt.title(f"Season {start_year}-{start_year + 1} Activity - {amp_name}"); plt.xlabel("Time"); plt.ylabel(amp_name)
        lt_all_s, _ = calculate_livetime(times_unix, livetime_threshold_seconds, season_start_unix, season_end_unix)
        label_all_events = f"All Events (station data) (Livetime: {format_duration_short(lt_all_s)})"
        plt.scatter(dt_times[base_seasonal_mask_for_plot], max_amps[base_seasonal_mask_for_plot], color="lightgray", s=10, label=label_all_events, alpha=0.7, edgecolor='k', linewidths=0.5)
        if cuts_to_plot_dict:
            for legend_label_base, cut_mask_for_series in cuts_to_plot_dict.items():
                if not (isinstance(cut_mask_for_series, np.ndarray) and cut_mask_for_series.dtype == bool and len(cut_mask_for_series) == len(times_unix)): continue
                final_series_mask = base_seasonal_mask_for_plot & cut_mask_for_series 
                if np.any(final_series_mask):
                    times_for_cut_seasonal = times_unix[cut_mask_for_series] 
                    lt_s, _ = calculate_livetime(times_for_cut_seasonal, livetime_threshold_seconds, season_start_unix, season_end_unix)
                    plot_label = f"{legend_label_base} (Livetime: {format_duration_short(lt_s)})"
                    plt.scatter(dt_times[final_series_mask], max_amps[final_series_mask], s=15, label=plot_label, marker=next(markers), alpha=0.8)
        if np.nanmin(max_amps[base_seasonal_mask_for_plot]) >= 0 and np.nanmax(max_amps[base_seasonal_mask_for_plot]) <=1.0 and "Chi" in amp_name : plt.ylim(0, 1.05)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y')); plt.gcf().autofmt_xdate(); plt.legend(fontsize=8); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
        filename = os.path.join(output_dir, f"{amp_name.replace(' ', '_')}_season_{start_year}_{start_year+1}.png"); plt.savefig(filename); plt.close(); ic(f"Saved: {filename}")

def plot_cuts_rates(times_unix, bin_size_seconds=30*60, output_dir=".", cuts_to_plot_dict=None):
    # ... (implementation from previous response) ...
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    valid_times_mask = np.isfinite(times_unix) & (times_unix > 0); times_unix_valid = times_unix[valid_times_mask]
    if len(times_unix_valid) == 0: return
    for start_year in range(2013, 2020):
        markers = itertools.cycle(("v", "s", "*", "d", "P", "X"))
        season_start_dt = datetime.datetime(start_year, 10, 1); season_end_dt = datetime.datetime(start_year + 1, 4, 30, 23, 59, 59)
        season_start_unix, season_end_unix = season_start_dt.timestamp(), season_end_dt.timestamp()
        seasonal_unix_mask = (times_unix_valid >= season_start_unix) & (times_unix_valid <= season_end_unix)
        if not np.any(seasonal_unix_mask): continue
        current_season_times_all = times_unix_valid[seasonal_unix_mask]
        if len(current_season_times_all) == 0: continue
        bins = np.arange(season_start_unix, season_end_unix + bin_size_seconds, bin_size_seconds)
        if len(bins) < 2 : continue
        bin_centers_unix = (bins[:-1] + bins[1:]) / 2.0; dt_bin_centers = [datetime.datetime.fromtimestamp(ts) for ts in bin_centers_unix]
        plt.figure(figsize=(12, 7)); plt.title(f"Season {start_year}-{start_year + 1} Event Rate"); plt.xlabel("Time"); plt.ylabel(f"Event Rate (Hz, {bin_size_seconds/60:.0f}min bins)")
        count_all, _ = np.histogram(current_season_times_all, bins=bins); rate_all = count_all / bin_size_seconds
        plt.plot(dt_bin_centers, rate_all, linestyle='-', marker='o', markersize=3, label="All Events Rate", color="lightgray", alpha=0.7)
        if cuts_to_plot_dict:
            for legend_label_base, cut_mask_for_series in cuts_to_plot_dict.items():
                if not (isinstance(cut_mask_for_series, np.ndarray) and cut_mask_for_series.dtype == bool and len(cut_mask_for_series) == len(times_unix)): continue
                times_for_series_cut = times_unix_valid[seasonal_unix_mask & cut_mask_for_series[valid_times_mask]]
                if len(times_for_series_cut) > 0:
                    count_cut, _ = np.histogram(times_for_series_cut, bins=bins); rate_cut = count_cut / bin_size_seconds
                    plt.plot(dt_bin_centers, rate_cut, linestyle='-', marker=next(markers), markersize=3, label=f'{legend_label_base} Rate')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y')); plt.gcf().autofmt_xdate(); plt.legend(fontsize=8); plt.yscale('log')
        min_rate = np.min(rate_all[rate_all > 0]) if np.any(rate_all > 0) else 1e-5
        plt.ylim(max(1e-5, min_rate / 2) , 10); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
        out_filename = os.path.join(output_dir, f"season_{start_year}_{start_year+1}_rate.png"); plt.savefig(out_filename); plt.close(); ic(f"Saved: {out_filename}")

def plot_concurrent_station_summary_strips(all_station_gti_lists, plot_start_time_unix, plot_end_time_unix, output_dir, filename_suffix, relevant_station_ids=None, title_prefix=""):
    # ... (implementation from previous response) ...
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filter_ids = relevant_station_ids if relevant_station_ids else all_station_gti_lists.keys()
    active_periods_to_plot = {st_id: merge_gti_list(list(gtis)) for st_id, gtis in all_station_gti_lists.items() if st_id in filter_ids and gtis}
    if not active_periods_to_plot: ic(f"No station data to plot for {filename_suffix}"); return
    time_points = {plot_start_time_unix, plot_end_time_unix}
    for gti_list in active_periods_to_plot.values():
        for start, end in gti_list:
            if start <= plot_end_time_unix and end >= plot_start_time_unix:
                time_points.add(max(start, plot_start_time_unix)); time_points.add(min(end, plot_end_time_unix))
    if not time_points or len(time_points) < 2: ic(f"Not enough time points for plot {filename_suffix}"); return
    sorted_unique_times = sorted([t for t in list(time_points) if plot_start_time_unix <= t <= plot_end_time_unix])
    if len(sorted_unique_times) <2 or sorted_unique_times[0] == sorted_unique_times[-1]: ic(f"Not enough distinct time points in plot range for {filename_suffix}"); return
    plot_x_times_dt = []; plot_y_num_active = []
    for i in range(len(sorted_unique_times) -1):
        t_interval_start, t_interval_end = sorted_unique_times[i], sorted_unique_times[i+1]
        if t_interval_start >= t_interval_end: continue
        test_point = t_interval_start; num_active = 0
        for gti_list in active_periods_to_plot.values():
            for gti_s, gti_e in gti_list:
                if test_point >= gti_s and test_point < gti_e: num_active += 1; break
        plot_x_times_dt.append(datetime.datetime.fromtimestamp(t_interval_start)); plot_y_num_active.append(num_active)
        plot_x_times_dt.append(datetime.datetime.fromtimestamp(t_interval_end)); plot_y_num_active.append(num_active)
    if not plot_x_times_dt: ic(f"No data to plot in concurrency strips for {filename_suffix}"); return
    num_total_stations_considered = len(active_periods_to_plot)
    plt.figure(figsize=(15, 8)); plt.plot(plot_x_times_dt, plot_y_num_active, drawstyle='steps-post', color='black', linewidth=1.5, label="Num. Concurrent Stations")
    colors = plt.cm.get_cmap('viridis', num_total_stations_considered +1)
    for level in range(1, num_total_stations_considered + 1):
        plt.fill_between(plot_x_times_dt, plot_y_num_active, where=[(y >= level) for y in plot_y_num_active], step='post', alpha=0.6, color=colors(level / num_total_stations_considered))
    plt.title(f"{title_prefix}Number of Concurrently Active Stations ({filename_suffix})"); plt.xlabel("Time"); plt.ylabel("Number of Concurrent Stations")
    plt.ylim(0, num_total_stations_considered + 0.5); plt.yticks(range(num_total_stations_considered + 1)) # Adjusted y-ticks slightly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M')); plt.gcf().autofmt_xdate(); plt.grid(True, linestyle=':', alpha=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    custom_handles = [handles[0]] # Start with the line plot handle
    custom_labels = [labels[0]] 
    for level in range(1, num_total_stations_considered + 1):
        custom_handles.append(plt.Rectangle((0,0),1,1,color=colors(level / num_total_stations_considered), alpha=0.6))
        custom_labels.append(f'{level}+ Station{"s" if level > 1 else ""}')
    plt.legend(custom_handles, custom_labels, fontsize=8, loc='upper left')
    output_filepath = os.path.join(output_dir, f"concurrent_stations_strip_{filename_suffix.replace(' ', '_')}.png"); plt.savefig(output_filepath); plt.close(); ic(f"Saved: {output_filepath}")


# --- Global Definitions ---
LIVETIME_THRESHOLD_SECONDS = 3600.0
REPORT_CUT_STAGES = collections.OrderedDict([
    ("Total (after initial time filters)", "mask_total"),
    ("After L1", "mask_l1"),
    ("After L1 + Storm", "mask_l1_storm"),
    ("After L1 + Storm + Burst", "mask_l1_storm_burst")
])

# --- Main Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply cuts, calculate individual and overlapping livetimes.')
    parser.add_argument('--stnID', type=int, required=False, default=None,
                        help="Station ID for single-station processing. If absent, runs overlap analysis.")
    parser.add_argument('--date', type=str, required=True,
                        help="Date string for data processing (e.g., YYYYMMDD).")
    parser.add_argument('--stations_for_overlap', nargs='+', type=int,
                        default=[13, 14, 15, 17, 18, 19, 30], # Example default list from your C00
                        help="List of station IDs for overlap analysis mode.")

    args = parser.parse_args()
    date_filter = args.date
    ic.enable()

    # --- Path Definitions ---
    base_project_path = 'HRAStationDataAnalysis'
    base_data_path = os.path.join(base_project_path, 'StationData')
    # Corrected path to align with C00 structure if nurFiles are directly under StationData
    station_data_folder = os.path.join(base_data_path, 'nurFiles', date_filter) # From your C00 structure for data files
    cuts_data_folder = os.path.join(base_data_path, 'cuts', date_filter) # From your C00 structure for cut files
    plot_folder_base = os.path.join(base_project_path, 'plots', date_filter)

    os.makedirs(cuts_data_folder, exist_ok=True)
    os.makedirs(plot_folder_base, exist_ok=True)

    # --- Mode Selection ---
    if args.stnID is not None:
        # --- SINGLE-STATION PROCESSING MODE ---
        current_station_id = args.stnID
        ic("\n\n" + "*"*20)
        ic(f"MODE: Single-Station Processing for Station {current_station_id}, Date Ref: {date_filter}")
        ic("*"*20)

        plot_folder_station = os.path.join(plot_folder_base, f'Station{current_station_id}')
        os.makedirs(plot_folder_station, exist_ok=True)
        station_livetime_output_dir = os.path.join(plot_folder_station, "livetime_data")
        os.makedirs(station_livetime_output_dir, exist_ok=True)

        try:
            time_files = sorted(glob.glob(os.path.join(station_data_folder, f'{date_filter}_Station{current_station_id}_Times*')))
            trace_files = sorted(glob.glob(os.path.join(station_data_folder, f'{date_filter}_Station{current_station_id}_Traces*')))
            eventid_files = sorted(glob.glob(os.path.join(station_data_folder, f'{date_filter}_Station{current_station_id}_EventIDs*')))

            if not time_files or not trace_files or not eventid_files:
                raise FileNotFoundError(f"Time, Trace, or EventID files missing for Station {current_station_id} on {date_filter}")

            times_list = [np.load(f) for f in time_files]; times_raw = np.concatenate(times_list, axis=0).squeeze()
            traces_list = [np.load(f) for f in trace_files]; traces_raw = np.concatenate(traces_list, axis=0)
            eventids_list = [np.load(f) for f in eventid_files]; eventids_raw = np.concatenate(eventids_list, axis=0).squeeze()

            if times_raw.ndim == 0: times_raw = np.array([times_raw.item()])
            if eventids_raw.ndim == 0: eventids_raw = np.array([eventids_raw.item()])
            if traces_raw.ndim == 2 and traces_raw.shape[0] == 4 and traces_raw.shape[1] == 256 : traces_raw = traces_raw.reshape(1,4,256)
            if traces_raw.ndim == 1 and traces_raw.size == 4*256: traces_raw = traces_raw.reshape(1,4,256)

        except Exception as e:
            ic(f"Error loading data for Station {current_station_id}: {e}. Aborting for this station.")
            exit(1)

        if times_raw.size == 0 or traces_raw.size == 0 or eventids_raw.size == 0 or \
           not (times_raw.shape[0] == traces_raw.shape[0] == eventids_raw.shape[0]):
            ic(f"Empty or mismatched data arrays for Station {current_station_id} after loading. Aborting.")
            exit(1)

        zerotime_mask = (times_raw != 0)
        min_datetime_threshold = datetime.datetime(2013, 1, 1).timestamp()
        pretime_mask = (times_raw >= min_datetime_threshold)
        initial_valid_mask = zerotime_mask & pretime_mask

        base_times_for_cuts = times_raw[initial_valid_mask]
        base_traces_for_cuts = traces_raw[initial_valid_mask]
        base_event_ids_for_cuts = eventids_raw[initial_valid_mask]

        if base_times_for_cuts.size == 0:
            ic(f"No data for Station {current_station_id} after initial time filters. Saving empty report and aborting.")
            _save_pickle_atomic({}, os.path.join(station_livetime_output_dir, f"livetime_gti_St{current_station_id}_{date_filter}.pkl"))
            exit(0)

        ic(f"Data for cuts: Times {base_times_for_cuts.shape}, Traces {base_traces_for_cuts.shape}, EventIDs {base_event_ids_for_cuts.shape}")

        # --- Apply Cuts with Incremental Saving ---
        cut_file_path = os.path.join(cuts_data_folder, f'{date_filter}_Station{current_station_id}_Cuts.npy')
        # This dictionary will hold all masks, loaded or computed.
        # It will be saved after each new computation.
        current_all_cut_masks = {} 
        
        L1_mask_final, storm_mask_final, burst_mask_final = None, None, None # Initialize to None

        if os.path.exists(cut_file_path):
            ic(f"Attempting to load existing cut masks from: {cut_file_path}")
            try:
                loaded_data = np.load(cut_file_path, allow_pickle=True).item()
                if isinstance(loaded_data, dict):
                    current_all_cut_masks = loaded_data # Start with what's in the file
                else:
                    ic("Warning: Cuts file did not contain a dictionary. Will recalculate all.")
            except Exception as e:
                ic(f"Error loading or parsing cuts file {cut_file_path}: {e}. Will recalculate all.")
        
        # Check and assign L1_mask
        temp_L1 = current_all_cut_masks.get('L1_mask')
        if temp_L1 is not None and isinstance(temp_L1, np.ndarray) and len(temp_L1) == len(base_times_for_cuts):
            L1_mask_final = temp_L1
            ic("Loaded valid L1_mask from file.")
        else:
            if temp_L1 is not None: ic("L1_mask from file is invalid. Will recalculate.")
            L1_mask_final = None # Ensure it's None to trigger recalculation

        # Check and assign storm_mask
        temp_storm = current_all_cut_masks.get('storm_mask')
        if temp_storm is not None and isinstance(temp_storm, np.ndarray) and len(temp_storm) == len(base_times_for_cuts):
            storm_mask_final = temp_storm
            ic("Loaded valid storm_mask from file.")
        else:
            if temp_storm is not None: ic("storm_mask from file is invalid. Will recalculate.")
            storm_mask_final = None

        # Check and assign burst_mask
        temp_burst = current_all_cut_masks.get('burst_mask')
        if temp_burst is not None and isinstance(temp_burst, np.ndarray) and len(temp_burst) == len(base_times_for_cuts):
            burst_mask_final = temp_burst
            ic("Loaded valid burst_mask from file.")
        else:
            if temp_burst is not None: ic("burst_mask from file is invalid. Will recalculate.")
            burst_mask_final = None
            
        # Calculate L1_mask if not loaded/valid
        if L1_mask_final is None:
            ic(f"Calculating L1 cut for {len(base_times_for_cuts)} events...")
            L1_mask_final = L1_cut(base_traces_for_cuts, power_cut=0.3) #
            current_all_cut_masks['L1_mask'] = L1_mask_final
            ic(f"Saving L1_mask to: {cut_file_path}")
            np.save(cut_file_path, current_all_cut_masks, allow_pickle=True)
        
        # Calculate storm_mask if not loaded/valid
        if storm_mask_final is None:
            ic(f"Calculating storm cut (Amp > 0.4V, Win: 1hr, Freq >= 2)...") # User had 0.4 in last C00
            storm_mask_final = cluster_cut(base_times_for_cuts, base_traces_for_cuts, base_event_ids_for_cuts,
                                           amplitude_threshold=0.4,
                                           time_period=datetime.timedelta(seconds=3600),
                                           cut_frequency=2) #
            current_all_cut_masks['storm_mask'] = storm_mask_final
            ic(f"Saving storm_mask to: {cut_file_path}")
            np.save(cut_file_path, current_all_cut_masks, allow_pickle=True)

        # Calculate burst_mask if not loaded/valid
        if burst_mask_final is None:
            ic(f"Calculating burst cut (Amp > 0.2V, Win: 60s, Freq >= 2)...")
            burst_mask_final = cluster_cut(base_times_for_cuts, base_traces_for_cuts, base_event_ids_for_cuts,
                                           amplitude_threshold=0.2,
                                           time_period=datetime.timedelta(seconds=60),
                                           cut_frequency=2) #
            current_all_cut_masks['burst_mask'] = burst_mask_final
            ic(f"Saving burst_mask to: {cut_file_path}")
            np.save(cut_file_path, current_all_cut_masks, allow_pickle=True)

        ic(f"Final Cut results for Station {current_station_id} (on {len(base_times_for_cuts)} events):")
        ic(f"  L1_mask passed: {np.sum(L1_mask_final)} ({np.sum(L1_mask_final)/len(base_times_for_cuts)*100:.2f}%)")
        ic(f"  storm_mask passed: {np.sum(storm_mask_final)} ({np.sum(storm_mask_final)/len(base_times_for_cuts)*100:.2f}%)")
        ic(f"  burst_mask passed: {np.sum(burst_mask_final)} ({np.sum(burst_mask_final)/len(base_times_for_cuts)*100:.2f}%)")

        station_specific_report = collections.OrderedDict()
        report_masks = {
            "Total (after initial time filters)": np.ones_like(base_times_for_cuts, dtype=bool),
            "After L1": L1_mask_final,
            "After L1 + Storm": L1_mask_final & storm_mask_final,
            "After L1 + Storm + Burst": L1_mask_final & storm_mask_final & burst_mask_final
        } #
        for stage_label in REPORT_CUT_STAGES.keys():
            current_stage_mask = report_masks[stage_label]
            times_survived_stage = base_times_for_cuts[current_stage_mask]
            lt_s, active_periods = calculate_livetime(times_survived_stage, LIVETIME_THRESHOLD_SECONDS)
            station_specific_report[stage_label] = (lt_s, active_periods)
            ic(f"Station {current_station_id}, {stage_label}: Livetime = {format_duration(lt_s)}")

        station_gti_file_to_save = os.path.join(station_livetime_output_dir, f"livetime_gti_St{current_station_id}_{date_filter}.pkl")
        _save_pickle_atomic(station_specific_report, station_gti_file_to_save)

        cuts_dict_for_plotting = collections.OrderedDict([
            ("L1 cut", L1_mask_final),
            ("L1+Storm cut", L1_mask_final & storm_mask_final),
            ("L1+Storm+Burst cut", L1_mask_final & storm_mask_final & burst_mask_final),
        ]) #
        plot_cuts_amplitudes(base_times_for_cuts, base_traces_for_cuts, "Max Amplitude", plot_folder_station, LIVETIME_THRESHOLD_SECONDS, cuts_dict_for_plotting) #
        plot_cuts_rates(base_times_for_cuts, output_dir=plot_folder_station, cuts_to_plot_dict=cuts_dict_for_plotting) #
        ic(f"Single-station processing for Station {current_station_id} complete.")
        gc.collect()

    else:
        # --- OVERLAP ANALYSIS MODE ---
        STATIONS_FOR_OVERLAP_ANALYSIS = args.stations_for_overlap
        ic("\n\n" + "*"*20)
        ic(f"MODE: Overlap Analysis for Stations: {STATIONS_FOR_OVERLAP_ANALYSIS}, Date Ref: {date_filter}")
        ic("*"*20)

        all_stations_loaded_reports = {}
        stations_with_missing_data = []

        for st_id_overlap in STATIONS_FOR_OVERLAP_ANALYSIS:
            station_livetime_input_dir = os.path.join(plot_folder_base, f"Station{st_id_overlap}", "livetime_data")
            station_gti_file_to_load = os.path.join(station_livetime_input_dir, f"livetime_gti_St{st_id_overlap}_{date_filter}.pkl")
            
            loaded_data = _load_pickle(station_gti_file_to_load) #
            if loaded_data is not None :
                all_stations_loaded_reports[st_id_overlap] = loaded_data
                if not loaded_data: 
                     ic(f"Loaded empty livetime/GTI data for Station {st_id_overlap} (likely no data post initial filters).")
                else:
                     ic(f"Successfully loaded livetime/GTI data for Station {st_id_overlap}")
            else:
                ic(f"WARNING: Livetime/GTI data file not found or failed to load for Station {st_id_overlap}: {station_gti_file_to_load}")
                stations_with_missing_data.append(st_id_overlap)
        
        if stations_with_missing_data:
            ic(f"Proceeding with overlap analysis, but NOTE: Data was missing or failed to load for stations: {stations_with_missing_data}.")
        
        final_station_ids_for_overlap = sorted(all_stations_loaded_reports.keys())

        if not final_station_ids_for_overlap:
            ic("No station livetime data available for any station. Cannot perform overlap analysis. Exiting.")
            exit(0)
        
        ic(f"Overlap analysis will use data from stations: {final_station_ids_for_overlap}")

        # --- Requirement 1: Combinatorial Livetime Report (from C00_eventSearchCuts.py) ---
        if len(final_station_ids_for_overlap) >= 1: 
            combo_livetime_file_path = os.path.join(plot_folder_base, f"combinatorial_livetimes_{date_filter}.txt")
            ic(f"Generating combinatorial livetime report to: {combo_livetime_file_path}")
            with open(combo_livetime_file_path, "w") as f_combo:
                f_combo.write(f"COMBINATORIAL LIVETIME REPORT (Date: {date_filter})\nStations considered: {final_station_ids_for_overlap}\nLivetime Threshold: {LIVETIME_THRESHOLD_SECONDS / 3600.0:.1f} hours\n\n")
                for stage_label in REPORT_CUT_STAGES.keys(): #
                    f_combo.write(f"--- Cut Stage: {stage_label} ---\n")
                    if len(final_station_ids_for_overlap) >=2:
                        for k in range(2, len(final_station_ids_for_overlap) + 1):
                            f_combo.write(f"  Combinations of {k} stations:\n"); found_any_for_k = False
                            for station_combo_ids_tuple in itertools.combinations(final_station_ids_for_overlap, k):
                                combo_gti_dict = {}; valid_combo = True
                                for st_id_in_combo in station_combo_ids_tuple:
                                    if stage_label in all_stations_loaded_reports.get(st_id_in_combo, {}): combo_gti_dict[st_id_in_combo] = all_stations_loaded_reports[st_id_in_combo][stage_label][1]
                                    else: valid_combo = False; break
                                if valid_combo and len(combo_gti_dict) == k :
                                    overlap_s, _ = calculate_stations_combination_overlap(combo_gti_dict) #
                                    if overlap_s > 1e-9 : f_combo.write(f"    {list(station_combo_ids_tuple)} : {format_duration(overlap_s)}\n"); found_any_for_k = True #
                            if not found_any_for_k and k > 1 : f_combo.write(f"    (No combinations of {k} stations had >0s overlap for this cut stage)\n")       
                    else: f_combo.write(f"  (Need at least 2 stations for combinatorial overlap)\n")
                    f_combo.write("\n")

        # --- Requirement 2 & 3: Enhanced Station Summary & N-Concurrent Data & Timestrip Plots (from C00_eventSearchCuts.py) ---
        summary_report_path = os.path.join(plot_folder_base, f"{date_filter}_station_summary_livetimes.txt")
        ic(f"Generating enhanced station summary report to: {summary_report_path}")
        with open(summary_report_path, "w") as f_stat:
            f_stat.write(f"STATION LIVETIMES REPORT\nDate Ref: {date_filter}\nLivetime Threshold: {LIVETIME_THRESHOLD_SECONDS / 3600.0:.1f} hours\n\nINDIVIDUAL STATION LIVETIMES REPORT\n") #
            for st_id_report in STATIONS_FOR_OVERLAP_ANALYSIS: 
                f_stat.write(f"Station {st_id_report}:\n")
                report_data = all_stations_loaded_reports.get(st_id_report)
                if report_data:
                    for stage_label, (lt_s, _) in report_data.items(): f_stat.write(f"  {stage_label}: {format_duration(lt_s)}\n") #
                else: f_stat.write(f"  Data not processed or not found for this station.\n")
                f_stat.write("\n")
            if final_station_ids_for_overlap:
                f_stat.write(f"CONCURRENT STATION LIVETIMES\nStations in this analysis: {final_station_ids_for_overlap}\n\n") #
                for stage_label in REPORT_CUT_STAGES.keys(): #
                    f_stat.write(f"  --- Cut Stage: {stage_label} ---\n")
                    gtis_for_all_relevant = {st_id: all_stations_loaded_reports[st_id][stage_label][1] for st_id in final_station_ids_for_overlap if stage_label in all_stations_loaded_reports.get(st_id,{}) and all_stations_loaded_reports[st_id][stage_label][1]}
                    if not gtis_for_all_relevant: f_stat.write("    No station data for N-concurrent analysis.\n"); continue
                    max_N_possible = len(gtis_for_all_relevant)
                    if max_N_possible == 0: continue
                    for N_val in range(1, max_N_possible + 1):
                        lt_N_s, gtis_N = calculate_N_or_more_stations_livetime(gtis_for_all_relevant, N_val) #
                        label_N = f"At least {N_val} station{'s' if N_val > 1 else ''}"
                        if N_val == 1: label_N += " (Union)"
                        if N_val == max_N_possible and max_N_possible > 1: label_N = f"All {max_N_possible} stations concurrently"
                        f_stat.write(f"    {label_N}: {format_duration(lt_N_s)}\n") #
                        
                        if N_val ==1 and lt_N_s > 0: 
                            for year_plot in range(2013, 2020):
                                season_s_dt = datetime.datetime(year_plot, 10, 1); season_e_dt = datetime.datetime(year_plot + 1, 4, 30, 23, 59, 59)
                                plot_s_unix, plot_e_unix = season_s_dt.timestamp(), season_e_dt.timestamp()
                                data_in_season_flag = any(any(max(g_s, plot_s_unix) < min(g_e, plot_e_unix) for g_s, g_e in g_list) for g_list in gtis_for_all_relevant.values())
                                if not data_in_season_flag : continue
                                plot_concurrent_station_summary_strips(gtis_for_all_relevant, plot_s_unix, plot_e_unix, plot_folder_base, f"{stage_label}_season_{year_plot}-{year_plot+1}", list(gtis_for_all_relevant.keys()), f"Season {year_plot}-{year_plot+1} - ") #
                    f_stat.write("\n")
        ic("Enhanced station summary report and timestrip plots (if any) generation complete.")
    ic("Script finished.")