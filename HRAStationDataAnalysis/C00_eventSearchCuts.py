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
from matplotlib.lines import Line2D

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

# --- GTI Manipulation Functions (Updated merge_gti_list) ---
def merge_gti_list(gti_list_input):
    """Merges a list of [start, end] time intervals. Sorts and handles overlaps/touches."""
    if not gti_list_input: 
        return []
    
    # Create a copy to sort in place without modifying the original list if it's passed by reference elsewhere
    gti_list = sorted(list(gti_list_input), key=lambda x: (x[0], x[1]))
    
    merged = []
    if not gti_list: # Still possible if input was e.g. [None] which became []
        return merged

    current_start, current_end = gti_list[0]
    
    for i in range(1, len(gti_list)):
        next_start, next_end = gti_list[i]
        # Merge if the next period starts before or EXACTLY where the current one ends
        if next_start <= current_end: 
            current_end = max(current_end, next_end) # Extend the current period
        else: 
            # Gap found, finalize the current period and start a new one
            merged.append([current_start, current_end])
            current_start, current_end = next_start, next_end
            
    merged.append([current_start, current_end]) # Add the last processed or single period
    return merged

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

def subtract_gti_lists(source_gti, subtraction_gti):
    """
    Subtracts a list of time intervals (subtraction_gti) from another (source_gti).
    Handles overlaps and splits intervals correctly.
    """
    if not source_gti:
        return []
    if not subtraction_gti:
        return source_gti

    # It's crucial that both lists are pre-merged to handle their own overlaps.
    merged_source = merge_gti_list(source_gti)
    merged_subtraction = merge_gti_list(subtraction_gti)
    
    current_gti = merged_source
    
    # Sequentially apply each subtraction interval to the list of good intervals
    for sub_start, sub_end in merged_subtraction:
        next_gti = []
        for src_start, src_end in current_gti:
            # No overlap: source is entirely before or after the subtraction interval
            if src_end <= sub_start or src_start >= sub_end:
                next_gti.append([src_start, src_end])
                continue

            # At this point, an overlap is guaranteed.
            
            # Add the part of the source interval that is *before* the subtraction
            if src_start < sub_start:
                next_gti.append([src_start, sub_start])
            
            # Add the part of the source interval that is *after* the subtraction
            if src_end > sub_end:
                next_gti.append([sub_end, src_end])
                
        # The result of this subtraction becomes the source for the next subtraction interval
        current_gti = next_gti 
        
    # The final list might have fragmented intervals that can be merged.
    return merge_gti_list(current_gti)

# --- Livetime and Overlap Calculation Functions  ---
def calculate_livetime(times_survived_input, threshold_seconds, start_bound=None, end_bound=None,
                       all_times_before_cuts=None, final_mask=None):
    """
    Calculates approximate livetime and active periods from a list of event timestamps.
    1. Defines initial active periods based on `times_survived_input`.
    2. If `all_times_before_cuts` and `final_mask` are given, it finds periods of consecutive
       "bad" events and subtracts them from the initial active periods.
    3. Merges the final periods to get final GTIs and total livetime.
    """
    ic(f"calculate_livetime called with {len(times_survived_input) if hasattr(times_survived_input, '__len__') else 'N/A'} survived events")

    # --- Part 1: Generate initial GTIs from the "good" (survived) events ---
    def _generate_raw_periods(times, threshold):
        # This helper contains the core logic for turning timestamps into intervals.
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if len(times) == 0:
            return []
            
        times_unique_sorted = np.unique(times)
        n = len(times_unique_sorted)
        raw_periods = []
        
        if n == 1:
            t_event = times_unique_sorted[0]
            duration_singular = 0.5 * threshold
            raw_periods.append([t_event - duration_singular / 2.0, t_event + duration_singular / 2.0])
        elif n > 1:
            # Connected events
            for i in range(n - 1):
                t_current, t_next = times_unique_sorted[i], times_unique_sorted[i+1]
                if (t_next - t_current) < threshold:
                    raw_periods.append([t_current, t_next])
            # Singular events
            for i in range(n):
                t_event = times_unique_sorted[i]
                connected_to_prev = (i > 0) and ((t_event - times_unique_sorted[i-1]) < threshold)
                connected_to_next = (i < n - 1) and ((times_unique_sorted[i+1] - t_event) < threshold)
                if not connected_to_prev and not connected_to_next:
                    duration_singular = 0.5 * threshold
                    raw_periods.append([t_event - duration_singular / 2.0, t_event + duration_singular / 2.0])
        return raw_periods

    # Apply bounds to the survived times
    times_survived_bounded = np.array(times_survived_input)
    if start_bound is not None:
        times_survived_bounded = times_survived_bounded[times_survived_bounded >= start_bound]
    if end_bound is not None:
        times_survived_bounded = times_survived_bounded[times_survived_bounded <= end_bound]

    if len(times_survived_bounded) == 0:
        ic("No survived events within the given bounds. Livetime: 0.0s")
        return 0.0, []

    raw_good_periods = _generate_raw_periods(times_survived_bounded, threshold_seconds)
    good_gtis = merge_gti_list(raw_good_periods)

    # --- Part 2: Subtract "bad" periods if exclusion data is provided ---
    final_gtis = good_gtis
    if all_times_before_cuts is not None and final_mask is not None:
        if len(all_times_before_cuts) == len(final_mask):
            bad_times_mask = ~final_mask
            times_to_exclude = all_times_before_cuts[bad_times_mask]

            if len(times_to_exclude) > 0:
                ic(f"Found {len(times_to_exclude)} events to be excluded from livetime.")
                
                # Apply the same bounds to the exclusion times
                if start_bound is not None:
                    times_to_exclude = times_to_exclude[times_to_exclude >= start_bound]
                if end_bound is not None:
                    times_to_exclude = times_to_exclude[times_to_exclude <= end_bound]

                if len(times_to_exclude) > 0:
                    # FIX: Use a small, fixed threshold to define veto periods around bad events,
                    # instead of the large livetime-linking threshold. This prevents incorrectly
                    # removing large good-time periods between sparse bad events.
                    BAD_EVENT_VETO_THRESHOLD_S = 300.0 # 5 minutes to string bad events together 
                    ic(f"Using a fixed threshold of {BAD_EVENT_VETO_THRESHOLD_S}s to generate exclusion intervals.")
                    
                    raw_bad_periods = _generate_raw_periods(times_to_exclude, BAD_EVENT_VETO_THRESHOLD_S)
                    bad_gtis = merge_gti_list(raw_bad_periods)
                    
                    if bad_gtis:
                        ic(f"Subtracting {len(bad_gtis)} bad time interval(s) from {len(good_gtis)} good interval(s).")
                        final_gtis = subtract_gti_lists(good_gtis, bad_gtis)
        else:
            ic(f"Warning: Mismatch between all_times_before_cuts ({len(all_times_before_cuts)}) and final_mask ({len(final_mask)}). Cannot perform exclusion.")

    # --- Part 3: Calculate total livetime from the final GTIs ---
    total_livetime_seconds = sum(end - start for start, end in final_gtis)
    ic(f"Final Livetime: {total_livetime_seconds:.2f}s ({format_duration_short(total_livetime_seconds)}). Num final periods: {len(final_gtis)}")
    
    return total_livetime_seconds, final_gtis


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

# --- Cut Function ---
def cluster_cut(times, max_amplitudes_per_event, event_ids, amplitude_threshold, time_period, cut_frequency):
    """
    Creates a boolean mask to identify and cut on events clustered in time.

    This function performs a rolling calculation to find periods where n events
    (where n >= cut_frequency) occur within a specified time_period and have
    amplitudes greater than or equal to the amplitude_threshold.

    Args:
        times (np.ndarray): A 1D numpy array of timestamps for each event.
        max_amplitudes_per_event (np.ndarray): A 1D numpy array of the maximum
                                               amplitude for each event.
        event_ids (np.ndarray): A 1D numpy array of unique event identifiers.
        amplitude_threshold (float): The minimum amplitude for an event to be
                                     considered in a cluster.
        time_period (float): The maximum time duration for a cluster of events.
        cut_frequency (int): The minimum number of events with high amplitude
                             to form a cluster.

    Returns:
        np.ndarray: A boolean numpy array (mask) of the same length as the input
                    arrays. Events that are part of a cluster are marked as True.
    """

    # Ensure the input arrays are numpy arrays
    times = np.asarray(times)
    max_amplitudes_per_event = np.asarray(max_amplitudes_per_event)
    event_ids = np.asarray(event_ids)

    # Sort the events by time
    sort_indices = np.argsort(times)
    sorted_times = times[sort_indices]
    sorted_amplitudes = max_amplitudes_per_event[sort_indices]
    
    # Initialize the boolean mask that will be returned
    mask = np.ones_like(times, dtype=bool)
    
    # Pointers for the sliding window
    start_index = 0
    
    # Counter for events in the window that are above the amplitude threshold
    high_amplitude_count = 0

    # Iterate through the sorted events with the end of the window
    for end_index in range(len(sorted_times)):
        
        # Check if the current event's amplitude is above the threshold
        if sorted_amplitudes[end_index] >= amplitude_threshold:
            high_amplitude_count += 1

        # Shrink the window from the left (start_index) if the time period is exceeded
        while sorted_times[end_index] - sorted_times[start_index] > time_period:
            # If the event being removed was a high-amplitude event, decrement the count
            if sorted_amplitudes[start_index] >= amplitude_threshold:
                high_amplitude_count -= 1
            start_index += 1

        # If the number of high-amplitude events in the window meets the frequency cut
        if high_amplitude_count >= cut_frequency:
            # Mark all events within the current window in the mask
            original_indices = sort_indices[start_index:end_index + 1]
            mask[original_indices] = False

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
def plot_cuts_amplitudes(times_unix, values_data, amp_name, output_dir=".",
                         livetime_threshold_seconds=3600.0,
                         cuts_to_plot_dict=None, is_max_amp_data=False,
                         final_cut_mask_for_gti_fill=None): # New argument
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    dt_times_all_events = np.array([datetime.datetime.fromtimestamp(t) if t > 0 else datetime.datetime(1970,1,1) for t in times_unix])
    
    max_amps_to_plot = None
    # ... (max_amps_to_plot calculation remains the same as your previous version) ...
    if is_max_amp_data:
        if values_data.ndim == 1 and len(values_data) == len(times_unix): max_amps_to_plot = np.array(values_data)
        else: ic(f"Values_data (max_amp_data) shape {values_data.shape} mismatch."); return
    elif values_data.ndim == 3 and values_data.shape[1:3] == (4, 256): max_amps_to_plot = np.max(np.abs(values_data), axis=(1, 2))
    elif values_data.ndim == 1 and len(values_data) == len(times_unix): max_amps_to_plot = np.array(values_data)
    else: ic(f"Unexpected 'values_data' shape {values_data.shape}."); return

    valid_data_mask_global = np.isfinite(times_unix) & np.isfinite(max_amps_to_plot)
    
    for start_year in range(2013, 2020):
        markers = itertools.cycle(("v", "s", "*", "d", "P", "X"))
        season_start_dt = datetime.datetime(start_year, 10, 1); season_end_dt = datetime.datetime(start_year + 1, 4, 30, 23, 59, 59)
        season_start_unix, season_end_unix = season_start_dt.timestamp(), season_end_dt.timestamp()
        
        seasonal_dt_mask_plotting = (dt_times_all_events >= season_start_dt) & (dt_times_all_events <= season_end_dt)
        base_seasonal_mask_for_plot = seasonal_dt_mask_plotting & valid_data_mask_global

        if not np.any(base_seasonal_mask_for_plot):
            continue
            
        plt.figure(figsize=(14,7))
        ax = plt.gca()
        plt.title(f"Season {start_year}-{start_year + 1} Activity - {amp_name}")
        plt.xlabel("Time"); plt.ylabel(amp_name)
        
        legend_handles = []

        # --- Plot Final Livetime Periods (Background Fill) ---
        active_periods_to_fill_final_cut = []
        if final_cut_mask_for_gti_fill is not None and \
           isinstance(final_cut_mask_for_gti_fill, np.ndarray) and \
           len(final_cut_mask_for_gti_fill) == len(times_unix):
            
            times_for_final_gti_calc = times_unix[final_cut_mask_for_gti_fill] # Apply the most restrictive cut
            _, active_periods_to_fill_final_cut = calculate_livetime(
                times_for_final_gti_calc,
                livetime_threshold_seconds,
                season_start_unix,
                season_end_unix,
                all_times_before_cuts=times_unix,
                final_mask=final_cut_mask_for_gti_fill
            )
        # This fill will be done after data points are plotted, to get y-axis range.
        # Store the periods for now.

        # --- Plot Data Points ---
        # "All Events (station data)" points and its livetime in legend
        lt_all_s, _ = calculate_livetime(times_unix, livetime_threshold_seconds, season_start_unix, season_end_unix)
        label_all_events = f"All Events (station data) (Livetime: {format_duration_short(lt_all_s)})"
        ax.scatter(dt_times_all_events[base_seasonal_mask_for_plot], 
                   max_amps_to_plot[base_seasonal_mask_for_plot], 
                   color="lightgray", s=10, label="_nolegend_", alpha=0.7, edgecolor='k', linewidths=0.5)
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markeredgecolor='k', markerfacecolor='lightgray', markersize=5, label=label_all_events))

        data_y_values_for_ylim = []
        if np.any(base_seasonal_mask_for_plot):
            data_y_values_for_ylim.extend(max_amps_to_plot[base_seasonal_mask_for_plot])

        # Plot for specific cuts
        if cuts_to_plot_dict:
            # Get the actual marker objects to use in legend, advance cycle manually
            marker_instances = [next(markers) for _ in cuts_to_plot_dict] 
            
            for idx, (legend_label_base, cut_mask_for_series) in enumerate(cuts_to_plot_dict.items()):
                if not (isinstance(cut_mask_for_series, np.ndarray) and cut_mask_for_series.dtype == bool and len(cut_mask_for_series) == len(times_unix)): continue
                
                final_series_mask_global = valid_data_mask_global & cut_mask_for_series
                final_series_mask_plotting = base_seasonal_mask_for_plot & cut_mask_for_series
                
                current_marker = marker_instances[idx % len(marker_instances)] # Use pre-cycled marker

                if np.any(final_series_mask_plotting):
                    # Use a color for the series points if desired, or let scatter auto-cycle
                    points_collection = ax.scatter(dt_times_all_events[final_series_mask_plotting], 
                               max_amps_to_plot[final_series_mask_plotting], 
                               s=15, label="_nolegend_", marker=current_marker, alpha=0.8)
                    if np.any(final_series_mask_plotting):
                         data_y_values_for_ylim.extend(max_amps_to_plot[final_series_mask_plotting])
                else: # No points, but still need a dummy for legend color
                    points_collection = ax.scatter([],[], s=15, marker=current_marker, alpha=0.8)


                times_for_this_cut_series = times_unix[final_series_mask_global]
                lt_s, _ = calculate_livetime(
                    times_for_this_cut_series, 
                    livetime_threshold_seconds, 
                    season_start_unix, 
                    season_end_unix,
                    all_times_before_cuts=times_unix,
                    final_mask=final_series_mask_global
                )

                plot_label_with_livetime = f"{legend_label_base} (Livetime: {format_duration_short(lt_s)})"
                legend_handles.append(Line2D([0], [0], marker=current_marker, color='w', 
                                             markerfacecolor=points_collection.get_facecolor()[0], # Get color from actual plot
                                             markeredgecolor=points_collection.get_edgecolor()[0],
                                             markersize=5, label=plot_label_with_livetime))
        
        # Determine y-limits from plotted data
        if data_y_values_for_ylim:
            min_y_data = np.nanmin([y for y in data_y_values_for_ylim if np.isfinite(y)])
            max_y_data = np.nanmax([y for y in data_y_values_for_ylim if np.isfinite(y)])
            if np.isnan(min_y_data) or np.isnan(max_y_data): # Fallback if all NaNs
                min_y_data, max_y_data = (0,1) if "Chi" in amp_name else (0, 0.5 if is_max_amp_data else 1)
            if min_y_data == max_y_data: min_y_data -= 0.1*(abs(min_y_data)+1e-6); max_y_data += 0.1*(abs(max_y_data)+1e-6)
            padding = (max_y_data - min_y_data) * 0.05 if (max_y_data - min_y_data) > 1e-6 else 0.1
            current_ymin, current_ymax = min_y_data - padding, max_y_data + padding
            ax.set_ylim(current_ymin, current_ymax)
        else: # Default if no points plotted
            current_ymin, current_ymax = (0, 1.05) if "Chi" in amp_name else (0, 0.5 if is_max_amp_data else 1.0)
            ax.set_ylim(current_ymin, current_ymax)


        # Plot the final livetime fill using the determined y-limits
        if active_periods_to_fill_final_cut:
            for start_p, end_p in active_periods_to_fill_final_cut:
                dt_start = datetime.datetime.fromtimestamp(start_p)
                dt_end = datetime.datetime.fromtimestamp(end_p)
                ax.fill_betweenx(y=[current_ymin, current_ymax], x1=dt_start, x2=dt_end, 
                                 color='red', alpha=0.1, zorder=-2) # Low zorder
            legend_handles.append(plt.Rectangle((0,0),1,1,fc="red", alpha=0.1, label="Final Cut Active Periods"))

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
        plt.gcf().autofmt_xdate()
        ax.legend(handles=legend_handles, fontsize=8, loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 0.82, 1])

        filename = os.path.join(output_dir, f"{amp_name.replace(' ', '_')}_season_{start_year}_{start_year+1}.png")
        plt.savefig(filename); plt.close(); ic(f"Saved: {filename}")



def plot_cuts_rates(times_unix, bin_size_seconds=30*60, output_dir=".", 
                    cuts_to_plot_dict=None, livetime_threshold_seconds=3600.0,
                    final_cut_mask_for_gti_fill=None): # New argument
    # ... (setup and seasonal loop as before) ...
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    valid_times_mask_global = np.isfinite(times_unix) & (times_unix > 0)
    times_unix_valid_global = times_unix[valid_times_mask_global] 
    if len(times_unix_valid_global) == 0: ic("No valid times for rate plotting."); return

    for start_year in range(2013, 2020):
        markers = itertools.cycle(("v", "s", "*", "d", "P", "X"))
        season_start_dt = datetime.datetime(start_year, 10, 1); season_end_dt = datetime.datetime(start_year + 1, 4, 30, 23, 59, 59)
        season_start_unix, season_end_unix = season_start_dt.timestamp(), season_end_dt.timestamp()
        seasonal_unix_mask_for_plot = (times_unix_valid_global >= season_start_unix) & (times_unix_valid_global <= season_end_unix)
        if not np.any(seasonal_unix_mask_for_plot): continue
        
        bins = np.arange(season_start_unix, season_end_unix + bin_size_seconds, bin_size_seconds)
        if len(bins) < 2 : continue
        bin_centers_unix = (bins[:-1] + bins[1:]) / 2.0
        dt_bin_centers = [datetime.datetime.fromtimestamp(ts) for ts in bin_centers_unix]
        
        plt.figure(figsize=(14, 7))
        ax = plt.gca()
        plt.title(f"Season {start_year}-{start_year + 1} Event Rate")
        plt.xlabel("Time"); plt.ylabel(f"Event Rate (Hz, {bin_size_seconds/60:.0f}min bins)")
        
        legend_handles = []
        # Set y-limits first for fill_betweenx
        plot_ymin, plot_ymax = 1e-5, 10 # Default y-range for rate plot
        ax.set_yscale('log'); ax.set_ylim(plot_ymin, plot_ymax)


        # --- Plot Final Livetime Periods (Background Fill) ---
        if final_cut_mask_for_gti_fill is not None and \
           isinstance(final_cut_mask_for_gti_fill, np.ndarray) and \
           len(final_cut_mask_for_gti_fill) == len(times_unix):
            
            times_for_final_gti_calc = times_unix[final_cut_mask_for_gti_fill]
            _, active_periods_to_fill_final_cut = calculate_livetime(
                times_for_final_gti_calc,
                livetime_threshold_seconds,
                season_start_unix,
                season_end_unix,
                all_times_before_cuts=times_unix,
                final_mask=final_cut_mask_for_gti_fill
            )
            if active_periods_to_fill_final_cut:
                for start_p, end_p in active_periods_to_fill_final_cut:
                    dt_start = datetime.datetime.fromtimestamp(start_p)
                    dt_end = datetime.datetime.fromtimestamp(end_p)
                    ax.fill_betweenx(y=[plot_ymin, plot_ymax], x1=dt_start, x2=dt_end, 
                                     color='red', alpha=0.1, zorder=-2) 
                legend_handles.append(plt.Rectangle((0,0),1,1,fc="red", alpha=0.1, label="Final Cut Active Periods"))

        # --- Plot Rate Data Points ---
        current_season_times_all_for_hist = times_unix_valid_global[seasonal_unix_mask_for_plot]
        min_overall_rate_in_season = plot_ymax # For adjusting ylim later
        if len(current_season_times_all_for_hist) > 0:
            count_all, _ = np.histogram(current_season_times_all_for_hist, bins=bins)
            rate_all = count_all / bin_size_seconds
            # Use plot for scatter-like appearance, connect if desired
            line_all, = plt.plot(dt_bin_centers, rate_all, linestyle='None', marker='o', markersize=3, label="All Events Rate", color="lightgray", alpha=0.7) # Changed to plot
            legend_handles.append(line_all)
            if np.any(rate_all > 0): min_overall_rate_in_season = min(min_overall_rate_in_season, np.min(rate_all[rate_all>0]))


        if cuts_to_plot_dict:
            # Get marker instances for legend
            marker_instances_rate = [next(markers) for _ in cuts_to_plot_dict]
            for idx, (legend_label_base, cut_mask_for_series_orig_len) in enumerate(cuts_to_plot_dict.items()):
                if not (isinstance(cut_mask_for_series_orig_len, np.ndarray) and 
                        cut_mask_for_series_orig_len.dtype == bool and 
                        len(cut_mask_for_series_orig_len) == len(times_unix)): continue

                cut_mask_aligned_with_valid_global = cut_mask_for_series_orig_len[valid_times_mask_global]
                times_for_series_cut_in_season = times_unix_valid_global[seasonal_unix_mask_for_plot & cut_mask_aligned_with_valid_global]
                
                current_marker_rate = marker_instances_rate[idx % len(marker_instances_rate)]

                if len(times_for_series_cut_in_season) > 0:
                    count_cut, _ = np.histogram(times_for_series_cut_in_season, bins=bins)
                    rate_cut = count_cut / bin_size_seconds
                    line_cut, = plt.plot(dt_bin_centers, rate_cut, linestyle='None', marker=current_marker_rate, markersize=3, label=f'{legend_label_base} Rate') # Changed to plot
                    legend_handles.append(line_cut)
                    if np.any(rate_cut > 0): min_overall_rate_in_season = min(min_overall_rate_in_season, np.min(rate_cut[rate_cut>0]))

        if min_overall_rate_in_season < plot_ymax : # Ensure min_overall_rate is valid
             ax.set_ylim(max(plot_ymin, min_overall_rate_in_season / 2.0) , plot_ymax)


        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
        plt.gcf().autofmt_xdate()
        ax.legend(handles=legend_handles, fontsize=8, loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        
        out_filename = os.path.join(output_dir, f"season_{start_year}_{start_year+1}_rate_with_livetime.png")
        plt.savefig(out_filename); plt.close(); ic(f"Saved: {out_filename}")

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
# LIVETIME_THRESHOLD_SECONDS = 3600.0   # 1 hour
LIVETIME_THRESHOLD_SECONDS = 86400.0       # 1 day
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
    parser.add_argument('--date_processing', type=str, required=False, default=None, help="Date for processing, if different from --date.")

    args = parser.parse_args()
    date_filter = args.date
    date_save = args.date_processing if args.date_processing else date_filter
    ic.enable()

    # --- Path Definitions ---
    base_project_path = 'HRAStationDataAnalysis'
    base_data_path = os.path.join(base_project_path, 'StationData')
    # Corrected path to align with C00 structure if nurFiles are directly under StationData
    station_data_folder = os.path.join(base_data_path, 'nurFiles', date_filter) # From your C00 structure for data files
    cuts_data_folder = os.path.join(base_data_path, 'cuts', date_save) # From your C00 structure for cut files
    plot_folder_base = os.path.join(base_project_path, 'plots', date_save)

    os.makedirs(cuts_data_folder, exist_ok=True)
    os.makedirs(plot_folder_base, exist_ok=True)

    # --- Mode Selection ---
    if args.stnID is not None:
        # --- SINGLE-STATION PROCESSING MODE ---
        current_station_id = args.stnID
        # ... (setup paths and initial ic messages) ...
        ic("\n\n" + "*"*20); ic(f"MODE: Single-Station Processing for St {current_station_id}, Date: {date_filter}"); ic("*"*20)
        plot_folder_station = os.path.join(plot_folder_base, f'Station{current_station_id}'); os.makedirs(plot_folder_station, exist_ok=True)
        station_livetime_output_dir = os.path.join(plot_folder_station, "livetime_data"); os.makedirs(station_livetime_output_dir, exist_ok=True)

        try:
            time_files = sorted(glob.glob(os.path.join(station_data_folder, f'{date_filter}_Station{current_station_id}_Times*')))
            trace_files = sorted(glob.glob(os.path.join(station_data_folder, f'{date_filter}_Station{current_station_id}_Traces*'))) # Paths to trace parts
            eventid_files = sorted(glob.glob(os.path.join(station_data_folder, f'{date_filter}_Station{current_station_id}_EventIDs*')))

            if not time_files or not trace_files or not eventid_files:
                raise FileNotFoundError(f"Core data files (Times, Traces, or EventIDs) missing for St {current_station_id}, Date {date_filter}")

            times_list = [np.load(f) for f in time_files]; times_raw = np.concatenate(times_list, axis=0).squeeze()
            eventids_list = [np.load(f) for f in eventid_files]; eventids_raw = np.concatenate(eventids_list, axis=0).squeeze()
            if times_raw.ndim == 0: times_raw = np.array([times_raw.item()])
            if eventids_raw.ndim == 0: eventids_raw = np.array([eventids_raw.item()])
            
            # --- Max Amplitudes & Traces Loading/Calculation ---
            max_amplitudes_parts_collected = []
            traces_parts_collected = [] # Still collect trace parts for L1/plotting

            ic("Processing Traces and MaxAmplitudes (part by part if necessary)...")
            for i, trace_file_path in enumerate(trace_files):
                expected_max_amp_f_path = trace_file_path.replace("_Traces_", "_MaxAmplitudes_")
                
                current_trace_part = np.load(trace_file_path)
                # Ensure current_trace_part has a leading dimension for N_events in part
                if current_trace_part.ndim == 2 and current_trace_part.shape == (4,256): current_trace_part = current_trace_part.reshape(1,4,256)
                elif current_trace_part.ndim == 1 and current_trace_part.size == 4*256: current_trace_part = current_trace_part.reshape(1,4,256)
                elif current_trace_part.ndim != 3 or current_trace_part.shape[1:3] != (4,256): # Check if not (N,4,256)
                    if current_trace_part.size == 0: # Empty part
                         ic(f"Trace part {trace_file_path} is empty.")
                         traces_parts_collected.append(np.array([]).reshape(0,4,256))
                         max_amplitudes_parts_collected.append(np.array([]))
                         if not os.path.exists(expected_max_amp_f_path): # Save empty if calc needed
                            np.save(expected_max_amp_f_path, np.array([]))
                         continue # Skip to next part
                    else:
                        ic(f"Warning: Trace part {trace_file_path} has unexpected shape {current_trace_part.shape}. Attempting reshape or skipping.")
                        # Attempt to reshape based on total size, if fails, this part is problematic
                        try:
                            num_events_in_part_candidate = current_trace_part.size // (4*256)
                            if current_trace_part.size % (4*256) == 0:
                                current_trace_part = current_trace_part.reshape(num_events_in_part_candidate, 4, 256)
                                ic(f"Reshaped trace part to {current_trace_part.shape}")
                            else:
                                raise ValueError("Cannot reshape to N,4,256")
                        except ValueError as e:
                            ic(f"Could not reshape trace part {trace_file_path}: {e}. Skipping this part for traces and max_amps.")
                            traces_parts_collected.append(np.array([]).reshape(0,4,256)) # Add empty placeholder
                            max_amplitudes_parts_collected.append(np.array([])) # Add empty placeholder
                            continue


                traces_parts_collected.append(current_trace_part)
                
                max_amp_part_data = None
                if os.path.exists(expected_max_amp_f_path):
                    try:
                        loaded_max_amp_part = np.load(expected_max_amp_f_path)
                        # Validate length against the number of events in the current trace part
                        if loaded_max_amp_part.ndim == 1 and loaded_max_amp_part.shape[0] == current_trace_part.shape[0]:
                            max_amp_part_data = loaded_max_amp_part
                            # ic(f"Loaded valid MaxAmplitudes part: {expected_max_amp_f_path}")
                        elif loaded_max_amp_part.ndim == 0 and current_trace_part.shape[0] == 1: # Scalar saved for single event part
                            max_amp_part_data = np.array([loaded_max_amp_part.item()])
                        elif current_trace_part.shape[0] == 0 and loaded_max_amp_part.size == 0: # both empty
                            max_amp_part_data = np.array([])
                        else:
                            ic(f"MaxAmplitudes part {expected_max_amp_f_path} event count mismatch (Trace evts: {current_trace_part.shape[0]}, MaxAmp evts: {loaded_max_amp_part.shape}). Recalculating.")
                    except Exception as e:
                        ic(f"Error loading MaxAmplitudes part {expected_max_amp_f_path}: {e}. Recalculating.")
                
                if max_amp_part_data is None: # Calculate if missing or validation failed
                    ic(f"Calculating MaxAmplitudes for part: {os.path.basename(trace_file_path)}")
                    if current_trace_part.shape[0] == 0: # Empty trace part
                        max_amp_part_data = np.array([])
                    else: # Should be (N_in_part, 4, 256)
                        max_amp_part_data = np.max(np.abs(current_trace_part), axis=(1, 2))
                    
                    np.save(expected_max_amp_f_path, max_amp_part_data)
                    ic(f"Saved calculated MaxAmplitudes part: {expected_max_amp_f_path}")
                max_amplitudes_parts_collected.append(max_amp_part_data)

            # Concatenate all parts
            traces_raw = np.concatenate(traces_parts_collected, axis=0) if traces_parts_collected and any(p.size > 0 for p in traces_parts_collected) else np.array([]).reshape(0,4,256)
            max_amplitudes_raw = np.concatenate(max_amplitudes_parts_collected, axis=0) if max_amplitudes_parts_collected and any(p.size>0 for p in max_amplitudes_parts_collected) else np.array([])
            
            # Final reshape/squeeze for single total event cases if needed
            if traces_raw.ndim == 2 and traces_raw.shape == (4,256) : traces_raw = traces_raw.reshape(1,4,256)
            if max_amplitudes_raw.ndim == 0 and max_amplitudes_raw.size == 1: max_amplitudes_raw = np.array([max_amplitudes_raw.item()])


        except Exception as e:
            ic(f"Error during data loading/MaxAmplitude processing for St {current_station_id}: {e}")
            exit(1)

        if times_raw.size == 0 or eventids_raw.size == 0 or max_amplitudes_raw.size == 0 or \
           not (times_raw.shape[0] == eventids_raw.shape[0] == max_amplitudes_raw.shape[0]):
            ic(f"Empty or mismatched data arrays for St {current_station_id} before trace check. T:{times_raw.shape}, E:{eventids_raw.shape}, MA:{max_amplitudes_raw.shape}")
            # If traces_raw is also needed for L1 and is empty/mismatched, that's an issue too.
            # For now, proceed if these core arrays for cluster_cut are okay.
            if traces_raw.size > 0 and times_raw.shape[0] != traces_raw.shape[0] :
                 ic(f"CRITICAL MISMATCH: Traces count {traces_raw.shape[0]} differs from Times/EventID/MaxAmp count {times_raw.shape[0]}. Aborting.")
                 exit(1)
            elif traces_raw.size == 0 and (times_raw.size > 0): # L1 cut and plotting will fail if traces_raw is empty but other data exists.
                 ic(f"Warning: Traces data is empty while other parameters are not. L1 cut and trace plotting will be affected.")


        # --- Initial Time Filtering (Applied to Times, EventIDs, MaxAmplitudes, and Traces) ---
        # ... (initial_valid_mask calculated from times_raw as before) ...
        # zerotime_mask = (times_raw != 0)
        # min_datetime_threshold = datetime.datetime(2013, 1, 1).timestamp()
        # pretime_mask = (times_raw >= min_datetime_threshold)
        # initial_valid_mask = zerotime_mask & pretime_mask

        # base_times_for_cuts = times_raw[initial_valid_mask]
        # base_event_ids_for_cuts = eventids_raw[initial_valid_mask]
        # base_max_amplitudes_for_cuts = max_amplitudes_raw[initial_valid_mask]
        # # Conditionally filter traces if they were loaded successfully
        # base_traces_for_cuts = traces_raw[initial_valid_mask] if traces_raw.size > 0 else np.array([]).reshape(0,4,256)


        # if base_times_for_cuts.size == 0:
        #     ic(f"No data for Station {current_station_id} after initial time filters. Saving empty report and aborting.")
        #     _save_pickle_atomic({}, os.path.join(station_livetime_output_dir, f"livetime_gti_St{current_station_id}_{date_filter}.pkl"))
        #     exit(0)


        # # Now, identify unique (Time, EventID) pairs from these base arrays
        # # This is simpler than trying to create a mask on original indices.
        # if base_times_for_cuts.size > 0:
        #     # Create pairs of (time, event_id) for finding unique combinations
        #     time_eventid_pairs = np.stack((base_times_for_cuts, base_event_ids_for_cuts), axis=-1)

        #     # Find unique pairs and the indices of their first occurrences
        #     # np.unique returns sorted unique values by default.
        #     # We need 'return_index=True' to get the indices of the first time each unique pair appears.
        #     # These indices will be relative to time_eventid_pairs (and thus to base_..._for_cuts arrays).
        #     _, unique_indices = np.unique(time_eventid_pairs, axis=0, return_index=True)

        #     # Sort these unique_indices to maintain the original time order of the first occurrences.
        #     unique_indices.sort() 

        #     ic(f"Identified {len(unique_indices)} events after removing Time+EventID duplicates from {len(base_times_for_cuts)} events.")

        #     # Apply this uniqueness filter
        #     base_times_for_cuts = base_times_for_cuts[unique_indices]
        #     base_event_ids_for_cuts = base_event_ids_for_cuts[unique_indices]
        #     base_max_amplitudes_for_cuts = base_max_amplitudes_for_cuts[unique_indices]
        #     base_traces_for_cuts = base_traces_for_cuts[unique_indices]
        from HRAStationDataAnalysis.C_utils import getTimeEventMasks
        # Use the utility function to get the initial valid mask and unique indices
        initial_valid_mask, unique_indices = getTimeEventMasks(times_raw, eventids_raw)
        base_times_for_cuts = times_raw[initial_valid_mask][unique_indices]
        base_event_ids_for_cuts = eventids_raw[initial_valid_mask][unique_indices]
        base_max_amplitudes_for_cuts = max_amplitudes_raw[initial_valid_mask][unique_indices]
        base_traces_for_cuts = traces_raw[initial_valid_mask][unique_indices] if traces_raw.size > 0 else np.array([]).reshape(0,4,256)
        if base_times_for_cuts.size == 0:
            ic(f"No data for Station {current_station_id} after initial time filters. Saving empty report and aborting.")
            _save_pickle_atomic({}, os.path.join(station_livetime_output_dir, f"livetime_gti_St{current_station_id}_{date_filter}.pkl"))
            exit(0)


        ic(f"Data for cuts: Times {base_times_for_cuts.shape}, EventIDs {base_event_ids_for_cuts.shape}, MaxAmps {base_max_amplitudes_for_cuts.shape}, Traces {base_traces_for_cuts.shape}")

        # --- Apply Cuts with Incremental Saving (L1 cut needs full traces) ---
        cut_file_path = os.path.join(cuts_data_folder, f'{date_filter}_Station{current_station_id}_Cuts.npy')
        current_all_cut_masks = {} 
        L1_mask_final, storm_mask_final, burst_mask_final = None, None, None

        if os.path.exists(cut_file_path):
            # ... (loading logic for current_all_cut_masks as in previous response) ...
            try:
                loaded_data = np.load(cut_file_path, allow_pickle=True).item()
                if isinstance(loaded_data, dict): current_all_cut_masks = loaded_data
            except: pass # Ignore errors, will recalc
        
        temp_L1 = current_all_cut_masks.get('L1_mask')
        if temp_L1 is not None and isinstance(temp_L1, np.ndarray) and len(temp_L1) == len(base_times_for_cuts): L1_mask_final = temp_L1; ic("Loaded L1_mask.")
        else: L1_mask_final = None; # Invalidate
        # ... (similar validation for storm_mask and burst_mask) ...
        temp_storm = current_all_cut_masks.get('storm_mask')
        if temp_storm is not None and isinstance(temp_storm, np.ndarray) and len(temp_storm) == len(base_times_for_cuts): storm_mask_final = temp_storm; ic("Loaded storm_mask.")
        else: storm_mask_final = None; 
        temp_burst = current_all_cut_masks.get('burst_mask')
        if temp_burst is not None and isinstance(temp_burst, np.ndarray) and len(temp_burst) == len(base_times_for_cuts): burst_mask_final = temp_burst; ic("Loaded burst_mask.")
        else: burst_mask_final = None; 

        if L1_mask_final is None:
            ic(f"Calculating L1 cut...")
            if base_traces_for_cuts.size == 0:
                ic("WARNING: No trace data available for L1 cut. Assuming all pass L1.")
                L1_mask_final = np.ones_like(base_times_for_cuts, dtype=bool)
            else:
                L1_mask_final = L1_cut(base_traces_for_cuts, power_cut=0.3)
            current_all_cut_masks['L1_mask'] = L1_mask_final
            np.save(cut_file_path, current_all_cut_masks, allow_pickle=True); ic("Saved L1_mask.")
        
        if storm_mask_final is None:
            ic(f"Calculating storm cut...")
            storm_mask_final = cluster_cut(base_times_for_cuts, base_max_amplitudes_for_cuts, base_event_ids_for_cuts, # USE MAX AMPS
                                           amplitude_threshold=0.3, 
                                           time_period=datetime.timedelta(seconds=3600).total_seconds(), 
                                           cut_frequency=2)
            current_all_cut_masks['storm_mask'] = storm_mask_final
            np.save(cut_file_path, current_all_cut_masks, allow_pickle=True); ic("Saved storm_mask.")

        if burst_mask_final is None:
            ic(f"Calculating burst cut...")
            burst_mask_final = cluster_cut(base_times_for_cuts, base_max_amplitudes_for_cuts, base_event_ids_for_cuts, # USE MAX AMPS
                                           amplitude_threshold=0.2, 
                                           time_period=datetime.timedelta(seconds=60).total_seconds(), 
                                           cut_frequency=2)
            current_all_cut_masks['burst_mask'] = burst_mask_final
            np.save(cut_file_path, current_all_cut_masks, allow_pickle=True); ic("Saved burst_mask.")

        # ... (Livetime calculation for this station and saving its .pkl: station_specific_report) ...
        # ... (Plotting individual station plots: plot_cuts_amplitudes will now get base_max_amplitudes_for_cuts for "Max Amplitude" plot) ...
        station_specific_report = collections.OrderedDict()
        report_masks = {
            "Total (after initial time filters)": np.ones_like(base_times_for_cuts, dtype=bool),
            "After L1": L1_mask_final,
            "After L1 + Storm": L1_mask_final & storm_mask_final,
            "After L1 + Storm + Burst": L1_mask_final & storm_mask_final & burst_mask_final
        }
        for stage_label in report_masks.keys():
            current_stage_mask = report_masks[stage_label]
            times_survived_stage = base_times_for_cuts[current_stage_mask]
            lt_s, active_periods = calculate_livetime(
                times_survived_stage, 
                LIVETIME_THRESHOLD_SECONDS,
                all_times_before_cuts=base_times_for_cuts, 
                final_mask=current_stage_mask
            )
            station_specific_report[stage_label] = (lt_s, active_periods)
        station_gti_file_to_save = os.path.join(station_livetime_output_dir, f"livetime_gti_St{current_station_id}_{date_filter}.pkl")
        _save_pickle_atomic(station_specific_report, station_gti_file_to_save) # Using your atomic save for pkl

        # Also want to save the livetime report for this station next to the gti file
        livetime_report_file = os.path.join(station_livetime_output_dir, f"livetime_report_St{current_station_id}_{date_filter}.txt")
        with open(livetime_report_file, "w") as f:
            f.write(f"Livetime Report for Station {current_station_id} on {date_filter}\n")
            f.write(f"Livetime Threshold: {LIVETIME_THRESHOLD_SECONDS / 3600.0:.1f} hours\n\n")
            for stage_label, (lt_s, active_periods) in station_specific_report.items():
                f.write(f"--- {stage_label} ---\n")
                t_delta = datetime.timedelta(seconds=lt_s)
                d = datetime.datetime(1, 1, 1) + t_delta
                f.write(f"Livetime: {d.year} years {d.month-1} months, {d.day-1} days, {d.hour} hours, {d.minute} minutes, {d.second} seconds\n")

            f.write("\n")
            f.close()
        ic(f"Saved station-specific GTI report to: {station_gti_file_to_save}")
        ic(f"Saved station-specific livetime report to: {livetime_report_file}")

        # Prepare cuts_dict_for_plotting for plotting
        cuts_dict_for_plotting = collections.OrderedDict([
            ("L1 cut", L1_mask_final),
            ("L1+Storm cut", np.logical_and(L1_mask_final,storm_mask_final)),
            ("L1+Storm+Burst cut", np.logical_and(np.logical_and(L1_mask_final, storm_mask_final), burst_mask_final)),
        ])

        # Define the final overall cut mask
        final_overall_mask = L1_mask_final & storm_mask_final & burst_mask_final # This should align with base_times_for_cuts
        # Plot Max Amplitudes using pre-calculated ones
        plot_cuts_amplitudes(base_times_for_cuts, 
                            base_max_amplitudes_for_cuts, # Or base_traces_for_cuts if not Max Amp plot
                            "Max Amplitude", 
                            plot_folder_station, 
                            LIVETIME_THRESHOLD_SECONDS, 
                            cuts_dict_for_plotting, 
                            is_max_amp_data=True, # True for Max Amp plot
                            final_cut_mask_for_gti_fill=final_overall_mask)
        
        # Plot Chi values if available (these don't depend on traces directly here)
        # ... (load ChiRCR, call plot_cuts_amplitudes with ChiRCR data and is_max_amp_data=False) ...
        plot_cuts_rates(base_times_for_cuts, 
                        output_dir=plot_folder_station, 
                        cuts_to_plot_dict=cuts_dict_for_plotting,
                        livetime_threshold_seconds=LIVETIME_THRESHOLD_SECONDS,
                        final_cut_mask_for_gti_fill=final_overall_mask)       
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