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

# --- Helper for Loading Data (from context) ---
def _load_pickle(filepath):
    """Loads data from a pickle file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            ic(f"Error loading pickle file {filepath}: {e}")
    return None

# --- Coincidence Event Cut Functions ---
def check_chi_cut(event_details, chi_threshold=0.3, min_triggers_passing=2):
    """
    Checks if a coincidence event passes the Chi cut.
    A coincidence passes if at least 'min_triggers_passing' of its constituent
    station triggers have a ChiRCR or Chi2016 value above 'chi_threshold'.
    """
    passing_station_triggers_count = 0
    if not isinstance(event_details, dict):
        return False

    for station_id_str, station_triggers_data in event_details.get("stations", {}).items():
        if not isinstance(station_triggers_data, dict):
            continue
            
        # Use SNR list length as reference for number of triggers, as it's fundamental
        num_triggers_in_station = len(station_triggers_data.get('SNR', []))

        chi_rcr_list = station_triggers_data.get('ChiRCR', [])
        chi_2016_list = station_triggers_data.get('Chi2016', [])

        for i in range(num_triggers_in_station):
            trigger_passed_chi = False
            # Check ChiRCR
            if i < len(chi_rcr_list) and chi_rcr_list[i] is not None and \
               not np.isnan(chi_rcr_list[i]) and chi_rcr_list[i] > chi_threshold:
                trigger_passed_chi = True
            
            # Check Chi2016 (can also pass if ChiRCR didn't)
            if not trigger_passed_chi and i < len(chi_2016_list) and \
               chi_2016_list[i] is not None and not np.isnan(chi_2016_list[i]) and \
               chi_2016_list[i] > chi_threshold:
                trigger_passed_chi = True
            
            if trigger_passed_chi:
                passing_station_triggers_count += 1
    
    return passing_station_triggers_count >= min_triggers_passing

def check_coincidence_cuts(event_details):
    """
    Main function to check if a coincidence event passes all defined analysis cuts.
    Currently only implements the Chi cut.
    """
    # Add other cut checks here in the future, e.g.:
    # passes_angle_cut = check_angle_cut(event_details, ...)
    # return passes_chi and passes_angle_cut

    passes_chi = check_chi_cut(event_details)
    return passes_chi


# --- Plotting Function 1: SNR vs Chi Parameters (no changes needed for this function based on new cuts) ---
def plot_snr_vs_chi(events_dict, output_dir, dataset_name):
    # ... (previous implementation) ...
    ic(f"Generating SNR vs Chi plots for {dataset_name}")
    os.makedirs(output_dir, exist_ok=True)

    chi_params = ['ChiRCR', 'Chi2016', 'ChiBad']
    fig, axs = plt.subplots(len(chi_params), 1, figsize=(12, 6 * len(chi_params)), sharex=True)
    if len(chi_params) == 1: axs = [axs]

    num_events = len(events_dict)
    if num_events == 0:
        ic(f"No events in {dataset_name} to plot for SNR vs Chi.")
        plt.close(fig); return
        
    colors_cmap = cm.get_cmap('jet', num_events if num_events > 1 else 2)

    for i, (event_id, event_data) in enumerate(events_dict.items()):
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
                    axs[plot_idx].scatter(point_d['SNR'], point_d[chi_param_name], color=event_color, s=30, alpha=0.8, zorder=5)

            valid_line_indices = ~np.isnan(snrs_event_sorted) & ~np.isnan(chi_values_event_sorted)
            if np.sum(valid_line_indices) > 1:
                    axs[plot_idx].plot(snrs_event_sorted[valid_line_indices], 
                                       chi_values_event_sorted[valid_line_indices], 
                                       linestyle='-', color=event_color, alpha=0.6, marker=None,
                                       label=f"Event {event_id}" if plot_idx == 0 and i < 15 else None)

    for plot_idx, chi_param_name in enumerate(chi_params):
        axs[plot_idx].set_ylabel(chi_param_name); axs[plot_idx].set_title(f'SNR vs {chi_param_name}')
        axs[plot_idx].grid(True, linestyle='--', alpha=0.6); axs[plot_idx].set_xscale('log')
        axs[plot_idx].set_xlim(3, 100); axs[plot_idx].set_ylim(0, 1)
    
    axs[-1].set_xlabel('SNR')
    handles, labels = axs[0].get_legend_handles_labels() # Get handles from the first plot for consistency
    if num_events > 0 and num_events <=15 and handles: 
        fig.legend(handles, labels, loc='center right', title="Events", bbox_to_anchor=(1.12, 0.5), fontsize='small')

    plt.suptitle(f'SNR vs Chi Parameters for {dataset_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.88 if num_events <=15 and handles else 0.98 , 0.96])
    
    plot_filename = os.path.join(output_dir, f"{dataset_name}_snr_vs_chi_params.png")
    plt.savefig(plot_filename, bbox_inches='tight'); ic(f"Saved SNR vs Chi plot: {plot_filename}"); plt.close(fig); gc.collect()


# --- Plotting Function 2: Parameter Histograms (Updated) ---
def plot_parameter_histograms(events_dict, output_dir, dataset_name):
    """Plots histograms for specified parameters, separated by cut status."""
    ic(f"Generating parameter histograms for {dataset_name} (with cut status)")
    os.makedirs(output_dir, exist_ok=True)

    params_to_histogram = ['SNR', 'ChiRCR', 'Chi2016', 'ChiBad', 'Zen', 'Azi']
    
    # Initialize dictionaries to hold values for different categories
    param_values_all = defaultdict(list)
    param_values_passing_cuts = defaultdict(list)
    param_values_failing_cuts = defaultdict(list)

    for event_id, event_data in events_dict.items():
        passes_cuts = event_data.get('passes_analysis_cuts', False) # Get the flag
        for station_id_str, station_triggers in event_data.get("stations", {}).items():
            for param_name in params_to_histogram:
                param_list = station_triggers.get(param_name, [])
                for val in param_list:
                    if val is not None and not np.isnan(val):
                        param_values_all[param_name].append(val)
                        if passes_cuts:
                            param_values_passing_cuts[param_name].append(val)
                        else:
                            param_values_failing_cuts[param_name].append(val)
    
    if not any(param_values_all.values()):
        ic(f"No valid data found for histograms in {dataset_name}.")
        return

    num_hist_params = len(params_to_histogram)
    cols = 3 if num_hist_params > 4 else (2 if num_hist_params > 1 else 1)
    rows = (num_hist_params + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)
    axs_flat = axs.flatten()

    for i, param_name in enumerate(params_to_histogram):
        ax = axs_flat[i]
        ax.set_title(f'{param_name}')
        ax.set_xlabel("Value")
        ax.set_ylabel('Frequency') # Default to non-log, adjust if log is applied

        values_all = param_values_all.get(param_name, [])
        values_pass = param_values_passing_cuts.get(param_name, [])
        values_fail = param_values_failing_cuts.get(param_name, [])

        has_data = False
        if values_all:
            ax.hist(values_all, bins=50, edgecolor='black', alpha=0.4, label='All Events', color='grey')
            has_data = True
        if values_pass:
            ax.hist(values_pass, bins=50, edgecolor='darkgreen', alpha=0.6, label='Pass Analysis Cuts', color='lightgreen')
            has_data = True
        if values_fail:
            ax.hist(values_fail, bins=50, edgecolor='darkred', alpha=0.6, label='Fail Analysis Cuts', color='lightcoral')
            has_data = True
        
        if has_data:
            if param_name in ['ChiRCR', 'Chi2016', 'ChiBad', 'SNR', 'Zen', 'Azi'] and any(v > 0 for v in values_all):
                 ax.set_yscale('log')
                 ax.set_ylabel('Frequency (log scale)')
            if param_name in ['ChiRCR', 'Chi2016', 'ChiBad']: ax.set_xlim(0, 1)
            if param_name == 'SNR': ax.set_xlim(3, 100); ax.set_xscale('log')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize='x-small')
        else:
            ax.text(0.5, 0.5, f"No data for\n{param_name}", ha='center', va='center', transform=ax.transAxes)
    
    for j in range(i + 1, len(axs_flat)): fig.delaxes(axs_flat[j])

    plt.suptitle(f'Parameter Histograms for {dataset_name} (by Cut Status)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_filename = os.path.join(output_dir, f"{dataset_name}_parameter_histograms_by_cut.png")
    plt.savefig(plot_filename); ic(f"Saved parameter histograms: {plot_filename}"); plt.close(fig); gc.collect()


# --- Plotting Function 3: Polar Plot (Zenith vs Azimuth) (no changes needed for this function based on new cuts) ---
def plot_polar_zen_azi(events_dict, output_dir, dataset_name):
    # ... (previous implementation) ...
    ic(f"Generating polar Zenith vs Azimuth plot for {dataset_name}")
    os.makedirs(output_dir, exist_ok=True); all_zen_values, all_azi_values = [], []
    for event_data in events_dict.values():
        for station_triggers in event_data.get("stations", {}).values():
            zen_list, azi_list = station_triggers.get('Zen', []), station_triggers.get('Azi', [])
            for k in range(len(zen_list)):
                if k < len(azi_list):
                    zen_val, azi_val = zen_list[k], azi_list[k]
                    if zen_val is not None and not np.isnan(zen_val) and azi_val is not None and not np.isnan(azi_val):
                        all_zen_values.append(zen_val); all_azi_values.append(azi_val)
    if not all_azi_values: ic(f"No valid Zenith/Azimuth data to plot for {dataset_name}."); return
    fig = plt.figure(figsize=(8, 8)); ax = plt.subplot(111, polar=True)
    ax.scatter(all_azi_values, np.degrees(all_zen_values), alpha=0.5, s=20, cmap='viridis', c=all_azi_values)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_rlabel_position(45)
    ax.set_rlim(0, 90); ax.set_rticks(np.arange(0, 91, 15))
    ax.set_title(f'Sky Plot: Zenith vs Azimuth for {dataset_name}\n(Zenith in degrees from center)', va='bottom', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    plot_filename = os.path.join(output_dir, f"{dataset_name}_polar_zen_azi.png")
    plt.savefig(plot_filename); ic(f"Saved polar Zenith vs Azimuth plot: {plot_filename}"); plt.close(fig); gc.collect()

# --- Plotting Function 4: Master Event Plot (Updated) ---
def plot_master_event_updated(events_dict, base_output_dir, dataset_name):
    """
    Generates a master plot for each event, saving into pass_cuts/fail_cuts subdirectories.
    """
    ic(f"Generating master event plots for {dataset_name}, separating by cut status.")
    
    # Define base master folder and subfolders for pass/fail
    master_folder_base = os.path.join(base_output_dir, f"{dataset_name}_master_event_plots")
    pass_cuts_folder = os.path.join(master_folder_base, "pass_cuts")
    fail_cuts_folder = os.path.join(master_folder_base, "fail_cuts")
    os.makedirs(pass_cuts_folder, exist_ok=True)
    os.makedirs(fail_cuts_folder, exist_ok=True)

    color_map = {13: 'tab:blue', 14: 'tab:orange', 15: 'tab:green',
                 17: 'tab:red', 18: 'tab:purple', 19: 'sienna', 30: 'tab:brown'}
    default_color = 'grey'; marker_list = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'X', '+']
    num_trace_channels = 4

    for event_id, event_details in events_dict.items():
        passes_analysis_cuts = event_details.get('passes_analysis_cuts', False)
        
        # Determine output directory for this specific event's plot
        current_event_plot_dir = pass_cuts_folder if passes_analysis_cuts else fail_cuts_folder
        
        # ... (Rest of the plotting logic from the original plot_master_event_updated function) ...
        # ... (This includes fig, gs, ax_scatter, ax_polar, trace_axs, spectrum_axs setup) ...
        # ... (event_time_str formatting, suptitle) ...
        # ... (text_info_lines, legend_handles_for_fig) ...
        # ... (global_trace_min/max calculation, y_margin, final_trace_ylim) ...
        # ... (The main plotting loop over stations and triggers) ...
        # ... (Finalizing subplots: labels, titles, grids, scales, text_box) ...
        # ... (Figure legend) ...

        # --- Start of copied plotting logic for a single event ---
        fig = plt.figure(figsize=(18, 20)); gs = gridspec.GridSpec(8, 2, figure=fig, hspace=0.8, wspace=0.3)
        ax_scatter = fig.add_subplot(gs[0:3, 0]); ax_polar = fig.add_subplot(gs[0:3, 1], polar=True)
        trace_axs = [fig.add_subplot(gs[3+i, 0]) for i in range(num_trace_channels)]
        spectrum_axs = [fig.add_subplot(gs[3+i, 1]) for i in range(num_trace_channels)]
        event_time_str = "Unknown Time"
        if "datetime" in event_details and event_details["datetime"] is not None:
            try: event_time_dt = datetime.datetime.fromtimestamp(event_details["datetime"]); event_time_str = event_time_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            except Exception as e: ic(f"Error formatting datetime for event {event_id}: {e}. Timestamp: {event_details['datetime']}")
        fig.suptitle(f"Master Plot: Event {event_id} ({dataset_name}) - {'PASS' if passes_analysis_cuts else 'FAIL'} CUTS\nTime: {event_time_str}", fontsize=18, y=0.98)
        text_info_lines = [f"Event ID: {event_id}", f"Time: {event_time_str}", f"Analysis Cuts: {'PASS' if passes_analysis_cuts else 'FAIL'}", "--- Station Triggers ---"]
        legend_handles_for_fig = {}; global_trace_min = float('inf'); global_trace_max = float('-inf')

        for station_id_str, station_data in event_details.get("stations", {}).items():
            all_traces_for_station_trigger = station_data.get("Traces", [])
            for trigger_idx, traces_for_one_trigger in enumerate(all_traces_for_station_trigger):
                if traces_for_one_trigger is not None and np.asarray(traces_for_one_trigger).any(): # Simpler check
                    padded_traces = (list(traces_for_one_trigger) + [None]*num_trace_channels)[:num_trace_channels]
                    for trace_array in padded_traces:
                        if trace_array is not None and hasattr(trace_array, "__len__") and len(trace_array) > 0:
                            current_min, current_max = np.nanmin(trace_array), np.nanmax(trace_array)
                            if not np.isnan(current_min): global_trace_min = min(global_trace_min, current_min)
                            if not np.isnan(current_max): global_trace_max = max(global_trace_max, current_max)
        if global_trace_min == float('inf'): global_trace_min = -1
        if global_trace_max == float('-inf'): global_trace_max = 1
        if global_trace_min == global_trace_max: global_trace_min -= 0.5; global_trace_max += 0.5
        y_margin = (global_trace_max - global_trace_min) * 0.1; final_trace_ylim = (global_trace_min - y_margin, global_trace_max + y_margin)

        for station_id_str, station_data in event_details.get("stations", {}).items():
            try: station_id_int = int(station_id_str)
            except ValueError: continue
            color = color_map.get(station_id_int, default_color)
            snr_values = station_data.get("SNR", []); num_triggers = len(snr_values)
            if num_triggers == 0: continue
            chi_rcr_values = (station_data.get("ChiRCR", []) + [np.nan] * num_triggers)[:num_triggers]
            chi_2016_values = (station_data.get("Chi2016", []) + [np.nan] * num_triggers)[:num_triggers]
            zen_values = (station_data.get("Zen", []) + [np.nan] * num_triggers)[:num_triggers]
            azi_values = (station_data.get("Azi", []) + [np.nan] * num_triggers)[:num_triggers]
            event_ids_for_station = (station_data.get("event_ids", []) + ["N/A"] * num_triggers)[:num_triggers] # Changed default for event_ids
            all_traces_for_station = station_data.get("Traces", [])

            for trigger_idx in range(num_triggers):
                marker = marker_list[trigger_idx % len(marker_list)]
                snr_val, chi_rcr_val, chi_2016_val = snr_values[trigger_idx], chi_rcr_values[trigger_idx], chi_2016_values[trigger_idx]
                zen_val, azi_val, current_event_id_val = zen_values[trigger_idx], azi_values[trigger_idx], event_ids_for_station[trigger_idx]
                traces_this_trigger = (all_traces_for_station[trigger_idx] if trigger_idx < len(all_traces_for_station) else [])
                padded_traces_this_trigger = (list(traces_this_trigger) + [None]*num_trace_channels)[:num_trace_channels]

                if snr_val is not None and not np.isnan(snr_val):
                    if chi_2016_val is not None and not np.isnan(chi_2016_val): ax_scatter.scatter(snr_val, chi_2016_val, c=color, marker=marker, s=60, alpha=0.9, zorder=3)
                    if chi_rcr_val is not None and not np.isnan(chi_rcr_val): ax_scatter.scatter(snr_val, chi_rcr_val, marker=marker, s=60, alpha=0.9, facecolors='none', edgecolors=color, linewidths=1.5, zorder=3)
                    if (chi_2016_val is not None and not np.isnan(chi_2016_val) and chi_rcr_val is not None and not np.isnan(chi_rcr_val)):
                        ax_scatter.annotate("", xy=(snr_val, chi_rcr_val), xytext=(snr_val, chi_2016_val), arrowprops=dict(arrowstyle="->",color=color,lw=1.2,shrinkA=3,shrinkB=3),zorder=2)
                if zen_val is not None and not np.isnan(zen_val) and azi_val is not None and not np.isnan(azi_val): ax_polar.scatter(azi_val, np.degrees(zen_val), c=color, marker=marker, s=60, alpha=0.9)
                
                for ch_idx in range(num_trace_channels):
                    trace_ch_data = padded_traces_this_trigger[ch_idx]
                    if trace_ch_data is not None and hasattr(trace_ch_data, "__len__") and len(trace_ch_data) > 0:
                        trace_ch_data_arr = np.asarray(trace_ch_data)
                        time_ax = np.linspace(0, (len(trace_ch_data_arr)-1)*0.5, len(trace_ch_data_arr)) # Assuming 0.5 ns sampling -> 2GSps
                        trace_axs[ch_idx].plot(time_ax, trace_ch_data_arr, c=color, ls='-' if trigger_idx % 2 == 0 else '--', alpha=0.7)
                        sampling_rate_hz = 2e9 # 2GSps
                        freq_ax_mhz = np.fft.rfftfreq(len(trace_ch_data_arr), d=1/sampling_rate_hz) / 1e6
                        spectrum = np.abs(fft.time2freq(trace_ch_data_arr, sampling_rate_hz))
                        if len(spectrum)>0: spectrum[0]=0 # Zero DC
                        spectrum_axs[ch_idx].plot(freq_ax_mhz, spectrum, c=color, ls=':' if trigger_idx % 2 == 0 else '-.', alpha=0.5)
                
                if station_id_int not in legend_handles_for_fig: legend_handles_for_fig[station_id_int] = Line2D([0], [0], marker='o', c=color, ls='None', markersize=8, label=f"St {station_id_int}")
                zen_d = f"{np.degrees(zen_val):.1f}°" if zen_val is not None and not np.isnan(zen_val) else "N/A"
                azi_d = f"{np.degrees(azi_val):.1f}°" if azi_val is not None and not np.isnan(azi_val) else "N/A"
                snr_fstr = f"{snr_val:.1f}" if snr_val is not None and not np.isnan(snr_val) else "N/A"
                ev_id_fstr = f"{int(current_event_id_val)}" if current_event_id_val is not None and not np.isnan(current_event_id_val) else "N/A" # Ensure EventID is int for display
                text_info_lines.append(f"  St{station_id_int} T{trigger_idx+1}: ID={ev_id_fstr}, SNR={snr_fstr}, Zen={zen_d}, Azi={azi_d}")
        
        ax_scatter.set_xlabel("SNR"); ax_scatter.set_ylabel("Chi value"); ax_scatter.set_title("SNR vs $\chi$ (Arrow: $\chi_{2016} \longrightarrow \chi_{RCR}$)")
        ax_scatter.set_xscale('log'); ax_scatter.set_xlim(3, 100); ax_scatter.set_ylim(0, 1); ax_scatter.grid(True, linestyle='--', alpha=0.6)
        ax_scatter.text(0.98, 0.02, "\n".join(text_info_lines[:30]),transform=ax_scatter.transAxes, ha='right', va='bottom', fontsize=7, family='monospace', linespacing=1.2, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="orange", alpha=0.7, lw=0.5))
        ax_polar.set_theta_zero_location("N"); ax_polar.set_theta_direction(-1); ax_polar.set_rlabel_position(22.5)
        ax_polar.set_rlim(0, 90); ax_polar.set_rticks(np.arange(0, 91, 30)); ax_polar.set_title("Zenith (radius) vs Azimuth (angle)"); ax_polar.grid(True, linestyle='--', alpha=0.5)
        for i in range(num_trace_channels):
            trace_axs[i].set_title(f"Trace - Ch {i}",fontsize=10); trace_axs[i].set_ylabel("Amp",fontsize=8); trace_axs[i].grid(True,ls=':',alpha=0.5); trace_axs[i].set_ylim(final_trace_ylim)
            if i < num_trace_channels -1 : trace_axs[i].set_xticklabels([])
            else: trace_axs[i].set_xlabel("Time (ns)", fontsize=8) # Changed unit to ns based on 0.5ns sampling
            spectrum_axs[i].set_title(f"Spectrum - Ch {i}",fontsize=10); spectrum_axs[i].set_ylabel("Mag",fontsize=8); spectrum_axs[i].grid(True,ls=':',alpha=0.5)
            if i < num_trace_channels -1 : spectrum_axs[i].set_xticklabels([])
            else: spectrum_axs[i].set_xlabel("Freq (MHz)", fontsize=8)
        if legend_handles_for_fig: fig.legend(handles=list(legend_handles_for_fig.values()), loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=min(len(legend_handles_for_fig),6), title="Stations", fontsize='medium')
        # --- End of copied plotting logic ---

        master_filename = os.path.join(current_event_plot_dir, f'master_event_{event_id}.png') # Save to correct subfolder
        try:
            plt.savefig(master_filename, dpi=150)
            # ic(f"Saved master plot: {master_filename}") # Reduce verbosity for many events
        except Exception as e:
            ic(f"Error saving master plot {master_filename}: {e}")
        plt.close(fig)
        gc.collect()
    ic(f"Finished master event plots for {dataset_name}. Check pass_cuts/ and fail_cuts/ subfolders.")


# --- Main Script ---
if __name__ == '__main__':
    ic.enable()

    # --- Configuration ---
    import configparser
    config = configparser.ConfigParser()
    # Ensure HRAStationDataAnalysis is in a path relative to where you run, or use absolute
    # Assuming the script is run from one level above HRAStationDataAnalysis
    config_path = os.path.join('HRAStationDataAnalysis', 'config.ini') 
    if not os.path.exists(config_path):
        # Fallback if script is inside HRAStationDataAnalysis
        config_path = 'config.ini' 
        if not os.path.exists(config_path):
            ic(f"CRITICAL: config.ini not found at primary or fallback path.")
            exit()
    config.read(config_path)
    date_of_data = config['PARAMETERS']['date']
    
    base_processed_data_dir = os.path.join("HRAStationDataAnalysis", "StationData", "processedNumpyData")
    processed_data_dir_for_date = os.path.join(base_processed_data_dir, date_of_data)
    
    dataset1_filename = f"{date_of_data}_CoincidenceDatetimes_with_all_params.pkl"
    dataset2_filename = f"{date_of_data}_CoincidenceRepeatStations_with_all_params.pkl" # Example, can add back if needed
    dataset3_filename = f"{date_of_data}_CoincidenceRepeatEventIDs_with_all_params.pkl" # Example

    dataset_paths = [
        os.path.join(processed_data_dir_for_date, dataset1_filename),
        os.path.join(processed_data_dir_for_date, dataset2_filename), 
        os.path.join(processed_data_dir_for_date, dataset3_filename)
    ]
    dataset_names = [
        "CoincidenceEvents", 
        "CoincidenceRepeatStations", 
        "CoincidenceRepeatEventIDs"
    ]
    dataset_plot_suffixes = [
        f"CoincidenceEvents_{date_of_data}",
        f"CoincidenceRepeatStations_{date_of_data}",
        f"CoincidenceRepeatEventIDs_{date_of_data}"
    ]
    
    output_plot_basedir = os.path.join("HRAStationDataAnalysis", "plots") # Main plots directory
    os.makedirs(output_plot_basedir, exist_ok=True)
    
    datasets_to_plot_info = []
    for i, d_path in enumerate(dataset_paths):
        data = _load_pickle(d_path)
        if data is not None:
            datasets_to_plot_info.append({"name": dataset_names[i], "data": data, "plot_dir_suffix": dataset_plot_suffixes[i]})
            ic(f"Loaded dataset: {d_path}")
        else:
            ic(f"Could not load dataset from: {d_path}. Please check path and file.")

    if not datasets_to_plot_info:
        ic("No datasets loaded. Exiting plotting script.")
        exit()

    # --- Apply Analysis Cuts and Generate Plots for each dataset ---
    for dataset_info in datasets_to_plot_info:
        dataset_name_label = dataset_info["name"]
        events_data_dict = dataset_info["data"]
        specific_dataset_plot_dir = os.path.join(output_plot_basedir, dataset_info["plot_dir_suffix"])
        os.makedirs(specific_dataset_plot_dir, exist_ok=True)

        ic(f"\n--- Processing dataset for cuts and plots: {dataset_name_label} ---")
        ic(f"Output will be in: {specific_dataset_plot_dir}")

        if not isinstance(events_data_dict, dict) or not events_data_dict:
            ic(f"Dataset '{dataset_name_label}' is empty or not a dictionary. Skipping.")
            continue

        # Apply analysis cuts to each event and store the result
        num_passing = 0
        num_failing = 0
        for event_id, event_details in events_data_dict.items():
            if isinstance(event_details, dict): # Ensure event_details is a dict
                 passes = check_coincidence_cuts(event_details)
                 event_details['passes_analysis_cuts'] = passes
                 if passes: num_passing +=1
                 else: num_failing +=1
            else: # Should not happen if data is structured correctly
                 event_details['passes_analysis_cuts'] = False # Default for malformed entries
                 num_failing +=1

        ic(f"Analysis cuts applied to '{dataset_name_label}': {num_passing} events passed, {num_failing} events failed.")

        # 1. SNR vs Chi parameters
        plot_snr_vs_chi(events_data_dict, specific_dataset_plot_dir, dataset_name_label)

        # 2. Parameter Histograms (now includes cut status)
        plot_parameter_histograms(events_data_dict, specific_dataset_plot_dir, dataset_name_label)

        # 3. Polar Plot (Zenith vs Azimuth)
        plot_polar_zen_azi(events_data_dict, specific_dataset_plot_dir, dataset_name_label)
        
        # 4. Master Event Plots (now saves to pass_cuts/fail_cuts subdirs)
        plot_master_event_updated(events_data_dict, specific_dataset_plot_dir, dataset_name_label) # Pass base output dir

        ic(f"--- Finished plots for: {dataset_name_label} ---")

    ic("\nAll plotting complete.")