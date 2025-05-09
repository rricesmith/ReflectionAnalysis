import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re # For master plot, if needed, though not directly for new plots
from matplotlib.lines import Line2D # For master plot legend
import matplotlib.cm as cm # For colormaps in plot 1
from icecream import ic # Keep user's preference for icecream
import datetime # For master plot title
import gc # For garbage collection
from collections import defaultdict # For histogram data collection
import matplotlib.gridspec as gridspec # For more complex subplot layouts
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

# --- Plotting Function 1: SNR vs Chi Parameters ---
def plot_snr_vs_chi(events_dict, output_dir, dataset_name):
    """
    Plots SNR vs. ChiRCR, Chi2016, ChiBad in 3 subplots.
    Each station trigger is plotted. Data points from the same event are
    plotted with the same color, and a line connects these points
    (sorted by station ID, then by trigger order within the station).
    """
    ic(f"Generating SNR vs Chi plots for {dataset_name}")
    os.makedirs(output_dir, exist_ok=True)

    chi_params = ['ChiRCR', 'Chi2016', 'ChiBad']
    fig, axs = plt.subplots(len(chi_params), 1, figsize=(12, 6 * len(chi_params)), sharex=True)
    if len(chi_params) == 1: # Make axs iterable if only one subplot
        axs = [axs]

    num_events = len(events_dict)
    if num_events == 0:
        ic(f"No events in {dataset_name} to plot for SNR vs Chi.")
        plt.close(fig)
        return
        
    colors_cmap = cm.get_cmap('jet', num_events if num_events > 1 else 2) # Ensure cmap has enough colors


    for i, (event_id, event_data) in enumerate(events_dict.items()):
        event_color = colors_cmap(i)
        
        event_plot_data_collected = [] # List of dicts: {'station_id': int, 'trigger_idx_orig': int, 'SNR': float, 'Chi...': float}
        
        for station_id_str, station_triggers_data in event_data.get("stations", {}).items():
            try:
                station_id_int = int(station_id_str)
            except ValueError:
                ic(f"Warning: Could not convert station ID '{station_id_str}' to int for event {event_id}. Skipping station.")
                continue

            snr_list = station_triggers_data.get('SNR', [])
            num_triggers_for_station = len(snr_list)
            if num_triggers_for_station == 0:
                continue

            # Prepare all chi parameter lists, ensuring they match num_triggers_for_station
            # and have np.nan for missing values.
            chi_data_for_station = {}
            for cp in chi_params:
                cp_list = station_triggers_data.get(cp, [])
                # Pad with NaNs if shorter, truncate if longer (should ideally be aligned)
                chi_data_for_station[cp] = (cp_list + [np.nan] * num_triggers_for_station)[:num_triggers_for_station]

            for trigger_idx in range(num_triggers_for_station):
                snr_val = snr_list[trigger_idx]
                if snr_val is None or np.isnan(snr_val):
                    continue # Skip triggers with invalid SNR

                point_data = {'station_id': station_id_int, 'trigger_idx_orig': trigger_idx, 'SNR': snr_val}
                for cp in chi_params:
                    point_data[cp] = chi_data_for_station[cp][trigger_idx]
                event_plot_data_collected.append(point_data)
        
        if not event_plot_data_collected:
            continue

        # Sort points for consistent line drawing: by station_id, then by original trigger index
        event_plot_data_collected.sort(key=lambda p: (p['station_id'], p['trigger_idx_orig']))

        # Extract sorted arrays for plotting lines for this event
        snrs_event_sorted = np.array([p['SNR'] for p in event_plot_data_collected])
        
        for plot_idx, chi_param_name in enumerate(chi_params):
            chi_values_event_sorted = np.array([p[chi_param_name] for p in event_plot_data_collected])
            
            # Filter out NaNs for line plotting segment by segment
            # Plot all individual points first
            for point_d in event_plot_data_collected:
                if point_d['SNR'] is not None and not np.isnan(point_d['SNR']) and \
                   point_d[chi_param_name] is not None and not np.isnan(point_d[chi_param_name]):
                    axs[plot_idx].scatter(point_d['SNR'], point_d[chi_param_name], color=event_color, s=30, alpha=0.8, zorder=5)

            # Then plot connecting lines for valid segments
            valid_line_indices = ~np.isnan(snrs_event_sorted) & ~np.isnan(chi_values_event_sorted)
            if np.sum(valid_line_indices) > 1: # Need at least two points for a line
                 axs[plot_idx].plot(snrs_event_sorted[valid_line_indices], 
                                    chi_values_event_sorted[valid_line_indices], 
                                    linestyle='-', color=event_color, alpha=0.6, marker=None, # No marker on line itself
                                    label=f"Event {event_id}" if plot_idx == 0 and i < 15 else None) # Limit legend

    for plot_idx, chi_param_name in enumerate(chi_params):
        axs[plot_idx].set_ylabel(chi_param_name)
        axs[plot_idx].set_title(f'SNR vs {chi_param_name}')
        axs[plot_idx].grid(True, linestyle='--', alpha=0.6)
        axs[plot_idx].set_xscale('log') # Chi parameters often benefit from log scale
        axs[plot_idx].set_xlim(3, 100) 
        axs[plot_idx].set_ylim(0, 1)
    
    axs[-1].set_xlabel('SNR')
    
    if num_events > 0 and num_events <=15 : 
        handles, labels = axs[0].get_legend_handles_labels()
        if handles:
             fig.legend(handles, labels, loc='center right', title="Events", bbox_to_anchor=(1.12, 0.5), fontsize='small')

    plt.suptitle(f'SNR vs Chi Parameters for {dataset_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.88 if num_events <=15 and handles else 0.98 , 0.96])
    
    plot_filename = os.path.join(output_dir, f"{dataset_name}_snr_vs_chi_params.png")
    plt.savefig(plot_filename, bbox_inches='tight')
    ic(f"Saved SNR vs Chi plot: {plot_filename}")
    plt.close(fig)
    gc.collect()


# --- Plotting Function 2: Parameter Histograms ---
def plot_parameter_histograms(events_dict, output_dir, dataset_name):
    """Plots histograms for specified parameters."""
    ic(f"Generating parameter histograms for {dataset_name}")
    os.makedirs(output_dir, exist_ok=True)

    params_to_histogram = ['SNR', 'ChiRCR', 'Chi2016', 'ChiBad', 'Zen', 'Azi']
    
    all_param_values = defaultdict(list)
    for event_id, event_data in events_dict.items():
        for station_id_str, station_triggers in event_data.get("stations", {}).items():
            for param_name in params_to_histogram:
                param_list = station_triggers.get(param_name, [])
                for val in param_list: # param_list contains values for each trigger
                    if val is not None and not np.isnan(val):
                        all_param_values[param_name].append(val)

    if not any(all_param_values.values()): # Check if any list in the defaultdict is non-empty
        ic(f"No valid data found for histograms in {dataset_name}.")
        return

    num_hist_params = len(params_to_histogram)
    cols = 3 if num_hist_params > 4 else (2 if num_hist_params > 1 else 1)
    rows = (num_hist_params + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False) # squeeze=False ensures axs is 2D
    axs_flat = axs.flatten()

    for i, param_name in enumerate(params_to_histogram):
        ax = axs_flat[i]
        values = all_param_values.get(param_name, [])
        if values:
            ax.hist(values, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
            ax.set_title(f'{param_name}')
            ax.set_xlabel("Value")
            ax.set_ylabel('Frequency')
            if param_name in ['ChiRCR', 'Chi2016', 'ChiBad', 'SNR', 'Zen', 'Azi'] and any(v > 0 for v in values): # Apply log for positive skewed data
                ax.set_yscale('log')
                ax.set_ylabel('Frequency (log scale)')
            if param_name in ['ChiRCR', 'Chi2016', 'ChiBad'] and any(v > 0 for v in values): # Set xlim for Chi parameters
                ax.set_xlim(0, 1)
            if param_name in ['SNR'] and any(v > 0 for v in values): 
                ax.set_xlim(3, 100)
                ax.set_xscale('log')
            ax.grid(True, linestyle='--', alpha=0.6)
        else:
            ax.text(0.5, 0.5, f"No data for\n{param_name}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{param_name}')
    
    for j in range(i + 1, len(axs_flat)): # Hide unused subplots
        fig.delaxes(axs_flat[j])

    plt.suptitle(f'Parameter Histograms for {dataset_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_filename = os.path.join(output_dir, f"{dataset_name}_parameter_histograms.png")
    plt.savefig(plot_filename)
    ic(f"Saved parameter histograms: {plot_filename}")
    plt.close(fig)
    gc.collect()

# --- Plotting Function 3: Polar Plot (Zenith vs Azimuth) ---
def plot_polar_zen_azi(events_dict, output_dir, dataset_name):
    """Plots a polar plot of Zenith (radial) vs. Azimuth (angular)."""
    ic(f"Generating polar Zenith vs Azimuth plot for {dataset_name}")
    os.makedirs(output_dir, exist_ok=True)

    all_zen_values = []
    all_azi_values = []

    for event_id, event_data in events_dict.items():
        for station_id_str, station_triggers in event_data.get("stations", {}).items():
            zen_list = station_triggers.get('Zen', [])
            azi_list = station_triggers.get('Azi', [])
            
            # Iterate through triggers for this station
            for k in range(len(zen_list)): # Assuming zen_list and azi_list are aligned by trigger
                if k < len(azi_list): # Ensure azi_list has a corresponding entry
                    zen_val = zen_list[k]
                    azi_val = azi_list[k]
                    if zen_val is not None and not np.isnan(zen_val) and \
                       azi_val is not None and not np.isnan(azi_val):
                        all_zen_values.append(zen_val)
                        all_azi_values.append(azi_val)
    
    if not all_azi_values: 
        ic(f"No valid Zenith/Azimuth data to plot for {dataset_name}.")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    scatter = ax.scatter(all_azi_values, np.degrees(all_zen_values), alpha=0.5, s=20, cmap='viridis', c=all_azi_values) # Color by azimuth
    
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)
    ax.set_rlim(0, 90) # Zenith from 0 (overhead) to 90 (horizon) in degrees
    ax.set_rticks(np.arange(0, 91, 15)) # Ticks every 15 degrees for Zenith
    ax.set_title(f'Sky Plot: Zenith vs Azimuth for {dataset_name}\n(Zenith in degrees from center)', va='bottom', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a colorbar if desired, though might be redundant if points are dense
    # cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", pad=0.1)
    # cbar.set_label('Azimuth (radians)')


    plot_filename = os.path.join(output_dir, f"{dataset_name}_polar_zen_azi.png")
    plt.savefig(plot_filename)
    ic(f"Saved polar Zenith vs Azimuth plot: {plot_filename}")
    plt.close(fig)
    gc.collect()

# --- Plotting Function 4: Master Event Plot (Updated) ---
def plot_master_event_updated(events_dict, output_dir, dataset_name):
    """
    Generates a master plot for each event, showing its parameters.
    Features 4 trace channel plots and 4 dummy spectrum plots.
    """
    ic(f"Generating master event plots for {dataset_name}")
    master_folder = os.path.join(output_dir, f"{dataset_name}_master_event_plots")
    os.makedirs(master_folder, exist_ok=True)

    color_map = {13: 'tab:blue', 14: 'tab:orange', 15: 'tab:green',
                   17: 'tab:red', 18: 'tab:purple', 19: 'sienna', 30: 'tab:brown'}
    default_color = 'grey'
    marker_list = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'X', '+']
    num_trace_channels = 4 # As per user request

    for event_id, event_details in events_dict.items():
        # Define figure and GridSpec layout
        # Taller figure to accommodate more subplots
        fig = plt.figure(figsize=(18, 20)) 
        # GridSpec: 8 rows total. Top plots take 3 rows. Bottom 4 rows for traces/FFTs. Last row for legend.
        gs = gridspec.GridSpec(8, 2, figure=fig, hspace=0.5, wspace=0.3)

        ax_scatter = fig.add_subplot(gs[0:3, 0])
        ax_polar = fig.add_subplot(gs[0:3, 1], polar=True)
        
        trace_axs = []
        for i in range(num_trace_channels):
            trace_axs.append(fig.add_subplot(gs[3+i, 0])) # Traces in the first column, rows 3,4,5,6

        spectrum_axs = []
        for i in range(num_trace_channels):
            spectrum_axs.append(fig.add_subplot(gs[3+i, 1])) # Spectrums in the second column, rows 3,4,5,6

        event_time_str = "Unknown Time"
        if "datetime" in event_details and event_details["datetime"] is not None:
            try:
                event_time_dt = datetime.datetime.fromtimestamp(event_details["datetime"])
                event_time_str = event_time_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            except Exception as e:
                ic(f"Error formatting datetime for event {event_id}: {e}. Timestamp: {event_details['datetime']}")
        
        fig.suptitle(f"Master Plot: Event {event_id} ({dataset_name})\nTime: {event_time_str}", fontsize=18, y=0.98)

        text_info_lines = [f"Event ID: {event_id}", f"Time: {event_time_str}", "--- Station Triggers ---"]
        legend_handles_for_fig = {}
        
        # --- Pre-calculate global y-limits for traces ---
        global_trace_min = float('inf')
        global_trace_max = float('-inf')
        all_event_traces_data = [] # Store (station_id_int, trigger_idx, channel_idx, trace_array)

        for station_id_str, station_data in event_details.get("stations", {}).items():
            all_traces_for_station_trigger = station_data.get("Traces", []) # List of triggers, each trigger has list of 4 channel traces
            for trigger_idx, traces_for_one_trigger in enumerate(all_traces_for_station_trigger):
                if traces_for_one_trigger.any() and isinstance(traces_for_one_trigger, (list, np.ndarray)):
                    # Ensure it's a list of 4 channels, even if some are None or empty
                    padded_traces_for_trigger = (list(traces_for_one_trigger) + [None]*num_trace_channels)[:num_trace_channels]
                    for channel_idx in range(num_trace_channels):
                        trace_array = padded_traces_for_trigger[channel_idx]
                        if trace_array is not None and hasattr(trace_array, "__len__") and len(trace_array) > 0:
                            all_event_traces_data.append({
                                "station_id_str": station_id_str, 
                                "trigger_idx": trigger_idx, 
                                "channel_idx": channel_idx, 
                                "trace_array": np.asarray(trace_array) # Ensure numpy array
                            })
                            current_min, current_max = np.min(trace_array), np.max(trace_array)
                            if not np.isnan(current_min): global_trace_min = min(global_trace_min, current_min)
                            if not np.isnan(current_max): global_trace_max = max(global_trace_max, current_max)
        
        if global_trace_min == float('inf'): global_trace_min = -1 # Default if no valid traces
        if global_trace_max == float('-inf'): global_trace_max = 1  # Default
        if global_trace_min == global_trace_max: # Avoid zero range
            global_trace_min -= 0.5 
            global_trace_max += 0.5
        
        y_margin = (global_trace_max - global_trace_min) * 0.1 # 10% margin
        final_trace_ylim = (global_trace_min - y_margin, global_trace_max + y_margin)


        # --- Plotting Loop ---
        for station_id_str, station_data in event_details.get("stations", {}).items():
            try:
                station_id_int = int(station_id_str)
            except ValueError:
                ic(f"Skipping station with non-integer ID '{station_id_str}' in event {event_id}")
                continue
            
            color = color_map.get(station_id_int, default_color)
            
            # Get parameter lists for this station
            snr_values = station_data.get("SNR", [])
            chi_rcr_values = (station_data.get("ChiRCR", []) + [np.nan] * len(snr_values))[:len(snr_values)]
            chi_2016_values = (station_data.get("Chi2016", []) + [np.nan] * len(snr_values))[:len(snr_values)]
            zen_values = (station_data.get("Zen", []) + [np.nan] * len(snr_values))[:len(snr_values)]
            azi_values = (station_data.get("Azi", []) + [np.nan] * len(snr_values))[:len(snr_values)]
            all_traces_for_station = station_data.get("Traces", []) # List of triggers; each trigger is list of 4 channel traces

            num_triggers = len(snr_values)
            if num_triggers == 0: continue

            for trigger_idx in range(num_triggers):
                marker = marker_list[trigger_idx % len(marker_list)] 

                snr_val = snr_values[trigger_idx]
                chi_rcr_val = chi_rcr_values[trigger_idx]
                chi_2016_val = chi_2016_values[trigger_idx]
                zen_val = zen_values[trigger_idx]
                azi_val = azi_values[trigger_idx]
                
                traces_for_this_trigger = (all_traces_for_station[trigger_idx] if trigger_idx < len(all_traces_for_station) else [])
                # Ensure traces_for_this_trigger is a list of 4 (padded with None if necessary)
                padded_traces_for_this_trigger = (list(traces_for_this_trigger) + [None]*num_trace_channels)[:num_trace_channels]


                # Scatter Plot (SNR vs Chi)
                if snr_val is not None and not np.isnan(snr_val):
                    if chi_2016_val is not None and not np.isnan(chi_2016_val):
                        ax_scatter.scatter(snr_val, chi_2016_val, color=color, marker=marker, s=60, alpha=0.9, zorder=3)
                    if chi_rcr_val is not None and not np.isnan(chi_rcr_val):
                        ax_scatter.scatter(snr_val, chi_rcr_val, color=color, marker=marker, s=60, alpha=0.9, facecolors='none', edgecolors=color, linewidths=1.5, zorder=3)
                    if (chi_2016_val is not None and not np.isnan(chi_2016_val) and
                        chi_rcr_val is not None and not np.isnan(chi_rcr_val)):
                        ax_scatter.annotate("", xy=(snr_val, chi_rcr_val), xytext=(snr_val, chi_2016_val),
                                            arrowprops=dict(arrowstyle="->", color=color, lw=1.2, shrinkA=3, shrinkB=3), zorder=2)

                # Polar Plot
                if zen_val is not None and not np.isnan(zen_val) and azi_val is not None and not np.isnan(azi_val):
                    ax_polar.scatter(azi_val, np.degrees(zen_val), color=color, marker=marker, s=60, alpha=0.9)

                # Trace and Spectrum Plots (Loop per channel)
                for channel_idx in range(num_trace_channels):
                    # Plot Trace for this channel
                    trace_data_one_channel = padded_traces_for_this_trigger[channel_idx]
                    if trace_data_one_channel is not None and hasattr(trace_data_one_channel, "__len__") and len(trace_data_one_channel) > 0:
                        trace_data_one_channel = np.asarray(trace_data_one_channel) # Ensure numpy array
                        # Assuming 256 samples, 50ns sampling -> 0.05 us per sample
                        # Time axis from 0 to (N-1)*dt
                        time_axis = np.linspace(0, (len(trace_data_one_channel) - 1) * 0.05, len(trace_data_one_channel))
                        trace_axs[channel_idx].plot(time_axis, trace_data_one_channel, color=color, 
                                                    linestyle='-' if trigger_idx % 2 == 0 else '--', 
                                                    alpha=0.7)
                    
                    # Plot FFT for this channel
                    if trace_data_one_channel is not None and hasattr(trace_data_one_channel, "__len__") and len(trace_data_one_channel) > 0:
                        sampling_rate = 2 # Assuming 2 GHz sampling rate
                        freq_axis = np.fft.rfftfreq(len(trace_data_one_channel), d=(1 / sampling_rate*units.GHz)) / units.MHz
                        spectrum_data = np.abs(fft.time2freq(trace_data_one_channel, sampling_rate=sampling_rate*units.GHz)) # FFT magnitude
                        spectrum_axs[channel_idx].plot(freq_axis, spectrum_data, color=color,
                                                       linestyle=':' if trigger_idx % 2 == 0 else '-.',
                                                       alpha=0.5)
                    else: # If no trace data, plot a placeholder text
                         spectrum_axs[channel_idx].text(0.5, 0.5, "No Data", ha="center", va="center", transform=spectrum_axs[channel_idx].transAxes, fontsize=9, color='lightgrey')


                # Add to figure legend map (once per station)
                if station_id_int not in legend_handles_for_fig:
                    legend_handles_for_fig[station_id_int] = Line2D([0], [0], marker='o', color=color, linestyle='None', # Use a consistent marker for legend
                                                       markersize=8, label=f"Station {station_id_int}")
                
                # Text info for this trigger
                zen_deg = f"{np.degrees(zen_val):.1f}" if zen_val is not None and not np.isnan(zen_val) else "N/A"
                azi_deg = f"{np.degrees(azi_val):.1f}" if azi_val is not None and not np.isnan(azi_val) else "N/A"
                snr_f = f"{snr_val:.1f}" if snr_val is not None and not np.isnan(snr_val) else "N/A"
                text_info_lines.append(f"  St{station_id_int} T{trigger_idx+1}: SNR={snr_f}, Zen={zen_deg}°, Azi={azi_deg}°")

        # --- Finalize Subplots ---
        # SNR vs Chi
        ax_scatter.set_xlabel("SNR")
        ax_scatter.set_ylabel("Chi value")
        ax_scatter.set_title("SNR vs $\chi$ (Arrow: $\chi_{2016} \longrightarrow \chi_{RCR}$)")
        ax_scatter.set_xscale('log') # As per original
        ax_scatter.set_xlim(3, 100)  # As per original
        ax_scatter.set_ylim(0, 1)    # As per original
        ax_scatter.grid(True, linestyle='--', alpha=0.6)
        # Add text box to ax_scatter
        ax_scatter.text(0.98, 0.02, "\n".join(text_info_lines[:15]), # Limit lines for readability
                        transform=ax_scatter.transAxes, ha='right', va='bottom',
                        fontsize=7, family='monospace', linespacing=1.2,
                        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="orange", alpha=0.7, lw=0.5))


        # Polar
        ax_polar.set_theta_zero_location("N")
        ax_polar.set_theta_direction(-1)
        ax_polar.set_rlabel_position(22.5)
        ax_polar.set_rlim(0, 90)
        ax_polar.set_rticks(np.arange(0, 91, 30))
        ax_polar.set_title("Zenith (radius) vs Azimuth (angle)")
        ax_polar.grid(True, linestyle='--', alpha=0.5)

        # Traces and Spectrums
        for i in range(num_trace_channels):
            trace_axs[i].set_title(f"Trace - Channel {i}", fontsize=10)
            trace_axs[i].set_ylabel("Amplitude", fontsize=8)
            trace_axs[i].grid(True, linestyle=':', alpha=0.5)
            trace_axs[i].set_ylim(final_trace_ylim) # Apply normalized y-limits
            if i < num_trace_channels -1 : trace_axs[i].set_xticklabels([]) # Remove x-labels for upper trace plots
            else: trace_axs[i].set_xlabel("Time ($\mu s$)", fontsize=8)
            if any(trace_axs[i].get_lines()): trace_axs[i].legend(fontsize='xx-small', loc='upper right')


            spectrum_axs[i].set_title(f"Spectrum - Channel {i}", fontsize=10)
            spectrum_axs[i].set_ylabel("Magnitude", fontsize=8)
            spectrum_axs[i].grid(True, linestyle=':', alpha=0.5)
            if i < num_trace_channels -1 : spectrum_axs[i].set_xticklabels([])
            else: spectrum_axs[i].set_xlabel("Frequency (MHz)", fontsize=8)
            if any(spectrum_axs[i].get_lines()): spectrum_axs[i].legend(fontsize='xx-small', loc='upper right')


        # Figure Legend (at the bottom, using the last gs row)
        if legend_handles_for_fig:
            fig.legend(handles=list(legend_handles_for_fig.values()), 
                       loc='lower center', 
                       bbox_to_anchor=(0.5, 0.01), # Positioned at the very bottom center of the figure
                       ncol=min(len(legend_handles_for_fig), 6), 
                       title="Stations", fontsize='medium')

        # plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust for suptitle and figure legend
        # tight_layout might conflict with GridSpec, manual adjustment or hspace/wspace in GridSpec is better.
        
        master_filename = os.path.join(master_folder, f'master_event_{event_id}.png')
        try:
            plt.savefig(master_filename, dpi=150)
            ic(f"Saved master plot: {master_filename}")
        except Exception as e:
            ic(f"Error saving master plot {master_filename}: {e}")
        plt.close(fig)
        gc.collect()


# --- Main Script ---
if __name__ == '__main__':
    ic.enable() 

    # --- Configuration ---
    # Read configuration and get date 
    import configparser
    config = configparser.ConfigParser() 
    config.read(os.path.join('HRAStationDataAnalysis', 'config.ini')) 
    date_of_data = config['PARAMETERS']['date']
    
    base_processed_data_dir = "HRAStationDataAnalysis/StationData/processedNumpyData"
    processed_data_dir_for_date = os.path.join(base_processed_data_dir, date_of_data) 
    
    dataset1_filename = f"{date_of_data}_CoincidenceDatetimes_with_all_params.pkl"
    dataset2_filename = f"{date_of_data}_CoincidenceRepeatStations_with_all_params.pkl"

    dataset1_path = os.path.join(processed_data_dir_for_date, dataset1_filename)
    dataset2_path = os.path.join(processed_data_dir_for_date, dataset2_filename)

    output_plot_basedir = f"HRAStationDataAnalysis/plots" 
    os.makedirs(output_plot_basedir, exist_ok=True)
    
    datasets_to_plot_info = []
    data1 = _load_pickle(dataset1_path)
    if data1 is not None:
        datasets_to_plot_info.append({"name": "CoincidenceEvents", "data": data1, "plot_dir_suffix": f"CoincidenceEvents_{date_of_data}"})
        ic(f"Loaded dataset 1: {dataset1_path}")
    else:
        ic(f"Could not load dataset from: {dataset1_path}. Please check path and file.")

    data2 = _load_pickle(dataset2_path)
    if data2 is not None:
        datasets_to_plot_info.append({"name": "CoincidenceRepeatStations", "data": data2, "plot_dir_suffix": f"CoincidenceRepeatStations_{date_of_data}"})
        ic(f"Loaded dataset 2: {dataset2_path}")

    else:
        ic(f"Could not load dataset from: {dataset2_path}. Please check path and file.")

    if not datasets_to_plot_info:
        ic("No datasets loaded. Exiting plotting script.")
        exit()

    # --- Generate Plots for each dataset ---
    for dataset_info in datasets_to_plot_info:
        dataset_name_label = dataset_info["name"]
        events_data_dict = dataset_info["data"]
        # Create a specific plot directory for this dataset and date
        specific_dataset_plot_dir = os.path.join(output_plot_basedir, dataset_info["plot_dir_suffix"])
        os.makedirs(specific_dataset_plot_dir, exist_ok=True)

        ic(f"\n--- Generating plots for: {dataset_name_label} ---")
        ic(f"Plots will be saved in: {specific_dataset_plot_dir}")

        if not isinstance(events_data_dict, dict) or not events_data_dict:
            ic(f"Dataset '{dataset_name_label}' is empty or not a dictionary. Skipping plots.")
            continue

        # 1. SNR vs Chi parameters
        plot_snr_vs_chi(events_data_dict, specific_dataset_plot_dir, dataset_name_label)

        # 2. Parameter Histograms
        plot_parameter_histograms(events_data_dict, specific_dataset_plot_dir, dataset_name_label)

        # 3. Polar Plot (Zenith vs Azimuth)
        plot_polar_zen_azi(events_data_dict, specific_dataset_plot_dir, dataset_name_label)
        
        # 4. Master Event Plots
        plot_master_event_updated(events_data_dict, specific_dataset_plot_dir, dataset_name_label)

        ic(f"--- Finished plots for: {dataset_name_label} ---")

    ic("\nAll plotting complete.")

