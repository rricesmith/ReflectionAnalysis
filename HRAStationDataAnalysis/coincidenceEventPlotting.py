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
    Adapted from user's provided function to work with the new events_dict structure.
    """
    ic(f"Generating master event plots for {dataset_name}")
    master_folder = os.path.join(output_dir, f"{dataset_name}_master_event_plots")
    os.makedirs(master_folder, exist_ok=True)

    color_map = {13: 'tab:blue', 14: 'tab:orange', 15: 'tab:green',
                   17: 'tab:red', 18: 'tab:purple', 19: 'sienna', 30: 'tab:brown',
                   # Fallback for other stations if any
                   } 
    default_color = 'grey' # Fallback color
    marker_list = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'X', '+']

    for event_id, event_details in events_dict.items():
        fig, axs = plt.subplots(2, 2, figsize=(15, 13)) 

        event_time_str = "Unknown Time"
        # Assuming 'datetime' key holds a single timestamp for the event
        if "datetime" in event_details and event_details["datetime"] is not None:
            try:
                event_time_dt = datetime.datetime.fromtimestamp(event_details["datetime"])
                event_time_str = event_time_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            except Exception as e:
                ic(f"Error formatting datetime for event {event_id}: {e}. Timestamp: {event_details['datetime']}")
        
        fig.suptitle(f"Master Plot: Event {event_id} ({dataset_name})\nTime: {event_time_str}", fontsize=16)

        ax_scatter = axs[0, 0]
        ax_polar = fig.add_subplot(2, 2, 2, polar=True) # Correct way to add polar subplot to existing figure
        ax_trace = axs[1, 0]
        ax_text_info = axs[1, 1] 
        ax_text_info.axis('off')
        text_info_lines = [f"Event ID: {event_id}", f"Time: {event_time_str}", "--- Station Triggers ---"]

        legend_handles_for_fig = {} # For the main figure legend: {station_id: Line2D_handle}

        station_plot_count = 0 # To manage layout if many stations
        for station_id_str, station_data in event_details.get("stations", {}).items():
            station_plot_count +=1
            try:
                station_id_int = int(station_id_str)
            except ValueError:
                ic(f"Skipping station with non-integer ID '{station_id_str}' in event {event_id}")
                continue
            
            color = color_map.get(station_id_int, default_color)
            
            snr_values = station_data.get("SNR", [])
            chi_rcr_values = station_data.get("ChiRCR", [])
            chi_2016_values = station_data.get("Chi2016", [])
            zen_values = station_data.get("Zen", [])
            azi_values = station_data.get("Azi", [])
            trace_values = station_data.get("Traces", [])
            ic(trace_values)

            num_triggers = len(snr_values)
            if num_triggers == 0 : continue

            # Pad lists if necessary
            chi_rcr_values = (chi_rcr_values + [np.nan] * num_triggers)[:num_triggers]
            chi_2016_values = (chi_2016_values + [np.nan] * num_triggers)[:num_triggers]
            zen_values = (zen_values + [np.nan] * num_triggers)[:num_triggers]
            azi_values = (azi_values + [np.nan] * num_triggers)[:num_triggers]
            trace_values = (trace_values + [None] * num_triggers)[:num_triggers]

            for trigger_idx in range(num_triggers):
                marker = marker_list[trigger_idx % len(marker_list)] 

                snr_val = snr_values[trigger_idx]
                chi_rcr_val = chi_rcr_values[trigger_idx]
                chi_2016_val = chi_2016_values[trigger_idx]
                zen_val = zen_values[trigger_idx]
                azi_val = azi_values[trigger_idx]
                trace_val = trace_values[trigger_idx]
                ic(trace_val)
                ic(trace_val[0])
                ic(trace_val[0][0])
                quit()

                # Scatter Plot
                if snr_val is not None and not np.isnan(snr_val):
                    # Plot Chi2016 point
                    if chi_2016_val is not None and not np.isnan(chi_2016_val):
                        ax_scatter.scatter(snr_val, chi_2016_val, color=color, marker=marker, s=60, alpha=0.9, label=f"St {station_id_int} T{trigger_idx+1} Chi16" if trigger_idx==0 else None, zorder=3)
                    # Plot ChiRCR point (perhaps hollow or different style)
                    if chi_rcr_val is not None and not np.isnan(chi_rcr_val):
                        ax_scatter.scatter(snr_val, chi_rcr_val, color=color, marker=marker, s=60, alpha=0.9, facecolors='none', edgecolors=color, linewidths=1.5, label=f"St {station_id_int} T{trigger_idx+1} ChiRCR" if trigger_idx==0 and (chi_2016_val is None or np.isnan(chi_2016_val)) else None, zorder=3)
                    # Arrow
                    if (chi_2016_val is not None and not np.isnan(chi_2016_val) and
                        chi_rcr_val is not None and not np.isnan(chi_rcr_val)):
                        ax_scatter.annotate("", xy=(snr_val, chi_rcr_val), xytext=(snr_val, chi_2016_val),
                                            arrowprops=dict(arrowstyle="->", color=color, lw=1.2, shrinkA=3, shrinkB=3), zorder=2)

                # Polar Plot
                if zen_val is not None and not np.isnan(zen_val) and \
                   azi_val is not None and not np.isnan(azi_val):
                    ax_polar.scatter(azi_val, np.degrees(zen_val), color=color, marker=marker, s=60, alpha=0.9, label=f"St {station_id_int} T{trigger_idx+1}" if trigger_idx==0 else None)

                # Trace Plot
                if trace_val is not None and hasattr(trace_val, "__len__") and len(trace_val) > 0 :
                    time_axis_trace = np.arange(0, 256, 0.5) # Assuming 256 samples and 0.5 microsecond time step
                    ax_trace.plot(time_axis_trace, trace_val, color=color, 
                                  linestyle='-' if trigger_idx % 2 == 0 else '--', # Vary linestyle for triggers
                                  alpha=0.8, label=f"St {station_id_int} T{trigger_idx+1}")
                
                # Add to figure legend map
                if station_id_int not in legend_handles_for_fig:
                    legend_handles_for_fig[station_id_int] = Line2D([0], [0], marker=marker_list[0], color=color, linestyle='None',
                                                       markersize=8, label=f"Station {station_id_int}")
                
                # Text info
                zen_deg = f"{np.degrees(zen_val):.1f}" if zen_val is not None and not np.isnan(zen_val) else "N/A"
                azi_deg = f"{np.degrees(azi_val):.1f}" if azi_val is not None and not np.isnan(azi_val) else "N/A"
                snr_f = f"{snr_val:.1f}" if snr_val is not None and not np.isnan(snr_val) else "N/A"
                text_info_lines.append(f"  St{station_id_int} T{trigger_idx+1}: SNR={snr_f}, Zen={zen_deg}°, Azi={azi_deg}°")

        # --- Finalize Subplots ---
        ax_scatter.set_xlabel("SNR")
        ax_scatter.set_ylabel("Chi")
        ax_scatter.set_title("SNR vs $\chi$ (Arrow: $\chi_{2016} \longrightarrow \chi_{RCR}$)")
        ax_scatter.set_xscale('log')
        ax_scatter.set_xlim(3, 100)
        ax_scatter.set_ylim(0, 1)
        ax_scatter.grid(True, linestyle='--', alpha=0.6)
        if any(ax_scatter.collections): ax_scatter.legend(fontsize='x-small', loc='best')


        ax_polar.set_theta_zero_location("N")
        ax_polar.set_theta_direction(-1)
        ax_polar.set_rlabel_position(22.5) # Degrees from North
        ax_polar.set_rlim(0, 90)
        ax_polar.set_rticks(np.arange(0, 91, 30))
        ax_polar.set_title("Zenith (radius) vs Azimuth (angle)")
        ax_polar.grid(True, linestyle='--', alpha=0.5)
        if any(ax_polar.collections): ax_polar.legend(fontsize='x-small', loc='lower left', bbox_to_anchor=(1.05, 0))


        ax_trace.set_xlabel("Time ($\mu s$)")
        ax_trace.set_ylabel("Amplitude (ADC counts or similar)")
        ax_trace.set_title("Time Traces")
        ax_trace.grid(True, linestyle='--', alpha=0.6)
        if any(ax_trace.get_lines()): ax_trace.legend(fontsize='x-small', loc='best')

        ax_text_info.text(0.02, 0.98, "\n".join(text_info_lines[:20]), # Limit lines
                          ha='left', va='top', fontsize=8, family='monospace',
                          bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="lightsteelblue", lw=1))
        ax_text_info.set_title("Trigger Summary", fontsize=10)

        if legend_handles_for_fig:
            fig.legend(handles=list(legend_handles_for_fig.values()), 
                       loc='lower center', ncol=min(len(legend_handles_for_fig), 6), 
                       bbox_to_anchor=(0.5, 0.01), title="Stations", fontsize='medium')

        plt.tight_layout(rect=[0, 0.05, 1, 0.93]) # Adjust for suptitle and figure legend
        
        master_filename = os.path.join(master_folder, f'master_event_{event_id}.png')
        try:
            plt.savefig(master_filename, dpi=150) # Slightly higher DPI for master plots
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
        # plot_snr_vs_chi(events_data_dict, specific_dataset_plot_dir, dataset_name_label)

        # 2. Parameter Histograms
        # plot_parameter_histograms(events_data_dict, specific_dataset_plot_dir, dataset_name_label)

        # 3. Polar Plot (Zenith vs Azimuth)
        # plot_polar_zen_azi(events_data_dict, specific_dataset_plot_dir, dataset_name_label)
        
        # 4. Master Event Plots
        plot_master_event_updated(events_data_dict, specific_dataset_plot_dir, dataset_name_label)

        ic(f"--- Finished plots for: {dataset_name_label} ---")

    ic("\nAll plotting complete.")

