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

# --- Helper for Loading Data ---
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
            
        num_triggers_in_station = len(station_triggers_data.get('SNR', []))
        chi_rcr_list = station_triggers_data.get('ChiRCR', [])
        chi_2016_list = station_triggers_data.get('Chi2016', [])

        for i in range(num_triggers_in_station):
            trigger_passed_chi = False
            if i < len(chi_rcr_list) and chi_rcr_list[i] is not None and \
               not np.isnan(chi_rcr_list[i]) and chi_rcr_list[i] > chi_threshold:
                trigger_passed_chi = True
            if not trigger_passed_chi and i < len(chi_2016_list) and \
               chi_2016_list[i] is not None and not np.isnan(chi_2016_list[i]) and \
               chi_2016_list[i] > chi_threshold:
                trigger_passed_chi = True
            if trigger_passed_chi:
                passing_station_triggers_count += 1
    
    return passing_station_triggers_count >= min_triggers_passing

def check_angle_cut(event_details, zenith_margin_deg=10.0, azimuth_margin_deg=20.0):
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

def check_coincidence_cuts(event_details):
    """
    Main function to check if a coincidence event passes all defined analysis cuts.
    Returns a dictionary with the pass/fail status for each cut.
    """
    results = {}
    results['chi_cut_passed'] = check_chi_cut(event_details)
    results['angle_cut_passed'] = check_angle_cut(event_details)
    
    # Add more cut checks here in the future:
    # results['another_cut_passed'] = check_another_cut(event_details)
    
    return results


# --- Plotting Function 1: SNR vs Chi Parameters ---
def plot_snr_vs_chi(events_dict, output_dir, dataset_name):
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
    handles, labels = axs[0].get_legend_handles_labels()
    if num_events > 0 and num_events <=15 and handles: 
        fig.legend(handles, labels, loc='center right', title="Events", bbox_to_anchor=(1.12, 0.5), fontsize='small')

    plt.suptitle(f'SNR vs Chi Parameters for {dataset_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.88 if num_events <=15 and handles else 0.98 , 0.96])
    
    plot_filename = os.path.join(output_dir, f"{dataset_name}_snr_vs_chi_params.png")
    plt.savefig(plot_filename, bbox_inches='tight'); ic(f"Saved SNR vs Chi plot: {plot_filename}"); plt.close(fig); gc.collect()


# --- Plotting Function 2: Parameter Histograms ---
def plot_parameter_histograms(events_dict, output_dir, dataset_name):
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
        if values_all: ax.hist(values_all, bins=50, edgecolor='black', alpha=0.4, label='All Events', color='grey'); has_data=True
        if values_pass: ax.hist(values_pass, bins=50, edgecolor='darkgreen', alpha=0.6, label='Pass Analysis Cuts', color='lightgreen'); has_data=True
        if values_fail: ax.hist(values_fail, bins=50, edgecolor='darkred', alpha=0.6, label='Fail Analysis Cuts', color='lightcoral'); has_data=True
        if has_data:
            if param_name in ['ChiRCR','Chi2016','ChiBad','SNR'] and any(v > 0 for v in values_all): ax.set_yscale('log'); ax.set_ylabel('Frequency (log scale)')
            if param_name in ['ChiRCR','Chi2016','ChiBad']: ax.set_xlim(0, 1)
            if param_name == 'SNR': ax.set_xlim(3, 100); ax.set_xscale('log')
            ax.grid(True, linestyle='--', alpha=0.6); ax.legend(fontsize='x-small')
        else: ax.text(0.5,0.5,f"No data for\n{param_name}",ha='center',va='center',transform=ax.transAxes); ax.set_title(f'{param_name}')
    for j in range(i + 1, len(axs_flat)): fig.delaxes(axs_flat[j])
    plt.suptitle(f'Parameter Histograms for {dataset_name} (by Cut Status)', fontsize=16); plt.tight_layout(rect=[0,0,1,0.96])
    plot_filename = os.path.join(output_dir, f"{dataset_name}_parameter_histograms_by_cut.png"); plt.savefig(plot_filename); ic(f"Saved: {plot_filename}"); plt.close(fig); gc.collect()


# --- Plotting Function 3: Polar Plot (Zenith vs Azimuth) ---
def plot_polar_zen_azi(events_dict, output_dir, dataset_name):
    ic(f"Generating polar Zenith vs Azimuth plot for {dataset_name}")
    os.makedirs(output_dir, exist_ok=True); all_zen_rad_values, all_azi_rad_values = [], []
    for event_data in events_dict.values():
        for station_triggers in event_data.get("stations", {}).values():
            zen_list_rad = station_triggers.get('Zen', []) # Assumed in RADIANS
            azi_list_rad = station_triggers.get('Azi', []) # Assumed in RADIANS
            for k in range(len(zen_list_rad)):
                if k < len(azi_list_rad):
                    zen_r_val, azi_r_val = zen_list_rad[k], azi_list_rad[k]
                    if zen_r_val is not None and not np.isnan(zen_r_val) and \
                       azi_r_val is not None and not np.isnan(azi_r_val):
                        all_zen_rad_values.append(zen_r_val) 
                        all_azi_rad_values.append(azi_r_val)
    
    if not all_azi_rad_values: ic(f"No valid Zenith/Azimuth data to plot for {dataset_name}."); return
    
    fig = plt.figure(figsize=(8, 8)); ax = plt.subplot(111, polar=True)
    
    all_zen_deg_values = np.degrees(np.array(all_zen_rad_values))
    # Azimuth is already in radians, Zenith converted to degrees for radial plot
    scatter = ax.scatter(np.array(all_azi_rad_values), # Azimuth (theta) in RADIANS
                         all_zen_deg_values,           # Zenith (r) in DEGREES
                         alpha=0.5, s=20, cmap='viridis', 
                         c=np.degrees(np.array(all_azi_rad_values))) # Color by Azimuth in degrees for interpretability
    
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_rlabel_position(45)
    ax.set_rlim(0, 90); ax.set_rticks(np.arange(0, 91, 15)) # Zenith ticks in degrees
    ax.set_title(f'Sky Plot: Zenith vs Azimuth for {dataset_name}\n(Zenith in degrees from center)', va='bottom', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plot_filename = os.path.join(output_dir, f"{dataset_name}_polar_zen_azi.png")
    plt.savefig(plot_filename); ic(f"Saved polar Zenith vs Azimuth plot: {plot_filename}"); plt.close(fig); gc.collect()


# --- Plotting Function 4: Master Event Plot (Updated) ---
def plot_master_event_updated(events_dict, base_output_dir, dataset_name):
    ic(f"Generating master event plots for {dataset_name}, separating by cut status.")
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
        if not isinstance(event_details, dict):
            ic(f"Warning: Event {event_id} data is not a dictionary. Skipping master plot.")
            continue

        cut_results = event_details.get('cut_results', {})
        passes_overall_analysis = event_details.get('passes_analysis_cuts', False)
        current_event_plot_dir = pass_cuts_folder if passes_overall_analysis else fail_cuts_folder
        
        fig = plt.figure(figsize=(18, 22)); 
        gs = gridspec.GridSpec(9, 2, figure=fig, hspace=1.0, wspace=0.3, height_ratios=[4, 4, 1, 1, 1, 1, 1, 1, 2])
        ax_scatter = fig.add_subplot(gs[0:2, 0]); 
        ax_polar = fig.add_subplot(gs[0:2, 1], polar=True)
        trace_axs = [fig.add_subplot(gs[2+i, 0]) for i in range(num_trace_channels)]
        spectrum_axs = [fig.add_subplot(gs[2+i, 1]) for i in range(num_trace_channels)]
        ax_text_box = fig.add_subplot(gs[7, :]) 
        
        event_time_str = "Unknown Time"
        if "datetime" in event_details and event_details["datetime"] is not None:
            try: event_time_dt = datetime.datetime.fromtimestamp(event_details["datetime"]); event_time_str = event_time_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            except Exception as e: ic(f"Error formatting datetime for event {event_id}: {e}. Timestamp: {event_details['datetime']}")
        
        fig.suptitle(f"Master Plot: Event {event_id} ({dataset_name}) - OVERALL: {'PASS' if passes_overall_analysis else 'FAIL'}\nTime: {event_time_str}", fontsize=16, y=0.98)

        text_info_lines = [f"Event ID: {event_id} -- Overall: {'PASS' if passes_overall_analysis else 'FAIL'}"]
        text_info_lines.append(f"Cut Status -> Chi: {'Passed' if cut_results.get('chi_cut_passed') else 'Failed'}, Angle: {'Passed' if cut_results.get('angle_cut_passed') else 'Failed'}")
        text_info_lines.append("--- Station Triggers ---")
        
        legend_handles_for_fig = {}; global_trace_min = float('inf'); global_trace_max = float('-inf')
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

        for station_id_str, station_data in event_details.get("stations", {}).items():
            try: station_id_int = int(station_id_str)
            except ValueError: continue
            color = color_map.get(station_id_int, default_color)
            snr_values = station_data.get("SNR", []); num_triggers = len(snr_values)
            if num_triggers == 0: continue
            
            zen_values_rad = (station_data.get("Zen", []) + [np.nan] * num_triggers)[:num_triggers]
            azi_values_rad = (station_data.get("Azi", []) + [np.nan] * num_triggers)[:num_triggers]
            pol_angle_values_rad = (station_data.get("PolAngle", []) + [np.nan] * num_triggers)[:num_triggers]
            pol_angle_err_values_rad = (station_data.get("PolAngleErr", []) + [np.nan] * num_triggers)[:num_triggers] # NEW: Get Polarization Angle Error
            
            chi_rcr_values = (station_data.get("ChiRCR", []) + [np.nan] * num_triggers)[:num_triggers]
            chi_2016_values = (station_data.get("Chi2016", []) + [np.nan] * num_triggers)[:num_triggers]
            event_ids_for_station = (station_data.get("event_ids", []) + ["N/A"] * num_triggers)[:num_triggers]
            all_traces_for_station = station_data.get("Traces", [])

            for trigger_idx in range(num_triggers):
                marker = marker_list[trigger_idx % len(marker_list)]
                snr_val, chi_rcr_val, chi_2016_val = snr_values[trigger_idx], chi_rcr_values[trigger_idx], chi_2016_values[trigger_idx]
                zen_rad, azi_rad = zen_values_rad[trigger_idx], azi_values_rad[trigger_idx]
                pol_rad = pol_angle_values_rad[trigger_idx] 
                pol_err_rad = pol_angle_err_values_rad[trigger_idx] # NEW: Get PolAngle Error for this trigger
                current_event_id_val = event_ids_for_station[trigger_idx]
                traces_this_trigger = (all_traces_for_station[trigger_idx] if trigger_idx < len(all_traces_for_station) else [])
                padded_traces_this_trigger = (list(traces_this_trigger) + [None]*num_trace_channels)[:num_trace_channels]

                if snr_val is not None and not np.isnan(snr_val):
                    if chi_2016_val is not None and not np.isnan(chi_2016_val): ax_scatter.scatter(snr_val, chi_2016_val, c=color, marker=marker, s=60, alpha=0.9, zorder=3)
                    if chi_rcr_val is not None and not np.isnan(chi_rcr_val): ax_scatter.scatter(snr_val, chi_rcr_val, marker=marker, s=60, alpha=0.9, facecolors='none', edgecolors=color, linewidths=1.5, zorder=3)
                
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
                            spectrum_axs[ch_idx].plot(freq_ax_mhz, spectrum, c=color, ls=':' if trigger_idx % 2 == 0 else '-.', alpha=0.5)
                
                if station_id_int not in legend_handles_for_fig: legend_handles_for_fig[station_id_int] = Line2D([0], [0], marker='o', c=color, ls='None', markersize=8, label=f"St {station_id_int}")
                
                zen_d_text = f"{np.degrees(zen_rad):.1f}°" if zen_rad is not None and not np.isnan(zen_rad) else "N/A"
                azi_d_text = f"{(np.degrees(azi_rad) % 360):.1f}°" if azi_rad is not None and not np.isnan(azi_rad) else "N/A"
                snr_fstr = f"{snr_val:.1f}" if snr_val is not None and not np.isnan(snr_val) else "N/A"
                ev_id_fstr = f"{int(current_event_id_val)}" if current_event_id_val not in ["N/A", np.nan, None] else "N/A"
                
                # NEW: Format polarization angle with its error
                pol_angle_full_text = "N/A"
                if pol_rad is not None and not np.isnan(pol_rad):
                    pol_angle_full_text = f"{np.degrees(pol_rad):.1f}"
                    if pol_err_rad is not None and not np.isnan(pol_err_rad) and isinstance(pol_err_rad, (float, int)):
                         pol_angle_full_text += f" ± {np.degrees(pol_err_rad):.1f}°"
                    else:
                         pol_angle_full_text += "°"

                # CHANGED: Added Pol(arization) angle with error to the text line
                text_info_lines.append(f"  St{station_id_int} T{trigger_idx+1}: ID={ev_id_fstr}, SNR={snr_fstr}, Zen={zen_d_text}, Azi={azi_d_text}, Pol={pol_angle_full_text}")

        # === START: MODIFIED SNR vs CHI PLOT SETUP ===
        ax_scatter.set_xlabel("SNR")
        ax_scatter.set_ylabel(r"$\chi$") # Use Chi symbol for y-axis
        # ax_scatter.set_title(r"SNR vs. $\chi$") # Simplified title
        ax_scatter.set_xscale('log')
        ax_scatter.set_xlim(3, 100)
        ax_scatter.set_ylim(0, 1)
        ax_scatter.grid(True, linestyle='--', alpha=0.6)

        # Create handles for the two legends
        # Legend 1: Chi Type (filled vs outline)
        chi_2016_handle = Line2D([0], [0], marker='o', color='k', label=r'$\chi_{2016}$ (Filled)',
                                 linestyle='None', markersize=8, markerfacecolor='k')
        chi_RCR_handle = Line2D([0], [0], marker='o', color='k', label=r'$\chi_{RCR}$ (Outline)',
                              linestyle='None', markersize=8, markerfacecolor='none', markeredgecolor='k')
        
        # Legend 2: Station Colors (handles are already in legend_handles_for_fig)
        station_handles = list(legend_handles_for_fig.values())
        
        # Add the legends to the ax_scatter plot
        # Create the first legend (for stations) and add it manually
        if station_handles:
            leg1 = ax_scatter.legend(handles=station_handles, loc='upper right', title="Stations")
            ax_scatter.add_artist(leg1)
        
        # Create the second legend (for chi types). This will be placed automatically.
        ax_scatter.legend(handles=[chi_2016_handle, chi_RCR_handle], loc='lower left', title=r"$\chi$ Type")
        # === END: MODIFIED SNR vs CHI PLOT SETUP ===
        
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
        
        # REMOVED: The old figure-wide legend is no longer needed as the info is in the scatter plot's legends.
        # if legend_handles_for_fig: 
        #     ax_legend = fig.add_subplot(gs[8, :])
        #     ax_legend.axis('off')
        #     ax_legend.legend(handles=list(legend_handles_for_fig.values()), loc='center', ncol=min(len(legend_handles_for_fig), 8), title="Stations", fontsize='medium')

        master_filename = os.path.join(current_event_plot_dir, f'master_event_{event_id}.png')
        try: 
            plt.savefig(master_filename, dpi=150, bbox_inches='tight')
        except Exception as e: 
            ic(f"Error saving master plot {master_filename}: {e}")
        plt.close(fig); gc.collect()
    ic(f"Finished master event plots for {dataset_name}. Check pass_cuts/ and fail_cuts/ subfolders.")


# --- Main Script ---
if __name__ == '__main__':
    ic.enable()
    import configparser
    config = configparser.ConfigParser()
    config_path = os.path.join('HRAStationDataAnalysis', 'config.ini') 
    if not os.path.exists(config_path): config_path = 'config.ini'
    if not os.path.exists(config_path): ic(f"CRITICAL: config.ini not found."); exit()
    config.read(config_path); date_of_data = config['PARAMETERS']['date']
    date_of_process = config['PARAMETERS']['date_processing']
    base_processed_data_dir = os.path.join("HRAStationDataAnalysis", "StationData", "processedNumpyData")
    processed_data_dir_for_date = os.path.join(base_processed_data_dir, date_of_data)
    
    # CHANGED: Point to the file that includes polarization data
    input_file = f"{date_of_process}_CoincidenceDatetimes_with_all_params_recalcZenAzi_calcPol.pkl"
    dataset_paths = [os.path.join(processed_data_dir_for_date, input_file)]
    
    dataset_names = ["CoincidenceEvents"]
    dataset_plot_suffixes = [f"CoincidenceEvents_{date_of_process}"]
    output_plot_basedir = os.path.join("HRAStationDataAnalysis", "plots")
    os.makedirs(output_plot_basedir, exist_ok=True)
    datasets_to_plot_info = []
    
    for i, d_path in enumerate(dataset_paths):
        if not os.path.exists(d_path):
            ic(f"Warning: Input file not found: {d_path}. Skipping.")
            # Fallback to the previous file if the polarization one doesn't exist
            fallback_file = f"{date_of_process}_CoincidenceDatetimes_with_all_params_recalcZenAzi.pkl"
            d_path = os.path.join(processed_data_dir_for_date, fallback_file)
            if not os.path.exists(d_path):
                ic(f"Error: Fallback file also not found: {d_path}. Cannot proceed.")
                continue
            else:
                ic(f"Found and using fallback file: {d_path}")

        data = _load_pickle(d_path)
        if data is not None: 
            datasets_to_plot_info.append({"name": dataset_names[i], "data": data, "plot_dir_suffix": dataset_plot_suffixes[i]})
            ic(f"Loaded: {d_path}")
        else: 
            ic(f"Could not load data from: {d_path}.")

    if not datasets_to_plot_info: ic("No datasets loaded. Exiting."); exit()

    for dataset_info in datasets_to_plot_info:
        dataset_name_label = dataset_info["name"]
        events_data_dict = dataset_info["data"]
        specific_dataset_plot_dir = os.path.join(output_plot_basedir, dataset_info["plot_dir_suffix"])
        os.makedirs(specific_dataset_plot_dir, exist_ok=True)

        ic(f"\n--- Processing dataset for cuts and plots: {dataset_name_label} ---")
        if not isinstance(events_data_dict, dict) or not events_data_dict:
            ic(f"Dataset '{dataset_name_label}' is empty or not a dict. Skipping."); continue

        num_passing_overall = 0; num_failing_overall = 0
        for event_id, event_details_loopvar in events_data_dict.items():
            if isinstance(event_details_loopvar, dict):
                 cut_results_dict = check_coincidence_cuts(event_details_loopvar)
                 event_details_loopvar['cut_results'] = cut_results_dict
                 event_details_loopvar['passes_analysis_cuts'] = all(cut_results_dict.values())
                 
                 if event_details_loopvar['passes_analysis_cuts']: num_passing_overall +=1
                 else: num_failing_overall +=1
            elif event_details_loopvar is not None : 
                 event_details_loopvar_placeholder = {'passes_analysis_cuts': False, 
                                       'cut_results': {'error': 'Malformed event data'}}
                 if isinstance(events_data_dict, dict):
                    events_data_dict[event_id] = event_details_loopvar_placeholder
                 num_failing_overall +=1

        ic(f"Analysis cuts applied to '{dataset_name_label}': {num_passing_overall} events passed, {num_failing_overall} events failed overall.")

        # plot_snr_vs_chi(events_data_dict, specific_dataset_plot_dir, dataset_name_label)
        # plot_parameter_histograms(events_data_dict, specific_dataset_plot_dir, dataset_name_label)
        # plot_polar_zen_azi(events_data_dict, specific_dataset_plot_dir, dataset_name_label)
        plot_master_event_updated(events_data_dict, specific_dataset_plot_dir, dataset_name_label)

        ic(f"--- Finished plots for: {dataset_name_label} ---")
    ic("\nAll plotting complete.")