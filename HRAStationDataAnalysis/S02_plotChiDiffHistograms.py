import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import glob
import pickle
import json
import configparser
from icecream import ic
from HRAStationDataAnalysis.C_utils import getTimeEventMasks
from scipy.optimize import curve_fit
from scipy.integrate import quad

# --- Configuration ---
def load_config():
    config = configparser.ConfigParser()
    config_path = 'HRAStationDataAnalysis/config.ini'
    if not os.path.exists(config_path):
        # Fallback or error
        ic(f"Config file not found at {config_path}")
        return None
    config.read(config_path)
    return config

# --- Data Loading Utilities ---
def load_station_data(folder, date, station_id, data_name):
    file_pattern = os.path.join(folder, f'{date}_Station{station_id}_{data_name}*')
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        return np.array([])
    data_arrays = [np.load(f, allow_pickle=True) for f in file_list]
    data_arrays = [arr for arr in data_arrays if arr.size > 0]
    if not data_arrays:
        return np.array([])
    return np.concatenate(data_arrays, axis=0)

def load_coincidence_events(filepath, requested_event_ids):
    if not os.path.exists(filepath):
        ic(f"Warning: Coincidence file not found: {filepath}")
        return {}
    try:
        with open(filepath, 'rb') as handle:
            events_dict = pickle.load(handle)
    except Exception as exc:
        ic(f"Error loading coincidence file {filepath}: {exc}")
        return {}

    coincidence_events = {}
    for event_id in requested_event_ids:
        event_obj = events_dict.get(event_id)
        if event_obj is None:
            event_obj = events_dict.get(str(event_id))
        if event_obj is None:
            continue
        coincidence_events[event_id] = event_obj
    return coincidence_events

def get_all_cut_masks(data_dict, cuts, cut_type='rcr'):
    snr = data_dict['snr']
    chircr = data_dict['ChiRCR']
    chi2016 = data_dict['Chi2016']
    
    if cut_type == 'rcr':
        chi_rcr_snr_cut_values = np.interp(snr, cuts['chi_rcr_line_snr'], cuts['chi_rcr_line_chi'])
        chi_diff = chircr - chi2016
        masks = {}
        masks['snr_cut'] = snr < cuts['snr_max']
        masks['snr_line_cut'] = chircr > chi_rcr_snr_cut_values
        masks['chi_diff_cut'] = (chi_diff > cuts['chi_diff_threshold']) & (chi_diff < cuts.get('chi_diff_max', 999))
        masks['all_cuts'] = masks['snr_cut'] & masks['snr_line_cut'] & masks['chi_diff_cut']
        
    elif cut_type == 'backlobe':
        chi_2016_snr_cut_values = np.interp(snr, cuts['chi_2016_line_snr'], cuts['chi_2016_line_chi'])
        chi_diff = chircr - chi2016
        masks = {}
        masks['snr_cut'] = snr < cuts['snr_max']
        masks['snr_line_cut'] = chi2016 > chi_2016_snr_cut_values
        masks['chi_diff_cut'] = (chi_diff < -cuts['chi_diff_threshold']) & (chi_diff > -cuts.get('chi_diff_max', 999))
        masks['all_cuts'] = masks['snr_cut'] & masks['snr_line_cut'] & masks['chi_diff_cut']
    
    return masks

def main():
    ic.configureOutput(prefix='Chi-Diff Hist | ')
    
    # --- Setup ---
    config = load_config()
    if config is None:
        return

    date = config['PARAMETERS']['date']
    date_processing = config['PARAMETERS']['date_processing']
    station_data_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'
    plot_folder = f'HRAStationDataAnalysis/plots/{date_processing}/Histograms/'
    os.makedirs(plot_folder, exist_ok=True)

    station_ids = [13, 14, 15, 17, 18, 19, 30]
    
    # Cuts Definition
    cuts = {
        'snr_max': 50,
        'chi_rcr_line_snr': np.array([0, 7, 8.5, 15, 20, 30, 100]),
        'chi_rcr_line_chi': np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]),
        'chi_diff_threshold': 0.0,
        'chi_diff_max': 0.2,
        'chi_2016_line_snr': np.array([0, 7, 8.5, 15, 20, 30, 100]),
        'chi_2016_line_chi': np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75])
    }

    # --- Load Coincidence Events ---
    coincidence_pickle_path = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/9.24.25_CoincidenceDatetimes_passing_cuts_with_all_params_recalcZenAzi_calcPol.pkl"
    requested_coincidence_event_ids = [
        3047, 3432, 10195, 10231, 10273, 10284, 10444, 10449, 
        10466, 10471, 10554, 11197, 11220, 11230, 11236, 11243
    ]
    coincidence_events_raw = load_coincidence_events(coincidence_pickle_path, requested_coincidence_event_ids)
    
    # Process Coincidence Events
    coinc_rcr_events = []
    coinc_bl_events = []
    
    # We need to flatten coincidence events into a list of (Station, EventID, ChiDiff)
    # And apply cuts
    
    for event_id, event_details in coincidence_events_raw.items():
        stations_info = event_details.get('stations', {}) if isinstance(event_details, dict) else {}
        for station_key, station_payload in stations_info.items():
            try:
                st_id = int(station_key)
            except: continue
            
            if st_id not in station_ids: continue
            
            snr = np.asarray(station_payload.get('SNR', []), dtype=float)
            chi2016 = np.asarray(station_payload.get('Chi2016', []), dtype=float)
            chircr = np.asarray(station_payload.get('ChiRCR', []), dtype=float)
            
            if snr.size == 0: continue
            
            # Create a mini data dict for cut checking
            mini_data = {'snr': snr, 'Chi2016': chi2016, 'ChiRCR': chircr}
            
            masks_rcr = get_all_cut_masks(mini_data, cuts, cut_type='rcr')
            masks_bl = get_all_cut_masks(mini_data, cuts, cut_type='backlobe')
            
            # Check which events pass
            pass_rcr = masks_rcr['all_cuts']
            pass_bl = masks_bl['all_cuts']
            
            for i in range(len(snr)):
                chi_diff = chircr[i] - chi2016[i]
                evt_tuple = (st_id, event_id, chi_diff, snr[i], chircr[i], chi2016[i])
                
                if pass_rcr[i]:
                    coinc_rcr_events.append(evt_tuple)
                if pass_bl[i]:
                    coinc_bl_events.append(evt_tuple)

    # --- Load Backlobe 2016 Events ---
    json_path = 'StationDataAnalysis/2016FoundEvents.json'
    found_events_json = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            found_events_json = json.load(f)
    
    # --- Load Data and Filter ---
    data_passing_rcr = []
    data_passing_bl = []
    backlobe_2016_events = []
    
    for station_id in station_ids:
        # Load Raw Data
        snr_array = load_station_data(station_data_folder, date, station_id, 'SNR')
        Chi2016_array = load_station_data(station_data_folder, date, station_id, 'Chi2016')
        ChiRCR_array = load_station_data(station_data_folder, date, station_id, 'ChiRCR')
        times = load_station_data(station_data_folder, date, station_id, 'Time')
        event_ids_raw = load_station_data(station_data_folder, date, station_id, 'EventIDs')
        
        if snr_array.size == 0: continue
        
        # Apply Initial Mask (same as S01)
        initial_mask, unique_indices = getTimeEventMasks(times, event_ids_raw)
        
        # Masked Data
        masked_snr = snr_array[initial_mask]
        masked_chi2016 = Chi2016_array[initial_mask]
        masked_chircr = ChiRCR_array[initial_mask]
        masked_times = times[initial_mask]
        masked_event_ids = event_ids_raw[initial_mask]
        
        # Load Passing Events Indices
        npz_file = f'{plot_folder}../PassingEvents_Station{station_id}_{date}.npz'
        if not os.path.exists(npz_file):
            ic(f"Warning: NPZ file not found: {npz_file}")
            continue
            
        passing_data = np.load(npz_file)
        
        # Extract RCR Passing
        # The 'unique_index' in npz refers to indices in the 'masked' arrays (post-initial_mask)
        # Wait, let's verify this assumption.
        # In S01: final_indices = unique_indices[cuts_mask]
        # passing_events_to_save['all_cuts']['unique_index'] = unique_indices[masks_rcr['all_cuts']]
        # Here unique_indices is the array of indices into masked data?
        # No, unique_indices from getTimeEventMasks are indices into the *compressed* array (after mask application)?
        # Let's assume unique_indices are indices into the array *after* initial_mask is applied.
        
        # Let's look at S01 again.
        # initial_mask, unique_indices = getTimeEventMasks(times, event_ids_raw)
        # temp_times = times[initial_mask][unique_indices]
        # This implies unique_indices are indices into times[initial_mask].
        
        # So, to get the event:
        # event = times[initial_mask][unique_index_from_npz]
        
        # RCR Passing
        rcr_indices = passing_data['all_cuts']['unique_index']
        for idx in rcr_indices:
            chi_diff = masked_chircr[idx] - masked_chi2016[idx]
            evt_tuple = (station_id, masked_event_ids[idx], chi_diff, masked_snr[idx], masked_chircr[idx], masked_chi2016[idx])
            data_passing_rcr.append(evt_tuple)
            
        # BL Passing
        bl_indices = passing_data['backlobe_all_cuts']['unique_index']
        for idx in bl_indices:
            chi_diff = masked_chircr[idx] - masked_chi2016[idx]
            evt_tuple = (station_id, masked_event_ids[idx], chi_diff, masked_snr[idx], masked_chircr[idx], masked_chi2016[idx])
            data_passing_bl.append(evt_tuple)
            
        # Backlobe 2016 Matching
        station_key = f"Station{station_id}Found"
        if station_key in found_events_json:
            target_times = set(found_events_json[station_key])
            # We need to find events in our data that match these times.
            # We should search in the *masked* data to be consistent with what we are plotting?
            # Or raw data? S01 searches in raw data.
            # "bl_2016_entry['Backlobe']['snr'] = snr_array[found_indices]"
            # So it uses raw data.
            
            # Efficient matching
            for i, t in enumerate(times):
                if t in target_times:
                    chi_diff = ChiRCR_array[i] - Chi2016_array[i]
                    evt_tuple = (station_id, event_ids_raw[i], chi_diff, snr_array[i], ChiRCR_array[i], Chi2016_array[i])
                    backlobe_2016_events.append(evt_tuple)

    # --- Plotting ---
    bins = np.linspace(-0.2, 0.2, 31) # Adjust as needed
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Chi Difference Histograms (RCR - BL) - {date}', fontsize=16)
    
    # Helper to extract ChiDiffs
    def get_diffs(event_list):
        return [e[2] for e in event_list]

    # 1. Coincidence Events
    ax = axs[0, 0]
    ax.hist(get_diffs(coinc_rcr_events), bins=bins, histtype='step', label='Coinc RCR Pass', color='purple', linewidth=2)
    ax.hist(get_diffs(coinc_bl_events), bins=bins, histtype='step', label='Coinc BL Pass', color='orange', linewidth=2)
    ax.set_title('Coincidence Events')
    ax.set_xlabel(r'RCR-$\chi$ - BL-$\chi$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Data Passing Cuts
    ax = axs[0, 1]
    ax.hist(get_diffs(data_passing_rcr), bins=bins, histtype='step', label='Data RCR Pass', color='purple', linewidth=2)
    ax.hist(get_diffs(data_passing_bl), bins=bins, histtype='step', label='Data BL Pass', color='orange', linewidth=2)
    ax.set_title('Data Passing Cuts')
    ax.set_xlabel(r'RCR-$\chi$ - BL-$\chi$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Backlobe 2016
    ax = axs[1, 0]
    ax.hist(get_diffs(backlobe_2016_events), bins=bins, histtype='step', label='Backlobe 2016', color='green', linewidth=2)
    ax.set_title('Backlobe 2016 Events')
    ax.set_xlabel(r'RCR-$\chi$ - BL-$\chi$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Combined with Prioritization
    # Priority: Backlobe 2016 > Coincidence > Data
    # Uniqueness key: (StationID, EventID)
    
    final_events = {} # Key -> (Category, ChiDiff)
    
    # Add Data first (lowest priority)
    for e in data_passing_rcr + data_passing_bl:
        key = (e[0], e[1])
        final_events[key] = ('Data', e[2])
        
    # Add Coincidence (overwrites Data)
    for e in coinc_rcr_events:
        key = (e[0], e[1])
        final_events[key] = ('Coincidence-RCR', e[2])
        
    for e in coinc_bl_events:
        key = (e[0], e[1])
        final_events[key] = ('Coincidence-BL', e[2])
        
    # Add Backlobe 2016 (overwrites Coincidence and Data)
    for e in backlobe_2016_events:
        key = (e[0], e[1])
        final_events[key] = ('Backlobe 2016', e[2])
        
    # Separate for plotting
    combined_data = []
    combined_coinc_rcr = []
    combined_coinc_bl = []
    combined_bl2016 = []
    
    for cat, diff in final_events.values():
        if cat == 'Data': combined_data.append(diff)
        elif cat == 'Coincidence-RCR': combined_coinc_rcr.append(diff)
        elif cat == 'Coincidence-BL': combined_coinc_bl.append(diff)
        elif cat == 'Backlobe 2016': combined_bl2016.append(diff)
        
    ax = axs[1, 1]
    ax.hist([combined_data, combined_coinc_rcr, combined_coinc_bl, combined_bl2016], bins=bins, histtype='step', stacked=False, 
            label=['Data', 'Coinc RCR', 'Coinc BL', 'Backlobe 2016'], 
            color=['gray', 'purple', 'orange', 'green'], linewidth=2)
    ax.set_title('Combined (Prioritized)')
    ax.set_xlabel(r'RCR-$\chi$ - BL-$\chi$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f'{plot_folder}ChiDiff_Histograms_Combined.png'
    plt.savefig(save_path)
    ic(f"Saved plot to {save_path}")
    plt.close()

    # --- New Plot: Data Only with Gaussian Fit ---
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Data for new plot: combined_data (Data category only)
    data_to_plot = np.array(combined_data)

    # --- Statistics Calculation ---
    bl_region_data = data_to_plot[data_to_plot < 0]
    rcr_region_data = data_to_plot[data_to_plot >= 0]

    bl_mean = np.mean(bl_region_data) if bl_region_data.size > 0 else np.nan
    bl_std = np.std(bl_region_data) if bl_region_data.size > 0 else np.nan

    print(f"\n--- Data Statistics ---")
    print(f"BL Region Data Points ({len(bl_region_data)}): {bl_region_data}")
    print(f"RCR Region Data Points ({len(rcr_region_data)}): {rcr_region_data}")
    print(f"BL Region Mean: {bl_mean:.4f}")
    print(f"BL Region Std Dev: {bl_std:.4f}")
    print(f"-----------------------\n")
    
    # Histogram
    counts, bin_edges, patches = ax2.hist(data_to_plot, bins=bins, histtype='bar', edgecolor='black', zorder=1)
    
    # Color bins
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for c, p in zip(bin_centers, patches):
        if c < 0:
            p.set_facecolor('gray')
        else:
            p.set_facecolor('red')

    # Legend 1: Regions
    legend_elements1 = [
        Patch(facecolor='gray', edgecolor='black', label='BL Region'),
        Patch(facecolor='red', edgecolor='black', label='RCR Region')
    ]
    leg1 = ax2.legend(handles=legend_elements1, loc='upper right', bbox_to_anchor=(0.98, 0.98), framealpha=1)
    ax2.add_artist(leg1)
            
    # Fit Gaussian to left side
    # Select data for fit
    mask_left = bin_centers < 0
    X_fit = bin_centers[mask_left]
    Y_fit = counts[mask_left]
    
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
        
    # Initial guess
    if len(Y_fit) > 0:
        p0 = [max(Y_fit), 0, 0.1]
        
        try:
            popt, pcov = curve_fit(gaussian, X_fit, Y_fit, p0=p0)
            
            # Draw Gaussian
            x_plot = np.linspace(min(bin_edges), max(bin_edges), 1000)
            y_plot = gaussian(x_plot, *popt)
            gauss_line, = ax2.plot(x_plot, y_plot, color='blue', linewidth=2, label='Gaussian Fit', zorder=6)
            
            # Shade region > 0
            x_shade = np.linspace(0, max(bin_edges), 500)
            y_shade = gaussian(x_shade, *popt)
            gauss_fill = ax2.fill_between(x_shade, y_shade, color='blue', alpha=0.25, linestyle='--', hatch='//', label='Gaussian>0', zorder=5)
            
            # Calculate sum of area > 0
            area, _ = quad(lambda x: gaussian(x, *popt), 0, np.inf)
            bin_width = bin_edges[1] - bin_edges[0]
            expected_events = area / bin_width
            
            # Legend 2: Gaussian
            leg2 = ax2.legend(handles=[gauss_line, gauss_fill], loc='upper right', bbox_to_anchor=(0.98, 0.86), framealpha=1)
            
            ax2.text(0.94, 0.66, f'Expected > 0: {expected_events:.1f}', transform=ax2.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))
            
        except Exception as e:
            ic(f"Gaussian fit failed: {e}")
    else:
        ic("Not enough data for Gaussian fit on left side.")

    ax2.set_title('Data Only - Gaussian Fit to Background')
    ax2.set_xlabel(r'RCR-$\chi$ - BL-$\chi$')
    ax2.set_ylabel('N-Events')
    ax2.grid(True, alpha=0.3)
    
    save_path_2 = f'{plot_folder}ChiDiff_Data_Gaussian.png'
    plt.savefig(save_path_2)
    ic(f"Saved data plot to {save_path_2}")
    plt.close()

if __name__ == "__main__":
    main()
