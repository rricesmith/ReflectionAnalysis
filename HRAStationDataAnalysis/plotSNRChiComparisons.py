import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import os
import glob
import gc
import h5py
import pickle
from icecream import ic
import configparser
from C_utils import getTimeEventMasks
from HRASimulation.HRAEventObject import HRAevent # Assuming HRAEventObject.py is in the python path

# --- Utility Functions ---

def loadHRAfromH5(filename):
    """
    Loads a list of HRAevent objects from an HDF5 file.
    """
    ic(f"Loading HRA event list from: {filename}")
    eventList = []
    with h5py.File(filename, 'r') as hf:
        for key in hf.keys():
            obj_bytes = hf[key][0]
            event = pickle.loads(obj_bytes)
            eventList.append(event)
    ic(f"Successfully loaded {len(eventList)} events.")
    return eventList

def load_station_data(folder, date, station_id, data_name):
    """
    Loads and concatenates data files for a specific station and data type.
    """
    file_pattern = os.path.join(folder, f'{date}_Station{station_id}_{data_name}*')
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        ic(f"Warning: No files found for {data_name} with pattern: {file_pattern}")
        return np.array([])
    
    data_arrays = [np.load(f, allow_pickle=True) for f in file_list]
    data_arrays = [arr for arr in data_arrays if arr.size > 0]
    if not data_arrays:
        return np.array([])
        
    return np.concatenate(data_arrays, axis=0)

def get_sim_data(HRAeventList, direct_weight_name, reflected_weight_name, direct_stations, reflected_stations, sigma=4.5):
    """
    Extracts SNR, Chi, and weights from the HRAevent list, separating them
    into direct and reflected triggers using separate weight names.
    """
    direct_data = {'snr': [], 'Chi2016': [], 'ChiRCR': [], 'weights': []}
    reflected_data = {'snr': [], 'Chi2016': [], 'ChiRCR': [], 'weights': []}

    for event in HRAeventList:
        direct_weight = event.getWeight(direct_weight_name, primary=True, sigma=sigma)
        if not np.isnan(direct_weight) and direct_weight > 0:
            triggered_direct = [st_id for st_id in direct_stations if event.hasTriggered(st_id, sigma)]
            if triggered_direct:
                split_weight = direct_weight / len(triggered_direct)
                for st_id in triggered_direct:
                    snr, chi_dict = event.getSNR(st_id), event.getChi(st_id)
                    if snr is not None and chi_dict:
                        direct_data['snr'].append(snr)
                        direct_data['Chi2016'].append(chi_dict.get('Chi2016', np.nan))
                        direct_data['ChiRCR'].append(chi_dict.get('ChiRCR', np.nan))
                        direct_data['weights'].append(split_weight)

        reflected_weight = event.getWeight(reflected_weight_name, primary=True, sigma=sigma)
        if not np.isnan(reflected_weight) and reflected_weight > 0:
            triggered_reflected = [st_id for st_id in reflected_stations if event.hasTriggered(st_id, sigma)]
            if triggered_reflected:
                split_weight = reflected_weight / len(triggered_reflected)
                for st_id in triggered_reflected:
                    snr, chi_dict = event.getSNR(st_id), event.getChi(st_id)
                    if snr is not None and chi_dict:
                        reflected_data['snr'].append(snr)
                        reflected_data['Chi2016'].append(chi_dict.get('Chi2016', np.nan))
                        reflected_data['ChiRCR'].append(chi_dict.get('ChiRCR', np.nan))
                        reflected_data['weights'].append(split_weight)

    for data_dict in [direct_data, reflected_data]:
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key])
            
    return direct_data, reflected_data

def calculate_cut_stats_table(data_dict, cuts, is_sim, title):
    """Calculates the percentage of events/weight passing each cut individually and all together."""
    if is_sim:
        total_val = np.sum(data_dict['weights'])
        if total_val == 0: return f"{title}\nNo weight in dataset."
    else:
        total_val = len(data_dict['snr'])
        if total_val == 0: return f"{title}\nNo events in dataset."
    
    snr, chi2016, chircr = data_dict['snr'], data_dict['Chi2016'], data_dict['ChiRCR']
    weights = data_dict.get('weights', np.ones_like(snr))

    masks = {
        f"Chi16 > {cuts['Chi2016_min']}": chi2016 > cuts['Chi2016_min'],
        f"Chi16 < {cuts['Chi2016_max']}": chi2016 < cuts['Chi2016_max'],
        f"ChiRCR > {cuts['ChiRCR_min']}": chircr > cuts['ChiRCR_min'],
        f"SNR < {cuts['snr_max']}": snr < cuts['snr_max'],
        f"ChiDiff > {cuts['chi_diff_min']}": (chircr - chi2016) > cuts['chi_diff_min']
    }
    
    full_mask = np.ones_like(snr, dtype=bool)
    
    lines = [f"{title}:"]
    for name, mask in masks.items():
        passing_val = np.sum(weights[mask])
        lines.append(f"- {name:<15}: {passing_val/total_val:>7.2%}")
        full_mask &= mask
        
    passing_all = np.sum(weights[full_mask])
    lines.append(f"- {'All Cuts':<15}: {passing_all/total_val:>7.2%}")
    return "\n".join(lines)


# --- Plotting Functions ---

def set_plot_labels(ax, xlabel, ylabel, title, xlim, ylim, xscale='linear', yscale='linear'):
    """Set common labels and properties for a subplot."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(visible=True, which='both', axis='both')

def draw_cut_visuals(ax, plot_key, cuts_dict):
    """Draws lines and shaded regions for cuts on a given subplot."""
    if plot_key == 'snr_vs_chi2016':
        ax.axhline(y=cuts_dict['Chi2016_min'], color='k', linestyle='--', linewidth=1.5)
        ax.axhline(y=cuts_dict['Chi2016_max'], color='k', linestyle='--', linewidth=1.5)
        ax.fill_between(ax.get_xlim(), cuts_dict['Chi2016_min'], cuts_dict['Chi2016_max'], color='gray', alpha=0.2)
        ax.axvline(x=cuts_dict['snr_max'], color='m', linestyle='--', linewidth=1.5)
        ax.fill_betweenx(ax.get_ylim(), cuts_dict['snr_max'], ax.get_xlim()[1], color='m', alpha=0.1)
    elif plot_key == 'snr_vs_chircr':
        ax.axhline(y=cuts_dict['ChiRCR_min'], color='k', linestyle='--', linewidth=1.5)
        ax.fill_between(ax.get_xlim(), cuts_dict['ChiRCR_min'], 1, color='gray', alpha=0.2)
        ax.axvline(x=cuts_dict['snr_max'], color='m', linestyle='--', linewidth=1.5)
        ax.fill_betweenx(ax.get_ylim(), cuts_dict['snr_max'], ax.get_xlim()[1], color='m', alpha=0.1)
    elif plot_key == 'chi_vs_chi':
        ax.axhline(y=cuts_dict['ChiRCR_min'], color='k', linestyle='--', linewidth=1.5)
        ax.axvline(x=cuts_dict['Chi2016_min'], color='k', linestyle='--', linewidth=1.5)
        ax.axvline(x=cuts_dict['Chi2016_max'], color='k', linestyle='--', linewidth=1.5)
        x_vals = np.array([0, 1.0])
        ax.plot(x_vals, x_vals + cuts_dict['chi_diff_min'], color='purple', linestyle='--', linewidth=1.5)
        x_fill = np.linspace(cuts_dict['Chi2016_min'], cuts_dict['Chi2016_max'], 100)
        y_lower = np.maximum(cuts_dict['ChiRCR_min'], x_fill + cuts_dict['chi_diff_min'])
        ax.fill_between(x_fill, y_lower, 1, color='gray', alpha=0.3, interpolate=True)
    elif plot_key == 'snr_vs_chidiff':
        ax.axhline(y=cuts_dict['chi_diff_min'], color='purple', linestyle='--', linewidth=1.5)
        ax.fill_between(ax.get_xlim(), cuts_dict['chi_diff_min'], ax.get_ylim()[1], color='purple', alpha=0.1)
        ax.axvline(x=cuts_dict['snr_max'], color='m', linestyle='--', linewidth=1.5)
        ax.fill_betweenx(ax.get_ylim(), cuts_dict['snr_max'], ax.get_xlim()[1], color='m', alpha=0.1)

def plot_2x2_grid(fig, axs, base_data, base_plot_type, cuts_dict, overlays=None, hist_bins_dict=None):
    """
    Master function to generate a 2x2 grid. Plots a base layer (scatter or hist)
    and then adds any number of overlay layers on top.
    """
    im = None
    
    plot_configs = {
        'snr_vs_chi2016': {'xlabel': 'SNR', 'ylabel': 'Chi2016', 'xlim': (3, 100), 'ylim': (0, 1), 'xscale': 'log'},
        'snr_vs_chircr': {'xlabel': 'SNR', 'ylabel': 'ChiRCR', 'xlim': (3, 100), 'ylim': (0, 1), 'xscale': 'log'},
        'chi_vs_chi': {'xlabel': 'Chi2016', 'ylabel': 'ChiRCR', 'xlim': (0, 1), 'ylim': (0, 1)},
        'snr_vs_chidiff': {'xlabel': 'SNR', 'ylabel': 'ChiRCR - Chi2016', 'xlim': (3, 100), 'ylim': (-0.4, 0.4), 'xscale': 'log'}
    }
    
    # Map data keys to plot configs
    data_map = {
        'snr_vs_chi2016': {'x': base_data['snr'], 'y': base_data['Chi2016']},
        'snr_vs_chircr': {'x': base_data['snr'], 'y': base_data['ChiRCR']},
        'chi_vs_chi': {'x': base_data['Chi2016'], 'y': base_data['ChiRCR']},
        'snr_vs_chidiff': {'x': base_data['snr'], 'y': base_data['ChiRCR'] - base_data['Chi2016']}
    }

    for i, (key, p) in enumerate(plot_configs.items()):
        ax = axs.flatten()[i]
        set_plot_labels(ax, p['xlabel'], p['ylabel'], f"{p['ylabel']} vs {p['xlabel']}", p.get('xlim'), p.get('ylim'), p.get('xscale', 'linear'))
        
        # Plot Base Layer
        if base_plot_type == 'scatter':
            ax.scatter(data_map[key]['x'], data_map[key]['y'], s=2, alpha=0.7, c='blue')
        elif base_plot_type == 'hist' and np.sum(base_data.get('weights', 0)) > 0:
            h, xedges, yedges, im_temp = ax.hist2d(data_map[key]['x'], data_map[key]['y'], bins=hist_bins_dict[key], weights=base_data['weights'], norm=colors.LogNorm())
            if im is None: im = im_temp # Capture the first mappable for the colorbar

        # Plot Overlay Layers
        if overlays:
            for overlay in overlays:
                overlay_map = {
                    'snr_vs_chi2016': {'x': overlay['data']['snr'], 'y': overlay['data']['Chi2016']},
                    'snr_vs_chircr': {'x': overlay['data']['snr'], 'y': overlay['data']['ChiRCR']},
                    'chi_vs_chi': {'x': overlay['data']['Chi2016'], 'y': overlay['data']['ChiRCR']},
                    'snr_vs_chidiff': {'x': overlay['data']['snr'], 'y': overlay['data']['ChiRCR'] - overlay['data']['Chi2016']}
                }
                x_data, y_data = overlay_map[key]['x'], overlay_map[key]['y']
                
                if overlay['style'].get('color_by_weight'):
                    weights = overlay['data']['weights']
                    sort_indices = np.argsort(weights)
                    x_data, y_data, weights = x_data[sort_indices], y_data[sort_indices], weights[sort_indices]
                    ax.scatter(x_data, y_data, c=weights, cmap='hot', alpha=overlay['style']['alpha'], s=overlay['style']['s'])
                else:
                    ax.scatter(x_data, y_data, **overlay['style'])

        if cuts_dict:
            draw_cut_visuals(ax, key, cuts_dict)
        if key == 'chi_vs_chi':
            ax.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=1)

    return im

if __name__ == "__main__":
    # --- Configuration and Setup ---
    config = configparser.ConfigParser()
    config.read('HRAStationDataAnalysis/config.ini')
    date = config['PARAMETERS']['date']
    date_processing = config['PARAMETERS']['date_processing']
    station_id = 13

    sim_file = config['SIMULATION']['sim_file']
    direct_weight_name = config['SIMULATION']['direct_weight_name']
    reflected_weight_name = config['SIMULATION']['reflected_weight_name']
    sim_sigma = float(config['SIMULATION']['sigma'])

    station_data_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'
    plot_folder = f'HRAStationDataAnalysis/plots/{date_processing}/'
    os.makedirs(plot_folder, exist_ok=True)
    
    ic.configureOutput(prefix=f'Stn{station_id} | ')
    ic(f"Processing date: {date}")

    # --- Data Loading ---
    ic("Loading station data...")
    times = load_station_data(station_data_folder, date, station_id, 'Times')
    event_ids = load_station_data(station_data_folder, date, station_id, 'EventID')
    
    if times.size == 0: exit("Error: Times or EventID data is missing.")

    snr_array = load_station_data(station_data_folder, date, station_id, 'SNR')
    Chi2016_array = load_station_data(station_data_folder, date, station_id, 'Chi2016')
    ChiRCR_array = load_station_data(station_data_folder, date, station_id, 'ChiRCR')
    
    if Chi2016_array.size == 0 or ChiRCR_array.size == 0:
        exit("Error: Chi2016 or ChiRCR data is missing or empty.")

    # --- Masking Data ---
    ic("Applying time and event masks to data...")
    initial_mask, unique_indices = getTimeEventMasks(times, event_ids)
    
    data_dict = {
        'snr': snr_array[initial_mask][unique_indices],
        'Chi2016': Chi2016_array[initial_mask][unique_indices],
        'ChiRCR': ChiRCR_array[initial_mask][unique_indices]
    }
    ic(f"Data events after masking: {len(data_dict['snr'])}")
    gc.collect()

    # --- Simulation Data Loading ---
    HRAeventList = loadHRAfromH5(sim_file)
    direct_stations = [13, 14, 15, 17, 18, 19, 30]
    reflected_stations = [113, 114, 115, 117, 118, 119, 130]
    
    sim_direct, sim_reflected = get_sim_data(HRAeventList, direct_weight_name, reflected_weight_name, direct_stations, reflected_stations, sigma=sim_sigma)
    ic(f"Processed {len(sim_direct['snr'])} direct and {len(sim_reflected['snr'])} reflected simulation triggers.")

    # --- Define Cuts & Bins ---
    cuts = {
        'Chi2016_min': 0.55, 'Chi2016_max': 0.73,
        'ChiRCR_min': 0.75, 'snr_max': 35,
        'chi_diff_min': 0.08
    }
    cut_string = (f"Cuts: {cuts['Chi2016_min']}<Chi16<{cuts['Chi2016_max']}, ChiRCR>{cuts['ChiRCR_min']}, SNR<{cuts['snr_max']}, ChiRCR-Chi16>{cuts['chi_diff_min']}")
    
    log_bins = np.logspace(np.log10(3), np.log10(100), 31)
    linear_bins = np.linspace(0, 1, 31)
    diff_bins = np.linspace(-0.4, 0.4, 31)
    hist_bins = {
        'snr_vs_chi2016': [log_bins, linear_bins], 'snr_vs_chircr': [log_bins, linear_bins],
        'chi_vs_chi': [linear_bins, linear_bins], 'snr_vs_chidiff': [log_bins, diff_bins]
    }
    
    # --- Plotting Section ---
    data_overlay_config = {
        'data': data_dict,
        'style': {'marker': '.', 's': 5, 'alpha': 0.8, 'c': 'orangered'}
    }

    # Plot 1: Data Only
    ic("Generating 2x2 scatter plot for data...")
    fig1, axs1 = plt.subplots(2, 2, figsize=(12, 13))
    fig1.suptitle(f'Data: Chi Comparison for Station {station_id} on {date}\n{cut_string}', fontsize=14)
    plot_2x2_grid(fig1, axs1, data_dict, 'scatter', cuts)
    stats_str = calculate_cut_stats_table(data_dict, cuts, is_sim=False, title="Data Stats")
    fig1.text(0.5, 0.01, stats_str, ha='center', va='bottom', fontsize=10, fontfamily='monospace')
    fig1.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(f'{plot_folder}Data_SNR_Chi_2x2_WithCuts_Station{station_id}_{date}.png')
    plt.close(fig1)

    # Plot 2 & 3: Sim Histograms with Data Overlay
    sim_datasets_for_hist = {'Reflected': sim_reflected, 'Direct': sim_direct}
    for name, sim_data in sim_datasets_for_hist.items():
        ic(f"Generating 2x2 histogram for Sim {name} with Data overlay...")
        if len(sim_data['snr']) == 0: continue
        
        fig_sim, axs_sim = plt.subplots(2, 2, figsize=(13, 14))
        fig_sim.suptitle(f'Data vs Simulation ({name} Triggers)\n{cut_string}', fontsize=14)
        im = plot_2x2_grid(fig_sim, axs_sim, sim_data, 'hist', cuts, overlays=[data_overlay_config], hist_bins_dict=hist_bins)
        
        stats_str = calculate_cut_stats_table(sim_data, cuts, is_sim=True, title=f"Sim ({name}) Stats")
        fig_sim.text(0.5, 0.01, stats_str, ha='center', va='bottom', fontsize=10, fontfamily='monospace')
        
        if im:
            fig_sim.tight_layout(rect=[0, 0.1, 0.9, 0.95])
            cbar_ax = fig_sim.add_axes([0.91, 0.15, 0.02, 0.7])
            fig_sim.colorbar(im, cax=cbar_ax, label='Weighted Counts (Evts/Yr)')
        else:
            fig_sim.tight_layout(rect=[0, 0.1, 1, 0.95])

        plt.savefig(f'{plot_folder}Data_over_Sim_{name}_SNR_Chi_2x2_{date}.png')
        plt.close(fig_sim)

    # Plot 4: Composite Sim Plot (Direct Hist + Reflected Scatter)
    ic("Generating composite simulation plot...")
    fig_both, axs_both = plt.subplots(2, 2, figsize=(13, 14))
    fig_both.suptitle(f'Simulation: Direct (Hist) vs Reflected (Scatter)\n{cut_string}', fontsize=14)
    reflected_overlay_config = {
        'data': sim_reflected,
        'style': {'s': 8, 'alpha': 0.3, 'color_by_weight': True}
    }
    im_both = plot_2x2_grid(fig_both, axs_both, sim_direct, 'hist', cuts, overlays=[reflected_overlay_config], hist_bins_dict=hist_bins)
    
    direct_stats = calculate_cut_stats_table(sim_direct, cuts, True, "Direct Sim")
    reflected_stats = calculate_cut_stats_table(sim_reflected, cuts, True, "Reflected Sim")
    fig_both.text(0.5, 0.01, f"{direct_stats}\n\n{reflected_stats}", ha='center', va='bottom', fontsize=10, fontfamily='monospace')
    
    if im_both:
        fig_both.tight_layout(rect=[0, 0.15, 0.9, 0.95])
        cbar_ax = fig_both.add_axes([0.91, 0.2, 0.02, 0.7])
        fig_both.colorbar(im_both, cax=cbar_ax, label='Direct Weighted Counts (Evts/Yr)')
    else:
        fig_both.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.savefig(f'{plot_folder}Sim_Composite_SNR_Chi_2x2_{date}.png')
    plt.close(fig_both)

    # Plot 5: Data over Composite Sim Plot
    ic("Generating data over composite simulation plot...")
    fig_all, axs_all = plt.subplots(2, 2, figsize=(13, 14))
    fig_all.suptitle(f'Data vs Composite Simulation\n{cut_string}', fontsize=14)
    im_all = plot_2x2_grid(fig_all, axs_all, sim_direct, 'hist', cuts, overlays=[reflected_overlay_config, data_overlay_config], hist_bins_dict=hist_bins)
    
    fig_all.text(0.5, 0.01, f"{direct_stats}\n\n{reflected_stats}", ha='center', va='bottom', fontsize=10, fontfamily='monospace')
    
    if im_all:
        fig_all.tight_layout(rect=[0, 0.15, 0.9, 0.95])
        cbar_ax = fig_all.add_axes([0.91, 0.2, 0.02, 0.7])
        fig_all.colorbar(im_all, cax=cbar_ax, label='Direct Weighted Counts (Evts/Yr)')
    else:
        fig_all.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.savefig(f'{plot_folder}Data_over_Sim_Composite_SNR_Chi_2x2_{date}.png')
    plt.close(fig_all)

    ic("Processing complete.")
