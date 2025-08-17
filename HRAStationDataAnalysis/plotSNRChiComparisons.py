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
    This is based on the method used in HRASNRPlots.py.
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
    into direct and reflected triggers. It uses separate weight names for each dataset.
    An event's 'direct' weight is split among its direct triggers, and its
    'reflected' weight is split among its reflected triggers.
    """
    direct_data = {'snr': [], 'chi_2016': [], 'chi_rcr': [], 'weights': []}
    reflected_data = {'snr': [], 'chi_2016': [], 'chi_rcr': [], 'weights': []}

    for event in HRAeventList:
        # Process for the 'direct' dataset using the direct weight name
        direct_weight = event.getWeight(direct_weight_name, primary=True, sigma=sigma)
        if not np.isnan(direct_weight) and direct_weight > 0:
            triggered_direct = [st_id for st_id in direct_stations if event.hasTriggered(st_id, sigma)]
            if triggered_direct:
                split_weight = direct_weight / len(triggered_direct)
                for st_id in triggered_direct:
                    snr = event.getSNR(st_id)
                    chi_dict = event.getChi(st_id)
                    if snr is not None and chi_dict:
                        direct_data['snr'].append(snr)
                        direct_data['chi_2016'].append(chi_dict.get('2016', np.nan))
                        direct_data['chi_rcr'].append(chi_dict.get('RCR', np.nan))
                        direct_data['weights'].append(split_weight)

        # Process for the 'reflected' dataset using the reflected weight name
        reflected_weight = event.getWeight(reflected_weight_name, primary=True, sigma=sigma)
        if not np.isnan(reflected_weight) and reflected_weight > 0:
            triggered_reflected = [st_id for st_id in reflected_stations if event.hasTriggered(st_id, sigma)]
            if triggered_reflected:
                split_weight = reflected_weight / len(triggered_reflected)
                for st_id in triggered_reflected:
                    snr = event.getSNR(st_id)
                    chi_dict = event.getChi(st_id)
                    if snr is not None and chi_dict:
                        reflected_data['snr'].append(snr)
                        reflected_data['chi_2016'].append(chi_dict.get('2016', np.nan))
                        reflected_data['chi_rcr'].append(chi_dict.get('RCR', np.nan))
                        reflected_data['weights'].append(split_weight)

    # Convert lists to numpy arrays for easier handling
    for data_dict in [direct_data, reflected_data]:
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key])
            
    return direct_data, reflected_data

def apply_cuts_and_get_stats(data, cuts_dict, is_sim=False):
    """
    Applies a set of cuts to the data and returns the mask and passing statistics.
    """
    snr = data['snr']
    chi_2016 = data['chi_2016']
    chi_rcr = data['chi_rcr']
    
    cut_mask = (
        (chi_2016 > cuts_dict['chi_2016_min']) &
        (chi_2016 < cuts_dict['chi_2016_max']) &
        (chi_rcr > cuts_dict['chi_rcr_min']) &
        (snr < cuts_dict['snr_max']) &
        ((chi_rcr - chi_2016) > cuts_dict['chi_diff_min'])
    )
    
    if is_sim:
        weights = data['weights']
        total_val = np.sum(weights)
        passing_val = np.sum(weights[cut_mask])
    else:
        total_val = len(snr)
        passing_val = np.sum(cut_mask)

    pass_percentage = (passing_val / total_val * 100) if total_val > 0 else 0
    
    return passing_val, total_val, pass_percentage

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
        ax.axhline(y=cuts_dict['chi_2016_min'], color='k', linestyle='--', linewidth=1.5)
        ax.axhline(y=cuts_dict['chi_2016_max'], color='k', linestyle='--', linewidth=1.5)
        ax.fill_between(ax.get_xlim(), cuts_dict['chi_2016_min'], cuts_dict['chi_2016_max'], color='gray', alpha=0.2)
        ax.axvline(x=cuts_dict['snr_max'], color='m', linestyle='--', linewidth=1.5)
        ax.fill_betweenx(ax.get_ylim(), cuts_dict['snr_max'], ax.get_xlim()[1], color='m', alpha=0.1)
    elif plot_key == 'snr_vs_chircr':
        ax.axhline(y=cuts_dict['chi_rcr_min'], color='k', linestyle='--', linewidth=1.5)
        ax.fill_between(ax.get_xlim(), cuts_dict['chi_rcr_min'], 1, color='gray', alpha=0.2)
        ax.axvline(x=cuts_dict['snr_max'], color='m', linestyle='--', linewidth=1.5)
        ax.fill_betweenx(ax.get_ylim(), cuts_dict['snr_max'], ax.get_xlim()[1], color='m', alpha=0.1)
    elif plot_key == 'chi_vs_chi':
        ax.axhline(y=cuts_dict['chi_rcr_min'], color='k', linestyle='--', linewidth=1.5)
        ax.axvline(x=cuts_dict['chi_2016_min'], color='k', linestyle='--', linewidth=1.5)
        ax.axvline(x=cuts_dict['chi_2016_max'], color='k', linestyle='--', linewidth=1.5)
        x_vals = np.array([0, 1.0])
        ax.plot(x_vals, x_vals + cuts_dict['chi_diff_min'], color='purple', linestyle='--', linewidth=1.5)
        x_fill = np.linspace(cuts_dict['chi_2016_min'], cuts_dict['chi_2016_max'], 100)
        y_lower = np.maximum(cuts_dict['chi_rcr_min'], x_fill + cuts_dict['chi_diff_min'])
        ax.fill_between(x_fill, y_lower, 1, color='gray', alpha=0.3, interpolate=True)
    elif plot_key == 'snr_vs_chidiff':
        ax.axhline(y=cuts_dict['chi_diff_min'], color='purple', linestyle='--', linewidth=1.5)
        ax.fill_between(ax.get_xlim(), cuts_dict['chi_diff_min'], ax.get_ylim()[1], color='purple', alpha=0.1)
        ax.axvline(x=cuts_dict['snr_max'], color='m', linestyle='--', linewidth=1.5)
        ax.fill_betweenx(ax.get_ylim(), cuts_dict['snr_max'], ax.get_xlim()[1], color='m', alpha=0.1)

def plot_2x2_grid(fig, axs, data, cuts_dict, plot_type='scatter', overlay_data=None, bins=None):
    """
    Master function to generate a 2x2 grid of plots.
    Can create scatter plots, 2D histograms, or both overlaid.
    """
    snr, chi2016, chircr = data['snr'], data['chi_2016'], data['chi_rcr']
    weights = data.get('weights', None) # Use .get for safety with data dict
    is_sim = weights is not None
    
    plot_configs = {
        'snr_vs_chi2016': {'ax': axs[0, 0], 'x': snr, 'y': chi2016, 'xlabel': 'SNR', 'ylabel': 'Chi2016', 'xlim': (3, 100), 'ylim': (0, 1), 'xscale': 'log'},
        'snr_vs_chircr': {'ax': axs[0, 1], 'x': snr, 'y': chircr, 'xlabel': 'SNR', 'ylabel': 'ChiRCR', 'xlim': (3, 100), 'ylim': (0, 1), 'xscale': 'log'},
        'chi_vs_chi': {'ax': axs[1, 0], 'x': chi2016, 'y': chircr, 'xlabel': 'Chi2016', 'ylabel': 'ChiRCR', 'xlim': (0, 1), 'ylim': (0, 1)},
        'snr_vs_chidiff': {'ax': axs[1, 1], 'x': snr, 'y': chircr - chi2016, 'xlabel': 'SNR', 'ylabel': 'ChiRCR - Chi2016', 'xlim': (3, 100), 'ylim': (-1, 1), 'xscale': 'log'}
    }

    im = None
    for key, p in plot_configs.items():
        ax = p['ax']
        set_plot_labels(ax, p['xlabel'], p['ylabel'], f"{p['ylabel']} vs {p['xlabel']}", p.get('xlim'), p.get('ylim'), p.get('xscale', 'linear'), p.get('yscale', 'linear'))
        
        if plot_type == 'scatter':
            ax.scatter(p['x'], p['y'], s=2, alpha=0.7, c='blue')
        
        elif plot_type == 'hist' and is_sim and np.sum(weights) > 0:
            # FIX: Removed cmin to prevent error when all weights are below the minimum
            h, xedges, yedges, im = ax.hist2d(p['x'], p['y'], bins=bins, weights=weights, norm=colors.LogNorm())

        if overlay_data:
             ax.scatter(overlay_data[key]['x'], overlay_data[key]['y'], s=3, alpha=0.8, c='orangered', marker='.')

        if cuts_dict:
            draw_cut_visuals(ax, key, cuts_dict)
        
        if key == 'chi_vs_chi': # Add identity line
            ax.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=1)

    return im # Return image from hist2d for colorbar

if __name__ == "__main__":
    # --- Configuration and Setup ---
    config = configparser.ConfigParser()
    config.read('HRAStationDataAnalysis/config.ini')
    date = config['PARAMETERS']['date']
    date_processing = config['PARAMETERS']['date_processing']
    station_id = 13 # Hardcoded

    # New config parameters for simulation
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
    
    if times.size == 0 or event_ids.size == 0:
        ic("Error: Times or EventID data is missing. Cannot proceed.")
        exit()

    snr_array = load_station_data(station_data_folder, date, station_id, 'SNR')
    chi_2016_array = load_station_data(station_data_folder, date, station_id, 'Chi2016')
    chi_rcr_array = load_station_data(station_data_folder, date, station_id, 'ChiRCR')
    
    # FIX: Add check for empty Chi arrays
    if chi_2016_array.size == 0 or chi_rcr_array.size == 0:
        ic("Error: Chi2016 or ChiRCR data is missing or empty. Cannot proceed.")
        exit()

    ic(f"Initial raw data events loaded: {len(times)}")

    # --- Masking Data ---
    ic("Applying time and event masks to data...")
    initial_mask, unique_indices = getTimeEventMasks(times, event_ids)
    
    snr_array = snr_array[initial_mask][unique_indices]
    chi_2016_array = chi_2016_array[initial_mask][unique_indices]
    chi_rcr_array = chi_rcr_array[initial_mask][unique_indices]
    
    data_dict = {'snr': snr_array, 'chi_2016': chi_2016_array, 'chi_rcr': chi_rcr_array}
    ic(f"Data events after masking: {len(data_dict['snr'])}")
    gc.collect()

    # --- Simulation Data Loading and Processing ---
    HRAeventList = loadHRAfromH5(sim_file)
    direct_stations = [13, 14, 15, 17, 18, 19, 30]
    reflected_stations = [113, 114, 115, 117, 118, 119, 130]
    
    sim_direct, sim_reflected = get_sim_data(
        HRAeventList, direct_weight_name, reflected_weight_name, 
        direct_stations, reflected_stations, sigma=sim_sigma
    )

    # Combine direct and reflected for a total simulation dataset
    sim_both = {key: np.concatenate((sim_direct[key], sim_reflected[key])) for key in sim_direct}
    ic(f"Processed {len(sim_direct['snr'])} direct and {len(sim_reflected['snr'])} reflected simulation triggers.")

    # --- Define Cuts ---
    cuts = {
        'chi_2016_min': 0.55, 'chi_2016_max': 0.73,
        'chi_rcr_min': 0.75, 'snr_max': 35,
        'chi_diff_min': 0.08 # ChiRCR - Chi2016
    }
    cut_string = (f"Cuts: {cuts['chi_2016_min']}<Chi16<{cuts['chi_2016_max']}, ChiRCR>{cuts['chi_rcr_min']}, SNR<{cuts['snr_max']}, ChiRCR-Chi16>{cuts['chi_diff_min']}")
    
    # --- Plotting Section ---
    
    # Plot 1: Original 2x2 Plot for Data
    ic("Generating 2x2 scatter plot for data with cuts...")
    pass_count, total_count, pass_pct = apply_cuts_and_get_stats(data_dict, cuts, is_sim=False)
    fig1, axs1 = plt.subplots(2, 2, figsize=(14, 14))
    title1 = (f'Data: Chi Comparison for Station {station_id} on {date}\n'
              f'{cut_string}\n'
              f'Events Passing: {pass_count}/{total_count} ({pass_pct:.2f}%)')
    fig1.suptitle(title1, fontsize=14)
    plot_2x2_grid(fig1, axs1, data_dict, cuts, plot_type='scatter')
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    savename1 = f'{plot_folder}Data_SNR_Chi_2x2_WithCuts_Station{station_id}_{date}.png'
    plt.savefig(savename1)
    ic(f'Saved data 2x2 plot to {savename1}')
    plt.close(fig1)

    # --- New Simulation Plots ---
    log_bins = np.logspace(np.log10(3), np.log10(100), 31)
    linear_bins = np.linspace(0, 1, 31)
    diff_bins = np.linspace(-1, 1, 31)
    hist_bins = {
        'snr_vs_chi2016': [log_bins, linear_bins], 'snr_vs_chircr': [log_bins, linear_bins],
        'chi_vs_chi': [linear_bins, linear_bins], 'snr_vs_chidiff': [log_bins, diff_bins]
    }

    sim_datasets = {'Reflected': sim_reflected, 'Direct': sim_direct, 'Both': sim_both}

    for name, sim_data in sim_datasets.items():
        ic(f"Generating 2x2 histogram for {name} simulation data...")
        if len(sim_data['snr']) == 0:
            ic(f"Skipping {name} plot, no data.")
            continue
        
        pass_w, total_w, pass_pct_w = apply_cuts_and_get_stats(sim_data, cuts, is_sim=True)
        
        fig_sim, axs_sim = plt.subplots(2, 2, figsize=(15, 14))
        title_sim = (f'Simulation ({name} Triggers): Chi Comparison\n'
                     f'{cut_string}\n'
                     f'Weight Passing: {pass_pct_w:.2f}%')
        fig_sim.suptitle(title_sim, fontsize=14)
        
        im = plot_2x2_grid(fig_sim, axs_sim, sim_data, cuts, plot_type='hist', bins=hist_bins['snr_vs_chi2016']) # Use one bin set for all for simplicity
        
        if im: # Only add a colorbar if hist2d returned an image
            fig_sim.tight_layout(rect=[0, 0, 0.9, 0.95])
            cbar_ax = fig_sim.add_axes([0.92, 0.15, 0.02, 0.7])
            #fig_sim.colorbar(im, cax=cbar_ax, label='Weighted Counts (Evts/Yr)')

        # Add legend with passing percentages
        legend_text = f'Passing Weight: {pass_pct_w:.2f}%'
        axs_sim[1,0].text(0.05, 0.95, legend_text, transform=axs_sim[1,0].transAxes, fontsize=12,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        savename_sim = f'{plot_folder}Sim_{name}_SNR_Chi_2x2_Hist_{date}.png'
        plt.savefig(savename_sim)
        ic(f'Saved {name} sim 2x2 plot to {savename_sim}')
        plt.close(fig_sim)

    # --- New Data over Sim Plot ---
    ic("Generating data over simulation histogram plot...")
    fig_overlay, axs_overlay = plt.subplots(2, 2, figsize=(15, 14))
    title_overlay = (f'Data vs Simulation (Both Triggers) for Station {station_id} on {date}\n'
                     f'{cut_string}')
    fig_overlay.suptitle(title_overlay, fontsize=14)
    
    overlay_plot_data = {
        'snr_vs_chi2016': {'x': data_dict['snr'], 'y': data_dict['chi_2016']},
        'snr_vs_chircr': {'x': data_dict['snr'], 'y': data_dict['chi_rcr']},
        'chi_vs_chi': {'x': data_dict['chi_2016'], 'y': data_dict['chi_rcr']},
        'snr_vs_chidiff': {'x': data_dict['snr'], 'y': data_dict['chi_rcr'] - data_dict['chi_2016']}
    }

    im_overlay = plot_2x2_grid(fig_overlay, axs_overlay, sim_both, cuts, plot_type='hist', 
                               overlay_data=overlay_plot_data, bins=hist_bins['snr_vs_chi2016'])
    
    if im_overlay: # Only add a colorbar if hist2d returned an image
        fig_overlay.tight_layout(rect=[0, 0, 0.9, 0.95])
        cbar_ax_overlay = fig_overlay.add_axes([0.92, 0.15, 0.02, 0.7])
        #fig_overlay.colorbar(im_overlay, cax=cbar_ax_overlay, label='Sim Weighted Counts (Evts/Yr)')

    # Create custom legend for overlay plot
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Data',
                          markerfacecolor='orangered', markersize=10),
                       plt.Rectangle((0,0),1,1,fc="lightblue", label='Sim Hist')]
    axs_overlay[1,0].legend(handles=legend_elements, loc='upper left')

    savename_overlay = f'{plot_folder}Data_over_Sim_SNR_Chi_2x2_{date}.png'
    plt.savefig(savename_overlay)
    ic(f'Saved data over sim plot to {savename_overlay}')
    plt.close(fig_overlay)

    ic("Processing complete.")
