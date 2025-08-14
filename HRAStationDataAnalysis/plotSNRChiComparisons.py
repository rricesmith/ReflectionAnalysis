import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import gc
from icecream import ic
import configparser
from C_utils import getTimeEventMasks

def load_station_data(folder, date, station_id, data_name):
    """
    Loads and concatenates data files for a specific station and data type.
    This uses the new universal format.
    """
    file_pattern = os.path.join(folder, f'{date}_Station{station_id}_{data_name}*')
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        ic(f"Warning: No files found for {data_name} with pattern: {file_pattern}")
        return np.array([])
    
    data_arrays = [np.load(f, allow_pickle=True) for f in file_list]
    # Filter out any empty arrays that might result from empty files
    data_arrays = [arr for arr in data_arrays if arr.size > 0]
    if not data_arrays:
        return np.array([])
        
    return np.concatenate(data_arrays, axis=0)

def set_plot_labels(ax, xlabel, ylabel, title, xlim, ylim, xscale='linear', yscale='linear'):
    """Set common labels and properties for a subplot."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(visible=True, which='both', axis='both')

def plot_snr_vs_chi(ax, snr, chi, name, color, markersize=2):
    """Helper function to create a SNR vs. Chi scatter plot."""
    ax.scatter(snr, chi, s=markersize, color=color, alpha=0.7)
    set_plot_labels(ax, 'SNR', f'{name} Chi', f'SNR vs {name} Chi', xlim=(3, 100), ylim=(0, 1), xscale='log')

def plot_chi_vs_chi(ax, chi1, chi2, name1, name2, color1, color2, markersize=2):
    """Helper function to create a Chi vs. Chi scatter plot."""
    ax.scatter(chi1, chi2, s=markersize, alpha=0.7)
    ax.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=1) # Identity line
    set_plot_labels(ax, f'{name1} Chi', f'{name2} Chi', f'{name1} Chi vs {name2} Chi', xlim=(0, 1), ylim=(0, 1))


def plot_snr_vs_chi_diff(ax, snr, chi1, chi2, name1, name2, color1, color2, markersize=2):
    """Helper function to create a SNR vs. Chi difference plot."""
    ax.scatter(snr, chi1 - chi2, s=markersize, alpha=0.7)
    set_plot_labels(ax, 'SNR', f'({name1} - {name2}) Chi', f'SNR vs ({name1} - {name2}) Chi', xlim=(3, 100), ylim=(-1, 1), xscale='log')
    ax.text(40, 0.8, f'More {name1}', fontsize=10, color=color1, ha='center', va='center')
    ax.text(40, -0.8, f'More {name2}', fontsize=10, color=color2, ha='center', va='center')


if __name__ == "__main__":
    # --- Configuration and Setup ---
    config = configparser.ConfigParser()
    # Assuming the script is run from the root of the project
    config.read('HRAStationDataAnalysis/config.ini')
    date = config['PARAMETERS']['date']
    date_processing = config['PARAMETERS']['date_processing']
    station_id = 13 # Hardcoded as in the original script

    station_data_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'
    plot_folder = f'HRAStationDataAnalysis/plots/{date_processing}/'
    os.makedirs(plot_folder, exist_ok=True)
    
    ic.configureOutput(prefix=f'Stn{station_id} | ')
    ic(f"Processing date: {date}")

    # --- Data Loading ---
    ic("Loading data...")
    times = load_station_data(station_data_folder, date, station_id, 'Times')
    event_ids = load_station_data(station_data_folder, date, station_id, 'EventID')
    
    # Check if essential time/event data is present
    if times.size == 0 or event_ids.size == 0:
        ic("Error: Times or EventID data is missing or empty. Cannot proceed.")
        exit()

    snr_array = load_station_data(station_data_folder, date, station_id, 'SNR')
    chi_2016_array = load_station_data(station_data_folder, date, station_id, 'Chi2016')
    chi_rcr_array = load_station_data(station_data_folder, date, station_id, 'ChiRCR')
    chi_rcr_bad_array = load_station_data(station_data_folder, date, station_id, 'ChiBad')
    
    ic(f"Initial raw events loaded: {len(times)}")

    # --- Masking Data ---
    ic("Applying time and event masks...")
    initial_mask, unique_indices = getTimeEventMasks(times, event_ids)
    
    # Apply the initial mask first to align arrays for unique indexing
    times = times[initial_mask]
    event_ids = event_ids[initial_mask]
    snr_array = snr_array[initial_mask]
    chi_2016_array = chi_2016_array[initial_mask]
    chi_rcr_array = chi_rcr_array[initial_mask]
    chi_rcr_bad_array = chi_rcr_bad_array[initial_mask]

    # Apply the unique mask to the already-filtered data
    times = times[unique_indices]
    event_ids = event_ids[unique_indices]
    snr_array = snr_array[unique_indices]
    chi_2016_array = chi_2016_array[unique_indices]
    chi_rcr_array = chi_rcr_array[unique_indices]
    chi_rcr_bad_array = chi_rcr_bad_array[unique_indices]

    ic(f"Events after masking: {len(times)}")
    gc.collect()

    # --- Plotting ---
    colors = {'2016': 'blue', 'RCR': 'green', 'Bad RCR': 'red'}
    markersize = 2

    # Plot 1: Original 3x3 Comparison Matrix
    ic("Generating 3x3 comparison plot...")
    fig1, axs1 = plt.subplots(3, 3, figsize=(18, 18))
    fig1.suptitle(f'Full Chi Comparison Matrix for Station {station_id} on {date}', fontsize=20)
    fig1.subplots_adjust(hspace=0.4, wspace=0.4)

    # Diagonal: SNR vs Chi
    plot_snr_vs_chi(axs1[0, 0], snr_array, chi_2016_array, '2016', colors['2016'], markersize)
    plot_snr_vs_chi(axs1[1, 1], snr_array, chi_rcr_array, 'RCR', colors['RCR'], markersize)
    plot_snr_vs_chi(axs1[2, 2], snr_array, chi_rcr_bad_array, 'Bad RCR', colors['Bad RCR'], markersize)

    # Lower Triangle: Chi vs Chi
    plot_chi_vs_chi(axs1[1, 0], chi_2016_array, chi_rcr_array, '2016', 'RCR', colors['2016'], colors['RCR'], markersize)
    plot_chi_vs_chi(axs1[2, 0], chi_2016_array, chi_rcr_bad_array, '2016', 'Bad RCR', colors['2016'], colors['Bad RCR'], markersize)
    plot_chi_vs_chi(axs1[2, 1], chi_rcr_array, chi_rcr_bad_array, 'RCR', 'Bad RCR', colors['RCR'], colors['Bad RCR'], markersize)

    # Upper Triangle: SNR vs Chi Difference
    plot_snr_vs_chi_diff(axs1[0, 1], snr_array, chi_rcr_array, chi_2016_array, 'RCR', '2016', colors['RCR'], colors['2016'], markersize)
    plot_snr_vs_chi_diff(axs1[0, 2], snr_array, chi_rcr_bad_array, chi_2016_array, 'Bad RCR', '2016', colors['Bad RCR'], colors['2016'], markersize)
    plot_snr_vs_chi_diff(axs1[1, 2], snr_array, chi_rcr_array, chi_rcr_bad_array, 'RCR', 'Bad RCR', colors['RCR'], colors['Bad RCR'], markersize)

    savename1 = f'{plot_folder}SNR_Chi_Matrix_Station{station_id}_{date}.png'
    plt.savefig(savename1)
    ic(f'Saved 3x3 plot to {savename1}')
    plt.close(fig1)

    # --- Cuts and 2x2 Plot ---
    ic("Applying cuts and generating 2x2 plot...")
    
    # Define cuts
    chi_2016_min_cut = 0.5
    chi_2016_max_cut = 0.7
    chi_rcr_min_cut = 0.7
    snr_max_cut = 35
    chi_diff_min_cut = 0.09 # This is ChiRCR - Chi2016

    # Create a boolean mask for events passing all cuts
    cut_mask = (
        (chi_2016_array > chi_2016_min_cut) &
        (chi_2016_array < chi_2016_max_cut) &
        (chi_rcr_array > chi_rcr_min_cut) &
        (snr_array < snr_max_cut) &
        ((chi_rcr_array - chi_2016_array) > chi_diff_min_cut)
    )
    num_passing_events = np.sum(cut_mask)
    total_events = len(chi_2016_array)
    
    cut_string = (f"Cuts: {chi_2016_min_cut}<Chi16<{chi_2016_max_cut}, ChiRCR>{chi_rcr_min_cut}, "
                  f"SNR<{snr_max_cut}, ChiRCR-Chi16>{chi_diff_min_cut}")
    ic(f"Events passing cuts: {num_passing_events}/{total_events}")


    # Plot 2: New 2x2 Plot with Cuts Visualized
    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 14))
    title = (f'Chi2016 vs ChiRCR for Station {station_id} on {date}\n'
             f'{cut_string}\n'
             f'Events Passing Cuts: {num_passing_events} ({num_passing_events/total_events:.2%})')
    fig2.suptitle(title, fontsize=14)
    fig2.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)

    # Top-left: SNR vs Chi2016
    plot_snr_vs_chi(axs2[0, 0], snr_array, chi_2016_array, '2016', colors['2016'], markersize)
    axs2[0, 0].axhline(y=chi_2016_min_cut, color='k', linestyle='--', linewidth=1.5)
    axs2[0, 0].axhline(y=chi_2016_max_cut, color='k', linestyle='--', linewidth=1.5)
    axs2[0, 0].fill_between(axs2[0, 0].get_xlim(), chi_2016_min_cut, chi_2016_max_cut, color='gray', alpha=0.2, label='Chi2016 Cut')
    axs2[0, 0].axvline(x=snr_max_cut, color='m', linestyle='--', linewidth=1.5, label=f'SNR Cut (<{snr_max_cut})')
    axs2[0, 0].fill_betweenx(axs2[0, 0].get_ylim(), snr_max_cut, axs2[0, 0].get_xlim()[1], color='m', alpha=0.1)


    # Top-right: SNR vs ChiRCR
    plot_snr_vs_chi(axs2[0, 1], snr_array, chi_rcr_array, 'RCR', colors['RCR'], markersize)
    axs2[0, 1].axhline(y=chi_rcr_min_cut, color='k', linestyle='--', linewidth=1.5)
    axs2[0, 1].fill_between(axs2[0, 1].get_xlim(), chi_rcr_min_cut, 1, color='gray', alpha=0.2, label='ChiRCR Cut')
    axs2[0, 1].axvline(x=snr_max_cut, color='m', linestyle='--', linewidth=1.5, label=f'SNR Cut (<{snr_max_cut})')
    axs2[0, 1].fill_betweenx(axs2[0, 1].get_ylim(), snr_max_cut, axs2[0, 1].get_xlim()[1], color='m', alpha=0.1)

    # Bottom-left: Chi2016 vs ChiRCR
    plot_chi_vs_chi(axs2[1, 0], chi_2016_array, chi_rcr_array, '2016', 'RCR', colors['2016'], colors['RCR'], markersize)
    axs2[1, 0].axhline(y=chi_rcr_min_cut, color='k', linestyle='--', linewidth=1.5)
    axs2[1, 0].axvline(x=chi_2016_min_cut, color='k', linestyle='--', linewidth=1.5)
    axs2[1, 0].axvline(x=chi_2016_max_cut, color='k', linestyle='--', linewidth=1.5)
    # Add Chi Diff cut line (y = x + 0.9)
    x_vals = np.array([0, 0.1])
    axs2[1, 0].plot(x_vals, x_vals + chi_diff_min_cut, color='purple', linestyle='--', linewidth=1.5, label=f'ChiDiff Cut (>{chi_diff_min_cut})')
    # Shade the intersection of all cuts
    x_fill = np.linspace(chi_2016_min_cut, chi_2016_max_cut, 100)
    y_lower_bound = np.maximum(chi_rcr_min_cut, x_fill + chi_diff_min_cut)
    axs2[1, 0].fill_between(x_fill, y_lower_bound, 1, color='gray', alpha=0.3, interpolate=True)


    # Bottom-right: SNR vs (RCR - 2016) Chi Difference
    plot_snr_vs_chi_diff(axs2[1, 1], snr_array, chi_rcr_array, chi_2016_array, 'RCR', '2016', colors['RCR'], colors['2016'], markersize)
    axs2[1, 1].axhline(y=chi_diff_min_cut, color='purple', linestyle='--', linewidth=1.5, label=f'ChiDiff Cut (>{chi_diff_min_cut})')
    axs2[1, 1].fill_between(axs2[1, 1].get_xlim(), chi_diff_min_cut, axs2[1, 1].get_ylim()[1], color='purple', alpha=0.1)
    axs2[1, 1].axvline(x=snr_max_cut, color='m', linestyle='--', linewidth=1.5, label=f'SNR Cut (<{snr_max_cut})')
    axs2[1, 1].fill_betweenx(axs2[1, 1].get_ylim(), snr_max_cut, axs2[1, 1].get_xlim()[1], color='m', alpha=0.1)

    savename2 = f'{plot_folder}SNR_Chi_2x2_WithCuts_Station{station_id}_{date}.png'
    plt.savefig(savename2)
    ic(f'Saved 2x2 plot to {savename2}')
    plt.close(fig2)

    ic("Processing complete.")
