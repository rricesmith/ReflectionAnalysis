import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import configparser
from icecream import ic

# Import your existing modules
from HRASimulation.HRAEventObject import HRAevent
from HRASimulation.HRANurToNpy import loadHRAfromH5
import HRASimulation.HRAAnalysis as HRAAnalysis

def get_snr_and_weights(HRAeventList, weight_name, station_ids, sigma=4.5):
    """
    Extracts SNR values and event weights for events that triggered on specified stations.

    Args:
        HRAeventList (list): The list of HRA event objects.
        weight_name (str): The name of the weight to use for each event.
        station_ids (list): A list of station IDs to get SNRs for.
        sigma (float): The significance threshold for checking triggers.

    Returns:
        tuple: A tuple containing two lists (snrs, weights).
    """
    snrs = []
    weights = []

    for event in HRAeventList:
        # Check if the event has a valid weight for this coincidence level
        event_weight = event.getWeight(weight_name, sigma=sigma)
        if event_weight > 0:
            # Check which of the specified stations have triggered
            for station_id in station_ids:
                if event.hasTriggered(station_id, sigma):
                    snr = event.getSNR(station_id)
                    if snr is not None:
                        snrs.append(snr)
                        weights.append(event_weight)
    
    return snrs, weights

def plot_snr_distribution(snrs_lists, weights_lists, titles, main_title, savename):
    """
    Plots weighted 1D histograms of SNR distributions for multiple station lists.

    Args:
        snrs_lists (list): A list of lists, where each inner list contains SNR values.
        weights_lists (list): A list of lists, where each inner list contains weights.
        titles (list): A list of titles for each subplot.
        main_title (str): The main title for the entire figure.
        savename (str): The path to save the plot image.
    """
    num_subplots = len(snrs_lists)
    fig, axs = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5), sharey=True)
    if num_subplots == 1:
        axs = [axs] # Make it iterable for a single subplot

    for i in range(num_subplots):
        axs[i].hist(snrs_lists[i], bins=50, weights=weights_lists[i], histtype='step', linewidth=2)
        axs[i].set_xlabel('SNR')
        axs[i].set_title(titles[i])
        axs[i].set_yscale('log')
        if i == 0:
            axs[i].set_ylabel('Weighted Counts (Evts/Yr)')
    
    plt.suptitle(main_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ic(f"Saving 1D plot: {savename}")
    plt.savefig(savename)
    plt.close(fig)

def plot_2d_snr_histogram(snrs1, weights1, snrs2, weights2, title1, title2, main_title, savename):
    """
    Plots a weighted 2D histogram of SNRs from two different station lists.

    Args:
        snrs1 (list): SNR values for the x-axis.
        weights1 (list): Weights corresponding to snrs1.
        snrs2 (list): SNR values for the y-axis.
        weights2 (list): Weights corresponding to snrs2.
        title1 (str): Label for the x-axis.
        title2 (str): Label for the y-axis.
        main_title (str): The main title for the plot.
        savename (str): The path to save the plot image.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Combine the data for the 2D histogram
    all_snrs_x = snrs1 + snrs2
    all_snrs_y = snrs1 + snrs2
    all_weights = weights1 + weights2

    # Define bins for the histogram
    bins = [np.linspace(0, 50, 51), np.linspace(0, 50, 51)]

    # Create the 2D histogram
    hist, xedges, yedges = np.histogram2d(snrs1, snrs2, bins=bins, weights=weights1)

    # Use LogNorm for better visualization of a wide range of values
    im = ax.imshow(hist.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   norm=colors.LogNorm())

    ax.set_xlabel(f'SNR ({title1})')
    ax.set_ylabel(f'SNR ({title2})')
    plt.colorbar(im, ax=ax, label='Weighted Counts (Evts/Yr)')
    plt.suptitle(main_title)
    ic(f"Saving 2D plot: {savename}")
    plt.savefig(savename)
    plt.close(fig)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('HRASimulation/config.ini')
    numpy_folder = config['FOLDERS']['numpy_folder']
    save_folder = config['FOLDERS']['save_folder']
    plot_sigma = float(config['PLOTPARAMETERS']['trigger_sigma'])

    # Create a dedicated folder for these new SNR plots
    snr_plot_folder = os.path.join(save_folder, 'SNR_Plots')
    os.makedirs(snr_plot_folder, exist_ok=True)

    ic("Loading HRA event list...")
    HRAeventList = loadHRAfromH5(f'{numpy_folder}HRAeventList.h5')

    # Define the station lists for direct and reflected triggers (excluding special stations)
    direct_stations = [13, 14, 15, 17, 18, 19, 30]
    reflected_stations = [113, 114, 115, 117, 118, 119, 130]
    
    # Loop through each coincidence level you are interested in
    for i in range(2, 8):
        # This weight name corresponds to the coincidence with reflected signal required
        weight_name = f'{i}_coincidence_refl'

        # Check if the first event has this weight to avoid errors
        if not HRAeventList[0].hasWeight(weight_name, sigma=plot_sigma):
            ic(f"Weight '{weight_name}' not found for sigma={plot_sigma}. Skipping coincidence level {i}.")
            continue
        
        ic(f"Processing coincidence level {i}...")

        # --- 1. Generate 1D SNR Distribution Plot ---
        
        # Get SNRs and weights for both direct and reflected triggers
        direct_snrs, direct_weights = get_snr_and_weights(HRAeventList, weight_name, direct_stations, sigma=plot_sigma)
        reflected_snrs, reflected_weights = get_snr_and_weights(HRAeventList, weight_name, reflected_stations, sigma=plot_sigma)

        # Define parameters for the 1D plot
        snrs_to_plot = [direct_snrs, reflected_snrs]
        weights_to_plot = [direct_weights, reflected_weights]
        subplot_titles = ['Direct Triggers', 'Reflected Triggers']
        main_plot_title = f'SNR Distribution for {i}-Fold Coincidence (Reflected Required)'
        save_path_1d = os.path.join(snr_plot_folder, f'snr_dist_{i}coinc_refl_1d.png')
        
        # Create and save the plot
        plot_snr_distribution(snrs_to_plot, weights_to_plot, subplot_titles, main_plot_title, save_path_1d)

        # --- 2. Generate 2D SNR Histogram ---

        main_plot_title_2d = f'2D SNR Histogram for {i}-Fold Coincidence (Reflected Required)'
        save_path_2d = os.path.join(snr_plot_folder, f'snr_hist_{i}coinc_refl_2d.png')
        
        # Create and save the 2D plot
        plot_2d_snr_histogram(direct_snrs, direct_weights, reflected_snrs, reflected_weights, 
                                'Direct', 'Reflected', main_plot_title_2d, save_path_2d)

    ic("\nAll SNR plots have been generated!")