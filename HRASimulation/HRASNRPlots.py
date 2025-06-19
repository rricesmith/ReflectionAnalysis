import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import configparser
from icecream import ic
import h5py
import pickle

# Import your existing modules
from HRASimulation.HRAEventObject import HRAevent
from HRASimulation.HRANurToNpy import loadHRAfromH5
import HRASimulation.HRAAnalysis as HRAAnalysis
from NuRadioReco.utilities import units


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

    # Define bins for the histogram
    bins = np.logspace(np.log10(3), np.log10(100), 21)

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
    diameter = config['SIMPARAMETERS']['diameter']
    max_distance = float(diameter) / 2 * units.km
    plot_sigma = float(config['PLOTPARAMETERS']['trigger_sigma'])

    # Create a dedicated folder for these new SNR plots
    snr_plot_folder = os.path.join(save_folder, 'SNR_Plots')
    os.makedirs(snr_plot_folder, exist_ok=True)

    ic("Loading HRA event list...")
    HRAeventList_path = f'{numpy_folder}HRAeventList.h5'
    HRAeventList = loadHRAfromH5(HRAeventList_path)

    # Flag to check if we need to resave the event list
    weights_were_added = False

    # Define the station lists for direct and reflected triggers (excluding special stations)
    direct_stations = [13, 14, 15, 17, 18, 19, 30]
    reflected_stations = [113, 114, 115, 117, 118, 119, 130]
    
    # Loop through each coincidence level you are interested in
    for i in range(2, 8):
        weight_name = f'{i}_coincidence_reflReq'

        # Check if weights exist, if not, calculate them
        if not HRAeventList[0].hasWeight(weight_name, sigma=plot_sigma):
            ic(f"Weight '{weight_name}' not found. Calculating now...")
            weights_were_added = True

            # Define stations to exclude and stations to force for reflection requirement
            bad_stations = [32, 52, 132, 152]
            force_stations = reflected_stations

            # Get the trigger rate for this specific coincidence requirement
            trigger_rate_coincidence = HRAAnalysis.getCoincidencesTriggerRates(
                HRAeventList, bad_stations, force_stations=force_stations, sigma=plot_sigma
            )

            if i in trigger_rate_coincidence and np.any(trigger_rate_coincidence[i] > 0):
                # Set a new trigger name for events that meet the criteria
                HRAAnalysis.setNewTrigger(HRAeventList, weight_name, bad_stations=bad_stations, sigma=plot_sigma)
                
                # Calculate and set the weight for each event based on the trigger rate
                HRAAnalysis.setHRAeventListRateWeight(
                    HRAeventList, trigger_rate_coincidence[i], weight_name=weight_name, 
                    max_distance=max_distance, sigma=plot_sigma
                )
                ic(f"Successfully calculated and added weights for '{weight_name}'.")
            else:
                ic(f"No events found for {i}-fold coincidence with reflection required. Skipping.")
                continue

        ic(f"Processing coincidence level {i}...")

        # --- 1. Generate 1D SNR Distribution Plot ---
        direct_snrs, direct_weights = get_snr_and_weights(HRAeventList, weight_name, direct_stations, sigma=plot_sigma)
        reflected_snrs, reflected_weights = get_snr_and_weights(HRAeventList, weight_name, reflected_stations, sigma=plot_sigma)

        snrs_to_plot = [direct_snrs, reflected_snrs]
        weights_to_plot = [direct_weights, reflected_weights]
        subplot_titles = ['Direct Triggers', 'Reflected Triggers']
        main_plot_title = f'SNR Distribution for {i}-Fold Coincidence (Reflected Required)'
        save_path_1d = os.path.join(snr_plot_folder, f'snr_dist_{i}coinc_reflReq_1d.png')
        
        plot_snr_distribution(snrs_to_plot, weights_to_plot, subplot_titles, main_plot_title, save_path_1d)

        # --- 2. Generate 2D SNR Histogram ---
        main_plot_title_2d = f'2D SNR Histogram for {i}-Fold Coincidence (Reflected Required)'
        save_path_2d = os.path.join(snr_plot_folder, f'snr_hist_{i}coinc_reflReq_2d.png')
        
        plot_2d_snr_histogram(direct_snrs, direct_weights, reflected_snrs, reflected_weights, 
                                'Direct', 'Reflected', main_plot_title_2d, save_path_2d)

    # If new weights were added, resave the HRAeventList
    if weights_were_added:
        ic("New weights were added, resaving HRAeventList to H5 file...")
        with h5py.File(HRAeventList_path, 'w') as hf:
            for i, obj in enumerate(HRAeventList):
                if not isinstance(obj, (np.ndarray, str, int, float)):
                    obj_bytes = pickle.dumps(obj)
                    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                    dset = hf.create_dataset(f'object_{i}', (1,), dtype=dt)
                    dset[0] = np.frombuffer(obj_bytes, dtype='uint8')
                else:
                    hf.create_dataset(f'object_{i}', data=obj)
        ic("HRAeventList successfully updated and saved.")

    ic("\nAll SNR plots have been generated!")