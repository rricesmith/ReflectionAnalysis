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


def get_snr_and_weights(HRAeventList, weight_name, direct_stations, reflected_stations, sigma=4.5):
    """
    Extracts SNR values and weights for events triggering direct or reflected stations.

    This function ensures all returned arrays have the same length by padding with 0
    for an SNR type if it wasn't triggered in an event. An event is included if it
    has a valid weight and triggers at least one direct or reflected station. The
    maximum SNR for each type is taken for a given event.

    Args:
        HRAeventList (list): The list of HRA event objects.
        weight_name (str): The name of the weight to use for each event.
        direct_stations (list): A list of direct station IDs.
        reflected_stations (list): A list of reflected station IDs.
        sigma (float): The significance threshold for checking triggers.

    Returns:
        tuple: A tuple of numpy arrays (direct_snrs, reflected_snrs, weights).
    """
    direct_snrs_list = []
    reflected_snrs_list = []
    weights_list = []

    for event in HRAeventList:
        event_weight = event.getWeight(weight_name, sigma=sigma)
        if event_weight > 0:
            current_direct_snr = 0
            current_reflected_snr = 0

            # Find max SNR for direct stations
            for station_id in direct_stations:
                if event.hasTriggered(station_id, sigma):
                    snr = event.getSNR(station_id)
                    if snr is not None and snr > current_direct_snr:
                        current_direct_snr = snr
            
            # Find max SNR for reflected stations
            for station_id in reflected_stations:
                if event.hasTriggered(station_id, sigma):
                    snr = event.getSNR(station_id)
                    if snr is not None and snr > current_reflected_snr:
                        current_reflected_snr = snr

            # Add to list if at least one of them triggered
            if current_direct_snr > 0 or current_reflected_snr > 0:
                direct_snrs_list.append(current_direct_snr)
                reflected_snrs_list.append(current_reflected_snr)
                weights_list.append(event_weight)

    return np.array(direct_snrs_list), np.array(reflected_snrs_list), np.array(weights_list)

def plot_snr_distribution(direct_snrs, reflected_snrs, weights, main_title, savename, bins):
    """
    Plots weighted 1D histograms of SNR distributions using masked arrays.

    Args:
        direct_snrs (np.ndarray): Array of SNR values for direct triggers.
        reflected_snrs (np.ndarray): Array of SNR values for reflected triggers.
        weights (np.ndarray): Array of weights for each event.
        main_title (str): The main title for the entire figure.
        savename (str): The path to save the plot image.
        bins (np.ndarray): The bins to use for the histogram.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Mask for direct triggers (where direct_snr > 0)
    direct_mask = direct_snrs > 0
    axs[0].hist(direct_snrs[direct_mask], bins=bins, weights=weights[direct_mask], histtype='step', linewidth=2)
    axs[0].set_xlabel('SNR')
    axs[0].set_ylabel('Weighted Counts (Evts/Yr)')
    axs[0].set_title('Direct Triggers')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    # Mask for reflected triggers (where reflected_snr > 0)
    reflected_mask = reflected_snrs > 0
    axs[1].hist(reflected_snrs[reflected_mask], bins=bins, weights=weights[reflected_mask], histtype='step', linewidth=2, color='C1')
    axs[1].set_xlabel('SNR')
    axs[1].set_title('Reflected Triggers')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    
    plt.suptitle(main_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ic(f"Saving 1D plot: {savename}")
    plt.savefig(savename)
    plt.close(fig)

def plot_2d_snr_histogram(direct_snrs, reflected_snrs, weights, main_title, savename, bins):
    """
    Plots a weighted 2D histogram using plt.hist2d and masked arrays.

    Args:
        direct_snrs (np.ndarray): SNR values for the x-axis.
        reflected_snrs (np.ndarray): SNR values for the y-axis.
        weights (np.ndarray): Weights corresponding to the events.
        main_title (str): The main title for the plot.
        savename (str): The path to save the plot image.
        bins (np.ndarray): The bins to use for the histogram.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Combined mask to include only events with BOTH direct and reflected triggers
    combined_mask = (direct_snrs > 0) & (reflected_snrs > 0)

    # Use plt.hist2d which handles weights directly
    h, xedges, yedges, im = ax.hist2d(
        direct_snrs[combined_mask],
        reflected_snrs[combined_mask],
        bins=bins,
        weights=weights[combined_mask],
        norm=colors.LogNorm()
    )

    ax.set_xlabel('SNR (Direct)')
    ax.set_ylabel('SNR (Reflected)')
    ax.set_xscale('log')
    ax.set_yscale('log')
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

    snr_plot_folder = os.path.join(save_folder, 'SNR_Plots')
    os.makedirs(snr_plot_folder, exist_ok=True)

    ic("Loading HRA event list...")
    HRAeventList_path = f'{numpy_folder}HRAeventList.h5'
    HRAeventList = loadHRAfromH5(HRAeventList_path)

    weights_were_added = False
    direct_stations = [13, 14, 15, 17, 18, 19, 30]
    reflected_stations = [113, 114, 115, 117, 118, 119, 130]
    
    # Define the logarithmic bins to be used for all plots
    log_bins = np.logspace(np.log10(3), np.log10(100), 21)
    
    for i in range(2, 8):
        weight_name = f'{i}_coincidence_reflReq'

        if not HRAeventList[0].hasWeight(weight_name, sigma=plot_sigma):
            ic(f"Weight '{weight_name}' not found. Calculating now...")
            weights_were_added = True
            bad_stations = [32, 52, 132, 152]
            trigger_rate_coincidence = HRAAnalysis.getCoincidencesTriggerRates(
                HRAeventList, bad_stations, force_stations=reflected_stations, sigma=plot_sigma
            )
            if i in trigger_rate_coincidence and np.any(trigger_rate_coincidence[i] > 0):
                HRAAnalysis.setNewTrigger(HRAeventList, weight_name, bad_stations=bad_stations, sigma=plot_sigma)
                HRAAnalysis.setHRAeventListRateWeight(
                    HRAeventList, trigger_rate_coincidence[i], weight_name=weight_name, 
                    max_distance=max_distance, sigma=plot_sigma
                )
                ic(f"Successfully calculated and added weights for '{weight_name}'.")
            else:
                ic(f"No events found for {i}-fold coincidence with reflection required. Skipping.")
                continue

        ic(f"Processing coincidence level {i}...")

        # Get SNR and weight arrays using the revised function
        direct_snrs, reflected_snrs, weights = get_snr_and_weights(
            HRAeventList, weight_name, direct_stations, reflected_stations, sigma=plot_sigma
        )
        
        # Check if any data was returned before plotting
        if len(weights) == 0:
            ic(f"No events with valid triggers found for weight '{weight_name}'. Skipping plots for this level.")
            continue

        # --- 1. Generate 1D SNR Distribution Plot ---
        main_plot_title_1d = f'SNR Distribution for {i}-Fold Coincidence (Reflected Required)'
        save_path_1d = os.path.join(snr_plot_folder, f'snr_dist_{i}coinc_reflReq_1d.png')
        plot_snr_distribution(direct_snrs, reflected_snrs, weights, main_plot_title_1d, save_path_1d, bins=log_bins)

        # --- 2. Generate 2D SNR Histogram ---
        main_plot_title_2d = f'2D SNR Histogram for {i}-Fold Coincidence (Reflected Required)'
        save_path_2d = os.path.join(snr_plot_folder, f'snr_hist_{i}coinc_reflReq_2d.png')
        plot_2d_snr_histogram(direct_snrs, reflected_snrs, weights, main_plot_title_2d, save_path_2d, bins=log_bins)

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