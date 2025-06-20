import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import configparser
from icecream import ic
import h5py
import pickle
import itertools

# Import your existing modules
from HRASimulation.HRAEventObject import HRAevent
from HRASimulation.HRANurToNpy import loadHRAfromH5
import HRASimulation.HRAAnalysis as HRAAnalysis
from NuRadioReco.utilities import units


def get_snr_and_weights_for_1d_plot(HRAeventList, weight_name, direct_stations, reflected_stations, sigma=4.5):
    """
    Extracts SNRs and weights for 1D histograms.
    Each direct or reflected trigger is a separate entry.
    """
    direct_snrs_list = []
    reflected_snrs_list = []
    direct_weights_list = []
    reflected_weights_list = []

    for event in HRAeventList:
        event_weight = event.getWeight(weight_name, sigma=sigma)
        if event_weight > 0:
            # Find all triggered SNRs for this event
            event_direct_snrs = [
                event.getSNR(st_id) for st_id in direct_stations
                if event.hasTriggered(st_id, sigma) and event.getSNR(st_id) is not None
            ]
            event_reflected_snrs = [
                event.getSNR(st_id) for st_id in reflected_stations
                if event.hasTriggered(st_id, sigma) and event.getSNR(st_id) is not None
            ]

            # Create separate entries for 1D histograms
            for snr in event_direct_snrs:
                direct_snrs_list.append(snr)
                reflected_snrs_list.append(0) # Placeholder
                direct_weights_list.append(event_weight)
                reflected_weights_list.append(0) # Placeholder
            for snr in event_reflected_snrs:
                direct_snrs_list.append(0) # Placeholder
                reflected_snrs_list.append(snr)
                direct_weights_list.append(0) # Placeholder
                reflected_weights_list.append(event_weight)

    return np.array(direct_snrs_list), np.array(reflected_snrs_list), np.array(direct_weights_list), np.array(reflected_weights_list)

def get_snr_pairs_for_2d_plot(HRAeventList, weight_name, direct_stations, reflected_stations, sigma=4.5):
    """
    Extracts paired (direct, reflected) SNR values and a single weight for each pair.
    Only creates data points for events that have BOTH a direct and a reflected trigger.
    """
    direct_snrs_list = []
    reflected_snrs_list = []
    weights_list = []

    for event in HRAeventList:
        event_weight = event.getWeight(weight_name, sigma=sigma)
        if event_weight > 0:
            # Find all triggered SNRs for this event
            event_direct_snrs = [
                event.getSNR(st_id) for st_id in direct_stations
                if event.hasTriggered(st_id, sigma) and event.getSNR(st_id) is not None
            ]
            event_reflected_snrs = [
                event.getSNR(st_id) for st_id in reflected_stations
                if event.hasTriggered(st_id, sigma) and event.getSNR(st_id) is not None
            ]

            # If there's at least one of each, create all combination pairs
            if event_direct_snrs and event_reflected_snrs:
                for d_snr, r_snr in itertools.product(event_direct_snrs, event_reflected_snrs):
                    direct_snrs_list.append(d_snr)
                    reflected_snrs_list.append(r_snr)
                    weights_list.append(event_weight)

    return np.array(direct_snrs_list), np.array(reflected_snrs_list), np.array(weights_list)


def plot_snr_distribution(direct_snrs, reflected_snrs, direct_weights, reflected_weights, main_title, savename, bins):
    """
    Plots weighted 1D histograms of SNR distributions.
    This function remains unchanged.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Use a mask to plot only the relevant data for each subplot
    direct_mask = direct_weights > 0
    axs[0].hist(direct_snrs[direct_mask], bins=bins, weights=direct_weights[direct_mask], histtype='step', linewidth=2)
    axs[0].set_xlabel('SNR')
    axs[0].set_ylabel('Weighted Counts (Evts/Yr)')
    axs[0].set_title('Direct Triggers')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    reflected_mask = reflected_weights > 0
    axs[1].hist(reflected_snrs[reflected_mask], bins=bins, weights=reflected_weights[reflected_mask], histtype='step', linewidth=2, color='C1')
    axs[1].set_xlabel('SNR')
    axs[1].set_title('Reflected Triggers')
    axs[1].set_xscale('log')
    # y-scale is shared

    plt.suptitle(main_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ic(f"Saving 1D plot: {savename}")
    plt.savefig(savename)
    plt.close(fig)

def plot_2d_snr_histogram(direct_snrs, reflected_snrs, weights, main_title, savename, bins):
    """
    Plots a weighted 2D histogram of paired SNR data.
    Signature and logic are now simplified.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Data is already correctly paired, no complex masking needed.
    # The cmin parameter handles bins with zero counts to prevent the LogNorm error.
    h, xedges, yedges, im = ax.hist2d(
        direct_snrs,
        reflected_snrs,
        bins=bins,
        weights=weights,
        norm=colors.LogNorm(),
        cmin=1e-5  # Prevents ValueError for empty bins on a log scale
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

        # --- 1D Plot Data Generation and Plotting ---
        direct_snrs_1d, reflected_snrs_1d, direct_weights_1d, reflected_weights_1d = get_snr_and_weights_for_1d_plot(
            HRAeventList, weight_name, direct_stations, reflected_stations, sigma=plot_sigma
        )
        
        if len(direct_weights_1d) > 0 or len(reflected_weights_1d) > 0:
            main_plot_title_1d = f'SNR Distribution for {i}-Fold Coincidence (Reflected Required)'
            save_path_1d = os.path.join(snr_plot_folder, f'snr_dist_{i}coinc_reflReq_1d.png')
            plot_snr_distribution(direct_snrs_1d, reflected_snrs_1d, direct_weights_1d, reflected_weights_1d, main_plot_title_1d, save_path_1d, bins=log_bins)
        else:
            ic(f"No events with valid triggers found for 1D plots for weight '{weight_name}'. Skipping.")


        # --- 2D Plot Data Generation and Plotting ---
        direct_snrs_2d, reflected_snrs_2d, weights_2d = get_snr_pairs_for_2d_plot(
            HRAeventList, weight_name, direct_stations, reflected_stations, sigma=plot_sigma
        )
        
        if len(weights_2d) > 0:
            main_plot_title_2d = f'2D SNR Histogram for {i}-Fold Coincidence (Reflected Required)'
            save_path_2d = os.path.join(snr_plot_folder, f'snr_hist_{i}coinc_reflReq_2d.png')
            plot_2d_snr_histogram(direct_snrs_2d, reflected_snrs_2d, weights_2d, main_plot_title_2d, save_path_2d, bins=log_bins)
        else:
            ic(f"No events with both direct and reflected triggers found for 2D plot for weight '{weight_name}'. Skipping.")


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