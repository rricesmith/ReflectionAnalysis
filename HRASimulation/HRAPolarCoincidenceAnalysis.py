
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import configparser
import os
from icecream import ic
from NuRadioReco.utilities import units
from HRASimulation.HRANurToNpy import loadHRAfromH5
from HRASimulation.HRAAnalysis import getCoincidencesTriggerRates

def getRawCoincidenceAnglesWeights(HRAEventList, weight_name, n, station_ids, bad_stations):
    # Rather than average stations, since we are looking at coincidences, we will return all angles and weights for events that triggered the right stations
    zenith_list = []
    recon_zenith_list = []
    azimuth_list = []
    recon_azimuth_list = []

    raw_weights_list = []

    smallest_diff_recon_zen_list = []
    smallest_diff_recon_azi_list = []
    smallest_diff_weights = []

    for event in HRAEventList:
        if not event.hasCoincidence(num=n, bad_stations=bad_stations):
            continue

        
        recon_zens = []
        recon_azis = []

        n_weights_to_add = 0
        
        for station_id in station_ids:
            if event.hasTriggered(station_id):
                zenith_list.append(event.getAngles()[0])
                recon_zenith_list.append(event.recon_zenith[station_id])
                azimuth_list.append(event.getAngles()[1])
                recon_azimuth_list.append(event.recon_azimuth[station_id])

                recon_zens.append(event.recon_zenith[station_id])
                recon_azis.append(event.recon_azimuth[station_id])

                n_weights_to_add += 1

        # Add weight for each angle added, divided equally between stations
        for _ in range(n_weights_to_add):
            raw_weights_list.append(event.getWeight(weight_name) / n_weights_to_add)

        # Find the smallest difference between the different reconstructions for this event
        if len(recon_zens) > 1:
            recon_zens = np.array(recon_zens)
            recon_azis = np.array(recon_azis)
            zen_diff_matrix = np.abs(recon_zens[:, None] - recon_zens[None, :])
            azi_diff_matrix = np.abs(recon_azis[:, None] - recon_azis[None, :])
            # Set diagonal to large value to ignore zero differences
            np.fill_diagonal(zen_diff_matrix, np.inf)
            np.fill_diagonal(azi_diff_matrix, np.inf)
            min_zen_diff = np.min(zen_diff_matrix)
            min_azi_diff = np.min(azi_diff_matrix)

            smallest_diff_recon_zen_list.append(min_zen_diff)
            smallest_diff_recon_azi_list.append(min_azi_diff)
            smallest_diff_weights.append(event.getWeight(weight_name))

    zenith = np.array(zenith_list)
    n = 1.37
    zenith = np.arcsin(np.sin(zenith) / n)  # Convert to in-ice angle
    recon_zenith = np.array(recon_zenith_list)
    azimuth = np.array(azimuth_list)
    recon_azimuth = np.array(recon_azimuth_list)
    weights = np.array(raw_weights_list)    
    smallest_diff_recon_zen = np.array(smallest_diff_recon_zen_list)
    smallest_diff_recon_azi = np.array(smallest_diff_recon_azi_list)
    diff_weights = np.array(smallest_diff_weights)

    ic(f'Got {len(zenith)} angles for {weight_name} with stations {station_ids}')
    return zenith, recon_zenith, azimuth, recon_azimuth, weights, smallest_diff_recon_zen, smallest_diff_recon_azi, diff_weights


def plot_polar_histogram(azimuth, zenith, weights, title, savename, colorbar_label='Weighted count'):
    """
    Create a polar 2D histogram of azimuth and zenith angles
    """
    # Convert angles to appropriate formats for polar plot
    if max(zenith) < np.pi/2 or max(azimuth) < np.pi*2:
        zenith = np.rad2deg(zenith)
        azimuth = np.rad2deg(azimuth)
    
    # Create bins
    azimuth_bins = np.linspace(0, 360, 37)  # 10 degree bins
    zenith_bins = np.linspace(0, 90, 19)    # 5 degree bins
    
    # Create 2D histogram
    hist, _, _ = np.histogram2d(azimuth, zenith, bins=(azimuth_bins, zenith_bins), weights=weights)
    
    # Create meshgrid for polar plot
    A, R = np.meshgrid(np.deg2rad(azimuth_bins), zenith_bins)
    
    # Create the plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # Set normalization
    if np.any(weights > 0):
        norm = matplotlib.colors.LogNorm(vmin=np.min(weights[np.nonzero(weights)]), vmax=np.max(weights))
    else:
        norm = matplotlib.colors.Normalize()
    
    pcm = ax.pcolormesh(A, R, hist.T, norm=norm, cmap='viridis')
    
    # Set polar plot properties
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_yticks(np.arange(0, 91, 15))
    ax.set_yticklabels([f'{i}°' for i in range(0, 91, 15)])
    
    # Add colorbar
    fig.colorbar(pcm, ax=ax, label=colorbar_label)
    ax.set_title(title)
    
    # Save the plot
    fig.savefig(savename)
    ic(f'Saved {savename}')
    plt.close(fig)


def plot_angle_differences(zenith_diff, azimuth_diff, weights, title, savename):
    """
    Create 1D histograms of angle differences
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    # Zenith difference histogram
    diff_bins_zen = np.linspace(-90, 90, 37)
    ax[0].hist(zenith_diff, bins=diff_bins_zen, weights=weights, alpha=0.7, edgecolor='black')
    ax[0].set_xlabel('True - Reconstructed Zenith (deg)')
    ax[0].set_ylabel('Weighted count')
    ax[0].set_title('Zenith Difference')
    ax[0].grid(True, alpha=0.3)
    
    # Azimuth difference histogram
    diff_bins_azi = np.linspace(-180, 180, 37)
    ax[1].hist(azimuth_diff, bins=diff_bins_azi, weights=weights, alpha=0.7, edgecolor='black')
    ax[1].set_xlabel('True - Reconstructed Azimuth (deg)')
    ax[1].set_ylabel('Weighted count')
    ax[1].set_title('Azimuth Difference')
    ax[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(savename)
    ic(f'Saved {savename}')
    plt.close(fig)


def plot_smallest_differences(smallest_diff_zen, smallest_diff_azi, diff_weights, title, savename):
    """
    Create 1D histograms of smallest differences between reconstructions
    """
    if len(smallest_diff_zen) == 0 or len(smallest_diff_azi) == 0:
        ic("No coincidence events with multiple reconstructions found")
        return
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    # Convert to degrees if in radians
    if max(smallest_diff_zen) < np.pi/2:
        smallest_diff_zen = np.rad2deg(smallest_diff_zen)
    if max(smallest_diff_azi) < np.pi:
        smallest_diff_azi = np.rad2deg(smallest_diff_azi)
    
    # Smallest zenith difference histogram
    diff_bins_zen = np.linspace(0, max(smallest_diff_zen)*1.1, 25)
    ax[0].hist(smallest_diff_zen, bins=diff_bins_zen, weights=diff_weights, alpha=0.7, edgecolor='black')
    ax[0].set_xlabel('Smallest Zenith Difference Between Stations (deg)')
    ax[0].set_ylabel('Weighted count')
    ax[0].set_title('Smallest Zenith Difference')
    ax[0].grid(True, alpha=0.3)
    
    # Smallest azimuth difference histogram
    diff_bins_azi = np.linspace(0, max(smallest_diff_azi)*1.1, 25)
    ax[1].hist(smallest_diff_azi, bins=diff_bins_azi, weights=diff_weights, alpha=0.7, edgecolor='black')
    ax[1].set_xlabel('Smallest Azimuth Difference Between Stations (deg)')
    ax[1].set_ylabel('Weighted count')
    ax[1].set_title('Smallest Azimuth Difference')
    ax[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(savename)
    ic(f'Saved {savename}')
    plt.close(fig)


def histCoincidenceAngles(zenith, recon_zenith, azimuth, recon_azimuth, weights, 
                         smallest_diff_recon_zen, smallest_diff_recon_azi, diff_weights,
                         title, savename_base, colorbar_label='Weighted count'):
    """
    Create all coincidence angle plots similar to histAngleRecon
    """
    # Convert angles to degrees if they're in radians
    if max(zenith) < np.pi/2 or max(azimuth) < np.pi*2:
        zenith = np.rad2deg(zenith)
        azimuth = np.rad2deg(azimuth) 
        recon_zenith = np.rad2deg(recon_zenith)
        recon_azimuth = np.rad2deg(recon_azimuth)
    
    # 1. Polar 2D histogram of true zenith/azimuth
    plot_polar_histogram(azimuth, zenith, weights, 
                        f'{title} - True Directions', 
                        savename_base.replace('.png', '_true_polar.png'),
                        colorbar_label)
    
    # 2. Polar 2D histogram of reconstructed zenith/azimuth
    plot_polar_histogram(recon_azimuth, recon_zenith, weights, 
                        f'{title} - Reconstructed Directions', 
                        savename_base.replace('.png', '_recon_polar.png'),
                        colorbar_label)
    
    # 3. 1D histograms of angle differences
    zenith_diff = zenith - recon_zenith
    azimuth_diff = azimuth - recon_azimuth
    plot_angle_differences(zenith_diff, azimuth_diff, weights,
                          f'{title} - Angle Differences',
                          savename_base.replace('.png', '_angle_diffs.png'))
    
    # 4. 1D histograms of smallest differences between reconstructions
    plot_smallest_differences(smallest_diff_recon_zen, smallest_diff_recon_azi, diff_weights,
                            f'{title} - Smallest Reconstruction Differences',
                            savename_base.replace('.png', '_smallest_diffs.png'))


def load_HRA_data():
    """
    Load HRA event data using configuration file, similar to HRAAnalysis.py
    """
    config = configparser.ConfigParser()
    config.read('HRASimulation/config.ini')
    
    sim_folder = config['FOLDERS']['sim_folder']
    numpy_folder = config['FOLDERS']['numpy_folder']
    save_folder = config['FOLDERS']['save_folder']
    
    # Create save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    if not os.path.exists(numpy_folder):
        raise FileNotFoundError(f"Numpy folder {numpy_folder} does not exist")
    
    # Load HRA event list
    HRAeventList = loadHRAfromH5(f'{numpy_folder}HRAeventList.h5')
    
    ic(f"Loaded {len(HRAeventList)} HRA events")
    
    return HRAeventList, save_folder


if __name__ == "__main__":
    ####### NOTE ##########
    # This script currently doesn't look at sigma trigger. Something that could be added


    # Load HRA event data
    try:
        HRAeventList, save_folder = load_HRA_data()
    except FileNotFoundError as e:
        ic(f"Error loading data: {e}")
        exit(1)
    
    # Create coincidence analysis folder
    coincidence_save_folder = f'{save_folder}coincidence_angles/'
    os.makedirs(coincidence_save_folder, exist_ok=True)
    
    # Define station combinations for analysis
    # Direct stations
    direct_stations = [13, 14, 15, 17, 18, 19, 30]
    # Reflected stations  
    reflected_stations = [113, 114, 115, 117, 118, 119, 130]
    # Combined stations (both direct and reflected)
    combined_stations = direct_stations + reflected_stations
    

    station_ids = [13, 14, 15, 17, 18, 19, 30, 113, 114, 115, 117, 118, 119, 130]
    bad_stations = [32, 52, 132, 152]
    trigger_rate_coincidence = getCoincidencesTriggerRates(HRAeventList, bad_stations)
    for i in trigger_rate_coincidence:
        if i < 2:
            continue
        weight_name = f'{i}_coincidence_wrefl'
    
        # Get coincidence angles and weights
        zenith, recon_zenith, azimuth, recon_azimuth, weights, \
        smallest_diff_recon_zen, smallest_diff_recon_azi, diff_weights = \
            getRawCoincidenceAnglesWeights(HRAeventList, weight_name, i, station_ids, bad_stations)
        
        # Skip if no events found
        if len(zenith) == 0:
            ic(f"No events found for weight {weight_name}")
            continue
        
        # Create plots
        title = f'Coincidence Analysis - n={i} Stations, Refl Required'
        savename_base = f'{coincidence_save_folder}coincidence_{weight_name}_reflreq.png'
        
        histCoincidenceAngles(zenith, recon_zenith, azimuth, recon_azimuth, weights,
                            smallest_diff_recon_zen, smallest_diff_recon_azi, diff_weights,
                            title, savename_base)
        
        ic(f"Completed analysis for {weight_name}")
                
    station_ids = direct_stations
    bad_stations = [32, 52, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    trigger_rate_coincidence = getCoincidencesTriggerRates(HRAeventList, bad_stations)
    for i in trigger_rate_coincidence:
        if i < 2:
            continue
        weight_name = f'{i}_coincidence_norefl'
        
    
        # Get coincidence angles and weights
        zenith, recon_zenith, azimuth, recon_azimuth, weights, \
        smallest_diff_recon_zen, smallest_diff_recon_azi, diff_weights = \
            getRawCoincidenceAnglesWeights(HRAeventList, weight_name, i, station_ids, bad_stations)
        
        # Skip if no events found
        if len(zenith) == 0:
            ic(f"No events found for weight {weight_name}")
            continue

        # Create plots
        title = f'Coincidence Analysis - n={i} Stations, Direct Only'
        savename_base = f'{coincidence_save_folder}coincidence_{weight_name}_directonly.png'

        histCoincidenceAngles(zenith, recon_zenith, azimuth, recon_azimuth, weights,
                            smallest_diff_recon_zen, smallest_diff_recon_azi, diff_weights,
                            title, savename_base)

        ic(f"Completed analysis for {weight_name}")



    ic("Coincidence analysis complete!")
    