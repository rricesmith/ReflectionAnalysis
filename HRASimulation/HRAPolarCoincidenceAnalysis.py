
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import configparser
import os
from icecream import ic
from NuRadioReco.utilities import units
from HRASimulation.S02_HRANurToNpy import loadHRAfromH5
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
            wrap_mask = (azi_diff_matrix > np.pi) & (azi_diff_matrix < 1000)
            azi_diff_matrix[wrap_mask] = 2 * np.pi - azi_diff_matrix[wrap_mask]  # Account for wrap-around
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


def getSummedCoincidenceAnglesWeights(HRAEventList, station_ids, bad_stations, min_coincidence=2):
    """
    Get angles and weights for events, summing across all coincidence levels
    """
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
        # Check if event has any coincidence >= min_coincidence
        has_coincidence = False
        total_weight = 0
        
        # Add highest weight for which event has coincidence
        for i in range(min_coincidence, 7): 
            n = 7 + min_coincidence - i # Go reverse from min to max
            weight_name = f'{n}_coincidence_wrefl' if 113 in station_ids else f'{n}_coincidence_norefl'
            if event.hasCoincidence(num=n, bad_stations=bad_stations):
                has_coincidence = True
                total_weight = event.getWeight(weight_name)
                break
        
        if not has_coincidence:
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
            raw_weights_list.append(total_weight / n_weights_to_add)

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
            wrap_mask = (azi_diff_matrix > np.pi) & (azi_diff_matrix < 1000)
            azi_diff_matrix[wrap_mask] = 2 * np.pi - azi_diff_matrix[wrap_mask]  # Account for wrap-around
            min_azi_diff = np.min(azi_diff_matrix)

            smallest_diff_recon_zen_list.append(min_zen_diff)
            smallest_diff_recon_azi_list.append(min_azi_diff)
            smallest_diff_weights.append(total_weight)

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

    ic(f'Got {len(zenith)} angles for summed coincidences with stations {station_ids}')
    return zenith, recon_zenith, azimuth, recon_azimuth, weights, smallest_diff_recon_zen, smallest_diff_recon_azi, diff_weights


def plot_polar_histogram(azimuth, zenith, weights, title, savename, colorbar_label='Weighted count'):
    """
    Create a polar 2D histogram of azimuth and zenith angles
    """
    # Convert angles to appropriate formats for polar plot
    # if max(zenith) < np.pi/2 or max(azimuth) < np.pi*2:
    # zenith = np.rad2deg(zenith)
    # azimuth = np.rad2deg(azimuth)
    
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
    ax[0].grid(True, alpha=0.3)
    
    # Azimuth difference histogram
    diff_bins_azi = np.linspace(-180, 180, 37)
    ax[1].hist(azimuth_diff, bins=diff_bins_azi, weights=weights, alpha=0.7, edgecolor='black')
    ax[1].set_xlabel('True - Reconstructed Azimuth (deg)')
    ax[1].set_ylabel('Weighted count')
    ax[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, y=1.02)  # Add vertical spacing above plots
    plt.tight_layout()
    fig.savefig(savename, bbox_inches='tight')  # Ensure title is included
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
    
    # Convert to degrees - these are calculated in radians in the main function
    smallest_diff_zen = np.rad2deg(smallest_diff_zen)
    smallest_diff_azi = np.rad2deg(smallest_diff_azi)
    
    # Smallest zenith difference histogram
    diff_bins_zen = np.linspace(0, max(smallest_diff_zen)*1.1, 25)
    ax[0].hist(smallest_diff_zen, bins=diff_bins_zen, weights=diff_weights, alpha=0.7, edgecolor='black')
    ax[0].set_xlabel('Smallest Zenith Difference (deg)')
    ax[0].set_ylabel('Weighted count')
    ax[0].grid(True, alpha=0.3)
    
    # Smallest azimuth difference histogram
    diff_bins_azi = np.linspace(0, max(smallest_diff_azi)*1.1, 25)
    ax[1].hist(smallest_diff_azi, bins=diff_bins_azi, weights=diff_weights, alpha=0.7, edgecolor='black')
    ax[1].set_xlabel('Smallest Azimuth Difference (deg)')
    ax[1].set_ylabel('Weighted count')
    ax[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, y=1.02)  # Add vertical spacing above plots
    plt.tight_layout()
    fig.savefig(savename, bbox_inches='tight')  # Ensure title is included
    ic(f'Saved {savename}')
    plt.close(fig)


def plot_smallest_differences_with_cuts(smallest_diff_zen, smallest_diff_azi, diff_weights, title, savename, 
                                       zen_cut=20, azi_cut=45):
    """
    Create 1D histograms of smallest differences between reconstructions with efficiency cuts
    """
    if len(smallest_diff_zen) == 0 or len(smallest_diff_azi) == 0:
        ic("No coincidence events with multiple reconstructions found")
        return
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    # Convert to degrees - these are calculated in radians in the main function
    smallest_diff_zen_deg = np.rad2deg(smallest_diff_zen)
    smallest_diff_azi_deg = np.rad2deg(smallest_diff_azi)
    
    # Calculate efficiencies
    total_weight = np.sum(diff_weights)
    zen_pass_mask = smallest_diff_zen_deg < zen_cut
    azi_pass_mask = smallest_diff_azi_deg < azi_cut
    combined_pass_mask = zen_pass_mask & azi_pass_mask
    
    zen_efficiency = 100*np.sum(diff_weights[zen_pass_mask]) / total_weight if total_weight > 0 else 0
    azi_efficiency = 100*np.sum(diff_weights[azi_pass_mask]) / total_weight if total_weight > 0 else 0
    combined_efficiency = 100*np.sum(diff_weights[combined_pass_mask]) / total_weight if total_weight > 0 else 0
    
    # Smallest zenith difference histogram
    diff_bins_zen = np.linspace(0, max(smallest_diff_zen_deg)*1.1, 25)
    ax[0].hist(smallest_diff_zen_deg, bins=diff_bins_zen, weights=diff_weights, alpha=0.7, edgecolor='black')
    ax[0].axvline(zen_cut, color='red', linestyle='--', linewidth=2, label=f'Cut at {zen_cut}°\nEff: {zen_efficiency:.2f}%')
    ax[0].set_xlabel('Smallest Zenith Difference (deg)')
    ax[0].set_ylabel('Weighted count')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()
    
    # Smallest azimuth difference histogram
    diff_bins_azi = np.linspace(0, max(smallest_diff_azi_deg)*1.1, 25)
    ic(smallest_diff_azi_deg)
    ax[1].hist(smallest_diff_azi_deg, bins=diff_bins_azi, weights=diff_weights, alpha=0.7, edgecolor='black')
    ax[1].axvline(azi_cut, color='red', linestyle='--', linewidth=2, label=f'Cut at {azi_cut}°\nEff: {azi_efficiency:.2f}%')
    ax[1].set_xlabel('Smallest Azimuth Difference (deg)')
    ax[1].set_ylabel('Weighted count')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()
    
    # Add combined efficiency to title
    title_with_eff = f'{title}\nCombined Cut Efficiency: {combined_efficiency:.2f}%'
    plt.suptitle(title_with_eff, y=1.08)  # Extra space for multi-line title
    plt.tight_layout()
    fig.savefig(savename, bbox_inches='tight')  # Ensure title is included
    ic(f'Saved {savename} with efficiencies: zen={zen_efficiency:.2f}%, azi={azi_efficiency:.2f}%, combined={combined_efficiency:.2f}%')
    plt.close(fig)


def plot_failed_cuts_2d_differences(zenith_diffs_list, azimuth_diffs_list, weights_list, title, savename_base, 
                                    zen_cut=20, azi_cut=45):
    """
    Create 2D histograms of True-Reconstructed differences between two stations for failed cut events
    zenith_diffs_list and azimuth_diffs_list should be lists of length 2 containing the differences for each station
    station_ids_list should be the list of station IDs to determine axis assignment
    """
    if len(zenith_diffs_list) != 2 or len(azimuth_diffs_list) != 2:
        ic("Error: Need exactly 2 stations for 2D difference plots")
        return
    
    if len(zenith_diffs_list[0]) == 0:
        ic("No failed cut events found")
        return
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    # Convert to degrees if needed
    zen_diff_1 = np.rad2deg(zenith_diffs_list[0]) if np.max(np.abs(zenith_diffs_list[0])) < np.pi else zenith_diffs_list[0]
    zen_diff_2 = np.rad2deg(zenith_diffs_list[1]) if np.max(np.abs(zenith_diffs_list[1])) < np.pi else zenith_diffs_list[1]
    azi_diff_1 = np.rad2deg(azimuth_diffs_list[0]) if np.max(np.abs(azimuth_diffs_list[0])) < np.pi else azimuth_diffs_list[0]
    azi_diff_2 = np.rad2deg(azimuth_diffs_list[1]) if np.max(np.abs(azimuth_diffs_list[1])) < np.pi else azimuth_diffs_list[1]
    
    # Use default ordering without axis assignment requirements
    zen_x, zen_y = zen_diff_1, zen_diff_2
    azi_x, azi_y = azi_diff_1, azi_diff_2
    x_label_zen = f'True - Reconstructed Zenith (deg)'
    y_label_zen = f'True - Reconstructed Zenith (deg)'
    x_label_azi = f'True - Reconstructed Azimuth (deg)'
    y_label_azi = f'True - Reconstructed Azimuth (deg)'
    
    # Zenith difference 2D histogram
    zen_bins = np.linspace(-90, 90, 31)
    hist_zen, _, _ = np.histogram2d(zen_x, zen_y, bins=(zen_bins, zen_bins), weights=weights_list)
    
    # Mask zero values to make them white
    hist_zen_masked = np.ma.masked_where(hist_zen.T == 0, hist_zen.T)
    
    X_zen, Y_zen = np.meshgrid(zen_bins, zen_bins)
    pcm_zen = ax[0].pcolormesh(X_zen, Y_zen, hist_zen_masked, cmap='viridis')
    ax[0].set_xlabel(x_label_zen)
    ax[0].set_ylabel(y_label_zen)
    ax[0].set_title('Zenith Differences')
    ax[0].grid(True, alpha=0.3)
    
    # Add diagonal cut lines for zenith (y = x ± zen_cut)
    x_range = np.linspace(-90, 90, 100)
    ax[0].plot(x_range, x_range + zen_cut, 'r--', linewidth=2, alpha=0.8, label=f'Cut: |y-x| = {zen_cut}°')
    ax[0].plot(x_range, x_range - zen_cut, 'r--', linewidth=2, alpha=0.8)
    ax[0].legend()
    
    fig.colorbar(pcm_zen, ax=ax[0], label='Weighted count')
    
    # Azimuth difference 2D histogram
    azi_bins = np.linspace(-180, 180, 31)
    hist_azi, _, _ = np.histogram2d(azi_x, azi_y, bins=(azi_bins, azi_bins), weights=weights_list)
    
    # Mask zero values to make them white
    hist_azi_masked = np.ma.masked_where(hist_azi.T == 0, hist_azi.T)
    
    X_azi, Y_azi = np.meshgrid(azi_bins, azi_bins)
    pcm_azi = ax[1].pcolormesh(X_azi, Y_azi, hist_azi_masked, cmap='viridis')
    ax[1].set_xlabel(x_label_azi)
    ax[1].set_ylabel(y_label_azi)
    ax[1].set_title('Azimuth Differences')
    ax[1].grid(True, alpha=0.3)
    
    # Add diagonal cut lines for azimuth (y = x ± azi_cut)
    x_range_azi = np.linspace(-180, 180, 100)
    ax[1].plot(x_range_azi, x_range_azi + azi_cut, 'r--', linewidth=2, alpha=0.8, label=f'Cut: |y-x| = {azi_cut}°')
    ax[1].plot(x_range_azi, x_range_azi - azi_cut, 'r--', linewidth=2, alpha=0.8)
    ax[1].legend()
    
    fig.colorbar(pcm_azi, ax=ax[1], label='Weighted count')
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    fig.savefig(savename_base, bbox_inches='tight')
    ic(f'Saved {savename_base}')
    plt.close(fig)


def plot_failed_cuts_snr_vs_differences(snr_list, zenith_diffs_list, azimuth_diffs_list, weights_list, title, savename_base, 
                                        recon_zenith_list=None, recon_azimuth_list=None, zen_cut=20, azi_cut=45):
    """
    Create 2D histograms with SNR on x-axis (log scale 3-100) and absolute True-Recon differences on y-axis
    Also overlay scatter plots showing reconstructed angle differences between stations
    """
    if len(zenith_diffs_list) == 0 or len(snr_list) == 0:
        ic("No failed cut events found for SNR plots")
        return
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    # Flatten SNR and angle differences for all stations
    snr_flat = np.concatenate(snr_list)
    zen_diff_flat = np.abs(np.concatenate(zenith_diffs_list))
    azi_diff_flat = np.abs(np.concatenate(azimuth_diffs_list))
    weights_flat = np.concatenate([weights_list] * len(zenith_diffs_list))
    
    # Convert to degrees if needed
    zen_diff_flat = np.rad2deg(zen_diff_flat) if np.max(zen_diff_flat) < np.pi else zen_diff_flat
    azi_diff_flat = np.rad2deg(azi_diff_flat) if np.max(azi_diff_flat) < np.pi else azi_diff_flat
    
    # Create log bins for SNR from 3 to 100
    snr_bins = np.logspace(np.log10(3), np.log10(100), 21)
    
    # SNR vs absolute zenith difference
    zen_bins = np.linspace(0, max(zen_diff_flat)*1.1, 21)
    hist_zen, _, _ = np.histogram2d(snr_flat, zen_diff_flat, bins=(snr_bins, zen_bins), weights=weights_flat)
    
    # Mask zero values to make them white
    hist_zen_masked = np.ma.masked_where(hist_zen.T == 0, hist_zen.T)
    
    X_zen, Y_zen = np.meshgrid(snr_bins, zen_bins)
    pcm_zen = ax[0].pcolormesh(X_zen, Y_zen, hist_zen_masked, cmap='viridis')
    
    # Add scatter overlay for zenith - difference in reconstructed values between stations
    # For each event, plot two points with same y-value (absolute reconstructed difference) but different x-values (SNRs)
    if recon_zenith_list is not None:
        scatter_snr_zen_1 = []
        scatter_snr_zen_2 = []
        scatter_recon_diff_zen = []
        scatter_weights_zen = []
        
        # Process each event (both stations have same weight, so we can use first station's weight)
        for i in range(len(weights_list)):
            # Get the absolute difference between reconstructed zenith angles for this event
            recon_zen_1 = np.rad2deg(recon_zenith_list[0][i])
            recon_zen_2 = np.rad2deg(recon_zenith_list[1][i])
            recon_zen_diff = np.abs(recon_zen_1 - recon_zen_2)
            
            scatter_snr_zen_1.append(snr_list[0][i])
            scatter_snr_zen_2.append(snr_list[1][i])
            scatter_recon_diff_zen.append(recon_zen_diff)
            scatter_recon_diff_zen.append(recon_zen_diff)  # Same y-value for both points
            scatter_weights_zen.append(weights_list[i])
            scatter_weights_zen.append(weights_list[i])  # Same weight for both points
        
        # Combine SNR values for scatter plot
        scatter_snr_zen = np.concatenate([scatter_snr_zen_1, scatter_snr_zen_2])
        
        # Create scatter plot overlay with different colormap (plasma for contrast with viridis)
        if len(scatter_snr_zen) > 0:
            scatter_zen = ax[0].scatter(scatter_snr_zen, scatter_recon_diff_zen, c=scatter_weights_zen, 
                                      cmap='plasma', alpha=0.7, s=15, edgecolors='white', linewidth=0.3, label='Recon Difference')
            # Add a separate colorbar for scatter points
            # cbar_scatter_zen = fig.colorbar(scatter_zen, ax=ax[0], pad=0.1)
            # cbar_scatter_zen.set_label('Event weight (scatter)', rotation=270, labelpad=15)
    
    ax[0].set_xlabel('SNR')
    ax[0].set_ylabel('|True - Reconstructed Zenith| (deg)')
    ax[0].set_title('SNR vs Absolute Zenith Difference')
    ax[0].set_xscale('log')
    ax[0].grid(True, alpha=0.3)
    
    # Add horizontal line for zenith cut
    ax[0].axhline(zen_cut, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Zenith Cut: {zen_cut}°')
    ax[0].legend()
    
    fig.colorbar(pcm_zen, ax=ax[0], label='Weighted count (histogram)')
    
    # SNR vs absolute azimuth difference
    azi_bins = np.linspace(0, max(azi_diff_flat)*1.1, 21)
    hist_azi, _, _ = np.histogram2d(snr_flat, azi_diff_flat, bins=(snr_bins, azi_bins), weights=weights_flat)
    
    # Mask zero values to make them white
    hist_azi_masked = np.ma.masked_where(hist_azi.T == 0, hist_azi.T)
    
    X_azi, Y_azi = np.meshgrid(snr_bins, azi_bins)
    pcm_azi = ax[1].pcolormesh(X_azi, Y_azi, hist_azi_masked, cmap='viridis')
    
    # Add scatter overlay for azimuth - difference in reconstructed values between stations
    if recon_azimuth_list is not None:
        scatter_snr_azi_1 = []
        scatter_snr_azi_2 = []
        scatter_recon_diff_azi = []
        scatter_weights_azi = []
        
        # Process each event
        for i in range(len(weights_list)):
            # Get the absolute difference between reconstructed azimuth angles for this event
            recon_azi_1 = np.rad2deg(recon_azimuth_list[0][i])
            recon_azi_2 = np.rad2deg(recon_azimuth_list[1][i])
            recon_azi_diff = np.abs(recon_azi_1 - recon_azi_2)
            # Handle azimuth wrap-around
            if recon_azi_diff > 180:
                recon_azi_diff = 360 - recon_azi_diff
            
            scatter_snr_azi_1.append(snr_list[0][i])
            scatter_snr_azi_2.append(snr_list[1][i])
            scatter_recon_diff_azi.append(recon_azi_diff)
            scatter_recon_diff_azi.append(recon_azi_diff)  # Same y-value for both points
            scatter_weights_azi.append(weights_list[i])
            scatter_weights_azi.append(weights_list[i])  # Same weight for both points
        
        # Combine SNR values for scatter plot
        scatter_snr_azi = np.concatenate([scatter_snr_azi_1, scatter_snr_azi_2])
        
        # Create scatter plot overlay
        if len(scatter_snr_azi) > 0:
            scatter_azi = ax[1].scatter(scatter_snr_azi, scatter_recon_diff_azi, c=scatter_weights_azi, 
                                      cmap='plasma', alpha=0.7, s=15, edgecolors='white', linewidth=0.3, label='Recon Difference')
            # Add a separate colorbar for scatter points
            # cbar_scatter_azi = fig.colorbar(scatter_azi, ax=ax[1], pad=0.1)
            # cbar_scatter_azi.set_label('Event weight (scatter)', rotation=270, labelpad=15)
    
    ax[1].set_xlabel('SNR')
    ax[1].set_ylabel('|True - Reconstructed Azimuth| (deg)')
    ax[1].set_title('SNR vs Absolute Azimuth Difference')
    ax[1].set_xscale('log')
    ax[1].grid(True, alpha=0.3)
    
    # Add horizontal line for azimuth cut
    ax[1].axhline(azi_cut, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Azimuth Cut: {azi_cut}°')
    ax[1].legend()
    
    fig.colorbar(pcm_azi, ax=ax[1], label='Weighted count (histogram)')
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    fig.savefig(savename_base, bbox_inches='tight')
    ic(f'Saved {savename_base}')
    plt.close(fig)


def histCoincidenceAngles(zenith, recon_zenith, azimuth, recon_azimuth, weights, 
                         smallest_diff_recon_zen, smallest_diff_recon_azi, diff_weights,
                         title, savename_base, colorbar_label='Weighted count'):
    """
    Create all coincidence angle plots similar to histAngleRecon
    """
    # Convert angles to degrees if they're in radians
    # if max(zenith) < np.pi/2 or max(azimuth) < np.pi*2:
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
    
    # 5. 1D histograms of smallest differences with efficiency cuts
    plot_smallest_differences_with_cuts(smallest_diff_recon_zen, smallest_diff_recon_azi, diff_weights,
                                      f'{title} - Smallest Reconstruction Differences with Cuts',
                                      savename_base.replace('.png', '_smallest_diffs_cuts.png'))


def getFailedCutEventsData(HRAEventList, weight_name, n, station_ids, bad_stations, zen_cut=20, azi_cut=45):
    """
    Get data for events that fail the zenith and azimuth cuts for the n=2 case
    Returns data structured for the new plotting functions
    """
    if n != 2:
        ic(f"This function is only for n=2 case, got n={n}")
        return None
    
    failed_zenith_diffs = [[], []]  # List for each of the 2 stations
    failed_azimuth_diffs = [[], []]  # List for each of the 2 stations
    failed_snr = [[], []]  # SNR for each of the 2 stations
    failed_recon_zenith = [[], []]  # Reconstructed zenith for each station
    failed_recon_azimuth = [[], []]  # Reconstructed azimuth for each station
    failed_weights = []
    
    for event in HRAEventList:
        if not event.hasCoincidence(num=n, bad_stations=bad_stations):
            continue
        
        # Get stations that triggered for this event
        triggered_stations = []
        for station_id in station_ids:
            if event.hasTriggered(station_id):
                triggered_stations.append(station_id)
        
        # Only process if exactly 2 stations triggered (n=2)
        if len(triggered_stations) != 2:
            continue
        
        # Get reconstructed angles for both stations
        recon_zens = []
        recon_azis = []
        station_snrs = []
        
        for station_id in triggered_stations:
            if event.recon_zenith[station_id] is not None and event.recon_azimuth[station_id] is not None:
                recon_zens.append(event.recon_zenith[station_id])
                recon_azis.append(event.recon_azimuth[station_id])
                station_snrs.append(event.getSNR(station_id))
        
        # Need exactly 2 valid reconstructions
        if len(recon_zens) != 2:
            continue
        
        # Calculate smallest differences
        zen_diff = abs(recon_zens[0] - recon_zens[1])
        azi_diff = abs(recon_azis[0] - recon_azis[1])
        # Handle azimuth wrap-around
        if azi_diff > np.pi:
            azi_diff = 2 * np.pi - azi_diff
        
        # Convert to degrees for cut check
        zen_diff_deg = np.rad2deg(zen_diff)
        azi_diff_deg = np.rad2deg(azi_diff)
        
        # Check if event fails cuts
        fails_zen_cut = zen_diff_deg >= zen_cut
        fails_azi_cut = azi_diff_deg >= azi_cut
        
        if fails_zen_cut or fails_azi_cut:
            
            # Calculate true vs reconstructed differences for each station
            true_zenith = event.getAngles()[0]
            true_azimuth = event.getAngles()[1]
            
            # Convert true angles to in-ice
            n_ice = 1.37
            true_zenith_ice = np.arcsin(np.sin(true_zenith) / n_ice)
            
            for i, station_id in enumerate(triggered_stations):
                zen_diff_true_recon = true_zenith_ice - recon_zens[i]
                azi_diff_true_recon = true_azimuth - recon_azis[i]
                
                # Handle azimuth wrap-around for true-recon difference
                if azi_diff_true_recon > np.pi:
                    azi_diff_true_recon -= 2 * np.pi
                elif azi_diff_true_recon < -np.pi:
                    azi_diff_true_recon += 2 * np.pi
                
                failed_zenith_diffs[i].append(zen_diff_true_recon)
                failed_azimuth_diffs[i].append(azi_diff_true_recon)
                failed_snr[i].append(station_snrs[i])
                failed_recon_zenith[i].append(recon_zens[i])
                failed_recon_azimuth[i].append(recon_azis[i])
            
            # Add weight for this event
            failed_weights.append(event.getWeight(weight_name))
    
    # Convert to numpy arrays
    for i in range(2):
        failed_zenith_diffs[i] = np.array(failed_zenith_diffs[i])
        failed_azimuth_diffs[i] = np.array(failed_azimuth_diffs[i])
        failed_snr[i] = np.array(failed_snr[i])
        failed_recon_zenith[i] = np.array(failed_recon_zenith[i])
        failed_recon_azimuth[i] = np.array(failed_recon_azimuth[i])
    failed_weights = np.array(failed_weights)
    
    ic(f"Found {len(failed_weights)} events that fail cuts for {weight_name}")
    
    return failed_zenith_diffs, failed_azimuth_diffs, failed_snr, failed_weights, failed_recon_zenith, failed_recon_azimuth


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
        
        # Add new plots for failed cuts (n=2 case only)
        if i == 2:
            failed_zen_diffs, failed_azi_diffs, failed_snr, failed_weights, failed_recon_zen, failed_recon_azi = \
                getFailedCutEventsData(HRAeventList, weight_name, i, station_ids, bad_stations)

            if failed_weights is not None and len(failed_weights) > 0:
                # Create 2D histograms of True-Reconstructed differences between stations
                failed_title = f'Failed Cuts Analysis - n={i} Stations, Refl Required'
                failed_savename_2d = f'{coincidence_save_folder}failed_cuts_{weight_name}_reflreq_2d_diffs.png'
                plot_failed_cuts_2d_differences(failed_zen_diffs, failed_azi_diffs, failed_weights, 
                                               failed_title, failed_savename_2d)
                
                # Create SNR vs angle difference plots
                failed_savename_snr = f'{coincidence_save_folder}failed_cuts_{weight_name}_reflreq_snr_vs_diffs.png'
                plot_failed_cuts_snr_vs_differences(failed_snr, failed_zen_diffs, failed_azi_diffs, 
                                                   failed_weights, failed_title, failed_savename_snr,
                                                   failed_recon_zen, failed_recon_azi)
                
                ic(f"Completed failed cuts analysis for {weight_name}")
        
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
        
        # Add new plots for failed cuts (n=2 case only)
        if i == 2:
            failed_zen_diffs, failed_azi_diffs, failed_snr, failed_weights, failed_recon_zen, failed_recon_azi = \
                getFailedCutEventsData(HRAeventList, weight_name, i, station_ids, bad_stations)

            if failed_weights is not None and len(failed_weights) > 0:
                # Create 2D histograms of True-Reconstructed differences between stations
                failed_title = f'Failed Cuts Analysis - n={i} Stations, Direct Only'
                failed_savename_2d = f'{coincidence_save_folder}failed_cuts_{weight_name}_directonly_2d_diffs.png'
                plot_failed_cuts_2d_differences(failed_zen_diffs, failed_azi_diffs, failed_weights, 
                                               failed_title, failed_savename_2d)
                
                # Create SNR vs angle difference plots
                failed_savename_snr = f'{coincidence_save_folder}failed_cuts_{weight_name}_directonly_snr_vs_diffs.png'
                plot_failed_cuts_snr_vs_differences(failed_snr, failed_zen_diffs, failed_azi_diffs, 
                                                   failed_weights, failed_title, failed_savename_snr,
                                                   failed_recon_zen, failed_recon_azi)
                
                ic(f"Completed failed cuts analysis for {weight_name}")

        ic(f"Completed analysis for {weight_name}")

    
    # Add analysis for sum of all coincidences with reflections
    ic("Starting summed coincidence analysis with reflections")
    station_ids = [13, 14, 15, 17, 18, 19, 30, 113, 114, 115, 117, 118, 119, 130]
    bad_stations = [32, 52, 132, 152]
    
    zenith, recon_zenith, azimuth, recon_azimuth, weights, \
    smallest_diff_recon_zen, smallest_diff_recon_azi, diff_weights = \
        getSummedCoincidenceAnglesWeights(HRAeventList, station_ids, bad_stations)
    
    if len(zenith) > 0:
        title = f'Summed Coincidence Analysis - All Stations, Refl Required'
        savename_base = f'{coincidence_save_folder}coincidence_summed_wrefl_reflreq.png'
        
        histCoincidenceAngles(zenith, recon_zenith, azimuth, recon_azimuth, weights,
                            smallest_diff_recon_zen, smallest_diff_recon_azi, diff_weights,
                            title, savename_base)
        
        ic("Completed summed analysis with reflections")
    else:
        ic("No events found for summed coincidences with reflections")
    
    # Add analysis for sum of all coincidences - direct only
    ic("Starting summed coincidence analysis - direct only")
    station_ids = direct_stations
    bad_stations = [32, 52, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    
    zenith, recon_zenith, azimuth, recon_azimuth, weights, \
    smallest_diff_recon_zen, smallest_diff_recon_azi, diff_weights = \
        getSummedCoincidenceAnglesWeights(HRAeventList, station_ids, bad_stations)
    
    if len(zenith) > 0:
        title = f'Summed Coincidence Analysis - All Stations, Direct Only'
        savename_base = f'{coincidence_save_folder}coincidence_summed_norefl_directonly.png'
        
        histCoincidenceAngles(zenith, recon_zenith, azimuth, recon_azimuth, weights,
                            smallest_diff_recon_zen, smallest_diff_recon_azi, diff_weights,
                            title, savename_base)
        
        ic("Completed summed analysis - direct only")
    else:
        ic("No events found for summed coincidences - direct only")



    ic("Coincidence analysis complete!")
    