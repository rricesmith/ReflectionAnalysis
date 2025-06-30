import os
import numpy as np
import matplotlib.pyplot as plt
import configparser
from icecream import ic

# Assuming HRAEventObject and HRANurToNpy are in a package named HRASimulation
# You might need to adjust the import path based on your project structure
from HRASimulation.HRAEventObject import HRAevent
from HRASimulation.HRANurToNpy import loadHRAfromH5
from NuRadioReco.utilities import units

def analyze_coincident_pair(HRAeventList, station_pair, weight_name, save_folder, ZenLim=None, AziLim=None, sigma=4.5):
    """
    Analyzes events in coincidence between two stations, plots reconstructed
    angle correlations, and prints details for events within specific angle ranges.

    Args:
        HRAeventList (list): A list of HRAevent objects.
        station_pair (list or tuple): A pair of station IDs, e.g., [17, 113].
        weight_name (str): The name of the weight to use for histograms, e.g., '2_coincidence_wrefl'.
        save_folder (str): The path to the directory where plots and output files will be saved.
        ZenLim (list or tuple, optional): A range [min, max] for zenith in degrees. Defaults to None.
        AziLim (list or tuple, optional): A range [min, max] for azimuth in degrees. Defaults to None.
        sigma (float, optional): The sigma value for getting weights. Defaults to 4.5.
    """
    ic(f"Analyzing coincidence for station pair: {station_pair}")
    ic(f"Using weight: '{weight_name}'")

    st1, st2 = station_pair
    
    # Lists to store data for coincident events
    recon_zen1, recon_zen2 = [], []
    recon_azi1, recon_azi2 = [], []
    weights = []
    events_in_angle_range = []

    # Loop through all events to find coincidences
    for event in HRAeventList:
        # Check for coincidence by seeing if both stations have reconstruction data
        if st1 in event.recon_zenith and st2 in event.recon_zenith:
            
            # Append reconstructed angles (converted to degrees)
            zen1_deg = np.rad2deg(event.recon_zenith[st1])
            zen2_deg = np.rad2deg(event.recon_zenith[st2])
            azi1_deg = np.rad2deg(event.recon_azimuth[st1])
            azi2_deg = np.rad2deg(event.recon_azimuth[st2])

            recon_zen1.append(zen1_deg)
            recon_zen2.append(zen2_deg)
            recon_azi1.append(azi1_deg)
            recon_azi2.append(azi2_deg)

            # Append weight for the event
            # The user wants to use a single weight key for the pair.
            event_weight = event.getWeight(weight_name, sigma=sigma)
            weights.append(event_weight if event_weight is not None and not np.isnan(event_weight) else 0)

            # Check if the event falls within the specified angle limits
            if ZenLim is not None and AziLim is not None:
                zen_in_range = (ZenLim[0] <= zen1_deg <= ZenLim[1]) and \
                               (ZenLim[0] <= zen2_deg <= ZenLim[1])
                
                azi_in_range = (AziLim[0] <= azi1_deg <= AziLim[1]) and \
                               (AziLim[0] <= azi2_deg <= AziLim[1])

                if zen_in_range and azi_in_range:
                    events_in_angle_range.append(event)

    if not recon_zen1:
        ic(f"No coincident events found for station pair {station_pair}. Exiting.")
        return

    # --- Plotting 2D Histograms ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Reconstructed Angle Correlation for Stations {st1} & {st2}', fontsize=16)

    # Zenith vs Zenith
    bins_zen = np.linspace(0, 90, 46)
    ax1.hist2d(recon_zen1, recon_zen2, bins=bins_zen, weights=weights, cmap='viridis')
    ax1.plot([0, 90], [0, 90], 'r--', label='y=x')
    ax1.set_xlabel(f'Station {st1} Recon. Zenith [deg]')
    ax1.set_ylabel(f'Station {st2} Recon. Zenith [deg]')
    ax1.set_title('Zenith Correlation')
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.legend()

    # Azimuth vs Azimuth
    bins_azi = np.linspace(0, 360, 73)
    ax2.hist2d(recon_azi1, recon_azi2, bins=bins_azi, weights=weights, cmap='viridis')
    ax2.plot([0, 360], [0, 360], 'r--', label='y=x')
    ax2.set_xlabel(f'Station {st1} Recon. Azimuth [deg]')
    ax2.set_ylabel(f'Station {st2} Recon. Azimuth [deg]')
    ax2.set_title('Azimuth Correlation')
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(save_folder, f'recon_angle_correlation_{st1}_{st2}.png')
    plt.savefig(plot_filename)
    ic(f"Saved histogram plot to {plot_filename}")
    plt.close(fig)

    # --- Writing Event Details to File ---
    if ZenLim and AziLim and events_in_angle_range:
        output_filename = os.path.join(save_folder, f'event_details_{st1}_{st2}_zen{ZenLim[0]}-{ZenLim[1]}_azi{AziLim[0]}-{AziLim[1]}.txt')
        with open(output_filename, 'w') as f:
            f.write(f"# Event details for coincident triggers between stations {st1} and {st2}\n")
            f.write(f"# Zenith Range: {ZenLim} deg, Azimuth Range: {AziLim} deg\n")
            f.write("-" * 80 + "\n")
            
            for event in events_in_angle_range:
                f.write(f"Event ID: {event.getEventID()}\n")
                f.write(f"  Core Position (x, y): ({event.coreas_x/units.m:.2f} m, {event.coreas_y/units.m:.2f} m)\n")
                f.write(f"  True Energy: {event.energy/units.EeV:.4f} EeV\n")
                f.write(f"  True Zenith: {np.rad2deg(event.zenith):.2f} deg\n")
                f.write(f"  True Azimuth: {np.rad2deg(event.azimuth):.2f} deg\n")
                
                # Station 1 details
                f.write(f"  Station {st1}:\n")
                f.write(f"    Recon Zenith: {np.rad2deg(event.recon_zenith.get(st1, np.nan)):.2f} deg\n")
                f.write(f"    Recon Azimuth: {np.rad2deg(event.recon_azimuth.get(st1, np.nan)):.2f} deg\n")
                f.write(f"    SNR: {event.getSNR(st1):.2f}\n")
                # f.write(f"    Chi2: {event.chi2.get(st1, 'N/A')}\n") # .get() handles cases where chi2 might be missing

                # Station 2 details
                f.write(f"  Station {st2}:\n")
                f.write(f"    Recon Zenith: {np.rad2deg(event.recon_zenith.get(st2, np.nan)):.2f} deg\n")
                f.write(f"    Recon Azimuth: {np.rad2deg(event.recon_azimuth.get(st2, np.nan)):.2f} deg\n")
                f.write(f"    SNR: {event.getSNR(st2):.2f}\n")
                # f.write(f"    Chi2: {event.chi2.get(st2, 'N/A')}\n")
                
                f.write("-" * 80 + "\n")
        
        ic(f"Saved event details to {output_filename}")
    elif ZenLim and AziLim:
        ic("No events found within the specified angle limits.")


if __name__ == "__main__":
    # --- Configuration ---
    config = configparser.ConfigParser()
    # Ensure the config file path is correct
    config.read('HRASimulation/config.ini')

    numpy_folder = config['FOLDERS']['numpy_folder']
    save_folder = config['FOLDERS']['save_folder']
    
    # Create a specific sub-folder for this analysis
    coincidence_save_folder = os.path.join(save_folder, 'CoincidenceAnalysis/')
    os.makedirs(coincidence_save_folder, exist_ok=True)

    # --- Load Data ---
    ic("Loading HRA event list...")
    # Make sure the filename matches what you have
    HRAeventList = loadHRAfromH5(os.path.join(numpy_folder, 'HRAeventList.h5'))
    ic(f"Loaded {len(HRAeventList)} events.")

    # --- Set Analysis Parameters ---
    # The station pair to analyze
    station_pair_to_analyze = [17, 113]
    
    # The weight key to use for histograms
    # This should be pre-calculated and stored in the HRAevent objects
    weight_key = '2_coincidence_wrefl'
    
    # Define the angle limits for detailed event printing
    zenith_limits = [42.0, 46.0]  # degrees
    azimuth_limits = [230.0, 250.0]  # degrees

    # --- Run Analysis ---
    analyze_coincident_pair(
        HRAeventList=HRAeventList,
        station_pair=station_pair_to_analyze,
        weight_name=weight_key,
        save_folder=coincidence_save_folder,
        ZenLim=zenith_limits,
        AziLim=azimuth_limits
    )

    ic("Analysis complete.")
