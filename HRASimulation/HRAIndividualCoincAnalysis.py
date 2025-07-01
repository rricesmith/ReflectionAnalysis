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

# --- MODIFICATION: Renamed function and updated 'station_pair' to 'station_pairs' ---
def analyze_coincident_pairs(HRAeventList, station_pairs, weight_name, save_folder, ZenLim=None, AziLim=None, sigma=4.5, SNR_threshold=0):
    """
    Analyzes events in coincidence between station pairs, plots reconstructed
    angle correlations, and prints details for events within specific angle ranges.

    Args:
        HRAeventList (list): A list of HRAevent objects.
        station_pairs (list): A list of station ID pairs, e.g., [[17, 113], [15, 118]].
                              Can also accept a single pair like [17, 113].
        weight_name (str): The name of the weight to use for histograms, e.g., '2_coincidence_wrefl'.
        save_folder (str): The path to the directory where plots and output files will be saved.
        ZenLim (list or tuple, optional): A range [min, max] for zenith in degrees. Defaults to None.
        AziLim (list or tuple, optional): A range [min, max] for azimuth in degrees. Defaults to None.
        sigma (float, optional): The sigma value for getting weights. Defaults to 4.5.
        SNR_threshold (float, optional): The minimum SNR for both stations in a pair. Defaults to 0.
    """
    # --- MODIFICATION: Handle both single pair and list of pairs input ---
    # Standardize input to be a list of lists for consistent processing
    if not isinstance(station_pairs[0], list):
        station_pairs = [station_pairs]

    # --- MODIFICATION: Create a string for filenames from all pairs ---
    pair_str = "_".join([f"{p[0]}-{p[1]}" for p in station_pairs])
    
    ic(f"Analyzing coincidence for station pairs: {pair_str}")
    ic(f"Using weight: '{weight_name}'")
    
    for type in ['Recon', 'True']:
        recon_zen1, recon_zen2 = [], []
        recon_azi1, recon_azi2 = [], []
        weights = []
        events_in_angle_range = []

        # --- MODIFICATION: Loop through events and then check each pair ---
        for event in HRAeventList:
            # Loop through all specified pairs to find a coincidence
            for st1, st2 in station_pairs:
                # Check for coincidence by seeing if both stations have reconstruction data
                if st1 in event.recon_zenith and st2 in event.recon_zenith:
                    
                    # Check if the event has SNR above the threshold for both stations
                    if not (event.getSNR(st1) >= SNR_threshold and event.getSNR(st2) >= SNR_threshold):
                        continue  # Skip to the next pair if SNR is too low

                    if type == 'Recon':
                        zen1_deg = np.rad2deg(event.recon_zenith[st1])
                        zen2_deg = np.rad2deg(event.recon_zenith[st2])
                        azi1_deg = np.rad2deg(event.recon_azimuth[st1])
                        azi2_deg = np.rad2deg(event.recon_azimuth[st2])
                    elif type == 'True':
                        zen1_deg = np.rad2deg(event.zenith)
                        zen2_deg = np.rad2deg(event.zenith)
                        azi1_deg = np.rad2deg(event.azimuth)
                        azi2_deg = np.rad2deg(event.azimuth)

                    recon_zen1.append(zen1_deg)
                    recon_zen2.append(zen2_deg)
                    recon_azi1.append(azi1_deg)
                    recon_azi2.append(azi2_deg)

                    event_weight = event.getWeight(weight_name, sigma=sigma)
                    weights.append(event_weight if event_weight is not None and not np.isnan(event_weight) else 0)

                    if ZenLim is not None and AziLim is not None:
                        zen_in_range = (ZenLim[0] <= zen1_deg <= ZenLim[1]) and \
                                    (ZenLim[0] <= zen2_deg <= ZenLim[1])
                        azi_in_range = (AziLim[0] <= azi1_deg <= AziLim[1]) and \
                                    (AziLim[0] <= azi2_deg <= AziLim[1])
                        if zen_in_range and azi_in_range:
                            events_in_angle_range.append((event, st1, st2)) # Also save which pair triggered it

                    # --- MODIFICATION: Break inner loop after finding the first valid pair for an event ---
                    # This ensures each event is only counted once.
                    break 

        if not recon_zen1:
            ic(f"No coincident events found for station pairs {pair_str} for type '{type}'. Skipping plots.")
            continue

        import matplotlib.cm as cm
        cmap = cm.viridis
        cmap.set_under('white')
        vmin_val = 1e-9

        # --- Plotting 2D Histograms (Angle Correlation) ---
        # --- MODIFICATION: Updated titles and labels for multiple pairs ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{type} Angle Correlation for Station Pairs {pair_str}', fontsize=16)

        bins_zen = np.linspace(0, 90, 46)
        ax1.hist2d(recon_zen1, recon_zen2, bins=bins_zen, weights=weights, cmap=cmap, vmin=vmin_val)
        ax1.plot([0, 90], [0, 90], 'r--', label='y=x')
        ax1.set_xlabel(f'Station 1 of Pair {type} Zenith [deg]')
        ax1.set_ylabel(f'Station 2 of Pair {type} Zenith [deg]')
        ax1.set_title('Zenith Correlation')
        ax1.set_aspect('equal')
        ax1.grid(True)
        ax1.legend()

        bins_azi = np.linspace(0, 360, 73)
        ax2.hist2d(recon_azi1, recon_azi2, bins=bins_azi, weights=weights, cmap=cmap, vmin=vmin_val)
        ax2.plot([0, 360], [0, 360], 'r--', label='y=x')
        ax2.set_xlabel(f'Station 1 of Pair {type} Azimuth [deg]')
        ax2.set_ylabel(f'Station 2 of Pair {type} Azimuth [deg]')
        ax2.set_title('Azimuth Correlation')
        ax2.set_aspect('equal')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # --- MODIFICATION: Use generated pair_str for filename ---
        plot_filename = os.path.join(save_folder, f'{type}_angle_correlation_{pair_str}.png')
        plt.savefig(plot_filename)
        ic(f"Saved histogram plot to {plot_filename}")
        plt.close(fig)

        # --- NEW PLOT: Individual Station Azimuth vs Zenith ---
        fig_indiv, (ax_st1, ax_st2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig_indiv.suptitle(f'{type} Azimuth vs Zenith for Station Pairs {pair_str}', fontsize=16)

        bins_azi_indiv = np.linspace(0, 360, 73)
        bins_zen_indiv = np.linspace(0, 90, 46)

        ax_st1.hist2d(recon_azi1, recon_zen1, bins=[bins_azi_indiv, bins_zen_indiv], weights=weights, cmap=cmap, vmin=vmin_val)
        ax_st1.set_xlabel(f'Azimuth [deg]')
        ax_st1.set_ylabel(f'Zenith [deg]')
        ax_st1.set_title(f'Station 1 of Pair')
        ax_st1.grid(True)

        ax_st2.hist2d(recon_azi2, recon_zen2, bins=[bins_azi_indiv, bins_zen_indiv], weights=weights, cmap=cmap, vmin=vmin_val)
        ax_st2.set_xlabel(f'Azimuth [deg]')
        ax_st2.set_title(f'Station 2 of Pair')
        ax_st2.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename_indiv = os.path.join(save_folder, f'{type}_azivzen_individual_{pair_str}.png')
        plt.savefig(plot_filename_indiv)
        ic(f"Saved individual station plot to {plot_filename_indiv}")
        plt.close(fig_indiv)

        # --- Writing Event Details to File ---
        if ZenLim and AziLim and events_in_angle_range:
            output_filename = os.path.join(save_folder, f'{type}_event_details_{pair_str}_zen{ZenLim[0]}-{ZenLim[1]}_azi{AziLim[0]}-{AziLim[1]}.txt')
            with open(output_filename, 'w') as f:
                f.write(f"# Event details for coincident triggers between station pairs {pair_str} ({type} values)\n")
                f.write(f"# Zenith Range: {ZenLim} deg, Azimuth Range: {AziLim} deg\n")
                f.write("-" * 80 + "\n")
                
                for event, st1, st2 in events_in_angle_range: # Unpack the tuple
                    f.write(f"Event ID: {event.getEventID()} (matched by pair {st1}-{st2})\n")
                    f.write(f"  Core Position (x, y): ({event.coreas_x/units.m:.2f} m, {event.coreas_y/units.m:.2f} m)\n")
                    f.write(f"  True Energy: {event.energy/units.EeV:.4f} EeV\n")
                    f.write(f"  True Zenith: {np.rad2deg(event.zenith):.2f} deg\n")
                    f.write(f"  True Azimuth: {np.rad2deg(event.azimuth):.2f} deg\n")
                    
                    f.write(f"  Station {st1}:\n")
                    f.write(f"    Recon Zenith: {np.rad2deg(event.recon_zenith.get(st1, np.nan)):.2f} deg\n")
                    f.write(f"    Recon Azimuth: {np.rad2deg(event.recon_azimuth.get(st1, np.nan)):.2f} deg\n")
                    f.write(f"    SNR: {event.getSNR(st1):.2f}\n")

                    f.write(f"  Station {st2}:\n")
                    f.write(f"    Recon Zenith: {np.rad2deg(event.recon_zenith.get(st2, np.nan)):.2f} deg\n")
                    f.write(f"    Recon Azimuth: {np.rad2deg(event.recon_azimuth.get(st2, np.nan)):.2f} deg\n")
                    f.write(f"    SNR: {event.getSNR(st2):.2f}\n")
                    
                    f.write("-" * 80 + "\n")
            
            ic(f"Saved event details to {output_filename}")
        elif ZenLim and AziLim:
            ic(f"No events found within the specified angle limits for type '{type}'.")

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('HRASimulation/config.ini')

    numpy_folder = config['FOLDERS']['numpy_folder']
    save_folder = config['FOLDERS']['save_folder']
    
    coincidence_save_folder = os.path.join(save_folder, 'CoincidenceAnalysis/')
    os.makedirs(coincidence_save_folder, exist_ok=True)

    ic("Loading HRA event list...")
    HRAeventList = loadHRAfromH5(os.path.join(numpy_folder, 'HRAeventList.h5'))
    ic(f"Loaded {len(HRAeventList)} events.")

    # --- MODIFICATION: Define list of pairs to be analyzed together ---
    # Example: Analyze events that are coincident in EITHER [17, 113] OR [15, 118]
    # station_pairs_to_analyze = [[17, 113], [15, 118], [30, 119], [18, 114]] 
    # station_pairs_to_analyze = [[17, 13], [15, 18], [30, 19], [18, 14]] 

    all_pairs = [[[17, 13], [15, 18], [30, 19], [18, 14]] , [[17, 113], [15, 118], [30, 119], [18, 114]]]

    weight_key = '2_coincidence_wrefl'
    
    zenith_limits = [42.0, 48.0]
    azimuth_limits = [300.0, 325.0]

    # --- MODIFICATION: Single call to the new function with the list of pairs ---
    for station_pairs_to_analyze in all_pairs:
        ic(f"Analyzing station group: {station_pairs_to_analyze}")
        analyze_coincident_pairs( # Note the function name change
            HRAeventList=HRAeventList,
            station_pairs=station_pairs_to_analyze, # Pass the list of pairs
            weight_name=weight_key,
            save_folder=coincidence_save_folder,
            ZenLim=zenith_limits,
            AziLim=azimuth_limits,
            SNR_threshold=0.0,
        )

    ic("Analysis complete.")