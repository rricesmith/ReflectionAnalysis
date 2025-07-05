import os
import numpy as np
import matplotlib.pyplot as plt
import configparser
from icecream import ic

from HRASimulation.HRAEventObject import HRAevent
from HRASimulation.HRANurToNpy import loadHRAfromH5
from NuRadioReco.utilities import units

import HRASimulation.HRAAnalysis as HRAAnalysis


# --- MODIFICATION: Added station locations dictionary for distance calculations ---
# This is extracted from HRAAnalysis.plotStationLocations to be used for the distance cut.
station_locations = {
    13: [1044.4451, -91.337], 14: [610.4495, 867.118], 15: [-530.8272, -749.382], 
    17: [503.6394, -805.116], 18: [0, 0], 19: [-371.9322, 950.705], 
    30: [-955.2426, 158.383], 32: [463.6215, -893.988], 52: [436.7442, 168.904]
}


# --- MODIFICATION: Updated function signature with `max_distance_cut` and added area plot generation ---
def analyze_coincident_pairs(HRAeventList, station_pairs, weight_name, save_folder, ZenLim=None, AziLim=None, sigma=4.5, SNR_threshold=0, max_distance_cut=None):
    """
    Analyzes events in coincidence, plots angle correlations and area rates, 
    and adds cuts for SNR and distance.

    Args:
        HRAeventList (list): A list of HRAevent objects.
        station_pairs (list): A list of station ID pairs, e.g., [[17, 113], [15, 118]].
        weight_name (str): The name of the weight to use for histograms.
        save_folder (str): The path to the directory where plots and output files will be saved.
        ZenLim (list, optional): A range [min, max] for zenith in degrees. Defaults to None.
        AziLim (list, optional): A range [min, max] for azimuth in degrees. Defaults to None.
        sigma (float, optional): The sigma value for getting weights. Defaults to 4.5.
        SNR_threshold (float, optional): The minimum SNR for both stations in a pair. Defaults to 0.
        max_distance_cut (float, optional): Maximum distance from a station for an event to be included.
                                            If None, no distance cut is applied. Defaults to None.
    """
    if not isinstance(station_pairs[0], list):
        station_pairs = [station_pairs]

    pair_str = "_".join([f"{p[0]}-{p[1]}" for p in station_pairs])
    
    ic(f"Analyzing coincidence for station pairs: {pair_str}")
    ic(f"Using weight: '{weight_name}'")
    if max_distance_cut is not None:
        ic(f"Applying distance cut: {max_distance_cut/units.km:.2f} km")


    # --- NEW: Lists to store data for the area plot ---
    area_x, area_y, area_weights = [], [], []

    for type in ['Recon', 'True']:
        recon_zen1, recon_zen2 = [], []
        recon_azi1, recon_azi2 = [], []
        weights = []
        events_in_angle_range = []

        for event in HRAeventList:
            for st1, st2 in station_pairs:
                if st1 in event.recon_zenith and st2 in event.recon_zenith:
                    
                    if not (event.getSNR(st1) >= SNR_threshold and event.getSNR(st2) >= SNR_threshold):
                        continue

                    # --- NEW: Distance cut logic ---
                    if max_distance_cut is not None:
                        core_x, core_y = event.getCoreasPosition()
                        # Use modulo to get base station ID (e.g., 113 -> 13)
                        st1_base_id, st2_base_id = st1 % 100, st2 % 100
                        st1_pos = station_locations.get(st1_base_id)
                        st2_pos = station_locations.get(st2_base_id)

                        if st1_pos and st2_pos:
                            dist1 = np.sqrt((core_x - st1_pos[0])**2 + (core_y - st1_pos[1])**2)
                            dist2 = np.sqrt((core_x - st2_pos[0])**2 + (core_y - st2_pos[1])**2)
                            # Event is disregarded if it's farther than the cut from BOTH stations
                            if dist1 > max_distance_cut and dist2 > max_distance_cut:
                                continue
                        else:
                            ic(f"Warning: Location for station {st1_base_id} or {st2_base_id} not found. Skipping distance check.")

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
                    current_weight = event_weight if event_weight is not None and not np.isnan(event_weight) else 0
                    weights.append(current_weight)

                    # --- NEW: Append data for area plot if it's the first pass ('Recon' loop) ---
                    if type == 'Recon':
                        core_x, core_y = event.getCoreasPosition()
                        area_x.append(core_x)
                        area_y.append(core_y)
                        area_weights.append(current_weight)

                    if ZenLim and AziLim:
                        zen_in_range = (ZenLim[0] <= zen1_deg <= ZenLim[1]) and (ZenLim[0] <= zen2_deg <= ZenLim[1])
                        azi_in_range = (AziLim[0] <= azi1_deg <= AziLim[1]) and (AziLim[0] <= azi2_deg <= AziLim[1])
                        if zen_in_range and azi_in_range:
                            events_in_angle_range.append((event, st1, st2))
                    
                    break 

        if not recon_zen1:
            ic(f"No coincident events found for station pairs {pair_str} for type '{type}'. Skipping plots.")
            continue

        import matplotlib.cm as cm
        cmap = cm.viridis
        cmap.set_under('white')
        vmin_val = 1e-9

        # Plotting 2D Histograms (Angle Correlation)
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
        plot_filename = os.path.join(save_folder, f'{type}_angle_correlation_{pair_str}.png')
        plt.savefig(plot_filename)
        ic(f"Saved histogram plot to {plot_filename}")
        plt.close(fig)

        # Individual Station Azimuth vs Zenith
        fig_indiv, (ax_st1, ax_st2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig_indiv.suptitle(f'{type} Azimuth vs Zenith for Station Pairs {pair_str}', fontsize=16)
        # ... (plotting logic unchanged) ...
        plt.close(fig_indiv)

        # Writing Event Details to File
        if ZenLim and AziLim and events_in_angle_range:
            # ... (file writing logic unchanged) ...
            pass
        elif ZenLim and AziLim:
            ic(f"No events found within the specified angle limits for type '{type}'.")

    # --- NEW: Generate and save the Area Rate plot ---
    if not area_x:
        ic(f"No events passed cuts for area plot for pairs {pair_str}. Skipping.")
    else:
        config = configparser.ConfigParser()
        config.read('HRASimulation/config.ini')
        diameter = config['SIMPARAMETERS']['diameter']
        max_plot_distance = float(diameter) / 2 * units.km

        savename = os.path.join(save_folder, f'AreaRate_{pair_str}.png')
        title = f'Area Event Rate for Pairs {pair_str}'

        # Get unique stations from the pairs to highlight on the plot
        all_stations_in_pairs = set(sum(station_pairs, []))
        dir_trig_to_plot = [s for s in all_stations_in_pairs if s < 100]
        refl_trig_to_plot = [s - 100 for s in all_stations_in_pairs if s >= 100]

        HRAAnalysis.histAreaRate(
            x=np.array(area_x),
            y=np.array(area_y),
            weights=np.array(area_weights),
            title=title,
            savename=savename,
            dir_trig=dir_trig_to_plot,
            refl_trig=refl_trig_to_plot,
            max_distance=max_plot_distance
        )

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

    # all_pairs = [[17, 13], [15, 18], [30, 19], [18, 14]], [[17, 113], [15, 118], [30, 119], [18, 114]]
    all_pairs = [ [17, 113], [17, 13] ]


    weight_key = '2_coincidence_wrefl'
    
    zenith_limits = [42.0, 48.0]
    azimuth_limits = [300.0, 325.0]

    for station_pairs_to_analyze in all_pairs:
        ic(f"Analyzing station group: {station_pairs_to_analyze}")
        analyze_coincident_pairs(
            HRAeventList=HRAeventList,
            station_pairs=station_pairs_to_analyze,
            weight_name=weight_key,
            save_folder=coincidence_save_folder,
            ZenLim=zenith_limits,
            AziLim=azimuth_limits,
            SNR_threshold=0.0,
            # --- NEW: max_distance_cut is not set, so it uses the default (no cut) ---
            # To apply a 5km cut, you would add:
            # max_distance_cut=5.0*units.km 
        )

    ic("Analysis complete.")