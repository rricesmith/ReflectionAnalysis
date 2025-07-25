import os
import numpy as np
import matplotlib.pyplot as plt
import configparser
from icecream import ic
import glob
from scipy.stats import binned_statistic_2d
# --- MODIFICATION: Import LogNorm for color scaling and Rectangle for plotting ---
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


# Import the HRAevent class from the provided file
from HRAEventObjectForAreaSensitivity import HRAevent

# Base station locations are used, and IDs > 100 (e.g., 113) map to their base (e.g., 13)
BASE_STATION_LOCATIONS = {
    13: [1044.4451, -91.337], 14: [610.4495, 867.118], 15: [-530.8272, -749.382],
    17: [503.6394, -805.116], 18: [0, 0], 19: [-371.9322, 950.705],
    30: [-955.2426, 158.383], 32: [463.6215, -893.988], 52: [436.7442, 168.904]
}

def load_hra_events_from_npy(filepath):
    """
    Loads a list of HRAevent objects from a .npy file.
    """
    return np.load(filepath, allow_pickle=True)

def get_max_trigger_sigma_for_event(event, station_ids):
    """
    Finds the highest sigma value that triggered for any station in the event.
    """
    station_set = set(station_ids)
    for sigma in event.trigger_sigmas:
        sigma_set = set(event.station_triggers[sigma])
        if station_set.issubset(sigma_set):
            return sigma  # Return the first sigma that includes all stations
    # If one of the stations never triggered, return 0
    return 0

def plot_sigma_sensitivity(event_list, station_ids, savename, vmin_plot=4):
    """
    Generates and saves a 2D histogram showing the highest trigger sigma
    for each x-y core position, with a configurable minimum plotting value.
    """
    if not event_list.any():
        ic(f"Event list for {savename} is empty. Skipping.")
        return

    # 1. Extract data from event list
    x_coords, y_coords, sigmas = [], [], []
    for event in event_list:
        x, y = event.getCoreasPosition()
        max_sigma = get_max_trigger_sigma_for_event(event, station_ids)
        x_coords.append(x)
        y_coords.append(y)
        sigmas.append(max_sigma)

    first_event = event_list[0]
    energy_eV = first_event.energy
    zenith_deg = np.rad2deg(first_event.zenith)
    azimuth_deg = np.rad2deg(first_event.azimuth)

    # 2. Create the 2D binned data for the plot
    bins = 100
    stat, x_edge, y_edge, _ = binned_statistic_2d(
        x_coords, y_coords, values=sigmas, statistic='max', bins=bins
    )
    stat_plot = np.nan_to_num(stat.T).astype(float)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- MODIFICATION: Setup for log scale and custom 'under' value display ---
    cmap = plt.get_cmap('viridis')
    cmap.set_bad('white')      # Sets color for NaN values (where sigma was 0)
    cmap.set_under('none')     # Makes cells with sigma < vmin_plot transparent

    # Set data with sigma=0 to NaN to be colored white by set_bad()
    plot_data = stat_plot.copy()
    plot_data[plot_data == 0] = np.nan

    # Create the logarithmic color normalization
    norm = LogNorm(vmin=vmin_plot, vmax=50)

    im = ax.pcolormesh(x_edge, y_edge, plot_data, shading='auto', cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Highest Triggering Sigma (Log Scale)', fontsize=12)

    # --- MODIFICATION: Add hollow rectangles for values below vmin_plot ---
    below_vmin_mask = (stat_plot > 0) & (stat_plot < vmin_plot)
    if np.any(below_vmin_mask):
        for i, j in zip(*np.where(below_vmin_mask)):
            y, x = y_edge[i], x_edge[j]
            height, width = y_edge[i+1] - y, x_edge[j+1] - x
            ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor='gray', lw=0.5))

    ax.set_aspect('equal')
    ax.set_xlabel('Core Position X [m]', fontsize=12)
    ax.set_ylabel('Core Position Y [m]', fontsize=12)
    ax.set_title(f'Trigger Sensitivity Map (Min Sigma: {vmin_plot})\nFile: {os.path.basename(savename)}', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Dynamic station plotting
    station_colors = ['blue', 'green', 'purple', 'orange']
    for i, st_id in enumerate(station_ids):
        pos = BASE_STATION_LOCATIONS.get(st_id % 100)
        if pos:
            ax.plot(pos[0], pos[1], marker='^', markersize=12,
                    color=station_colors[i % len(station_colors)],
                    markeredgecolor='k', label=f'Station {st_id}')

    # 4. Add text and arrow overlays
    info_text = (f'Energy: {energy_eV:.2e} eV\n'
                 f'Zenith: {zenith_deg:.2f}°\n'
                 f'Azimuth: {azimuth_deg:.2f}°')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))

    azi_rad = np.deg2rad(azimuth_deg)
    arrow_dx, arrow_dy = -np.cos(azi_rad), -np.sin(azi_rad)
    ax.arrow(0.85, 0.85, -0.08 * arrow_dx, -0.08 * arrow_dy,
             transform=ax.transAxes,
             width=0.01, head_width=0.03, head_length=0.04,
             fc='red', ec='red', label='Shower Azimuth')

    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(savename)
    ic(f"Saved sensitivity map to {savename}")
    plt.close(fig)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('HRASimulation/config.ini')

    sim_folder = config['FOLDERS']['area_sim_folder']
    save_folder = config['FOLDERS']['save_folder']

    sensitivity_save_folder = os.path.join(save_folder, 'SensitivityAnalysis/')
    os.makedirs(sensitivity_save_folder, exist_ok=True)

    search_path = os.path.join(sim_folder, '*_HRAeventList.npy')
    file_list = glob.glob(search_path)

    if not file_list:
        ic(f"No '*_HRAeventList.npy' files found in {sim_folder}. Exiting.")
    else:
        ic(f"Found {len(file_list)} event files to process.")

    BL_BL_stations = [13, 17]
    BL_RCR_stations = [17, 113]

    for f in file_list:
        ic(f"Processing file: {os.path.basename(f)}")

        basename = os.path.basename(f).replace('_HRAeventList.npy', '')

        HRAeventList = load_hra_events_from_npy(f)

        savename_1 = os.path.join(sensitivity_save_folder, f'SensitivityMap_{basename}_stns13_17.png')
        plot_sigma_sensitivity(HRAeventList, BL_BL_stations, savename_1)

        savename_2 = os.path.join(sensitivity_save_folder, f'SensitivityMap_{basename}_stns17_113.png')
        plot_sigma_sensitivity(HRAeventList, BL_RCR_stations, savename_2)

        savename_3 = os.path.join(sensitivity_save_folder, f'SensitivityMap_{basename}_stns13.png')
        plot_sigma_sensitivity(HRAeventList, [13], savename_3)
        savename_4 = os.path.join(sensitivity_save_folder, f'SensitivityMap_{basename}_stns17.png')
        plot_sigma_sensitivity(HRAeventList, [17], savename_4)
        savename_5 = os.path.join(sensitivity_save_folder, f'SensitivityMap_{basename}_stns113.png')
        plot_sigma_sensitivity(HRAeventList, [113], savename_5)


    ic("All files processed.")