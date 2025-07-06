import os
import numpy as np
import matplotlib.pyplot as plt
import configparser
from icecream import ic
import glob
from scipy.stats import binned_statistic_2d

# Assuming HRAEventObject is available, which was used to create the .npy files.
# We'll re-define a minimal version here for clarity on what attributes are used.
class HRAevent:
    def __init__(self, event_data):
        self.__dict__.update(event_data)

    def getCoreasPosition(self):
        return self.coreas_x, self.coreas_y

    def get_max_trigger_sigma(self, station_ids):
        max_sigma = 0
        if hasattr(self, 'station_parameters'):
            for station_id in station_ids:
                # The station_parameters attribute is a dictionary {station_id: {param: value}}
                station_params = self.station_parameters.get(str(station_id), {})
                sigma = station_params.get('highest_trigger_sigma', 0)
                if sigma > max_sigma:
                    max_sigma = sigma
        return max_sigma


def load_hra_events_from_npy(filepath):
    """Loads a list of HRAevent objects from a .npy file."""
    raw_event_list = np.load(filepath, allow_pickle=True)
    # The .npy file saves the __dict__ of each object. We reconstruct the objects.
    return [HRAevent(ev_dict) for ev_dict in raw_event_list]

def plot_sigma_sensitivity(event_list, station_ids, savename):
    """
    Generates and saves a 2D histogram showing the highest trigger sigma
    for each x-y core position.
    """
    if not event_list:
        ic(f"Event list for {savename} is empty. Skipping.")
        return

    # --- 1. Extract data from event list ---
    x_coords, y_coords, sigmas = [], [], []

    for event in event_list:
        x, y = event.getCoreasPosition()
        max_sigma = event.get_max_trigger_sigma(station_ids)

        x_coords.append(x)
        y_coords.append(y)
        sigmas.append(max_sigma)

    # Get shower parameters from the first event (constant for the sim)
    first_event = event_list[0]
    energy_eV = first_event.energy
    zenith_deg = np.rad2deg(first_event.zenith)
    azimuth_deg = np.rad2deg(first_event.azimuth)

    # --- 2. Create the 2D binned data for the plot ---
    # Using binned_statistic_2d to get the MAX sigma in each bin, not the sum
    # This avoids issues if multiple points fall in one bin.
    bins = 100 # Adjust binning as needed
    stat, x_edge, y_edge, _ = binned_statistic_2d(
        x_coords, y_coords, values=sigmas, statistic='max', bins=bins
    )
    # We need to transpose the result for pcolormesh
    stat = np.nan_to_num(stat.T)

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use pcolormesh to plot the binned data
    im = ax.pcolormesh(x_edge, y_edge, stat, shading='auto', cmap='viridis')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Highest Triggering Sigma', fontsize=12)

    ax.set_aspect('equal')
    ax.set_xlabel('Core Position X [m]', fontsize=12)
    ax.set_ylabel('Core Position Y [m]', fontsize=12)
    ax.set_title(f'Trigger Sensitivity Map\nFile: {os.path.basename(savename)}', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Plot station locations for context
    station_locations = {13: [1044.4451, -91.337], 17: [503.6394, -805.116]}
    for st_id, pos in station_locations.items():
        ax.plot(pos[0], pos[1], 'r^', markersize=10, label=f'Station {st_id}')

    # --- 4. Add text and arrow overlays ---
    
    # Add text box with event parameters
    info_text = (f'Energy: {energy_eV:.2e} eV\n'
                 f'Zenith: {zenith_deg:.2f}°\n'
                 f'Azimuth: {azimuth_deg:.2f}°')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))

    # Add arrow for azimuth direction
    # Azimuth is from North, so we convert to a standard angle for plotting
    azi_rad = np.deg2rad(azimuth_deg)
    arrow_dx = np.sin(azi_rad)
    arrow_dy = np.cos(azi_rad)
    
    ax.arrow(0.95, 0.95, -0.08 * arrow_dx, -0.08 * arrow_dy,
             transform=ax.transAxes,
             width=0.01, head_width=0.03, head_length=0.04,
             fc='red', ec='red', label='Shower Azimuth')

    ax.legend(loc='lower right')
    plt.tight_layout()

    # Save the figure
    plt.savefig(savename)
    ic(f"Saved sensitivity map to {savename}")
    plt.close(fig)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    # Ensure you have a config.ini with correct folder paths
    config.read('HRASimulation/config.ini') 
    
    area_sim_folder = config['FOLDERS']['area_sim_folder']
    save_folder = config['FOLDERS']['save_folder']
    
    sensitivity_save_folder = os.path.join(save_folder, 'SensitivityAnalysis/')
    os.makedirs(sensitivity_save_folder, exist_ok=True)

    # Find all HRAeventList files in the simulation output folder
    search_path = os.path.join(area_sim_folder, '*_HRAeventList.npy')
    file_list = glob.glob(search_path)
    
    if not file_list:
        ic(f"No '*_HRAeventList.npy' files found in {area_sim_folder}. Exiting.")
    else:
        ic(f"Found {len(file_list)} event files to process.")

    BL_BL_stations = [13, 17]
    BL_RCR_stations = [17, 113]

    for f in file_list:
        ic(f"Processing file: {os.path.basename(f)}")
        
        # Define a unique name for the output plot
        basename = os.path.basename(f).replace('_HRAeventList.npy', '')
        
        # Load the event data
        HRAeventList = load_hra_events_from_npy(f)
        
        # Generate and save the plot
        savename = os.path.join(sensitivity_save_folder, f'SensitivityMap_{basename}_13_17.png')
        plot_sigma_sensitivity(HRAeventList, BL_BL_stations, savename)
        savename = os.path.join(sensitivity_save_folder, f'SensitivityMap_{basename}_17_113.png')
        plot_sigma_sensitivity(HRAeventList, BL_RCR_stations, savename)


    ic("All files processed.")