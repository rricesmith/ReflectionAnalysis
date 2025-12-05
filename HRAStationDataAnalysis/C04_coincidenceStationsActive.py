import os
import sys
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import configparser
from icecream import ic
import itertools

# --- Configuration ---
STATIONS_TO_ANALYZE = [13, 14, 15, 17, 18, 19, 30]
LIVETIME_CUT_STAGE = "After L1 + Storm + Burst" # Use the most restrictive cut for "Active" status

def _load_pickle(filepath):
    """Loads data from a pickle file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            ic(f"Error loading pickle {filepath}: {e}")
    return None

def is_time_in_gti(time_unix, gti_list):
    """Checks if a timestamp is within any of the GTI intervals."""
    if not gti_list:
        return False
    for start, end in gti_list:
        if start <= time_unix <= end:
            return True
    return False

def load_station_positions(json_path):
    """Loads station positions from JSON file."""
    positions = {}
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Iterate through the "stations" dictionary
        # The keys are arbitrary strings, but the values contain "station_id"
        stations_data = data.get("stations", {})
        for entry in stations_data.values():
            st_id = entry.get("station_id")
            if st_id in STATIONS_TO_ANALYZE:
                positions[st_id] = {
                    "x": entry.get("pos_easting", 0.0),
                    "y": entry.get("pos_northing", 0.0)
                }
    except Exception as e:
        ic(f"Error loading station positions: {e}")
    return positions

def get_arrow_components(zenith_rad, azimuth_rad):
    """
    Converts Zenith and Azimuth to 2D arrow components (x, y).
    Assumes Azimuth 0 is North (positive y), increasing clockwise.
    Arrow length represents horizontal projection (sin(zenith)).
    """
    # Horizontal projection length
    r = np.sin(zenith_rad)
    
    # Azimuth 0 -> North (y=1, x=0)
    # Azimuth 90 -> East (y=0, x=1)
    # x = r * sin(azimuth)
    # y = r * cos(azimuth)
    
    dx = r * np.sin(azimuth_rad)
    dy = r * np.cos(azimuth_rad)
    
    return dx, dy

def main():
    ic.enable()
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Plot active stations for coincidence events.')
    parser.add_argument('--date', type=str, default=None, help="Date of data (e.g. 9.1.25)")
    parser.add_argument('--date_coincidence', type=str, default=None, help="Date of coincidence (e.g. 9.24.25)")
    parser.add_argument('--date_processing', type=str, default=None, help="Date for processing/plots")
    args = parser.parse_args()

    # --- Config Setup (similar to C03) ---
    config = configparser.ConfigParser()
    # Assuming running from root or HRAStationDataAnalysis
    config_path = 'config.ini'
    if not os.path.exists(config_path):
        config_path = os.path.join('HRAStationDataAnalysis', 'config.ini')
    
    # Defaults
    date_of_data = "9.1.25"
    date_of_coincidence = "9.24.25"
    date_of_process = "12.3.25n4"

    if os.path.exists(config_path):
        config.read(config_path)
        try:
            if config.has_option('PARAMETERS', 'date'):
                date_of_data = config['PARAMETERS']['date']
            if config.has_option('PARAMETERS', 'date_coincidence'):
                date_of_coincidence = config['PARAMETERS']['date_coincidence']
            if config.has_option('PARAMETERS', 'date_processing'):
                date_of_process = config['PARAMETERS']['date_processing']
        except KeyError as e:
            ic(f"Missing config parameter: {e}")
    else:
        ic(f"Config file not found at {config_path}. Using defaults.")

    # Override with args if provided
    if args.date: date_of_data = args.date
    if args.date_coincidence: date_of_coincidence = args.date_coincidence
    if args.date_processing: date_of_process = args.date_processing

    ic(f"Date Data: {date_of_data}")
    ic(f"Date Coincidence: {date_of_coincidence}")
    ic(f"Date Processing: {date_of_process}")

    base_project_path = 'HRAStationDataAnalysis'
    if not os.path.exists(base_project_path):
        # Maybe we are inside HRAStationDataAnalysis
        if os.path.exists('StationData'):
            base_project_path = '.'
        else:
            # Try to find where we are
            if os.path.basename(os.getcwd()) == 'HRAStationDataAnalysis':
                base_project_path = '.'
            else:
                ic("Could not locate HRAStationDataAnalysis folder. Assuming '.'")
                base_project_path = '.'

    base_processed_data_dir = os.path.join(base_project_path, "StationData", "processedNumpyData")
    processed_data_dir_for_date = os.path.join(base_processed_data_dir, date_of_data)
    
    # --- Load Station Positions ---
    # Path to HRAStationLayoutForCoREAS.json
    # Assuming workspace structure: root/HRASimulation/HRAStationLayoutForCoREAS.json
    # and root/HRAStationDataAnalysis/C04...
    # So from base_project_path (HRAStationDataAnalysis), go up one level then to HRASimulation
    
    layout_json_path = os.path.join(base_project_path, "..", "HRASimulation", "HRAStationLayoutForCoREAS.json")
    if not os.path.exists(layout_json_path):
        # Try absolute path or relative to current dir if running from root
        layout_json_path = os.path.join("HRASimulation", "HRAStationLayoutForCoREAS.json")
    
    if not os.path.exists(layout_json_path):
        ic(f"CRITICAL: Station layout JSON not found at {layout_json_path}")
        return

    station_positions = load_station_positions(layout_json_path)
    ic(f"Loaded positions for {len(station_positions)} stations.")

    # --- Load Coincidence Data ---
    prefixes = [
        f"{date_of_coincidence}_CoincidenceDatetimes_passing_cuts",
        f"{date_of_coincidence}_CoincidenceDatetimes",
    ]
    suffixes = [
        "with_all_params_recalcZenAzi_calcPol.pkl",
        "with_all_params_recalcZenAzi.pkl",
        "with_all_params.pkl",
    ]
    
    coincidence_file = None
    for p in prefixes:
        for s in suffixes:
            candidate = os.path.join(processed_data_dir_for_date, f"{p}_{s}")
            if os.path.exists(candidate):
                coincidence_file = candidate
                break
        if coincidence_file: break
    
    if not coincidence_file:
        ic("CRITICAL: No coincidence file found.")
        return

    ic(f"Loading coincidence events from: {coincidence_file}")
    events_dict = _load_pickle(coincidence_file)
    if not events_dict:
        ic("Failed to load events dictionary.")
        return

    # --- Load Active Periods (GTIs) ---
    # Path: HRAStationDataAnalysis/plots/{date_of_process}/Station{st_id}/livetime_data/livetime_gti_St{st_id}_{date_of_data}.pkl
    # Note: C00 uses date_filter (which is date_of_data) for the filename, and date_save (date_of_process) for the folder.
    
    station_gtis = {}
    plot_folder_base = os.path.join(base_project_path, 'plots', date_of_process)
    
    for st_id in STATIONS_TO_ANALYZE:
        gti_path = os.path.join(plot_folder_base, f"Station{st_id}", "livetime_data", f"livetime_gti_St{st_id}_{date_of_data}.pkl")
        data = _load_pickle(gti_path)
        if data and LIVETIME_CUT_STAGE in data:
            # data[stage] is (livetime_seconds, gti_list)
            station_gtis[st_id] = data[LIVETIME_CUT_STAGE][1]
            ic(f"Loaded GTIs for Station {st_id}")
        else:
            ic(f"Warning: No GTI data found for Station {st_id} at {gti_path}")
            station_gtis[st_id] = []

    # --- Output Directory ---
    output_dir = os.path.join(plot_folder_base, "StationActivityMaps")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Process Events ---
    ic(f"Processing {len(events_dict)} events...")
    
    for event_id, event_details in events_dict.items():
        # Get Event Time
        # event_details might have 'unix_time' or we infer from key if it's not there?
        # Usually event_details has 'unix_time' or similar.
        # Let's check C03... it uses event_details.get("unix_time") or similar?
        # Actually C03 iterates and gets timestamps.
        # Let's assume 'unix_time' is in event_details.
        
        event_time = event_details.get("unix_time")
        if event_time is None:
            # Try to get from stations
            stations_data = event_details.get("stations", {})
            if stations_data:
                first_st = next(iter(stations_data.values()))
                event_time = first_st.get("unix_time")
        
        if event_time is None:
            ic(f"Skipping event {event_id}: No timestamp found.")
            continue
            
        event_dt = datetime.datetime.fromtimestamp(event_time)
        
        # Determine Status
        triggered_stations = []
        active_stations = []
        offline_stations = []
        
        event_stations_data = event_details.get("stations", {})
        # event_stations_data keys are strings "13", "14" etc.
        
        # Map string keys to int for easier lookup
        event_triggered_ids = []
        for k in event_stations_data.keys():
            try:
                event_triggered_ids.append(int(k))
            except ValueError:
                pass
        
        print(f"\n--- Event {event_id} at {event_dt} ---")
        
        for st_id in STATIONS_TO_ANALYZE:
            status = "Offline"
            if st_id in event_triggered_ids:
                status = "Triggered"
                triggered_stations.append(st_id)
            elif is_time_in_gti(event_time, station_gtis.get(st_id, [])):
                status = "Active"
                active_stations.append(st_id)
            else:
                status = "Offline"
                offline_stations.append(st_id)
            
            print(f"Station {st_id}: {status}")

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot Offline
        if offline_stations:
            x = [station_positions[s]["x"] for s in offline_stations if s in station_positions]
            y = [station_positions[s]["y"] for s in offline_stations if s in station_positions]
            ax.scatter(x, y, c='lightgrey', s=200, label='Offline', edgecolors='grey')
            for s in offline_stations:
                if s in station_positions:
                    ax.text(station_positions[s]["x"], station_positions[s]["y"], str(s), ha='center', va='center', fontsize=8)

        # Plot Active
        if active_stations:
            x = [station_positions[s]["x"] for s in active_stations if s in station_positions]
            y = [station_positions[s]["y"] for s in active_stations if s in station_positions]
            ax.scatter(x, y, c='lightgreen', s=200, label='Active (Not Triggered)', edgecolors='green')
            for s in active_stations:
                if s in station_positions:
                    ax.text(station_positions[s]["x"], station_positions[s]["y"], str(s), ha='center', va='center', fontsize=8)

        # Plot Triggered
        if triggered_stations:
            x = [station_positions[s]["x"] for s in triggered_stations if s in station_positions]
            y = [station_positions[s]["y"] for s in triggered_stations if s in station_positions]
            ax.scatter(x, y, c='red', s=200, label='Triggered', edgecolors='darkred')
            
            for s in triggered_stations:
                if s in station_positions:
                    ax.text(station_positions[s]["x"], station_positions[s]["y"], str(s), ha='center', va='center', fontsize=8, color='white', fontweight='bold')
                    
                    # Draw Arrow
                    st_data = event_stations_data.get(str(s), {})
                    zen = st_data.get("Zenith")
                    azi = st_data.get("Azimuth")
                    
                    if zen is not None and azi is not None:
                        dx, dy = get_arrow_components(zen, azi)
                        # Scale arrow
                        arrow_len = 100 # meters, arbitrary scale for visibility
                        ax.arrow(station_positions[s]["x"], station_positions[s]["y"], 
                                 dx * arrow_len, dy * arrow_len, 
                                 head_width=20, head_length=30, fc='blue', ec='blue', alpha=0.7)

        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title(f"Event {event_id}\n{event_dt}")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal')
        
        # Save Plot
        plot_filename = f"Event_{event_id}_ActivityMap.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close(fig)
        # ic(f"Saved plot: {plot_path}")

    ic("Processing complete.")

if __name__ == "__main__":
    main()
