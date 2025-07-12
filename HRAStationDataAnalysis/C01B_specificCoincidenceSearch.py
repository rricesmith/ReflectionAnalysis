import numpy as np
import os
import glob
import pickle
import configparser
from datetime import datetime, timedelta

def find_all_station_parameters(base_path):
    """Finds all unique parameter names (e.g., 'Times', 'Traces') from filenames."""
    param_names = set()
    if not os.path.exists(base_path):
        return []
    for f in os.listdir(base_path):
        if f.endswith('.npy'):
            parts = f.split('_')
            if len(parts) > 2:
                param_name = parts[2].replace('.npy', '')
                if param_name not in ['Times', 'EventIDs']: # Already handled
                    param_names.add(param_name)
    return list(param_names)

def get_failed_cuts(original_index, station_cuts_data):
    """
    Checks an event's original index against all available cuts and returns
    a list of the names of the cuts that the event failed.

    Args:
        original_index (int): The index of the event in the original, uncut data file.
        station_cuts_data (dict): A dictionary where keys are cut names and values are
                                  boolean numpy arrays.

    Returns:
        list: A list of strings, where each string is the name of a cut that the
              event did not pass. Returns an empty list if the event passed all cuts.
    """
    failed_cuts = []
    for cut_name, cut_mask in station_cuts_data.items():
        if original_index < len(cut_mask) and not cut_mask[original_index]:
            failed_cuts.append(cut_name)
    return failed_cuts

def find_events_around_time(search_datetime, date, window_seconds=1.0):
    """
    Searches for all events from all stations within a specified time window
    around a given search datetime.

    This function loads all events, including those that would be cut, and
    annotates them with information about which cuts they failed.

    Args:
        search_datetime (datetime): The central time for the search.
        date (str): The date folder to process (e.g., '20231026').
        window_seconds (float): The number of seconds before and after the
                                search_datetime to include events from.

    Returns:
        A dictionary representing the coincidence event if any events are found,
        otherwise None. The dictionary contains the search parameters and a
        list of all event data found within the window.
    """
    station_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'nurFiles', date)
    cuts_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'cuts', date)
    
    if not os.path.exists(station_data_folder):
        print(f"Error: Data folder not found for date {date} at {station_data_folder}")
        return None

    search_start_time = search_datetime - timedelta(seconds=window_seconds)
    search_end_time = search_datetime + timedelta(seconds=window_seconds)

    station_ids = [13, 14, 15, 17, 18, 19, 30]
    all_found_events = []
    
    # Dynamically find all available parameter types (e.g., Traces, etc.)
    extra_params = find_all_station_parameters(station_data_folder)
    print(f"Searching for extra parameters: {extra_params}")

    for station_id in station_ids:
        # Load time and event ID data for the station
        time_files = sorted(glob.glob(os.path.join(station_data_folder, f'{date}_Station{station_id}_Times*.npy')))
        if not time_files:
            continue

        times_list = [np.load(f) for f in time_files]
        event_ids_list = [np.load(f.replace('_Times', '_EventIDs')) for f in time_files]
        
        # Load extra parameter data
        extra_param_data = {}
        for param in extra_params:
            param_files = sorted(glob.glob(os.path.join(station_data_folder, f'{date}_Station{station_id}_{param}*.npy')))
            if param_files:
                try:
                    extra_param_data[param] = [np.load(f, allow_pickle=True) for f in param_files]
                except Exception as e:
                    print(f"Warning: Could not load parameter '{param}' for station {station_id}. Skipping. Error: {e}")


        # Load all cuts for the station into a single dictionary
        station_cuts_data = {}
        cuts_file = os.path.join(cuts_data_folder, f'{date}_Station{station_id}_Cuts.npy')
        if os.path.exists(cuts_file):
            station_cuts_data = np.load(cuts_file, allow_pickle=True).item()

        # Iterate through each file part for the station
        for i in range(len(times_list)):
            file_times = times_list[i]
            file_event_ids = event_ids_list[i]
            
            # Find indices of events within the desired time window for this file part
            # Note: file_times are floating point seconds since epoch. search_start_time is a datetime object.
            start_epoch = search_start_time.timestamp()
            end_epoch = search_end_time.timestamp()
            
            indices_in_window = np.where((file_times >= start_epoch) & (file_times <= end_epoch))[0]

            for original_idx in indices_in_window:
                event_data = {
                    "station_id": station_id,
                    "original_index_in_file": original_idx,
                    "timestamp_sec": file_times[original_idx],
                    "datetime_utc": datetime.utcfromtimestamp(file_times[original_idx]),
                    "event_id": file_event_ids[original_idx],
                    "failed_cuts": get_failed_cuts(original_idx, station_cuts_data)
                }

                # Add data from extra parameters
                for param, param_list in extra_param_data.items():
                    if i < len(param_list) and original_idx < len(param_list[i]):
                        event_data[param] = param_list[i][original_idx]
                    else:
                        event_data[param] = 'Data not available'
                
                all_found_events.append(event_data)

    if all_found_events:
        # Sort events by time
        all_found_events.sort(key=lambda x: x['timestamp_sec'])
        
        coincidence_result = {
            "search_datetime_utc": search_datetime.isoformat(),
            "window_seconds": window_seconds,
            "numEvents": len(all_found_events),
            "stations_present": sorted(list(set(e['station_id'] for e in all_found_events))),
            "events": all_found_events
        }
        return coincidence_result
    
    return None

def generate_text_report(all_coincidences, output_path):
    """Writes a detailed, human-readable report of all found coincidences."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("            TARGETED COINCIDENCE EVENT SEARCH REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        if not all_coincidences:
            f.write("No coincidence events were found for the specified search times.\n")
            return

        for i, coincidence in enumerate(all_coincidences):
            f.write("-" * 80 + "\n")
            f.write(f"COINCIDENCE GROUP {i+1}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Search Center Time (UTC): {coincidence['search_datetime_utc']}\n")
            f.write(f"Search Window: +/- {coincidence['window_seconds']} seconds\n")
            f.write(f"Total Events Found: {coincidence['numEvents']}\n")
            f.write(f"Stations Present: {coincidence['stations_present']}\n\n")

            for event_idx, event in enumerate(coincidence['events']):
                f.write(f"--- Event {event_idx + 1} of {coincidence['numEvents']} ---\n")
                # Print all key-value pairs from the event dictionary
                for key, value in event.items():
                    if isinstance(value, np.ndarray):
                        # For numpy arrays (like traces), provide a summary
                        f.write(f"  {key}: array(shape={value.shape}, dtype={value.dtype})\n")
                        # You could optionally print the whole array if it's small:
                        # f.write(f"    Values: {np.array2string(value, max_line_width=120)}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
            f.write("\n")
        print(f"Text report saved to {output_path}")


if __name__ == '__main__':
    # --- CONFIGURATION ---
    
    # 1. Load configuration from .ini file
    config = configparser.ConfigParser() 
    config.read(os.path.join('HRAStationDataAnalysis', 'config.ini')) 
    SEARCH_DATE = config['PARAMETERS']['date']
    date_processing = config['PARAMETERS']['date_processing']
    
    # 2. Set the times you want to search for.
    #    Times should be in UTC. Use datetime(year, month, day, hour, minute, second, microsecond)
    SEARCH_TIMES_UTC = [
        datetime(2015, 12, 21, 21, 46, 11),
        datetime(2017, 2, 16, 19, 9, 51)
    ]

    # 3. Set the coincidence window in seconds.
    #    The search will look for events +/- this many seconds around each SEARCH_TIME.
    COINCIDENCE_WINDOW_SECONDS = 4.0

    # 4. Set the base name for the output files using the prefix from config.
    OUTPUT_BASENAME = f"{date_processing}_targeted_search"

    # --- END OF CONFIGURATION ---

    print(f"Configuration loaded: Searching date '{SEARCH_DATE}', Output prefix '{date_processing}'")

    # Create a directory for the output if it doesn't exist
    output_dir = os.path.join('HRAStationDataAnalysis', 'StationData', 'processedNumpyData', SEARCH_DATE)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path_npy = os.path.join(output_dir, f"{OUTPUT_BASENAME}.npy")
    output_path_txt = os.path.join(output_dir, f"{OUTPUT_BASENAME}_report.txt")
    
    all_found_coincidences = []

    print("Starting targeted coincidence search...")
    for search_time in SEARCH_TIMES_UTC:
        print(f"\nSearching around {search_time.isoformat()} UTC...")
        
        result = find_events_around_time(
            search_datetime=search_time,
            date=SEARCH_DATE,
            window_seconds=COINCIDENCE_WINDOW_SECONDS
        )
        
        if result:
            print(f"  > Found a coincidence group with {result['numEvents']} events from stations {result['stations_present']}.")
            all_found_coincidences.append(result)
        else:
            print(f"  > No events found within the window for this search time.")

    if all_found_coincidences:
        # Save the structured data to a numpy/pickle file
        with open(output_path_npy, 'wb') as f:
            pickle.dump(all_found_coincidences, f)
        print(f"\nCoincidence data saved to {output_path_npy}")

        # Generate the detailed text report
        generate_text_report(all_found_coincidences, output_path_txt)
    else:
        print("\nSearch complete. No coincidences were found for any of the specified times.")