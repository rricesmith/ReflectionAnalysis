import configparser
import numpy as np
import os
import datetime
from icecream import ic
import gc
import glob
from collections import defaultdict
import pickle # Using pickle for potentially complex dict structure
import tempfile # For atomic saving
import time # For timestamping save messages

def findCoincidenceDatetimes(date, cuts=True):
    """
    Finds all coincidence events between stations within a one-second window.

    For each station data file in the corresponding date folder, this function loads the event
    timestamps and records the station and the index of the event.
    Events are grouped if they occur within one second of the earliest event in the group.

    Each stored coincidence event is a dictionary with the following keys:
    - "numCoincidences": Number of events in the coincidence group.
    - "datetime": The representative event timestamp (the first event in the group).
    - "stations": Dictionary where each key is a station number, and its value is another dictionary.
                  This inner dictionary contains:
                    - 'indices': A list of indices corresponding to the events for that station.
                    - 'event_ids': A list of Event IDs for the events for that station.

    Args:
      date (str): The date folder to process, as read from the configuration.
      cuts (bool, optional): Whether to apply cuts. Defaults to True.

    Returns:
      A tuple of three dictionaries:
      (coincidence_datetimes, coincidence_with_repeated_stations, coincidence_with_repeated_eventIDs)
    """
    station_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'nurFiles', date)
    cuts_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'cuts', date)

    station_ids = [13, 14, 15, 17, 18, 19, 30]

    all_events = [] # Each event is represented as a tuple: (timestamp, station_id, event_index, event_id)

    # Load data for each station.
    for station_id in station_ids:
        # Load times
        file_list = sorted(glob.glob(station_data_folder + f'/{date}_Station{station_id}_Times*'))
        times = [np.load(f).squeeze() for f in file_list]
        times = np.concatenate(times, axis=0)

        # Load Event IDs
        event_id_files = sorted(glob.glob(station_data_folder + f'/{date}_Station{station_id}_EventIDs*'))
        event_ids = [np.load(f).squeeze() for f in event_id_files]
        event_ids = np.concatenate(event_ids, axis=0)

        # Ensure times and event_ids have the same length
        if len(times) != len(event_ids):
            ic(f"Error: Mismatch in length between Times ({len(times)}) and EventIDs ({len(event_ids)}) for station {station_id}.")
            continue

        # Initial filtering for times
        zerotime_mask = times != 0
        times = times[zerotime_mask]
        event_ids = event_ids[zerotime_mask] # Apply the same mask to event_ids

        pretime_mask = times >= datetime.datetime(2013, 1, 1).timestamp()
        times = times[pretime_mask]
        event_ids = event_ids[pretime_mask] # Apply the same mask to event_ids

        final_cuts_mask = np.ones(len(times), dtype=bool) # Initialize final cuts mask

        if cuts:
            cuts_file = os.path.join(cuts_data_folder, f'{date}_Station{station_id}_Cuts.npy')
            if os.path.exists(cuts_file):
                ic(f"Loading cuts file: {cuts_file}")
                cuts_data = np.load(cuts_file, allow_pickle=True)[()]
                for cut_key in cuts_data.keys():
                    ic(f"Applying cut: {cut_key}")
                    # Ensure cut_data matches the length of current times/event_ids
                    # before applying the mask
                    current_cut = cuts_data[cut_key][:len(times)]
                    final_cuts_mask &= current_cut
                times = times[final_cuts_mask]
                event_ids = event_ids[final_cuts_mask] # Apply the same mask to event_ids
            else:
                ic(f"Warning: Cuts file not found for station {station_id} on date {date}. No cuts applied for this station.")

        for idx, event_time in enumerate(times):
            # The idx here is the post-cut index, which is what we want for mapping later
            all_events.append((event_time, station_id, idx, event_ids[idx]))

    all_events.sort(key=lambda x: x[0]) # Sort all events by timestamp.

    coincidence_datetimes = {}
    coincidence_with_repeated_stations = {}
    coincidence_with_repeated_eventIDs = {} # New dictionary for events with repeated IDs within a station

    valid_counter = 0
    duplicate_station_counter = 0
    repeated_event_id_counter = 0

    n_events = len(all_events)
    i = 0
    one_second = 1 # One-second window for coincidence

    while i < n_events:
        current_group = [all_events[i]]
        j = i + 1
        # Include subsequent events only if their time is within one second of the first event in the group.
        while j < n_events and (all_events[j][0] - all_events[i][0]) <= one_second:
            current_group.append(all_events[j])
            j += 1

        # Only record a coincidence if at least 2 events are found.
        if len(current_group) > 1:
            # Build a dictionary that separates station information, including indices and Event IDs.
            stations_info = {}
            for ts, station_id, idx, event_id in current_group:
                if station_id not in stations_info:
                    stations_info[station_id] = {"indices": [], "event_ids": []}
                stations_info[station_id]["indices"].append(idx)
                stations_info[station_id]["event_ids"].append(event_id)

            # Check for repeated Event IDs within any single station in the current group
            has_repeated_event_ids_within_station = False
            for info in stations_info.values():
                if len(info["event_ids"]) > len(set(info["event_ids"])):
                    has_repeated_event_ids_within_station = True
                    break # Found a repeat, no need to check other stations in this group

            # Skip groups where all events come from the same station (unless they have repeated Event IDs)
            # The original logic was `len(set(stations)) == 1`. With the new `stations_info` structure,
            # this translates to `len(stations_info) == 1`.
            if len(stations_info) == 1 and not has_repeated_event_ids_within_station:
                i = j
                continue

            # Determine which dictionary to add this event to.
            if has_repeated_event_ids_within_station:
                target_dict = coincidence_with_repeated_eventIDs
                idx_counter = repeated_event_id_counter
                repeated_event_id_counter += 1
            elif any(len(info["indices"]) > 1 for info in stations_info.values()):
                # This condition checks if there are multiple events from the same station
                # but without repeated Event IDs. This is the original 'repeated stations' case.
                target_dict = coincidence_with_repeated_stations
                idx_counter = duplicate_station_counter
                duplicate_station_counter += 1
            else:
                # No repeated stations and no repeated event IDs within stations.
                target_dict = coincidence_datetimes
                idx_counter = valid_counter
                valid_counter += 1

            target_dict[idx_counter] = {
                "numCoincidences": len(current_group),
                "datetime": all_events[i][0],
                "stations": stations_info # stations_info now contains both indices and event_ids
            }
            # Skip over the events already grouped.
            i = j
        else:
            i += 1 # Move to the next event if it's not part of a coincidence

    return coincidence_datetimes, coincidence_with_repeated_stations, coincidence_with_repeated_eventIDs





def save_checkpoint(data, path):
    """
    Safely save data via pickle to a file using a temporary file 
    and atomic replace.
    """
    temp_file_path = None # Ensure variable is defined for potential error handling
    try:
        temp_dir = os.path.dirname(path)
        os.makedirs(temp_dir, exist_ok=True) # Ensure directory exists
        fd, temp_file_path = tempfile.mkstemp(suffix='.tmp', dir=temp_dir)
        
        with os.fdopen(fd, 'wb') as tf:
            pickle.dump(data, tf, protocol=pickle.HIGHEST_PROTOCOL)
            tf.flush(); os.fsync(tf.fileno()) # Ensure write to disk
            
        os.replace(temp_file_path, path) # Atomic replace
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Checkpoint saved successfully to {path}")
        temp_file_path = None # Prevent removal in finally if successful

    except Exception as e:
        print(f"Error saving checkpoint to {path}: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                print(f"Attempting to remove temporary file: {temp_file_path}")
                os.remove(temp_file_path)
            except OSError as oe:
                 print(f"Error removing temporary checkpoint file {temp_file_path}: {oe}")

def add_parameter_simple_checkpoint(
    events_dict, 
    parameter_name, 
    date, 
    cuts=True, 
    flag='base', 
    checkpoint_path=None # Path for saving progress after each station
    ):
    """
    Adds a parameter to events_dict, processing stations sequentially.
    - Modifies events_dict in-place.
    - Assumes caller loads the initial state from checkpoint_path if resuming.
    - Automatically skips processing for parameters already filled.
    - Saves the entire events_dict state to checkpoint_path after each station.
    """
    print(f"Starting parameter addition for '{parameter_name}' (Date: {date}, Flag: {flag})")
    if checkpoint_path:
        print(f"Checkpoints will be saved to: {checkpoint_path}")
        
    station_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'nurFiles', date)
    cuts_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'cuts', date)
    threshold = datetime.datetime(2013, 1, 1, tzinfo=datetime.timezone.utc).timestamp()

    unique_stations = set()
    for event_id, event in events_dict.items(): 
        for st in event.get("stations", {}).keys():
            try: unique_stations.add(int(st))
            except ValueError: print(f"Warning: Cannot convert station key '{st}' in event '{event_id}' to int.")

    unique_stations = sorted(list(unique_stations))
    if not unique_stations:
        print("No stations found in events_dict to process.")
        return 
    total_stations = len(unique_stations)
    print(f"Found {total_stations} unique stations to process: {unique_stations}")
    processed_station_count = 0

    for station in unique_stations:
        station_start_time = datetime.datetime.now()
        print(f"\nProcessing Station {station} ({processed_station_count + 1}/{total_stations})...")

        requests = defaultdict(list)
        needs_processing = False 
        for event_id, event in events_dict.items():
            station_data = event.get("stations", {}).get(station) or event.get("stations", {}).get(str(station))
            if station_data:
                st_key = station if station in event["stations"] else str(station)
                final_indices = station_data.get("indices")
                if final_indices:
                    param_list = station_data.get(parameter_name)
                    if not isinstance(param_list, list) or len(param_list) != len(final_indices):
                        param_list = [None] * len(final_indices)
                        station_data[parameter_name] = param_list
                    for pos, final_idx in enumerate(final_indices):
                        if param_list[pos] is None:
                            requests[final_idx].append((event, st_key, pos))
                            needs_processing = True 
        
        if not needs_processing:
            print(f"Station {station}: No processing needed (parameter '{parameter_name}' already filled).")
            processed_station_count += 1
            continue 
        print(f"Station {station}: Found {len(requests)} unique indices needing '{parameter_name}'.")

        cuts_mask_for_time_valid = None
        if cuts:
            # [ Cut loading logic - unchanged ]
            cuts_file = os.path.join(cuts_data_folder, f'{date}_Station{station}_Cuts.npy')
            if os.path.exists(cuts_file):
                try:
                    cuts_data_dict_local = np.load(cuts_file, allow_pickle=True)[()]
                    first_cut_key = next(iter(cuts_data_dict_local), None)
                    if first_cut_key:
                        num_cut_entries = len(cuts_data_dict_local[first_cut_key])
                        cuts_mask_for_time_valid = np.ones(num_cut_entries, dtype=bool)
                        for cut_name, cut_array in cuts_data_dict_local.items():
                            if len(cut_array) == num_cut_entries: cuts_mask_for_time_valid &= cut_array
                            else: print(f"Warning: Cut '{cut_name}' length mismatch {cuts_file}. Skip cut.")
                    else: print(f"Cuts file {cuts_file} is empty.")
                except Exception as e: print(f"Error loading cuts {cuts_file}: {e}")
            else: print(f"Warning: Cuts file not found: {cuts_file}.")

        final_event_counter = 0
        cumulative_time_valid_count = 0
        time_files = sorted(glob.glob(os.path.join(station_data_folder, f'{date}_Station{station}_Times*.npy')))
        
        if not time_files:
            print(f"No time files found for station {station}. Cannot fulfill {len(requests)} requests.")
            for final_idx, targets in requests.items():
                for event_ref, target_st_key, target_pos in targets:
                    event_ref["stations"][target_st_key][parameter_name][target_pos] = np.nan
        else:
            for fpath in time_files:
                if not requests: break 
                times, param_array, indices_in_file_all_cuts = None, None, None
                indices_in_file_time_valid, local_time_mask, local_cuts_mask = None, None, None
                try:
                    # [ File loading, cut application, param loading, request fulfillment logic - unchanged ]
                    try: # Load times
                        times = np.load(fpath).squeeze()
                        if times.ndim == 0: times = times.reshape(1,); 
                        if times.size == 0: continue 
                    except Exception as e: print(f"Error loading time file {fpath}: {e}. Skipping."); continue 
                    # Time cuts
                    local_time_mask = (times != 0) & (times >= threshold)
                    indices_in_file_time_valid = np.where(local_time_mask)[0]
                    num_time_valid_in_file = len(indices_in_file_time_valid); del local_time_mask; local_time_mask = None 
                    if num_time_valid_in_file == 0: continue 
                    # External cuts
                    indices_in_file_all_cuts = indices_in_file_time_valid 
                    if cuts_mask_for_time_valid is not None:
                        start_idx = cumulative_time_valid_count; end_idx = start_idx + num_time_valid_in_file
                        if end_idx <= len(cuts_mask_for_time_valid):
                            local_cuts_mask = cuts_mask_for_time_valid[start_idx:end_idx]
                            indices_in_file_all_cuts = indices_in_file_time_valid[local_cuts_mask]
                        else: # Handle mismatch
                            print(f"Warning: Cuts data length mismatch station {station}, file {fpath}.")
                            available_len = len(cuts_mask_for_time_valid) - start_idx
                            if available_len > 0:
                                local_cuts_mask = cuts_mask_for_time_valid[start_idx:]
                                indices_in_file_all_cuts = indices_in_file_time_valid[:available_len][local_cuts_mask]
                            else: indices_in_file_all_cuts = np.array([], dtype=int)
                        if local_cuts_mask is not None: del local_cuts_mask; local_cuts_mask = None
                        if indices_in_file_all_cuts is not indices_in_file_time_valid: del indices_in_file_time_valid; indices_in_file_time_valid = None 
                    num_all_cuts_in_file = len(indices_in_file_all_cuts)
                    # Check relevant requests
                    file_final_idx_start = final_event_counter; file_final_idx_end = final_event_counter + num_all_cuts_in_file
                    relevant_requests = {idx: req_list for idx, req_list in requests.items() if file_final_idx_start <= idx < file_final_idx_end}
                    # Load params and fulfill if relevant
                    if relevant_requests:
                        param_file_path = fpath.replace('_Times', f'_{parameter_name}')
                        if param_file_path == fpath and parameter_name != 'Times': # Handle filename failure
                             print(f"Warning: Param file name construction failed {fpath}/{parameter_name}")
                             for final_idx in relevant_requests:
                                for event_ref, target_st_key, target_pos in requests[final_idx]: event_ref["stations"][target_st_key][parameter_name][target_pos] = np.nan
                                del requests[final_idx] 
                        else: # Load param file
                            param_loaded = False
                            try:
                                if os.path.exists(param_file_path):
                                    param_array = np.load(param_file_path, allow_pickle=True)
                                    if isinstance(param_array, np.ndarray) and parameter_name != 'Traces': param_array = param_array.squeeze();
                                    if isinstance(param_array, np.ndarray) and param_array.ndim == 0: param_array = param_array.reshape(1,)
                                    param_loaded = True
                                else: print(f"Warning: Parameter file not found: {param_file_path}")
                            except Exception as e: print(f"Error loading param file {param_file_path}: {e}")
                            # Fulfill requests for this file
                            for i, original_index in enumerate(indices_in_file_all_cuts):
                                current_final_idx = file_final_idx_start + i
                                if current_final_idx in relevant_requests:
                                    value = np.nan
                                    if param_loaded:
                                        try: # Extract value safely
                                            is_sequence = isinstance(param_array, (np.ndarray, list, tuple)); length = -1
                                            if is_sequence:
                                                try: length = len(param_array)
                                                except TypeError: is_sequence = False
                                            if is_sequence and 0 <= original_index < length: value = param_array[original_index]
                                            elif not is_sequence: print(f"Warning: Param data {param_file_path} not sequence.")
                                            else: print(f"Error: Index {original_index} out of bounds {param_file_path} len {length}.")
                                        except Exception as e: print(f"Error extracting index {original_index} from {param_file_path}: {e}")
                                    # Assign value to all events needing it
                                    for event_ref, target_st_key, target_pos in requests[current_final_idx]:
                                        event_ref["stations"][target_st_key][parameter_name][target_pos] = value
                                    del requests[current_final_idx] # Remove fulfilled request
                    # Update counters
                    final_event_counter += num_all_cuts_in_file
                    cumulative_time_valid_count += num_time_valid_in_file
                finally: # Cleanup for file iteration
                    if times is not None: del times
                    if param_array is not None: del param_array
                    if indices_in_file_all_cuts is not None: del indices_in_file_all_cuts
                    if indices_in_file_time_valid is not None: del indices_in_file_time_valid
                    if local_time_mask is not None: del local_time_mask
                    if local_cuts_mask is not None: del local_cuts_mask
                    gc.collect()
            # --- End File Iteration Loop ---

        # Handle remaining requests after file loop
        if requests:
            print(f"Warning: Station {station}: {len(requests)} requests remain unfulfilled. Setting to NaN.")
            for final_idx, targets in requests.items():
                for event_ref, target_st_key, target_pos in targets: event_ref["stations"][target_st_key][parameter_name][target_pos] = np.nan
            requests.clear()

        # Cleanup station-level objects
        if cuts_mask_for_time_valid is not None: del cuts_mask_for_time_valid
        
        processed_station_count += 1
        station_end_time = datetime.datetime.now()
        print(f"Finished processing station {station}. Time taken: {station_end_time - station_start_time}")

        # *** Save Checkpoint After Station Completion ***
        if checkpoint_path: save_checkpoint(events_dict, checkpoint_path) 
        gc.collect() 

    print(f"\nFinished adding parameter '{parameter_name}'. Processed {processed_station_count}/{total_stations} stations.")

if __name__ == "__main__": 
    # Read configuration and get date 
    config = configparser.ConfigParser() 
    config.read(os.path.join('HRAStationDataAnalysis', 'config.ini')) 
    date = config['PARAMETERS']['date']

    # Define folder to save processed data.
    numpy_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'processedNumpyData', date)
    if not os.path.exists(numpy_folder):
        os.makedirs(numpy_folder)

    output_file = os.path.join(numpy_folder, f'{date}_CoincidenceDatetimes.npy')

    # Check for existing processed data, load if found; otherwise find coincidences and save.
    if os.path.exists(output_file):
        data = np.load(output_file, allow_pickle=True)
        coincidence_datetimes = data[0]
        coincidence_with_repeated_stations = data[1]
        ic("Loaded processed coincidences", len(coincidence_datetimes))
    else:
        coincidence_datetimes, coincidence_with_repeated_stations, coincidence_with_repeated_eventIDs = findCoincidenceDatetimes(date, cuts=True)
        np.save(output_file, [coincidence_datetimes, coincidence_with_repeated_stations, coincidence_with_repeated_eventIDs], allow_pickle=True)
        ic("Saved new coincidences", len(coincidence_datetimes))



    # Optional: ic first few coincidences for verification.
    for key in list(coincidence_datetimes.keys()):
        ic(key, coincidence_datetimes[key])
        if isinstance(coincidence_datetimes[key], dict):
            ic(coincidence_datetimes[key].keys())
    for key in list(coincidence_with_repeated_stations.keys()):
        ic(key, coincidence_with_repeated_stations[key])
        if isinstance(coincidence_with_repeated_stations[key], dict):
            ic(coincidence_with_repeated_stations[key].keys()) 

    for key in list(coincidence_with_repeated_eventIDs.keys()):
        ic(key, coincidence_with_repeated_eventIDs[key])
        if isinstance(coincidence_with_repeated_eventIDs[key], dict):
            ic(coincidence_with_repeated_eventIDs[key].keys())


