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
from HRAStationDataAnalysis.C_utils import getTimeEventMasks

def findCoincidenceDatetimes(date, date_cuts, cuts=True):
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
    cuts_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'cuts', date_cuts)

    station_ids = [13, 14, 15, 17, 18, 19, 30]

    all_events = [] # Each event is represented as a tuple: (timestamp, station_id, event_index, event_id)

    # Load data for each station.
    for station_id in station_ids:
        # Load times
        file_list = sorted(glob.glob(station_data_folder + f'/{date}_Station{station_id}_Times*'))
        times = [np.load(f) for f in file_list]
        times = np.concatenate(times, axis=0)

        # Load Event IDs
        event_id_files = sorted(glob.glob(station_data_folder + f'/{date}_Station{station_id}_EventIDs*'))
        event_ids = [np.load(f) for f in event_id_files]
        event_ids = np.concatenate(event_ids, axis=0)

        # Ensure times and event_ids have the same length
        if len(times) != len(event_ids):
            ic(f"Error: Mismatch in length between Times ({len(times)}) and EventIDs ({len(event_ids)}) for station {station_id}.")
            continue

        # Apply initial time and uniqueness cuts using the utility function.
        initial_mask, unique_indices = getTimeEventMasks(times, event_ids)
        times = times[initial_mask][unique_indices]
        event_ids = event_ids[initial_mask][unique_indices]

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
            if len(stations_info) == 1 and not has_repeated_event_ids_within_station:
                i = j
                continue

            # Determine which dictionary to add this event to.
            if has_repeated_event_ids_within_station:
                target_dict = coincidence_with_repeated_eventIDs
                idx_counter = repeated_event_id_counter
                repeated_event_id_counter += 1
            elif any(len(info["indices"]) > 1 for info in stations_info.values()):
                target_dict = coincidence_with_repeated_stations
                idx_counter = duplicate_station_counter
                duplicate_station_counter += 1
            else:
                target_dict = coincidence_datetimes
                idx_counter = valid_counter
                valid_counter += 1

            target_dict[idx_counter] = {
                "numCoincidences": len(current_group),
                "datetime": all_events[i][0],
                "stations": stations_info 
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
    temp_file_path = None 
    try:
        temp_dir = os.path.dirname(path)
        os.makedirs(temp_dir, exist_ok=True) 
        fd, temp_file_path = tempfile.mkstemp(suffix='.tmp', dir=temp_dir)
        
        with os.fdopen(fd, 'wb') as tf:
            pickle.dump(data, tf, protocol=pickle.HIGHEST_PROTOCOL)
            tf.flush(); os.fsync(tf.fileno()) 
            
        os.replace(temp_file_path, path) 
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Checkpoint saved successfully to {path}")
        temp_file_path = None 

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
    checkpoint_path=None
    ):
    """
    Adds a parameter to events_dict by re-applying filters to map final event
    indices back to their original file locations.
    - Modifies events_dict in-place.
    - Uses getTimeEventMasks for initial time and uniqueness filtering.
    - Saves the entire events_dict state to checkpoint_path after each station.
    """
    print(f"Starting parameter addition for '{parameter_name}' (Date: {date}, Flag: {flag})")
    if checkpoint_path:
        print(f"Checkpoints will be saved to: {checkpoint_path}")
        
    station_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'nurFiles', date)
    cuts_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'cuts', date)

    unique_stations = sorted(list({int(st) for event in events_dict.values() for st in event.get("stations", {}).keys()}))
    
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
                    if parameter_name not in station_data or not isinstance(station_data.get(parameter_name), list) or len(station_data[parameter_name]) != len(final_indices):
                        station_data[parameter_name] = [None] * len(final_indices)
                    
                    for pos, final_idx in enumerate(final_indices):
                        if station_data[parameter_name][pos] is None:
                            requests[final_idx].append((event, st_key, pos))
                            needs_processing = True

        if not needs_processing:
            print(f"Station {station}: No processing needed for parameter '{parameter_name}'.")
            processed_station_count += 1
            continue
        print(f"Station {station}: Found {len(requests)} unique indices needing '{parameter_name}'.")

        # This mask is generated based on data that has passed initial time and uniqueness filters.
        station_cuts_mask = None
        if cuts:
            cuts_file = os.path.join(cuts_data_folder, f'{date}_Station{station}_Cuts.npy')
            if os.path.exists(cuts_file):
                try:
                    cuts_data_dict = np.load(cuts_file, allow_pickle=True)[()]
                    if cuts_data_dict:
                        # Combine all boolean cut arrays from the file.
                        cut_arrays = list(cuts_data_dict.values())
                        station_cuts_mask = np.ones(len(cut_arrays[0]), dtype=bool)
                        for cut_array in cut_arrays:
                            if len(cut_array) == len(station_cuts_mask):
                                station_cuts_mask &= cut_array
                            else:
                                print(f"Warning: Cut length mismatch in {cuts_file}. Ignoring this specific cut.")
                except Exception as e:
                    print(f"Error loading or processing cuts file {cuts_file}: {e}")
            else:
                print(f"Warning: Cuts file not found: {cuts_file}.")

        final_event_counter = 0
        cumulative_pre_cuts_count = 0  # Tracks events post-getTimeEventMasks, pre-external-cuts
        
        time_files = sorted(glob.glob(os.path.join(station_data_folder, f'{date}_Station{station}_Times*.npy')))
        
        if not time_files:
            print(f"No data files found for station {station}. Cannot process {len(requests)} requests.")
            for final_idx, targets in requests.items():
                for event_ref, target_st_key, target_pos in targets:
                    event_ref["stations"][target_st_key][parameter_name][target_pos] = np.nan
        else:
            for fpath in time_files:
                if not requests: break
                
                try:
                    times = np.load(fpath)
                    event_ids_path = fpath.replace('_Times', '_EventIDs')
                    if not os.path.exists(event_ids_path):
                        print(f"Error: EventID file not found for {fpath}. Skipping.")
                        continue
                    event_ids = np.load(event_ids_path)

                    if len(times) != len(event_ids):
                        print(f"Error: Mismatch in length for {fpath}. Skipping.")
                        continue
                    if times.size == 0:
                        continue

                    # Apply initial time and uniqueness cuts from C_utils
                    initial_mask, unique_indices = getTimeEventMasks(times, event_ids)
                    
                    # Get original file indices that pass initial and unique cuts
                    indices_after_initial_cuts = np.where(initial_mask)[0]
                    indices_after_unique_cuts = indices_after_initial_cuts[unique_indices]
                    num_pre_cuts_in_file = len(indices_after_unique_cuts)

                    if num_pre_cuts_in_file == 0:
                        continue

                    indices_in_file_all_cuts = indices_after_unique_cuts
                    if station_cuts_mask is not None:
                        start_idx = cumulative_pre_cuts_count
                        end_idx = start_idx + num_pre_cuts_in_file
                        
                        if end_idx <= len(station_cuts_mask):
                            local_cuts_mask = station_cuts_mask[start_idx:end_idx]
                            indices_in_file_all_cuts = indices_after_unique_cuts[local_cuts_mask]
                        else:
                            print(f"Warning: Cuts data length mismatch for station {station}, file {fpath}. Applying partial or no cuts for this file.")
                            available_len = len(station_cuts_mask) - start_idx
                            if available_len > 0:
                                local_cuts_mask = station_cuts_mask[start_idx:]
                                indices_in_file_all_cuts = indices_after_unique_cuts[:available_len][local_cuts_mask]
                            else:
                                indices_in_file_all_cuts = np.array([], dtype=int)
                    
                    num_all_cuts_in_file = len(indices_in_file_all_cuts)

                    file_final_idx_start = final_event_counter
                    file_final_idx_end = final_event_counter + num_all_cuts_in_file
                    relevant_requests = {idx: req for idx, req in requests.items() if file_final_idx_start <= idx < file_final_idx_end}

                    if relevant_requests:
                        param_file_path = fpath.replace('_Times', f'_{parameter_name}')
                        param_array = None
                        if os.path.exists(param_file_path):
                            try:
                                param_array = np.load(param_file_path, allow_pickle=True)
                            except Exception as e:
                                print(f"Error loading parameter file {param_file_path}: {e}")

                        for i, original_index in enumerate(indices_in_file_all_cuts):
                            current_final_idx = file_final_idx_start + i
                            if current_final_idx in requests:
                                value_to_assign = np.nan
                                if param_array is not None and original_index < len(param_array):
                                    value_to_assign = param_array[original_index]
                                
                                for event_ref, target_st_key, target_pos in requests[current_final_idx]:
                                    event_ref["stations"][target_st_key][parameter_name][target_pos] = value_to_assign
                                del requests[current_final_idx]

                    final_event_counter += num_all_cuts_in_file
                    cumulative_pre_cuts_count += num_pre_cuts_in_file
                
                finally:
                    gc.collect()

        if requests:
            print(f"Warning: Station {station}: {len(requests)} requests remain unfulfilled. Setting to NaN.")
            for final_idx, targets in requests.items():
                for event_ref, target_st_key, target_pos in targets:
                    event_ref["stations"][target_st_key][parameter_name][target_pos] = np.nan

        processed_station_count += 1
        station_end_time = datetime.datetime.now()
        print(f"Finished processing station {station}. Time taken: {station_end_time - station_start_time}")

        if checkpoint_path:
            save_checkpoint(events_dict, checkpoint_path)
        gc.collect()

    print(f"\nFinished adding parameter '{parameter_name}'. Processed {processed_station_count}/{total_stations} stations.")


if __name__ == "__main__": 
    config = configparser.ConfigParser() 
    config.read(os.path.join('HRAStationDataAnalysis', 'config.ini')) 
    date = config['PARAMETERS']['date']
    date_cuts = config['PARAMETERS']['date_cuts']
    date_processing = config['PARAMETERS']['date_processing']
    ic("Processing date:", date)
    ic("Saving to date_processing:", date_processing)

    numpy_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'processedNumpyData', date)
    if not os.path.exists(numpy_folder):
        os.makedirs(numpy_folder)

    output_file = os.path.join(numpy_folder, f'{date_processing}_CoincidenceDatetimes.npy')

    if os.path.exists(output_file):
        data = np.load(output_file, allow_pickle=True)
        coincidence_datetimes = data[0]
        coincidence_with_repeated_stations = data[1]
        # Safely load third dict if present; older saves may lack it
        if len(data) > 2:
            coincidence_with_repeated_eventIDs = data[2]
        else:
            coincidence_with_repeated_eventIDs = {}
        ic("Loaded processed coincidences", len(coincidence_datetimes))
    else:
        coincidence_datetimes, coincidence_with_repeated_stations, coincidence_with_repeated_eventIDs = findCoincidenceDatetimes(date, date_cuts, cuts=True)
        np.save(output_file, [coincidence_datetimes, coincidence_with_repeated_stations, coincidence_with_repeated_eventIDs], allow_pickle=True)
        ic("Saved new coincidences", len(coincidence_datetimes))

    # --- Summary Printouts ---
    def _print_summary(label, events_dict):
        if not events_dict:
            print(f"[Summary] {label}: None found.")
            return
        combo_level_counts = defaultdict(int)
        station_participation = defaultdict(int)
        for ev in events_dict.values():
            stations_block = ev.get("stations", {})
            # keys may be int or str; normalize to int where possible
            norm_keys = []
            for k in stations_block.keys():
                try:
                    norm_keys.append(int(k))
                except Exception:
                    norm_keys.append(k)
            k_len = len(set(norm_keys))
            combo_level_counts[k_len] += 1
            for st in set(norm_keys):
                station_participation[st] += 1
        print(f"\n[Summary] {label} - total groups: {len(events_dict)}")
        for combo_size in sorted(combo_level_counts.keys()):
            print(f"  Groups with {combo_size} unique stations: {combo_level_counts[combo_size]}")
        print("  Station participation counts (number of groups each station appears in):")
        for st in sorted(station_participation.keys()):
            print(f"    Station {st}: {station_participation[st]}")

    _print_summary("Valid coincidences", coincidence_datetimes)
    _print_summary("Coincidences with repeated stations", coincidence_with_repeated_stations)
    _print_summary("Coincidences with repeated EventIDs within a station", coincidence_with_repeated_eventIDs)

    # Optional: ic first few coincidences for verification.
    # (Optional detailed dumps can be re-enabled if needed)
    # for key in list(coincidence_datetimes.keys())[:5]:
    #     ic(key, coincidence_datetimes[key])
    # for key in list(coincidence_with_repeated_stations.keys())[:5]:
    #     ic(key, coincidence_with_repeated_stations[key])
    # for key in list(coincidence_with_repeated_eventIDs.keys())[:5]:
    #     ic(key, coincidence_with_repeated_eventIDs[key])