import configparser
import numpy as np
import os
import datetime
from icecream import ic
import gc
import glob
from collections import defaultdict


def findCoincidenceDatetimes(date, cuts=True): 
    """ 
    Finds all coincidence events between stations within a one-second window.
    
    For each station data file in the corresponding date folder, this function loads the event
    timestamps and records the station and the index of the event.
    Instead of grouping events by exact timestamp, events are grouped if they occur within one second
    of the earliest event in the group.
    
    Each stored coincidence event is a dictionary with the following keys:
    - "numCoincidences": Number of events in the coincidence group.
    - "datetime": The representative event timestamp (the first event in the group).
    - "stations": Dictionary where each key is a station number, and its value is another dictionary.
                  Initially, this inner dictionary contains the key 'indices' with a list of indices 
                  corresponding to the events for that station.
    
    Args:
      date (str): The date folder to process, as read from the configuration.
    
    Returns:
      A tuple of two dictionaries: (coincidence_datetimes, coincidence_with_repeated_stations)
    """
    import os
    import numpy as np
    import datetime
    from icecream import ic
    import gc
    import glob

    station_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'nurFiles', date)
    cuts_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'cuts', date)

    # Use a subset of stations for the example.
    station_ids = [13, 14, 15, 17, 18, 19, 30]

    # Each event is represented as a tuple: (timestamp, station_id, event_index)
    all_events = []

    # Load data for each station.
    for station_id in station_ids:
        file_list = sorted(glob.glob(station_data_folder + f'/{date}_Station{station_id}_Times*'))
        times = [np.load(f) for f in file_list]
        times = np.concatenate(times, axis=0)
        times = times.squeeze()
        times = np.array(times)
        # Filter out zero timestamps and pre-time events.
        zerotime_mask = times != 0
        times = times[zerotime_mask]
        pretime_mask = times >= datetime.datetime(2013, 1, 1).timestamp()
        times = times[pretime_mask]

        if cuts:
            cuts_file = os.path.join(cuts_data_folder, f'{date}_Station{station_id}_Cuts.npy')
            if os.path.exists(cuts_file):
                ic(f"Loading cuts file: {cuts_file}")
                cuts_data = np.load(cuts_file, allow_pickle=True)
                cuts_data = cuts_data[()]
            else:
                ic(f"Warning: Cuts file not found for station {station_id} on date {date}.")
                continue
            final_cuts = np.ones(len(times), dtype=bool)
            for cut in cuts_data.keys():
                ic(f"Applying cut: {cut}")
                final_cuts &= cuts_data[cut]
            times = times[final_cuts]

        for idx, event_time in enumerate(times):
            if event_time == 0:
                continue
            ts = event_time  
            all_events.append((ts, station_id, idx))

    all_events.sort(key=lambda x: x[0])

    coincidence_datetimes = {}
    coincidence_with_repeated_stations = {}
    valid_counter = 0
    duplicate_counter = 0

    n_events = len(all_events)
    i = 0
    one_second = 1
    while i < n_events:
        current_group = [all_events[i]]
        j = i + 1
        while j < n_events and (all_events[j][0] - all_events[i][0]) <= one_second:
            current_group.append(all_events[j])
            j += 1

        if len(current_group) > 1:
            # Build a dictionary that separates station information.
            stations_info = {}
            for ts, station_id, idx in current_group:
                if station_id not in stations_info:
                    stations_info[station_id] = {"indices": []}
                stations_info[station_id]["indices"].append(idx)
            
            # Skip groups where all events come from the same station.
            if len(stations_info) == 1:
                i = j
                continue

            # Determine which dictionary to add this event to.
            if any(len(info["indices"]) > 1 for info in stations_info.values()):
                target_dict = coincidence_with_repeated_stations
                idx_counter = duplicate_counter
                duplicate_counter += 1
            else:
                target_dict = coincidence_datetimes
                idx_counter = valid_counter
                valid_counter += 1

            target_dict[idx_counter] = {
                "numCoincidences": len(current_group),
                "datetime": all_events[i][0],
                "stations": stations_info
            }
            i = j
        else:
            i += 1

    return coincidence_datetimes, coincidence_with_repeated_stations


def analyze_coincidence_events(coincidence_datetimes, coincidence_with_repeated_stations):
        """
        Analyzes and logs the number of coincidence events for different coincidence numbers,
        and counts unique stations while taking into account that each 'stations' entry may have multiple keys.

        Args:
            coincidence_datetimes (dict): Dictionary of coincidence events with unique stations.
            coincidence_with_repeated_stations (dict): Dictionary of coincidence events with potentially repeated stations.
        """
        # Analyze coincidence_datetimes (unique stations)
        coincidence_counts = {}
        for event_data in coincidence_datetimes.values():
            num_coincidences = event_data["numCoincidences"]
            coincidence_counts[num_coincidences] = coincidence_counts.get(num_coincidences, 0) + 1

        ic("Analysis of coincidence events with unique stations:")
        for num, count in sorted(coincidence_counts.items()):
            ic(f"Number of events with {num} coincidences: {count}")

        ic("\nAnalysis of coincidence events with potentially repeated stations:")
        repeated_coincidence_counts = {}
        for event_data in coincidence_with_repeated_stations.values():
            # Here the 'stations' dictionary may contain multiple keys (e.g., 'indices', 'metadata', etc.)
            # We count unique station IDs by iterating over its keys.
            unique_station_ids = list(event_data["stations"].keys())
            unique_stations_count = len(unique_station_ids)
            repeated_coincidence_counts[unique_stations_count] = repeated_coincidence_counts.get(unique_stations_count, 0) + 1

        for num, count in sorted(repeated_coincidence_counts.items()):
            ic(f"Number of events with {num} unique station coincidences: {count}")

def add_parameter_to_events(events_dict, parameter_name, date, cuts=True, flag='base'):
    """
    Loads the given parameter for each station present in the coincidence events.
    This refactored version uses a precomputed global mask (time + cuts) per station
    to map valid event indices from the concatenated times array back to individual files.
    The processing for each station is saved so that it is not repeated.
    """
    station_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'nurFiles', date)
    cuts_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'cuts', date)
    threshold = datetime.datetime(2013, 1, 1).timestamp()

    # Collect unique station IDs from events.
    unique_stations = set()
    for event in events_dict.values():
        for st in event["stations"].keys():
            unique_stations.add(int(st))
    unique_stations = list(unique_stations)

    # Temporary folder to store processed station data & global mask.
    temp_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'processedNumpyData', date, flag)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    for station in unique_stations:
        temp_save_file = os.path.join(temp_folder, f'temp_param_{parameter_name}_Station{station}.npy')
        global_mask_file = os.path.join(temp_folder, f'global_mask_{station}.npy')
        if os.path.exists(temp_save_file):
            ic(f"Parameter {parameter_name} for station {station} already processed. Loading saved values.")
            processed_events = np.load(temp_save_file, allow_pickle=True).item()
            for event_id, event in events_dict.items():
                st_key = None
                if station in event["stations"]:
                    st_key = station
                elif str(station) in event["stations"]:
                    st_key = str(station)
                if st_key is not None and parameter_name in processed_events[event_id]["stations"][st_key]:
                    event["stations"][st_key][parameter_name] = (
                        processed_events[event_id]["stations"][st_key][parameter_name]
                    )
            continue

        ic(f"Processing station {station} for parameter {parameter_name}")

        # Get list of times files.
        files = sorted(glob.glob(os.path.join(station_data_folder, f'{date}_Station{station}_Times*')))
        if not files:
            continue

        # === Step 1: Compute or load the global mask for valid events for this station.
        if os.path.exists(global_mask_file):
            ic(f"Loading global mask for station {station}")
            global_mask = np.load(global_mask_file, allow_pickle=True)
        else:
            ic(f"Building global mask for station {station}")
            all_times_list = []
            for fpath in files:
                times = np.load(fpath)
                times = np.array(times).squeeze()
                all_times_list.append(times)
            all_times = np.concatenate(all_times_list, axis=0)
            # Create a time mask: times != 0 and times >= threshold.
            time_mask = (all_times != 0) & (all_times >= threshold)
            # Initialize a mask for cuts.
            if cuts:
                cuts_file = os.path.join(cuts_data_folder, f'{date}_Station{station}_Cuts.npy')
                if os.path.exists(cuts_file):
                    ic(f"Loading cuts file: {cuts_file} for station {station}")
                    cuts_data = np.load(cuts_file, allow_pickle=True)[()]
                    # First, combine all cuts into one array for valid events.
                    num_valid = np.sum(time_mask)
                    combined_cut = np.ones(num_valid, dtype=bool)
                    for cut in cuts_data.keys():
                        # Ensure we use only the first num_valid entries.
                        combined_cut &= cuts_data[cut][:num_valid]
                else:
                    ic(f"Warning: Cuts file not found for station {station} on date {date}.")
                    combined_cut = np.ones(np.sum(time_mask), dtype=bool)
            else:
                combined_cut = np.ones(np.sum(time_mask), dtype=bool)
            # Build the global mask. For every position in all_times, if time_mask is True,
            # replace with the next value from combined_cut. Otherwise keep False.
            global_mask = np.zeros(all_times.shape, dtype=bool)
            valid_indices = np.where(time_mask)[0]
            for idx, pos in enumerate(valid_indices):
                global_mask[pos] = combined_cut[idx]
            np.save(global_mask_file, global_mask, allow_pickle=True)
            ic(f"Saved global mask for station {station} to {global_mask_file}")

        # === Step 2: For each times file, determine the mapping of valid events using the global mask.
        file_info_list = []
        global_valid_counter = 0  # Counter over valid events (per file) based on time_mask.
        current_global = 0        # Counter for processed events (after cuts) across files.
        for fpath in files:
            times = np.load(fpath)
            times = np.array(times).squeeze()
            # Local time mask from this file.
            local_time_mask = (times != 0) & (times >= threshold)
            valid_local_indices = np.where(local_time_mask)[0]
            num_valid_local = len(valid_local_indices)
            # Get corresponding segment from global_mask.
            global_segment = global_mask[global_valid_counter: global_valid_counter + num_valid_local]
            # Determine which of the valid local events survived the cuts.
            local_surviving = np.where(global_segment)[0]
            file_info_list.append({
                'file': fpath,
                'global_start': current_global,
                'global_end': current_global + len(local_surviving),
                'local_surviving': local_surviving,  # mapping: processed order -> index in valid_local_indices
            })
            current_global += len(local_surviving)
            global_valid_counter += num_valid_local

        # === Step 3: Build file_updates using the global mapping and update events.
        file_updates = {}
        for event in events_dict.values():
            # Try to find the key matching the station.
            st_key = None
            if station in event["stations"]:
                st_key = station
            elif str(station) in event["stations"]:
                st_key = str(station)
            else:
                continue
            indices_list = event["stations"][st_key].get("indices", [])
            # Initialize parameter list if missing or too short.
            if (parameter_name not in event["stations"][st_key] or
                    len(event["stations"][st_key][parameter_name]) < len(indices_list)):
                event["stations"][st_key][parameter_name] = [None] * len(indices_list)
            for pos, global_idx in enumerate(indices_list):
                if event["stations"][st_key][parameter_name][pos] is not None:
                    continue  # Already processed.
                # Find the file info in which this global index falls.
                for finfo in file_info_list:
                    if finfo['global_start'] <= global_idx < finfo['global_end']:
                        # Map from global index in the processed (surviving) events
                        # to local index in the file.
                        local_counter = global_idx - finfo['global_start']
                        local_index = int(finfo['local_surviving'][local_counter])
                        file_updates.setdefault(finfo['file'], []).append(
                            (event, st_key, pos, local_index)
                        )
                        break

        # === Step 4: For each times file with pending updates,
        # load the corresponding parameter file (matched by replacing "Times" with parameter_name)
        for tfile, updates in file_updates.items():
            param_file = tfile.replace('Times', parameter_name)
            if not os.path.exists(param_file):
                ic(f"Parameter file {param_file} not found for station {station} and parameter {parameter_name}.")
                continue
            with open(param_file, 'rb') as pf:
                param_array = np.load(pf)
            param_array = np.array(param_array)
            if parameter_name != 'Traces':
                param_array = param_array.squeeze()
            for (event, st_key, pos, local_index) in updates:
                try:
                    value = param_array[local_index]
                except IndexError:
                    value = np.nan
                event["stations"][st_key][parameter_name][pos] = value

        # Save the updated events_dict for this station.
        np.save(temp_save_file, events_dict, allow_pickle=True)
        ic(f"Saved parameter {parameter_name} for station {station} to {temp_save_file}")

    gc.collect()

def add_parameter_to_events_improved(events_dict, parameter_name, date, cuts=True, flag='base'):
    """
    Loads the given parameter for each station present in the coincidence events
    using an improved index mapping strategy.

    Args:
        events_dict (dict): Dictionary of events.
        parameter_name (str): Name of the parameter to load (e.g., 'Energy', 'Traces').
        date (str): Date string (YYYYMMDD) for locating data files.
        cuts (bool): Whether to apply external cuts.
        flag (str): Flag for naming cache directories.
    """
    station_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'nurFiles', date)
    cuts_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'cuts', date)
    cache_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'processedNumpyData', date, flag, 'index_maps')
    
    # Use UTC timestamp for threshold comparison
    threshold = datetime.datetime(2013, 1, 1, tzinfo=datetime.timezone.utc).timestamp()

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
        # ic(f"Created cache folder: {cache_folder}")

    # Collect unique station IDs from events.
    unique_stations = set()
    for event in events_dict.values():
        for st in event.get("stations", {}).keys():
             # Attempt conversion to int robustly
            try:
                unique_stations.add(int(st))
            except ValueError:
                print(f"Warning: Could not convert station key '{st}' to int.") # Or use ic
                # Decide how to handle non-integer keys if necessary
                
    unique_stations = sorted(list(unique_stations))
    # ic(f"Unique stations found: {unique_stations}")

    for station in unique_stations:
        # ic(f"Processing Station {station} for Parameter '{parameter_name}'")
        map_cache_file = os.path.join(cache_folder, f'station_{station}_index_map.npy')

        station_index_map = None
        # === Step 1: Build or Load the Index Map ===
        if os.path.exists(map_cache_file):
            try:
                station_index_map = np.load(map_cache_file, allow_pickle=True).item()
                # ic(f"Loaded index map for station {station} from {map_cache_file}")
            except Exception as e:
                # ic(f"Error loading cache file {map_cache_file}: {e}. Rebuilding map.")
                station_index_map = None

        if station_index_map is None:
            # ic(f"Building index map for station {station}...")
            station_index_map = {}
            final_event_counter = 0 # Global counter for events *after* all cuts
            cumulative_time_valid_count = 0 # Counter for events *after* time cuts, across files

            # Load cuts data ONCE for the station if needed
            cuts_mask_for_time_valid = None
            if cuts:
                cuts_file = os.path.join(cuts_data_folder, f'{date}_Station{station}_Cuts.npy')
                if os.path.exists(cuts_file):
                    try:
                        cuts_data_dict = np.load(cuts_file, allow_pickle=True)[()]
                        # Combine all cuts into a single boolean mask
                        # Find a representative cut to get the expected length
                        first_cut_key = next(iter(cuts_data_dict), None)
                        if first_cut_key:
                             num_cut_entries = len(cuts_data_dict[first_cut_key])
                             cuts_mask_for_time_valid = np.ones(num_cut_entries, dtype=bool)
                             for cut_array in cuts_data_dict.values():
                                 # Ensure consistent length before combining
                                 if len(cut_array) == num_cut_entries:
                                     cuts_mask_for_time_valid &= cut_array
                                 else:
                                     # ic(f"Warning: Cut array length mismatch in {cuts_file}. Expected {num_cut_entries}, got {len(cut_array)}. Skipping this cut.")
                                     print(f"Warning: Cut array length mismatch in {cuts_file}. Expected {num_cut_entries}, got {len(cut_array)}. Skipping this cut.")

                             # ic(f"Loaded and combined cuts for station {station} from {cuts_file}. Total cut entries: {num_cut_entries}")
                        else:
                            # ic(f"Cuts file {cuts_file} is empty or has no keys.")
                             print(f"Cuts file {cuts_file} is empty or has no keys.")

                    except Exception as e:
                        # ic(f"Error loading or processing cuts file {cuts_file}: {e}")
                        print(f"Error loading or processing cuts file {cuts_file}: {e}")

                else:
                    # ic(f"Warning: Cuts file not found: {cuts_file}. Proceeding without external cuts for station {station}.")
                    print(f"Warning: Cuts file not found: {cuts_file}. Proceeding without external cuts for station {station}.")


            # Get sorted list of time files for the station
            time_files = sorted(glob.glob(os.path.join(station_data_folder, f'{date}_Station{station}_Times*.npy')))
            if not time_files:
                # ic(f"No time files found for station {station} in {station_data_folder}. Skipping station.")
                print(f"No time files found for station {station} in {station_data_folder}. Skipping station.")
                continue

            for fpath in time_files:
                try:
                    times = np.load(fpath).squeeze()
                    if times.ndim == 0: # Handle scalar case if squeeze results in 0-dim array
                         times = times.reshape(1,) 
                    if times.size == 0: # Skip empty files
                        continue
                        
                except Exception as e:
                    # ic(f"Error loading time file {fpath}: {e}. Skipping file.")
                    print(f"Error loading time file {fpath}: {e}. Skipping file.")
                    continue

                # Apply time cuts locally
                local_time_mask = (times != 0) & (times >= threshold)
                indices_in_file_time_valid = np.where(local_time_mask)[0]
                num_time_valid_in_file = len(indices_in_file_time_valid)

                if num_time_valid_in_file == 0:
                    continue # No time-valid events in this file

                # Determine which of these time-valid events also pass external cuts
                indices_in_file_all_cuts = indices_in_file_time_valid # Default if no cuts apply
                if cuts_mask_for_time_valid is not None:
                    # Calculate the slice needed from the global cuts mask
                    start_idx = cumulative_time_valid_count
                    end_idx = cumulative_time_valid_count + num_time_valid_in_file

                    if end_idx <= len(cuts_mask_for_time_valid):
                        # Apply the relevant segment of the cuts mask
                        local_cuts_mask = cuts_mask_for_time_valid[start_idx:end_idx]
                        indices_in_file_all_cuts = indices_in_file_time_valid[local_cuts_mask]
                    else:
                        # This indicates an inconsistency between time data and cuts data length
                        # ic(f"Warning: Mismatch for station {station}. Time files suggest {end_idx} time-valid events so far, but cuts file only has {len(cuts_mask_for_time_valid)} entries. External cuts may be incomplete for file {fpath}.")
                        print(f"Warning: Mismatch for station {station}. Time files suggest {end_idx} time-valid events so far, but cuts file only has {len(cuts_mask_for_time_valid)} entries. External cuts may be incomplete for file {fpath}.")
                        # Decide how to handle: maybe only apply cuts where possible?
                        # For now, we'll take the intersection based on available cut data
                        available_len = len(cuts_mask_for_time_valid) - start_idx
                        if available_len > 0:
                            local_cuts_mask = cuts_mask_for_time_valid[start_idx:]
                            # Apply mask only to the part of indices_in_file_time_valid that corresponds to available cuts
                            indices_in_file_all_cuts = indices_in_file_time_valid[:available_len][local_cuts_mask]
                        else:
                             indices_in_file_all_cuts = np.array([], dtype=int) # No cut info available for these


                # Add entries to the map for events passing all cuts
                for idx_in_file in indices_in_file_all_cuts:
                    station_index_map[final_event_counter] = (fpath, idx_in_file)
                    final_event_counter += 1

                # Update the cumulative count of time-valid events
                cumulative_time_valid_count += num_time_valid_in_file

            # Save the newly built map
            try:
                np.save(map_cache_file, station_index_map, allow_pickle=True)
                # ic(f"Saved index map for station {station} to {map_cache_file}")
            except Exception as e:
                # ic(f"Error saving cache file {map_cache_file}: {e}")
                print(f"Error saving cache file {map_cache_file}: {e}")


        # === Step 2: Gather Required Parameter Indices ===
        # Structure: { param_file_path: { original_index: [ (event_ref, st_key, pos_in_event_list), ... ] } }
        updates_by_param_file = defaultdict(lambda: defaultdict(list))
        
        if not station_index_map: # Handle case where map is empty (no valid data)
             # ic(f"Index map for station {station} is empty. Skipping parameter loading for this station.")
             print(f"Index map for station {station} is empty. Skipping parameter loading for this station.")
             continue

        for event_id, event in events_dict.items():
            st_key = None
            # Find the correct key for the station (int or str)
            if station in event.get("stations", {}):
                st_key = station
            elif str(station) in event.get("stations", {}):
                st_key = str(station)
            
            if st_key is None:
                continue # This event doesn't involve the current station

            station_event_data = event["stations"][st_key]
            final_indices = station_event_data.get("indices")

            if final_indices is None or len(final_indices) == 0:
                continue # No indices specified for this station in this event

            # Ensure the parameter list exists and has the correct size
            if parameter_name not in station_event_data or not isinstance(station_event_data[parameter_name], list) or len(station_event_data[parameter_name]) != len(final_indices):
                 station_event_data[parameter_name] = [None] * len(final_indices)

            for pos, final_idx in enumerate(final_indices):
                # Only process if the value hasn't been filled yet
                if station_event_data[parameter_name][pos] is None:
                    try:
                        # Use the map to find the origin of this final index
                        time_file_path, original_index = station_index_map[final_idx]
                        
                        # Construct the corresponding parameter file path
                        param_file_path = time_file_path.replace('_Times', f'_{parameter_name}')
                        
                        # Check if file name actually changed (parameter might be 'Times')
                        if param_file_path == time_file_path and parameter_name != 'Times':
                             # ic(f"Warning: Parameter file name construction failed for {time_file_path} and parameter {parameter_name}")
                             print(f"Warning: Parameter file name construction failed for {time_file_path} and parameter {parameter_name}")
                             continue # Skip this index if path generation failed

                        # Store the request: we need the value at original_index from param_file_path
                        # to update the event at the specified position (pos)
                        updates_by_param_file[param_file_path][original_index].append(
                            (event, st_key, pos) # Pass reference to event, station key, and position
                        )
                    except KeyError:
                        # This final_idx from events_dict doesn't exist in our map.
                        # This indicates an inconsistency. Maybe cuts changed or data is corrupt?
                        # ic(f"Warning: Final index {final_idx} for station {st_key} in event {event_id} not found in the index map. Cannot load parameter '{parameter_name}'. Assigning NaN.")
                        print(f"Warning: Final index {final_idx} for station {st_key} in event {event_id} not found in the index map. Cannot load parameter '{parameter_name}'. Assigning NaN.")

                        station_event_data[parameter_name][pos] = np.nan # Or None, or raise error


        # === Step 3: Load Parameter Files and Update Events ===
        # ic(f"Station {station}: Found {len(updates_by_param_file)} parameter files to load.")
        for param_file_path, indices_to_load in updates_by_param_file.items():
            if not os.path.exists(param_file_path):
                # ic(f"Warning: Parameter file not found: {param_file_path}. Assigning NaN to related entries.")
                print(f"Warning: Parameter file not found: {param_file_path}. Assigning NaN to related entries.")
                # Assign NaN to all requests related to this missing file
                for original_index, targets in indices_to_load.items():
                    for event_ref, target_st_key, target_pos in targets:
                        event_ref["stations"][target_st_key][parameter_name][target_pos] = np.nan
                continue

            try:
                # Load the entire parameter array for this file
                param_array = np.load(param_file_path, allow_pickle=True) # Allow pickle for complex objects like traces

                # Basic check for shape consistency if needed, especially if not Traces
                if parameter_name != 'Traces':
                     param_array = param_array.squeeze() # Remove single dimensions often added during saving
                     if param_array.ndim == 0: # Handle scalar squeeze
                          param_array = param_array.reshape(1,)


                # ic(f"Loaded {param_file_path}, shape: {param_array.shape}")

                # Extract needed values and update events
                for original_index, targets in indices_to_load.items():
                    try:
                        # Get the specific value from the loaded array
                        value = param_array[original_index]
                    except IndexError:
                        # The original_index is out of bounds for the loaded parameter array
                        # ic(f"Error: Index {original_index} out of bounds for {param_file_path} (shape {param_array.shape}). Assigning NaN.")
                        print(f"Error: Index {original_index} out of bounds for {param_file_path} (shape {param_array.shape}). Assigning NaN.")
                        value = np.nan
                    except Exception as e:
                         # Catch other potential errors during value extraction
                         # ic(f"Error extracting index {original_index} from {param_file_path}: {e}. Assigning NaN.")
                         print(f"Error extracting index {original_index} from {param_file_path}: {e}. Assigning NaN.")
                         value = np.nan

                    # Assign the loaded value to all events/positions that need it
                    for event_ref, target_st_key, target_pos in targets:
                        event_ref["stations"][target_st_key][parameter_name][target_pos] = value

            except Exception as e:
                # ic(f"Error loading or processing parameter file {param_file_path}: {e}. Assigning NaN to related entries.")
                print(f"Error loading or processing parameter file {param_file_path}: {e}. Assigning NaN to related entries.")
                # Assign NaN if the whole file load failed
                for original_index, targets in indices_to_load.items():
                    for event_ref, target_st_key, target_pos in targets:
                        event_ref["stations"][target_st_key][parameter_name][target_pos] = np.nan
                        
        # Optional: Clear large objects for the station if memory is tight
        del station_index_map
        del updates_by_param_file
        # ic(f"Finished processing station {station}.")

    gc.collect() # Explicitly call garbage collector after processing all stations
    # ic("Finished adding parameter to all relevant events.")
    # The events_dict is modified in place, so no return is strictly necessary,
    # but returning it can sometimes be convenient.
    # return events_dict

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
        coincidence_datetimes, coincidence_with_repeated_stations = findCoincidenceDatetimes(date, cuts=True)
        np.save(output_file, [coincidence_datetimes, coincidence_with_repeated_stations], allow_pickle=True)
        ic("Saved new coincidences", len(coincidence_datetimes))

    # Optional: ic first few coincidences for verification.
    # for key in list(coincidence_datetimes.keys()):
    #     ic(key, coincidence_datetimes[key])

    # for key in list(coincidence_with_repeated_stations.keys()):
    #     ic(key, coincidence_with_repeated_stations[key])

    # Analyze the coincidence events.
    # analyze_coincidence_events(coincidence_datetimes, coincidence_with_repeated_stations)


    # Add parameters to events.
    parameters_to_add = ['Traces', 'SNR', 'ChiRCR', 'Chi2016', 'ChiBad', 'Zen', 'Azi']
    for param in parameters_to_add:
        ic(f"Adding parameter: {param}")
        # add_parameter_to_events(coincidence_datetimes, param, date, cuts=True, flag='no_repeats')
        # add_parameter_to_events(coincidence_with_repeated_stations, param, date, cuts=True, flag='repeats')
        add_parameter_to_events_improved(coincidence_datetimes, param, date, cuts=True, flag='no_repeats')
        add_parameter_to_events_improved(coincidence_with_repeated_stations, param, date, cuts=True, flag='repeats')
    # Optional: ic first few coincidences for verification.
    for key in list(coincidence_datetimes.keys()):
        ic(key, coincidence_datetimes[key])
        if isinstance(coincidence_datetimes[key], dict):
            ic(coincidence_datetimes[key].keys())
    for key in list(coincidence_with_repeated_stations.keys()):
        ic(key, coincidence_with_repeated_stations[key])
        if isinstance(coincidence_with_repeated_stations[key], dict):
            ic(coincidence_with_repeated_stations[key].keys()) 

    quit()
    # Save the updated events with parameters.
    np.save(output_file, [coincidence_datetimes, coincidence_with_repeated_stations], allow_pickle=True)
    ic("Updated events with parameters and saved.")



    # Make plots of the coincidences
    import HRAStationDataAnalysis.loadHRAConvertedData as loadHRAConvertedData

    station_data = loadHRAConvertedData.loadHRAConvertedData(date, cuts=True, SNR='SNR', ChiRCR='ChiRCR', Chi2016='Chi2016', ChiBad='ChiBad', Zen='Zen', Azi='Azi', Trace='Trace')
    # Data is a dictionary with keys 'times', 'SNR', 'ChiRCR', 'Chi2016', and 'ChiRCR_bad'.
    # Each key contains a dictionary where keys are station IDs and values are the corresponding data arrays.

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    plot_folder = os.path.join('HRAStationDataAnalysis', 'plots', date, 'coincidence')
    os.makedirs(plot_folder, exist_ok=True)

    def plot_events(events, title_suffix, plot_folder):
        # Define Chi keys to plot against SNR.
        chi_keys = ['ChiRCR', 'Chi2016', 'ChiBad']
        # Create one figure with three side-by-side subplots.
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, chi in zip(axes, chi_keys):
            ax.set_xlabel('SNR')
            ax.set_ylabel(chi)
            ax.set_title(f'SNR vs {chi} ({title_suffix})')
            ax.set_xscale('log')
            ax.set_xlim(3, 100)
            ax.set_ylim(0, 1)

        num_events = len(events)
        # Generate a distinct color for each event using a colormap.
        colors = cm.jet(np.linspace(0, 1, num_events))

        # Loop over the events.
        for i, (event_id, event) in enumerate(events.items()):
            stations = event["stations"]
            indices = event["indices"]
            # For each Chi plot, collect the SNR and Chi values from station_data
            # station_data contains keys 'SNR', 'ChiRCR', 'Chi2016', 'ChiRCR_bad'
            event_snr = []
            event_chi = {chi: [] for chi in chi_keys}
            for j, station in enumerate(stations):
                idx = indices[j]
                # Extract SNR point if available.
                try:
                    snr_val = station_data['SNR'][station][idx]
                except (KeyError, IndexError):
                    snr_val = np.nan
                event_snr.append(snr_val)
                # Extract each Chi value; if not available, fill with nan.
                for chi in chi_keys:
                    try:
                        chi_val = station_data[chi][station][idx]
                    except (KeyError, IndexError):
                        chi_val = np.nan
                    event_chi[chi].append(chi_val)
            
            # For each of the three properties, plot scatter points and connect them.
            for ax, chi in zip(axes, chi_keys):
                ax.scatter(event_snr,event_chi[chi], color=colors[i])
                ax.plot(event_snr, event_chi[chi], color=colors[i])
        
        plt.tight_layout()
        ic(f"Saving plot for {title_suffix}")
        plt.savefig(os.path.join(plot_folder, f'coincidence_{title_suffix}.png'))
        plt.close(fig)

    # Plot scatter plots for coincidence events with unique stations.
    plot_events(coincidence_datetimes, "Unique Stations Coincidences", plot_folder)

    # Plot scatter plots for coincidence events (including repeated stations).
    plot_events(coincidence_with_repeated_stations, "Repeated Stations Coincidences", plot_folder)

    def plot_master_events(events, station_data, plot_folder):
        import matplotlib.pyplot as plt

        master_folder = os.path.join(plot_folder, "master")
        os.makedirs(master_folder, exist_ok=True)

        # Define fixed color mapping for each station.
        # Stations are [13, 14, 15, 17, 18, 30].
        color_map = {13: 'tab:blue',
                     14: 'tab:orange',
                     15: 'tab:green',
                     17: 'tab:red',
                     18: 'tab:purple',
                     30: 'tab:brown'}

        # List of marker styles to cycle through if a station appears more than once.
        marker_list = ['o', 's', 'D', '^', 'v', '>', '<', 'p']

        # Iterate over each event.
        for event_id, event in events.items():
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            # Compute the average time from the event's 'Times' data.
            try:
                times = event["datetime"]  # 'Times' is a list/array with time stamps from all stations.
                ic(times)
                # avg_time = sum(times) / len(times)
                # event_time = datetime.datetime.fromtimestamp(avg_time)
                event_time = datetime.datetime.fromtimestamp(times)  # Use the first time as representative.
            except Exception:
                event_time = "Unknown Time"
            fig.suptitle(f"Master Event {event_id} - Average Time: {event_time}", fontsize=16)

            # Upper left: SNR vs Chi scatter
            ax_scatter = axs[0, 0]
            # Upper right: Polar plot
            ax_polar = plt.subplot(2, 2, 2, polar=True)
            # Bottom left: Time trace
            ax_trace = axs[1, 0]
            # Bottom right: Dummy plot
            ax_dummy = axs[1, 1]

            # To track repetition for each station in this event.
            occurrence_counter = {}
            # To build the super legend: one entry per station in the event.
            legend_entries = {}

            # Loop through each station in this event.
            for j, station in enumerate(event["stations"]):
                idx = event["indices"][j]
                # Setup repetition counter for marker selection.
                occurrence_counter.setdefault(station, 0)
                marker = marker_list[occurrence_counter[station] % len(marker_list)]
                occurrence_counter[station] += 1

                try:
                    snr_val      = station_data['SNR'][station][idx]
                    chi_rcr_val  = station_data['ChiRCR'][station][idx]
                    chi_2016_val = station_data['Chi2016'][station][idx]
                    zen_val      = station_data['Zen'][station][idx]
                    azi_val      = station_data['Azi'][station][idx]
                    trace_val    = station_data['Trace'][station][idx]
                except (KeyError, IndexError):
                    continue

                # Get the color for this station.
                color = color_map.get(station, 'black')

                # Upper left: Plot Chi2016 and ChiRCR, with arrow from Chi2016 to ChiRCR.
                ax_scatter.scatter(snr_val, chi_2016_val, color=color, marker=marker)
                ax_scatter.scatter(snr_val, chi_rcr_val, color=color, marker=marker)
                ax_scatter.annotate("",
                                    xy=(snr_val, chi_rcr_val),
                                    xytext=(snr_val, chi_2016_val),
                                    arrowprops=dict(arrowstyle="->", color=color))

                # Upper right: Polar plot using azi as the angle and zen as the radial coordinate.
                ax_polar.scatter(azi_val, zen_val, color=color, marker=marker)

                # Bottom left: Plot the time trace; x-axis: 256 point array spanning 0 to 128.
                x_vals = np.linspace(0, 128, 256)
                ax_trace.plot(x_vals, trace_val, color=color, marker=marker)

                # Build the legend entry if not already added.
                if station not in legend_entries:
                    legend_entries[station] = Line2D([0], [0], marker='o', color=color, linestyle='None',
                                                     markersize=8, label=f"Station {station}")

            ax_scatter.set_xlabel("SNR")
            ax_scatter.set_ylabel("Chi")
            ax_scatter.set_title("SNR vs Chi (Arrow: Chi2016 -> ChiRCR)")

            ax_polar.set_title("Polar: Zenith vs Azimuth")

            ax_trace.set_xlabel("Time")
            ax_trace.set_ylabel("Trace")
            ax_trace.set_title("Time Trace")

            ax_dummy.set_title("Dummy Plot")
            ax_dummy.text(0.5, 0.5, "Dummy Plot", horizontalalignment='center', verticalalignment='center')

            # Create a single super legend (only for stations present in this event)
            if legend_entries:
                handles = list(legend_entries.values())
                ax_scatter.legend(handles=handles, loc='best', title="Stations")

            plt.tight_layout()
            master_filename = os.path.join(master_folder, f'master_event_{event_id}.png')
            plt.savefig(master_filename)
            plt.close(fig)

    # Plot master plots for each individual coincidence event (unique stations)
    plot_master_events(coincidence_datetimes, station_data, plot_folder)
