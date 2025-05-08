import os
import glob
import numpy as np
import pickle
import re
import gc
import datetime
from collections import defaultdict
import time # For logging/timestamps if needed
from icecream import ic

# --- Helper for Caching ---
def _load_pickle(filepath):
    """Loads data from a pickle file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file {filepath}: {e}")
    return None

def _save_pickle(data, filepath):
    """Saves data to a pickle file, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Error saving pickle file {filepath}: {e}")

# --- Function 1: Create Final Index to GRCI Map ---
def create_final_idx_to_grci_map(
    date_str,
    station_id,
    time_threshold_timestamp,
    time_files_template, # e.g., "path/to/nurFiles/{date}/{date}_Station{station_id}_Times*.npy"
    external_cuts_file_path, # Full path to the combined external cuts .npy file for the station
    map_cache_file_path # Full path for saving/loading this map
    ):
    """
    Creates a mapping from a final_idx (post-all-cuts) to a GRCI
    (Global Raw Concatenated Index - index if all original station files were concatenated).

    Args:
        date_str (str): Date string (YYYYMMDD).
        station_id (Union[int, str]): Station identifier.
        time_threshold_timestamp (float): Timestamp for the time cut.
        time_files_template (str): Glob pattern template for time files.
        external_cuts_file_path (str): Path to the .npy file containing a boolean mask
                                       for external cuts. This mask applies to events
                                       that have ALREADY passed the time/threshold cut.
        map_cache_file_path (str): Path to save/load the computed map.

    Returns:
        dict: A dictionary mapping {final_idx: GRCI}, or None if an error occurs.
    """
    print(f"Attempting to create/load final_idx_to_grci_map for Station {station_id} on {date_str}")
    
    # 1. Caching
    cached_map = _load_pickle(map_cache_file_path)
    if cached_map is not None:
        print(f"Loaded final_idx_to_grci_map from cache: {map_cache_file_path}")
        return cached_map

    print(f"Cache not found. Building map for Station {station_id}...")

    # 2. Initialization
    global_raw_counter = 0  # This will be the GRCI
    time_passed_event_index = 0  # Index for events passing only time cuts
    
    # Stores GRCI for each event that passes the time cut, indexed by time_passed_event_index
    map_time_passed_idx_to_grci = {}

    # 3. File Iteration (Time files)
    # Resolve the glob pattern for time files
    actual_time_files_glob = time_files_template.format(date=date_str, station_id=station_id)
    sorted_time_files = sorted(glob.glob(actual_time_files_glob))

    if not sorted_time_files:
        print(f"Warning: No time files found for Station {station_id} using pattern: {actual_time_files_glob}")
        return {} # Return empty map if no time files

    print(f"Found {len(sorted_time_files)} time files for Station {station_id}.")

    for time_file_path in sorted_time_files:
        try:
            times_in_file = np.load(time_file_path)
            # Ensure times_in_file is 1D array
            if times_in_file.ndim > 1:
                times_in_file = times_in_file.squeeze()
            if times_in_file.ndim == 0: # Handle scalar case
                 times_in_file = np.array([times_in_file.item()])
            if times_in_file.size == 0:
                continue

            for timestamp in times_in_file:
                if timestamp != 0 and timestamp >= time_threshold_timestamp:
                    map_time_passed_idx_to_grci[time_passed_event_index] = global_raw_counter
                    time_passed_event_index += 1
                global_raw_counter += 1
            
            del times_in_file # Memory management
        except Exception as e:
            print(f"Error processing time file {time_file_path}: {e}")
            # Decide if to continue or return None/empty
    gc.collect()

    if time_passed_event_index == 0:
        print(f"No events passed time/threshold cut for Station {station_id}.")
        _save_pickle({}, map_cache_file_path) # Cache empty map
        return {}

    # 4. External Cuts Application
    combined_external_cut_mask = None
    if os.path.exists(external_cuts_file_path):
        try:
            # This assumes the cuts file contains a single boolean array
            # or a dictionary from which a combined mask can be derived.
            # For simplicity, let's assume it's a direct boolean mask.
            cuts_data = np.load(external_cuts_file_path, allow_pickle=True)
            if isinstance(cuts_data, dict): # If it's a dict of cuts
                print(f"Cuts file {external_cuts_file_path} is a dictionary. Combining cuts.")
                # Find a representative cut to get the expected length
                first_cut_key = next(iter(cuts_data), None)
                if first_cut_key:
                    num_cut_entries = len(cuts_data[first_cut_key])
                    if num_cut_entries != time_passed_event_index:
                        print(f"Warning: Length of cuts ({num_cut_entries}) in {external_cuts_file_path} "
                              f"does not match number of time-passed events ({time_passed_event_index}). "
                              f"Cuts may not be applied correctly.")
                        # Truncate or pad if necessary, or error out. For now, try to use intersection.
                        min_len = min(num_cut_entries, time_passed_event_index)
                        combined_external_cut_mask = np.ones(min_len, dtype=bool)
                        for cut_array in cuts_data.values():
                             combined_external_cut_mask &= cut_array[:min_len]
                        if time_passed_event_index > num_cut_entries: # More time events than cut entries
                            # Assume events beyond cut data pass (or fail, based on policy)
                            # For now, let's assume they fail if cut data is shorter
                            temp_mask = np.zeros(time_passed_event_index, dtype=bool)
                            temp_mask[:min_len] = combined_external_cut_mask
                            combined_external_cut_mask = temp_mask

                    else: # Lengths match
                        combined_external_cut_mask = np.ones(num_cut_entries, dtype=bool)
                        for cut_array in cuts_data.values():
                            combined_external_cut_mask &= cut_array
                else: # Empty cuts dictionary
                    print(f"Warning: Cuts file {external_cuts_file_path} is an empty dictionary. Assuming all pass external cuts.")
                    combined_external_cut_mask = np.ones(time_passed_event_index, dtype=bool)

            elif isinstance(cuts_data, np.ndarray) and cuts_data.dtype == bool:
                combined_external_cut_mask = cuts_data
                if len(combined_external_cut_mask) != time_passed_event_index:
                    print(f"Warning: Length of external_cuts_mask ({len(combined_external_cut_mask)}) "
                          f"does not match time_passed_event_index ({time_passed_event_index}). "
                          f"Cuts may not be applied correctly. Taking intersection.")
                    min_len = min(len(combined_external_cut_mask), time_passed_event_index)
                    mask_subset = combined_external_cut_mask[:min_len]
                    # Create a new mask of the correct full length
                    final_mask = np.zeros(time_passed_event_index, dtype=bool) # Assume fail if not covered
                    final_mask[:min_len] = mask_subset
                    combined_external_cut_mask = final_mask

            else:
                print(f"Warning: External cuts file {external_cuts_file_path} has unexpected format. Assuming all pass.")
                combined_external_cut_mask = np.ones(time_passed_event_index, dtype=bool)

        except Exception as e:
            print(f"Error loading or processing external cuts file {external_cuts_file_path}: {e}. Assuming all pass.")
            combined_external_cut_mask = np.ones(time_passed_event_index, dtype=bool)
    else:
        print(f"Warning: External cuts file not found: {external_cuts_file_path}. Assuming all pass external cuts.")
        combined_external_cut_mask = np.ones(time_passed_event_index, dtype=bool)

    # 5. Final Map Creation
    final_map_final_idx_to_grci = {}
    current_final_idx = 0
    for i in range(time_passed_event_index): # Iterate 0 to N-1 for time_passed_events
        if i < len(combined_external_cut_mask) and combined_external_cut_mask[i]:  # Event passes external cuts
            grci_for_this_event = map_time_passed_idx_to_grci[i]
            final_map_final_idx_to_grci[current_final_idx] = grci_for_this_event
            current_final_idx += 1
    
    print(f"Station {station_id}: {current_final_idx} events passed all cuts.")

    # 6. Caching
    _save_pickle(final_map_final_idx_to_grci, map_cache_file_path)
    print(f"Saved final_idx_to_grci_map to cache: {map_cache_file_path}")
    
    return final_map_final_idx_to_grci

# --- Function 2: Fetch Parameters by GRCI ---
def fetch_parameters_by_grci(
    list_of_grcis,
    station_id,
    date_str,
    parameter_name,
    parameter_files_template, # e.g. "path/data/{date}/{date}_Station{station_id}_{param_name}_*.npy"
    filename_event_count_regex # Compiled regex, e.g., re.compile(r"_(\d+)evts_")
    ):
    """
    Retrieves parameter values for a list of GRCIs for a specific station and parameter.
    Uses the '_Xevts_' convention in filenames.

    Args:
        list_of_grcis (list[int]): List of Global Raw Concatenated Indices.
        station_id (Union[int, str]): Station identifier.
        date_str (str): Date string (YYYYMMDD).
        parameter_name (str): Name of the parameter.
        parameter_files_template (str): Glob pattern template for parameter files.
        filename_event_count_regex (re.Pattern): Compiled regex to extract event count.

    Returns:
        dict: A dictionary mapping {GRCI: parameter_value}.
              GRCIs not found or with errors might be missing or have a NaN value.
    """
    if not list_of_grcis:
        return {}

    print(f"Fetching param '{parameter_name}' for {len(list_of_grcis)} GRCIs, Station {station_id}...")

    # 1. File Discovery and Sorting
    actual_param_files_glob = parameter_files_template.format(
        date=date_str, station_id=station_id, parameter_name=parameter_name
    )
    sorted_param_files = sorted(glob.glob(actual_param_files_glob))

    if not sorted_param_files:
        print(f"Warning: No parameter files found for {parameter_name}, Station {station_id} "
              f"using pattern: {actual_param_files_glob}")
        results_error = {grci: np.nan for grci in list_of_grcis} # Mark all as not found
        return results_error

    # 2. Preparation
    results = {}
    # Process GRCIs in sorted order to efficiently iterate through files
    # Keep track of original requested GRCIs to ensure all are handled
    unique_grcis_to_fetch_sorted = sorted(list(set(list_of_grcis)))
    
    grci_pointer = 0  # Points to the current GRCI in unique_grcis_to_fetch_sorted
    cumulative_grci_offset = 0  # GRCI at the start of the current file

    # 3. Iterate Through Data Files
    for param_file_path in sorted_param_files:
        if grci_pointer >= len(unique_grcis_to_fetch_sorted):
            break # All requested GRCIs have been processed or determined to be in earlier files

        match = filename_event_count_regex.search(os.path.basename(param_file_path))
        if not match:
            print(f"Warning: Could not extract event count from filename: {param_file_path}. Skipping file.")
            continue
        
        try:
            events_in_this_file = int(match.group(1))
            if events_in_this_file <= 0:
                 print(f"Warning: Invalid event count {events_in_this_file} in {param_file_path}. Skipping.")
                 continue
        except ValueError:
            print(f"Warning: Non-integer event count in {param_file_path}. Skipping.")
            continue

        file_grci_start = cumulative_grci_offset
        file_grci_end = cumulative_grci_offset + events_in_this_file
        
        # Determine which of the still-needed GRCIs fall into this file's range
        grcis_in_current_file_data = {} # Maps local_file_idx -> target_grci

        temp_grci_pointer = grci_pointer
        while temp_grci_pointer < len(unique_grcis_to_fetch_sorted):
            target_grci = unique_grcis_to_fetch_sorted[temp_grci_pointer]
            if target_grci >= file_grci_end:
                # This GRCI and subsequent ones are in later files
                break 
            
            if target_grci >= file_grci_start:
                local_file_idx = int(target_grci - file_grci_start) # Make sure it's int
                grcis_in_current_file_data[local_file_idx] = target_grci
                temp_grci_pointer += 1
            else:
                # This GRCI should have been in a previous file. This indicates an issue
                # or the GRCI is out of bounds of any file.
                print(f"Warning: Target GRCI {target_grci} is less than current file start {file_grci_start}. "
                      f"File: {param_file_path}. Marking as NaN.")
                results[target_grci] = np.nan 
                temp_grci_pointer += 1 # Move to next GRCI
                grci_pointer = temp_grci_pointer # Update main pointer
        
        if grcis_in_current_file_data:
            parameter_data_array = None # Ensure defined for finally
            try:
                # print(f"Loading {param_file_path} for {len(grcis_in_current_file_data)} values.")
                parameter_data_array = np.load(param_file_path, allow_pickle=True)
                if parameter_name != 'Traces': # Traces can be multi-dimensional
                    if parameter_data_array.ndim > 1:
                        parameter_data_array = parameter_data_array.squeeze()
                    if parameter_data_array.ndim == 0: # Handle scalar item
                        parameter_data_array = np.array([parameter_data_array.item()])


                for local_file_idx, target_grci in grcis_in_current_file_data.items():
                    if 0 <= local_file_idx < len(parameter_data_array):
                        results[target_grci] = parameter_data_array[local_file_idx]
                    else:
                        print(f"Error: Local index {local_file_idx} out of bounds for file {param_file_path} "
                              f"(len: {len(parameter_data_array)}) for GRCI {target_grci}. Setting to NaN.")
                        results[target_grci] = np.nan
                    # Advance the main grci_pointer if we've successfully processed or marked this GRCI
                    if target_grci == unique_grcis_to_fetch_sorted[grci_pointer]:
                         grci_pointer +=1

            except Exception as e:
                print(f"Error loading or processing parameter file {param_file_path}: {e}")
                for target_grci in grcis_in_current_file_data.values():
                    results[target_grci] = np.nan # Mark all GRCIs intended for this file as NaN
                    if target_grci == unique_grcis_to_fetch_sorted[grci_pointer]: # Ensure pointer advances
                         grci_pointer +=1
            finally:
                if parameter_data_array is not None:
                    del parameter_data_array
                gc.collect()
        
        cumulative_grci_offset += events_in_this_file

    # For any GRCIs that were requested but not found in any file (e.g., too large)
    while grci_pointer < len(unique_grcis_to_fetch_sorted):
        target_grci = unique_grcis_to_fetch_sorted[grci_pointer]
        if target_grci not in results: # If not already marked due to other errors
            print(f"Warning: GRCI {target_grci} was not found in any parameter file. Setting to NaN.")
            results[target_grci] = np.nan
        grci_pointer += 1
        
    # Ensure all originally requested GRCIs have an entry, even if it's NaN
    final_output = {grci: results.get(grci, np.nan) for grci in list_of_grcis}
    return final_output


# --- Function 3: Orchestrator ---
def add_parameter_orchestrator(
    events_dict,
    parameter_name,
    date_str,
    run_flag, # For organizing cache/checkpoints
    config # Dictionary with path templates and settings
    ):
    """
    Adds a specific parameter to the events_dict using the modular functions.

    Args:
        events_dict (dict): The main dictionary of events to be modified in-place.
        parameter_name (str): Name of the parameter to add.
        date_str (str): Date string (YYYYMMDD).
        run_flag (str): A flag for this run (e.g., 'base', 'test') for cache organization.
        config (dict): Configuration dictionary containing:
            'time_threshold_timestamp' (float)
            'time_files_template' (str)
            'external_cuts_file_template' (str) e.g. "path/cuts/{date}/{date}_Station{station_id}_Cuts.npy"
            'map_cache_template' (str) e.g. "cache/{date}/{flag}/maps/station_{station_id}_final_to_grci.pkl"
            'parameter_files_template' (str)
            'filename_event_count_regex_str' (str) e.g. r"_(\d+)evts_"
            'checkpoint_path_template' (str, optional) e.g. "checkpoints/{date}/{flag}/{dataset_name}_{param_name}.pkl"
            'dataset_name_for_checkpoint' (str, optional, used with checkpoint_path_template)

    Returns:
        dict: The modified events_dict.
    """
    print(f"\nOrchestrating addition of parameter '{parameter_name}' for date {date_str}, flag {run_flag}")

    # Extract config
    time_thresh = config['time_threshold_timestamp']
    time_files_tmpl = config['time_files_template']
    ext_cuts_tmpl = config['external_cuts_file_template']
    map_cache_tmpl = config['map_cache_template']
    param_files_tmpl = config['parameter_files_template']
    event_count_regex = re.compile(config['filename_event_count_regex_str'])
    
    checkpoint_file = None
    if 'checkpoint_path_template' in config and 'dataset_name_for_checkpoint' in config:
        checkpoint_file = config['checkpoint_path_template'].format(
            date=date_str, 
            flag=run_flag, 
            dataset_name=config['dataset_name_for_checkpoint'],
            parameter_name=parameter_name
        )
        # Attempt to load from checkpoint - This function assumes events_dict is ALREADY loaded
        # from a checkpoint by the CALLER if resumption is desired.
        # This function will SAVE checkpoints if path is provided.
        print(f"Checkpoint for events_dict (save only): {checkpoint_file}")


    # 1. Identify Unique Stations from events_dict
    unique_stations = set()
    for event_data in events_dict.values():
        for st_id_key in event_data.get("stations", {}).keys():
            try:
                unique_stations.add(int(st_id_key))
            except ValueError:
                print(f"Warning: Could not parse station ID '{st_id_key}' to int.")
    
    sorted_unique_stations = sorted(list(unique_stations))
    if not sorted_unique_stations:
        print("No stations found in events_dict.")
        return events_dict
        
    print(f"Processing {len(sorted_unique_stations)} stations: {sorted_unique_stations}")

    # 2. Loop Through Stations
    for station_id in sorted_unique_stations:
        print(f"\n--- Processing Station {station_id} for parameter '{parameter_name}' ---")
        
        # Resolve paths for this station
        current_map_cache_path = map_cache_tmpl.format(date=date_str, flag=run_flag, station_id=station_id)
        current_ext_cuts_path = ext_cuts_tmpl.format(date=date_str, station_id=station_id)

        # 2a. Get final_idx to GRCI map
        final_idx_to_grci_map = create_final_idx_to_grci_map(
            date_str, station_id, time_thresh, time_files_tmpl,
            current_ext_cuts_path, current_map_cache_path
        )

        if not final_idx_to_grci_map: # Handles None or empty dict
            print(f"Warning: No final_idx_to_GRCI map for station {station_id}. Skipping parameter addition for this station.")
            continue

        # 2b. Collect GRCIs needed for this station from events_dict
        grcis_to_fetch_for_station = []
        # Structure: {grci: [(event_dict_ref, station_key_in_event, list_position_in_param_array)]}
        map_grci_to_event_updates = defaultdict(list)
        
        events_needing_param_count = 0
        for event_id, event_data_ref in events_dict.items(): # Iterate directly over dict
            # Determine the correct key for the station (int or str)
            station_key_in_event = None
            if station_id in event_data_ref.get("stations", {}):
                station_key_in_event = station_id
            elif str(station_id) in event_data_ref.get("stations", {}):
                station_key_in_event = str(station_id)
            
            if station_key_in_event:
                station_event_data = event_data_ref["stations"][station_key_in_event]
                
                # Ensure parameter list exists and is of correct length
                indices_list = station_event_data.get("indices", [])
                if not indices_list: continue

                if parameter_name not in station_event_data or \
                   not isinstance(station_event_data[parameter_name], list) or \
                   len(station_event_data[parameter_name]) != len(indices_list):
                    station_event_data[parameter_name] = [None] * len(indices_list)

                # Check which entries are None and need fetching
                for pos, final_idx in enumerate(indices_list):
                    if station_event_data[parameter_name][pos] is None: # Only fetch if not already filled
                        events_needing_param_count+=1
                        grci = final_idx_to_grci_map.get(final_idx)
                        if grci is not None:
                            grcis_to_fetch_for_station.append(grci)
                            map_grci_to_event_updates[grci].append(
                                (event_data_ref, station_key_in_event, pos)
                            )
                        else:
                            print(f"Warning: final_idx {final_idx} not in map for station {station_id}. "
                                  f"Event {event_id}. Setting param '{parameter_name}' pos {pos} to NaN.")
                            station_event_data[parameter_name][pos] = np.nan
        
        if not grcis_to_fetch_for_station:
            if events_needing_param_count == 0:
                 print(f"Station {station_id}: No '{parameter_name}' values needed (all seem filled or no indices).")
            else: # This implies all needed final_idx were not in the map
                 print(f"Station {station_id}: No GRCIs to fetch for '{parameter_name}' (likely all final_idx missing from map).")
            # Save checkpoint even if no fetching, in case some NaNs were set
            if checkpoint_file: _save_pickle(events_dict, checkpoint_file)
            continue
        
        unique_grcis_to_fetch = sorted(list(set(grcis_to_fetch_for_station)))
        print(f"Station {station_id}: Need to fetch {len(unique_grcis_to_fetch)} unique GRCIs for '{parameter_name}'.")

        # 2c. Fetch parameter values for these GRCIs
        fetched_params_by_grci = fetch_parameters_by_grci(
            unique_grcis_to_fetch, station_id, date_str, parameter_name,
            param_files_tmpl, event_count_regex
        )

        # 2d. Update events_dict with fetched parameters
        updated_count = 0
        for grci, update_targets_list in map_grci_to_event_updates.items():
            value_to_assign = fetched_params_by_grci.get(grci, np.nan) # Default to NaN if fetch failed for a GRCI
            for event_data_ref, st_key, pos_in_list in update_targets_list:
                event_data_ref["stations"][st_key][parameter_name][pos_in_list] = value_to_assign
                updated_count +=1
        print(f"Station {station_id}: Updated {updated_count} entries for '{parameter_name}'.")

        # 2e. Save Checkpoint (for the entire events_dict)
        if checkpoint_file:
            print(f"Saving checkpoint after processing station {station_id} for '{parameter_name}'...")
            _save_pickle(events_dict, checkpoint_file)
        
        gc.collect() # Collect garbage after each station

    print(f"\nFinished orchestrating parameter '{parameter_name}'.")
    return events_dict

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    import configparser
    # Read configuration and get date 
    config = configparser.ConfigParser() 
    config.read(os.path.join('HRAStationDataAnalysis', 'config.ini')) 
    date = config['PARAMETERS']['date']


    # --- Mock Configuration ---
    data_path = "mock_data/HRAStationDataAnalysis/StationData"
    cache_path = "mock_data/cache"
    
    # Create mock directories
    os.makedirs(os.path.join(data_path, "nurFiles", date), exist_ok=True)
    os.makedirs(os.path.join(data_path, "cuts", date), exist_ok=True)
    os.makedirs(os.path.join(cache_path, date, "base", "maps"), exist_ok=True)
    os.makedirs(os.path.join(cache_path, date, "base", "checkpoints"), exist_ok=True)


    CONFIG = {
        'time_threshold_timestamp': datetime.datetime(2013, 1, 1, tzinfo=datetime.timezone.utc).timestamp(),
        'time_files_template': os.path.join(data_path, "nurFiles", "{date}", "{date}_Station{station_id}_Times_*.npy"),
        'external_cuts_file_template': os.path.join(data_path, "cuts", "{date}", "{date}_Station{station_id}_Cuts.npy"),
        'map_cache_template': os.path.join(cache_path, "{date}", "{flag}", "maps", "st_{station_id}_final_to_grci.pkl"),
        'parameter_files_template': os.path.join(data_path, "nurFiles", "{date}", "{date}_Station{station_id}_{parameter_name}_*_Xevts_*.npy").replace("_Xevts_", "_*evts_"), # Adjusted for glob
        'filename_event_count_regex_str': r"_(\d+)evts_",
        'checkpoint_path_template': os.path.join(cache_path, "{date}", "{flag}", "checkpoints", "{dataset_name}_{parameter_name}_events_dict.pkl"),
        'dataset_name_for_checkpoint': 'myDataset' # Example name
    }
    
    # First create map for all stations
    stations = [13, 14, 15, 17, 18, 19, 30]
    station_map = {}
    for station_id in stations:

        # --- Run Orchestrator ---
        map_cache_path = CONFIG['map_cache_template'].format(date=date, flag="base", station_id=station_id)
        if os.path.exists(map_cache_path): os.remove(map_cache_path) # Clear cache for test
    
        station_map[station_id] = create_final_idx_to_grci_map(
            date_str=date,
            station_id=station_id,
            time_threshold_timestamp=CONFIG['time_threshold_timestamp'],
            time_files_template=CONFIG['time_files_template'],
            external_cuts_file_path=CONFIG['external_cuts_file_template'].format(date=date, station_id=station_id),
            map_cache_file_path=map_cache_path
        )
        print(f"Station {station_id} final_idx_to_grci_map: {station_map[station_id]}")


    # Load coincidence events dictionary
    coincidence_events_path = os.path.join('HRAStationDataAnalysis', 'StationData', 'processedNumpyData', date, f'{date}_CoincidenceDatetimes.npy')
    if os.path.exists(coincidence_events_path):
        events_dict = np.load(coincidence_events_path, allow_pickle=True)
        coincidence_events = events_dict[0]
        coincidence_with_repeat_stations_events = events_dict[1]
    else:
        print(f"Warning: Coincidence events file not found at {coincidence_events_path}.")
        quit()


    parameters_to_add = ['Traces', 'SNR', 'ChiRCR', 'Chi2016', 'ChiBad', 'Zen', 'Azi']
    for param in parameters_to_add:
        ic(f"Adding parameter: {param}")
        coincidence_events = add_parameter_orchestrator(
            events_dict=coincidence_events,
            parameter_name=param,
            date_str=date,
            run_flag="base",
            config=CONFIG
        )
        print(f"Finished adding parameter: {param} for coincidence events.")

        # Save checkpoint after each parameter addition
        checkpoint_path = CONFIG['checkpoint_path_template'].format(
            date=date, flag="base", dataset_name=CONFIG['dataset_name_for_checkpoint'], parameter_name=param
        )
        _save_pickle(coincidence_events, checkpoint_path)
        print(f"Saved checkpoint for parameter '{param}' at {checkpoint_path}.")
        
    # Save the final events_dict with all parameters added
    final_events_dict_path = os.path.join('HRAStationDataAnalysis', 'StationData', 'processedNumpyData', date, f'{date}_CoincidenceDatetimes_with_all_params.pkl')
    _save_pickle(coincidence_events, final_events_dict_path)
    print(f"Final events_dict with all parameters saved at {final_events_dict_path}.")

    # Test the addition of SNR parameter
    # This is a mock test, assuming the SNR parameter is added correctly
    # Check if SNR was added correctly
    for event_id, event_data in coincidence_events.items():
        if 'SNR' in event_data['stations'][13]:
            print(f"Event {event_id} Station 13 SNR: {event_data['stations'][13]['SNR']}")
        else:
            print(f"Event {event_id} Station 13 SNR not found.")

    # Now process the coincidence_with_repeat_stations_events
    for param in parameters_to_add:
        ic(f"Adding parameter: {param} to repeat stations events")
        coincidence_with_repeat_stations_events = add_parameter_orchestrator(
            events_dict=coincidence_with_repeat_stations_events,
            parameter_name=param,
            date_str=date,
            run_flag="with_repeat",
            config=CONFIG
        )
        print(f"Finished adding parameter: {param} for repeat stations events.")

        # Save checkpoint after each parameter addition
        checkpoint_path = CONFIG['checkpoint_path_template'].format(
            date=date, flag="base", dataset_name=CONFIG['dataset_name_for_checkpoint'], parameter_name=param
        )
        _save_pickle(coincidence_with_repeat_stations_events, checkpoint_path)
        print(f"Saved checkpoint for parameter '{param}' at {checkpoint_path}.")

    