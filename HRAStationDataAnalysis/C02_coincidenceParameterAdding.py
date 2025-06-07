import os
import glob
import numpy as np
import pickle
import re
import gc
import datetime
from collections import defaultdict
import time # For logging/timestamps if needed
import tempfile # For atomic saving
from icecream import ic

# --- Helper for Caching ---
def _load_pickle(filepath):
    """Loads data from a pickle file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            ic(f"Error loading pickle file {filepath}: {e}")
    return None

def _save_pickle_atomic(data, filepath):
    """Saves data to a pickle file atomically, creating directories if needed."""
    temp_file_path = None # Ensure variable is defined for potential error handling
    try:
        base_dir = os.path.dirname(filepath)
        if base_dir: # Ensure base_dir is not empty if filepath is just a filename
            os.makedirs(base_dir, exist_ok=True)
        else: # Handle case where filepath is in current directory
            base_dir = '.' 

        # Create a temporary file in the same directory for atomic os.replace
        fd, temp_file_path = tempfile.mkstemp(suffix='.tmp', dir=base_dir)
        
        with os.fdopen(fd, 'wb') as tf:
            pickle.dump(data, tf, protocol=pickle.HIGHEST_PROTOCOL)
            # Ensure data is written to disk before renaming
            tf.flush()
            os.fsync(tf.fileno()) # Force write to disk

        # Atomically replace the old file with the new one
        os.replace(temp_file_path, filepath)
        ic(f"Atomically saved pickle file to {filepath}")
        temp_file_path = None # Indicate success, temp file is now the actual file
    except Exception as e:
        ic(f"Error saving pickle file atomically to {filepath}: {e}")
    finally:
        # If temp_file_path is still set, it means an error occurred before or during os.replace
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                ic(f"Removed temporary file {temp_file_path} after error.")
            except OSError as oe:
                ic(f"Error removing temporary file {temp_file_path}: {oe}")

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
    """
    ic(f"Attempting to create/load final_idx_to_grci_map for Station {station_id} on {date_str}")
    
    cached_map = _load_pickle(map_cache_file_path)
    if cached_map is not None:
        ic(f"Loaded final_idx_to_grci_map from cache: {map_cache_file_path}")
        return cached_map

    ic(f"Cache not found. Building map for Station {station_id}...")

    global_raw_counter = 0
    time_passed_event_index = 0
    map_time_passed_idx_to_grci = {}

    actual_time_files_glob = time_files_template.format(date=date_str, station_id=station_id)
    sorted_time_files = sorted(glob.glob(actual_time_files_glob))

    if not sorted_time_files:
        ic(f"Warning: No time files found for Station {station_id} using pattern: {actual_time_files_glob}")
        _save_pickle_atomic({}, map_cache_file_path) # Cache empty map
        return {}
    ic(f"Found {len(sorted_time_files)} time files for Station {station_id}.")

    for time_file_path in sorted_time_files:
        try:
            times_in_file = np.load(time_file_path)
            if times_in_file.ndim > 1: times_in_file = times_in_file.squeeze()
            if times_in_file.ndim == 0: times_in_file = np.array([times_in_file.item()])
            if times_in_file.size == 0: continue
            for timestamp in times_in_file:
                if timestamp != 0 and timestamp >= time_threshold_timestamp:
                    map_time_passed_idx_to_grci[time_passed_event_index] = global_raw_counter
                    time_passed_event_index += 1
                global_raw_counter += 1
            del times_in_file
        except Exception as e:
            ic(f"Error processing time file {time_file_path}: {e}")
    gc.collect()

    if time_passed_event_index == 0:
        ic(f"No events passed time/threshold cut for Station {station_id}.")
        _save_pickle_atomic({}, map_cache_file_path)
        return {}

    combined_external_cut_mask = np.ones(time_passed_event_index, dtype=bool) # Default to all pass
    if os.path.exists(external_cuts_file_path):
        try:
            cuts_data = np.load(external_cuts_file_path, allow_pickle=True)
            # Handle if cuts_data is a 0-d array containing a dict (common with np.save(..., dict_obj))
            if isinstance(cuts_data, np.ndarray) and cuts_data.ndim == 0 and isinstance(cuts_data.item(), dict):
                cuts_data = cuts_data.item()

            if isinstance(cuts_data, dict):
                ic(f"Cuts file {external_cuts_file_path} is a dictionary. Combining cuts.")
                first_cut_key = next(iter(cuts_data), None)
                if first_cut_key:
                    num_cut_entries = len(cuts_data[first_cut_key])
                    current_combined_mask = np.ones(num_cut_entries, dtype=bool)
                    for cut_array in cuts_data.values():
                        if len(cut_array) == num_cut_entries:
                            current_combined_mask &= cut_array
                        else:
                            ic(f"Warning: Cut array length mismatch in {external_cuts_file_path}. Skipping this cut.")
                    
                    if num_cut_entries == time_passed_event_index:
                        combined_external_cut_mask = current_combined_mask
                    else: # Length mismatch
                        ic(f"Warning: Combined cuts length ({num_cut_entries}) in {external_cuts_file_path} "
                              f"does not match time-passed events ({time_passed_event_index}). Adjusting mask.")
                        min_len = min(num_cut_entries, time_passed_event_index)
                        # Assume events outside the shorter range fail the cut
                        adjusted_mask = np.zeros(time_passed_event_index, dtype=bool) 
                        adjusted_mask[:min_len] = current_combined_mask[:min_len]
                        combined_external_cut_mask = adjusted_mask
                else:
                    ic(f"Warning: Cuts file {external_cuts_file_path} is an empty dictionary.")
            elif isinstance(cuts_data, np.ndarray) and cuts_data.dtype == bool:
                if len(cuts_data) == time_passed_event_index:
                    combined_external_cut_mask = cuts_data
                else:
                    ic(f"Warning: Length of external_cuts_mask ({len(cuts_data)}) "
                          f"does not match time_passed_event_index ({time_passed_event_index}). Adjusting.")
                    min_len = min(len(cuts_data), time_passed_event_index)
                    adjusted_mask = np.zeros(time_passed_event_index, dtype=bool)
                    adjusted_mask[:min_len] = cuts_data[:min_len]
                    combined_external_cut_mask = adjusted_mask
            else:
                ic(f"Warning: External cuts file {external_cuts_file_path} has unexpected format.")
        except Exception as e:
            ic(f"Error loading or processing external cuts file {external_cuts_file_path}: {e}.")
    else:
        ic(f"Warning: External cuts file not found: {external_cuts_file_path}.")

    final_map_final_idx_to_grci = {}
    current_final_idx = 0
    for i in range(time_passed_event_index):
        if i < len(combined_external_cut_mask) and combined_external_cut_mask[i]:
            grci_for_this_event = map_time_passed_idx_to_grci[i]
            final_map_final_idx_to_grci[current_final_idx] = grci_for_this_event
            current_final_idx += 1
    
    ic(f"Station {station_id}: {current_final_idx} events passed all cuts.")
    _save_pickle_atomic(final_map_final_idx_to_grci, map_cache_file_path)
    ic(f"Saved final_idx_to_grci_map to cache: {map_cache_file_path}")
    return final_map_final_idx_to_grci

# --- Function 2: Fetch Parameters by GRCI ---
def fetch_parameters_by_grci(
    list_of_grcis,
    station_id,
    date_str,
    parameter_name,
    parameter_files_template, 
    filename_event_count_regex 
    ):
    """
    Retrieves parameter values for a list of GRCIs.
    """
    if not list_of_grcis: return {}
    ic(f"Fetching param '{parameter_name}' for {len(list_of_grcis)} GRCIs, Station {station_id}...")

    actual_param_files_glob = parameter_files_template.format(
        date=date_str, station_id=station_id, parameter_name=parameter_name
    )
    sorted_param_files = sorted(glob.glob(actual_param_files_glob))

    if not sorted_param_files:
        ic(f"Warning: No files for {parameter_name}, St {station_id}, pattern: {actual_param_files_glob}")
        return {grci: np.nan for grci in list_of_grcis}

    results = {}
    unique_grcis_to_fetch_sorted = sorted(list(set(list_of_grcis)))
    grci_pointer = 0
    cumulative_grci_offset = 0

    for param_file_path in sorted_param_files:
        if grci_pointer >= len(unique_grcis_to_fetch_sorted): break
        match = filename_event_count_regex.search(os.path.basename(param_file_path))
        if not match:
            ic(f"Warning: No event count in filename: {param_file_path}. Skipping.")
            continue
        try:
            events_in_this_file = int(match.group(1))
            if events_in_this_file <= 0:
                 ic(f"Warning: Invalid event count {events_in_this_file} in {param_file_path}. Skipping.")
                 continue
        except ValueError:
            ic(f"Warning: Non-integer event count in {param_file_path}. Skipping.")
            continue

        file_grci_start = cumulative_grci_offset
        file_grci_end = cumulative_grci_offset + events_in_this_file
        grcis_in_current_file_data = {}
        
        temp_grci_pointer_for_file_scan = grci_pointer # Use a temp pointer for scanning within this file's range
        while temp_grci_pointer_for_file_scan < len(unique_grcis_to_fetch_sorted):
            target_grci = unique_grcis_to_fetch_sorted[temp_grci_pointer_for_file_scan]
            if target_grci >= file_grci_end: break
            if target_grci >= file_grci_start:
                local_file_idx = int(target_grci - file_grci_start)
                grcis_in_current_file_data[local_file_idx] = target_grci
                # Don't advance main grci_pointer here, only after successful fetch from array
            elif target_grci < file_grci_start and target_grci not in results: # Should ideally not happen if GRCIs are correct
                 ic(f"Warning: Target GRCI {target_grci} < file start {file_grci_start}. File: {param_file_path}. Marking NaN.")
                 results[target_grci] = np.nan
                 # If this GRCI was the one grci_pointer was at, advance grci_pointer
                 if temp_grci_pointer_for_file_scan == grci_pointer:
                     grci_pointer +=1 
            temp_grci_pointer_for_file_scan += 1
        
        if grcis_in_current_file_data:
            parameter_data_array = None
            try:
                parameter_data_array = np.load(param_file_path, allow_pickle=True)
                if parameter_name != 'Traces':
                    if parameter_data_array.ndim > 1: parameter_data_array = parameter_data_array.squeeze()
                    if parameter_data_array.ndim == 0: parameter_data_array = np.array([parameter_data_array.item()])

                for local_file_idx, target_grci in grcis_in_current_file_data.items():
                    if 0 <= local_file_idx < len(parameter_data_array):
                        results[target_grci] = parameter_data_array[local_file_idx]
                    else:
                        ic(f"Error: Local idx {local_file_idx} out of bounds for {param_file_path} (len {len(parameter_data_array)}) GRCI {target_grci}. NaN.")
                        results[target_grci] = np.nan
                    # Advance main pointer if this was the GRCI it was waiting for
                    if grci_pointer < len(unique_grcis_to_fetch_sorted) and \
                       target_grci == unique_grcis_to_fetch_sorted[grci_pointer]:
                        grci_pointer += 1
            except Exception as e:
                ic(f"Error loading/processing param file {param_file_path}: {e}")
                for target_grci_in_error_file in grcis_in_current_file_data.values():
                    results[target_grci_in_error_file] = np.nan
                    if grci_pointer < len(unique_grcis_to_fetch_sorted) and \
                       target_grci_in_error_file == unique_grcis_to_fetch_sorted[grci_pointer]:
                        grci_pointer += 1
            finally:
                if parameter_data_array is not None: del parameter_data_array
                gc.collect()
        cumulative_grci_offset += events_in_this_file

    while grci_pointer < len(unique_grcis_to_fetch_sorted):
        target_grci = unique_grcis_to_fetch_sorted[grci_pointer]
        if target_grci not in results:
            ic(f"Warning: GRCI {target_grci} not found in any file. Setting to NaN.")
            results[target_grci] = np.nan
        grci_pointer += 1
        
    return {grci: results.get(grci, np.nan) for grci in list_of_grcis}

# --- Function 3: Orchestrator ---
def add_parameter_orchestrator(
    events_dict,
    parameter_name,
    date_str,
    run_flag, 
    config 
    ):
    """
    Adds a specific parameter to the events_dict.
    """
    ic(f"\nOrchestrating addition of '{parameter_name}' for date {date_str}, flag {run_flag}")
    time_thresh = config['time_threshold_timestamp']
    time_files_tmpl = config['time_files_template']
    ext_cuts_tmpl = config['external_cuts_file_template']
    map_cache_tmpl = config['map_cache_template']
    param_files_tmpl = config['parameter_files_template']
    event_count_regex = re.compile(config['filename_event_count_regex_str'])
    
    checkpoint_file = None
    # The checkpoint_path in config is the one this orchestrator will save to.
    if 'checkpoint_path_template' in config and 'dataset_name_for_checkpoint' in config: # dataset_name_for_checkpoint is set by caller
        checkpoint_file = config['checkpoint_path_template'].format(
            date=date_str, 
            flag=run_flag, # run_flag is specific to this dataset type
            dataset_name=config['dataset_name_for_checkpoint'], # general name, differentiated by flag
            parameter_name=parameter_name
        )
        ic(f"Orchestrator will save progress to: {checkpoint_file}")

    unique_stations = set()
    for event_data in events_dict.values():
        for st_id_key in event_data.get("stations", {}).keys():
            try: unique_stations.add(int(st_id_key))
            except ValueError: ic(f"Warning: Could not parse station ID '{st_id_key}' to int.")
    
    sorted_unique_stations = sorted(list(unique_stations))
    if not sorted_unique_stations: ic("No stations in events_dict."); return events_dict
    ic(f"Processing {len(sorted_unique_stations)} stations: {sorted_unique_stations}")

    for station_id in sorted_unique_stations:
        ic(f"\n--- Processing Station {station_id} for parameter '{parameter_name}' ---")
        current_map_cache_path = map_cache_tmpl.format(date=date_str, flag=run_flag, station_id=station_id)
        current_ext_cuts_path = ext_cuts_tmpl.format(date=date_str, station_id=station_id)

        final_idx_to_grci_map = create_final_idx_to_grci_map(
            date_str, station_id, time_thresh, time_files_tmpl,
            current_ext_cuts_path, current_map_cache_path
        )
        if not final_idx_to_grci_map:
            ic(f"Warning: No map for station {station_id}. Skipping.")
            continue

        grcis_to_fetch_for_station = []
        map_grci_to_event_updates = defaultdict(list)
        events_needing_param_count = 0

        for event_id, event_data_ref in events_dict.items():
            station_key_in_event = None
            if station_id in event_data_ref.get("stations", {}): station_key_in_event = station_id
            elif str(station_id) in event_data_ref.get("stations", {}): station_key_in_event = str(station_id)
            
            if station_key_in_event:
                station_event_data = event_data_ref["stations"][station_key_in_event]
                indices_list = station_event_data.get("indices", [])
                if not indices_list: continue

                if parameter_name not in station_event_data or \
                   not isinstance(station_event_data[parameter_name], list) or \
                   len(station_event_data[parameter_name]) != len(indices_list):
                    station_event_data[parameter_name] = [None] * len(indices_list)

                for pos, final_idx in enumerate(indices_list):
                    if station_event_data[parameter_name][pos] is None:
                        events_needing_param_count += 1
                        grci = final_idx_to_grci_map.get(final_idx)
                        if grci is not None:
                            grcis_to_fetch_for_station.append(grci)
                            map_grci_to_event_updates[grci].append(
                                (event_data_ref, station_key_in_event, pos)
                            )
                        else:
                            ic(f"Warn: final_idx {final_idx} not in map for St {station_id}, Ev {event_id}. NaN for '{parameter_name}' pos {pos}.")
                            station_event_data[parameter_name][pos] = np.nan
        
        if not grcis_to_fetch_for_station:
            if events_needing_param_count == 0: ic(f"St {station_id}: No '{parameter_name}' values needed.")
            else: ic(f"St {station_id}: No GRCIs to fetch for '{parameter_name}' (all final_idx missing from map?).")
            if checkpoint_file: _save_pickle_atomic(events_dict, checkpoint_file) # Save if NaNs were set
            continue
        
        unique_grcis_to_fetch = sorted(list(set(grcis_to_fetch_for_station)))
        ic(f"St {station_id}: Fetching {len(unique_grcis_to_fetch)} unique GRCIs for '{parameter_name}'.")

        fetched_params_by_grci = fetch_parameters_by_grci(
            unique_grcis_to_fetch, station_id, date_str, parameter_name,
            param_files_tmpl, event_count_regex
        )
        updated_count = 0
        for grci, update_targets_list in map_grci_to_event_updates.items():
            value_to_assign = fetched_params_by_grci.get(grci, np.nan)
            for event_data_ref, st_key, pos_in_list in update_targets_list:
                event_data_ref["stations"][st_key][parameter_name][pos_in_list] = value_to_assign
                updated_count +=1
        ic(f"St {station_id}: Updated {updated_count} entries for '{parameter_name}'.")

        if checkpoint_file:
            ic(f"Saving checkpoint after St {station_id} for '{parameter_name}' to {checkpoint_file}")
            _save_pickle_atomic(events_dict, checkpoint_file)
        gc.collect()
    ic(f"\nFinished orchestrating parameter '{parameter_name}'.")
    return events_dict

# --- Main Script Execution ---
if __name__ == '__main__':
    import configparser
    config_parser = configparser.ConfigParser() 
    config_parser.read(os.path.join('HRAStationDataAnalysis', 'config.ini')) 
    date = config_parser['PARAMETERS']['date']
    date_processing = config_parser['PARAMETERS']['date_processing']
    ic(f"Running parameter addition for date {date} with processing date {date_processing}")

    data_path = "HRAStationDataAnalysis/StationData" # Relative to script location
    cache_path = "HRAStationDataAnalysis/cache"   # Relative to script location
    
    # Ensure base directories for data and cache exist if running standalone test
    os.makedirs(os.path.join(data_path, "nurFiles", date), exist_ok=True)
    os.makedirs(os.path.join(data_path, "cuts", date), exist_ok=True)
    # Cache dirs will be created by _save_pickle_atomic if needed

    CONFIG = {
        'time_threshold_timestamp': datetime.datetime(2013, 1, 1, tzinfo=datetime.timezone.utc).timestamp(),
        'time_files_template': os.path.join(data_path, "nurFiles", "{date}", "{date}_Station{station_id}_Times*.npy"),
        'external_cuts_file_template': os.path.join(data_path, "cuts", "{date}", "{date}_Station{station_id}_Cuts.npy"),
        'map_cache_template': os.path.join(cache_path, "{date}", "{flag}", "maps", "st_{station_id}_final_to_grci.pkl"),
        'parameter_files_template': os.path.join(data_path, "nurFiles", "{date}", "{date}_Station{station_id}_{parameter_name}*_Xevts_*.npy").replace("_Xevts_", "_*evts_"),
        'filename_event_count_regex_str': r"_(\d+)evts_",
        'checkpoint_path_template': os.path.join(cache_path, "{date}", "{flag}", "checkpoints", "{dataset_name}_{parameter_name}_events_dict.pkl"),
        'dataset_name_for_checkpoint': 'HRA_Events' # Generic name, will be combined with flag
    }
    
    ic("Pre-calculating GRCI maps for all relevant stations...")
    stations = [13, 14, 15, 17, 18, 19, 30] # Example stations from your script
    # You might run this for different flags if map dependencies change with flag
    # For now, assuming "base" flag maps are applicable or you'll adjust run_flag for map creation
    map_creation_flag = "base" # Or loop through flags if maps are flag-dependent
    for station_id in stations:
        map_cache_p = CONFIG['map_cache_template'].format(date=date, flag=map_creation_flag, station_id=station_id)
        # if os.path.exists(map_cache_p): os.remove(map_cache_p) # Optional: force rebuild for testing
        create_final_idx_to_grci_map(
            date_str=date, station_id=station_id,
            time_threshold_timestamp=CONFIG['time_threshold_timestamp'],
            time_files_template=CONFIG['time_files_template'],
            external_cuts_file_path=CONFIG['external_cuts_file_template'].format(date=date, station_id=station_id),
            map_cache_file_path=map_cache_p
        )
    ic("Finished pre-calculating GRCI maps.")

    # Load initial coincidence events dictionary
    initial_events_data_path = os.path.join(data_path, 'processedNumpyData', date, f'{date_processing}_CoincidenceDatetimes.npy')
    initial_coincidence_events = None
    initial_coincidence_with_repeat_stations_events = None
    initial_coincidence_with_repeated_eventIDs = None

    if os.path.exists(initial_events_data_path):
        loaded_npy_data = np.load(initial_events_data_path, allow_pickle=True)
        if len(loaded_npy_data) >= 2:
            initial_coincidence_events = loaded_npy_data[0]
            initial_coincidence_with_repeat_stations_events = loaded_npy_data[1]
            initial_coincidence_with_repeated_eventIDs = loaded_npy_data[2] if len(loaded_npy_data) > 2 else None
            ic(f"Successfully loaded initial event dictionaries from {initial_events_data_path}")
        else:
            ic(f"Error: Expected at least 2 dictionaries in {initial_events_data_path}, found {len(loaded_npy_data)}.")
            exit()
    else:
        ic(f"CRITICAL Error: Initial Coincidence events file not found at {initial_events_data_path}. Exiting.")
        exit()

    parameters_to_add = ['Traces', 'SNR', 'ChiRCR', 'Chi2016', 'ChiBad', 'Zen', 'Azi']
    
    datasets_to_process = [
        {"name": "CoincidenceEvents", "data_dict": initial_coincidence_events, "run_flag": "base", "final_save_name": f'{date_processing}_CoincidenceDatetimes_with_all_params.pkl'},
        {"name": "CoincidenceEventsWithRepeat", "data_dict": initial_coincidence_with_repeat_stations_events, "run_flag": "with_repeat", "final_save_name": f'{date_processing}_CoincidenceRepeatStations_with_all_params.pkl'},
        {"name": "CoincidenceEventsWithRepeatedEventIDs", "data_dict": initial_coincidence_with_repeated_eventIDs, "run_flag": "with_repeated_eventIDs", "final_save_name": f'{date_processing}_CoincidenceRepeatEventIDs_with_all_params.pkl'}
    ]

    for dataset_info in datasets_to_process:
        dataset_label = dataset_info["name"]
        working_events_dict = dataset_info["data_dict"] # This will be updated
        current_run_flag = dataset_info["run_flag"]
        final_save_filename = dataset_info["final_save_name"]
        
        ic(f"\nProcessing Dataset: {dataset_label} with run_flag: {current_run_flag}")

        for param_idx, param_name in enumerate(parameters_to_add):
            ic(f"--- {dataset_label}: Adding Parameter '{param_name}' ({param_idx + 1}/{len(parameters_to_add)}) ---")

            # Define checkpoint path for the current parameter and dataset combination
            # This is where the orchestrator will save its progress after each station
            current_param_checkpoint_path = CONFIG['checkpoint_path_template'].format(
                date=date, 
                flag=current_run_flag, 
                dataset_name=CONFIG['dataset_name_for_checkpoint'], 
                parameter_name=param_name
            )
            ic(f"Target checkpoint for this step: {current_param_checkpoint_path}")

            # Determine the state to start with for this parameter
            # Option 1: Resume from current parameter's checkpoint (if script crashed mid-parameter)
            if os.path.exists(current_param_checkpoint_path):
                ic(f"Resuming '{param_name}' for '{dataset_label}' from its own checkpoint: {current_param_checkpoint_path}")
                loaded_data = _load_pickle(current_param_checkpoint_path)
                if loaded_data is not None:
                    working_events_dict = loaded_data
                else:
                    ic(f"Failed to load {current_param_checkpoint_path}. Using state from previous parameter if available.")
                    # Fall through to load previous parameter's checkpoint or use initial
            
            # Option 2: If not resuming, and not the first parameter, start from previous parameter's result
            elif param_idx > 0:
                prev_param_name = parameters_to_add[param_idx - 1]
                prev_param_checkpoint_path = CONFIG['checkpoint_path_template'].format(
                    date=date, 
                    flag=current_run_flag, 
                    dataset_name=CONFIG['dataset_name_for_checkpoint'], 
                    parameter_name=prev_param_name
                )
                if os.path.exists(prev_param_checkpoint_path):
                    ic(f"Starting '{param_name}' for '{dataset_label}' from '{prev_param_name}'s checkpoint: {prev_param_checkpoint_path}")
                    loaded_data = _load_pickle(prev_param_checkpoint_path)
                    if loaded_data is not None:
                        working_events_dict = loaded_data
                    else:
                        ic(f"Failed to load {prev_param_checkpoint_path}. Using initial data for {dataset_label} for this parameter.")
                        # This case means previous checkpoint was corrupt, so we'd restart this dataset from its initial state for current param
                        # For safety, one might choose to revert to the absolute initial state of the dataset here.
                        # For now, it implies using the 'working_events_dict' which, if previous loads failed, is initial.
                        if dataset_label == "CoincidenceEvents": working_events_dict = initial_coincidence_events.copy() # Or deepcopy
                        else: working_events_dict = initial_coincidence_with_repeat_stations_events.copy()


                else: # Previous checkpoint doesn't exist (shouldn't happen if logic is correct, unless first run after a clean)
                    ic(f"Checkpoint for previous parameter '{prev_param_name}' not found. Using initial data for '{dataset_label}' for parameter '{param_name}'.")
                    if dataset_label == "CoincidenceEvents": working_events_dict = initial_coincidence_events.copy()
                    else: working_events_dict = initial_coincidence_with_repeat_stations_events.copy()
            else:
                # First parameter for this dataset, use the initially loaded version (or a fresh copy)
                ic(f"Starting first parameter '{param_name}' for '{dataset_label}' with its initial data.")
                if dataset_label == "CoincidenceEvents": working_events_dict = initial_coincidence_events.copy() 
                else: working_events_dict = initial_coincidence_with_repeat_stations_events.copy()


            # The orchestrator modifies working_events_dict in-place AND saves to current_param_checkpoint_path
            add_parameter_orchestrator(
                events_dict=working_events_dict,
                parameter_name=param_name,
                date_str=date,
                run_flag=current_run_flag, # Critical for map cache path inside orchestrator
                config=CONFIG # Orchestrator uses templates from here to form checkpoint path
            )
            ic(f"--- Finished processing '{param_name}' for '{dataset_label}'. State saved in {current_param_checkpoint_path} ---")

        # After all parameters are processed for the current dataset, save the final result
        final_dataset_save_path = os.path.join(data_path, 'processedNumpyData', date, final_save_filename)
        _save_pickle_atomic(working_events_dict, final_dataset_save_path)
        ic(f"Final version of '{dataset_label}' with all parameters saved to: {final_dataset_save_path}")
        
        # Optional: Clean up intermediate parameter checkpoints for this dataset if desired
        # for param_name_to_clean in parameters_to_add:
        #     intermediate_cp_path = CONFIG['checkpoint_path_template'].format(date=date, flag=current_run_flag, dataset_name=CONFIG['dataset_name_for_checkpoint'], parameter_name=param_name_to_clean)
        #     if os.path.exists(intermediate_cp_path):
        #         try: os.remove(intermediate_cp_path); ic(f"Cleaned up: {intermediate_cp_path}")
        #         except Exception as e: ic(f"Error cleaning {intermediate_cp_path}: {e}")


    ic("All datasets and parameters processed.")

