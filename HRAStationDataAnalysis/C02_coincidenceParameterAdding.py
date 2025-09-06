import os
import glob
import numpy as np
import pickle
import re
import gc
import datetime
from collections import defaultdict
import time
import tempfile
from icecream import ic
from HRAStationDataAnalysis.C_utils import getTimeEventMasks

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
    temp_file_path = None
    try:
        base_dir = os.path.dirname(filepath)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        else:
            base_dir = '.' 

        fd, temp_file_path = tempfile.mkstemp(suffix='.tmp', dir=base_dir)
        
        with os.fdopen(fd, 'wb') as tf:
            pickle.dump(data, tf, protocol=pickle.HIGHEST_PROTOCOL)
            tf.flush()
            os.fsync(tf.fileno())

        os.replace(temp_file_path, filepath)
        ic(f"Atomically saved pickle file to {filepath}")
        temp_file_path = None
    except Exception as e:
        ic(f"Error saving pickle file atomically to {filepath}: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                ic(f"Removed temporary file {temp_file_path} after error.")
            except OSError as oe:
                ic(f"Error removing temporary file {temp_file_path}: {oe}")

# --- Function 1: Create Final Index to GRCI Map (Refactored) ---
def create_final_idx_to_grci_map(
    date_str,
    station_id,
    time_files_template,
    event_id_files_template,
    external_cuts_file_path,
    map_cache_file_path
    ):
    """
    Creates a mapping from a final_idx (post-all-cuts) to a GRCI
    (Global Raw Concatenated Index) using getTimeEventMasks for pre-filtering.
    """
    ic(f"Attempting to create/load final_idx_to_grci_map for Station {station_id} on {date_str}")
    
    cached_map = _load_pickle(map_cache_file_path)
    if cached_map is not None:
        ic(f"Loaded final_idx_to_grci_map from cache: {map_cache_file_path}")
        return cached_map

    ic(f"Cache not found. Building map for Station {station_id}...")

    # Load and concatenate all time and event_id files for the station
    sorted_time_files = sorted(glob.glob(time_files_template.format(date=date_str, station_id=station_id)))
    sorted_eventid_files = sorted(glob.glob(event_id_files_template.format(date=date_str, station_id=station_id)))

    if not sorted_time_files:
        ic(f"Warning: No time files found for Station {station_id}.")
        _save_pickle_atomic({}, map_cache_file_path)
        return {}
    if len(sorted_time_files) != len(sorted_eventid_files):
        ic(f"Warning: Mismatch in file count between Time ({len(sorted_time_files)}) and EventID ({len(sorted_eventid_files)}) files for Station {station_id}.")
        _save_pickle_atomic({}, map_cache_file_path)
        return {}

    try:
        all_times = np.concatenate([np.load(f) for f in sorted_time_files])
        all_event_ids = np.concatenate([np.load(f) for f in sorted_eventid_files])
    except Exception as e:
        ic(f"Error loading or concatenating files for station {station_id}: {e}")
        return {}
    
    if len(all_times) != len(all_event_ids):
        ic(f"Warning: Mismatch in total event count between concatenated Time and EventID arrays for Station {station_id}.")
        return {}

    # Use utility to get time and uniqueness masks
    initial_mask, unique_indices = getTimeEventMasks(all_times, all_event_ids)
    
    # Get the Global Raw Concatenated Indices of events passing the initial time cut
    grcis_time_passed = np.where(initial_mask)[0]
    # Further filter by uniqueness to get the GRCIs of events for which external cuts are defined
    grcis_pre_external_cuts = grcis_time_passed[unique_indices]
    
    num_events_pre_external_cuts = len(grcis_pre_external_cuts)

    if num_events_pre_external_cuts == 0:
        ic(f"No events passed time/uniqueness cuts for Station {station_id}.")
        _save_pickle_atomic({}, map_cache_file_path)
        return {}

    # Load and apply the external cuts mask
    combined_external_cut_mask = np.ones(num_events_pre_external_cuts, dtype=bool)
    if os.path.exists(external_cuts_file_path):
        try:
            cuts_data = np.load(external_cuts_file_path, allow_pickle=True)
            if isinstance(cuts_data, np.ndarray) and cuts_data.ndim == 0 and isinstance(cuts_data.item(), dict):
                cuts_data = cuts_data.item()

            if isinstance(cuts_data, dict):
                first_cut_key = next(iter(cuts_data), None)
                if first_cut_key:
                    num_cut_entries = len(cuts_data[first_cut_key])
                    mask_from_dict = np.ones(num_cut_entries, dtype=bool)
                    for cut_array in cuts_data.values():
                        if len(cut_array) == num_cut_entries:
                            mask_from_dict &= cut_array
                    
                    if num_cut_entries != num_events_pre_external_cuts:
                        ic(f"Warning: Cuts length ({num_cut_entries}) in {external_cuts_file_path} doesn't match pre-filtered events ({num_events_pre_external_cuts}). Adjusting.")
                        min_len = min(num_cut_entries, num_events_pre_external_cuts)
                        combined_external_cut_mask = np.zeros(num_events_pre_external_cuts, dtype=bool)
                        combined_external_cut_mask[:min_len] = mask_from_dict[:min_len]
                    else:
                        combined_external_cut_mask = mask_from_dict
            elif isinstance(cuts_data, np.ndarray) and cuts_data.dtype == bool:
                if len(cuts_data) != num_events_pre_external_cuts:
                    ic(f"Warning: Cuts array length ({len(cuts_data)}) in {external_cuts_file_path} doesn't match pre-filtered events ({num_events_pre_external_cuts}). Adjusting.")
                    min_len = min(len(cuts_data), num_events_pre_external_cuts)
                    combined_external_cut_mask = np.zeros(num_events_pre_external_cuts, dtype=bool)
                    combined_external_cut_mask[:min_len] = cuts_data[:min_len]
                else:
                    combined_external_cut_mask = cuts_data
        except Exception as e:
            ic(f"Error processing external cuts {external_cuts_file_path}: {e}")
    
    # Apply the external cuts mask to get the final list of GRCIs
    final_grcis = grcis_pre_external_cuts[combined_external_cut_mask]
    
    # Create the final map: {final_idx: GRCI}
    final_map_final_idx_to_grci = {i: grci for i, grci in enumerate(final_grcis)}
    
    ic(f"Station {station_id}: {len(final_map_final_idx_to_grci)} events passed all cuts.")
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
        
        temp_grci_pointer_for_file_scan = grci_pointer
        while temp_grci_pointer_for_file_scan < len(unique_grcis_to_fetch_sorted):
            target_grci = unique_grcis_to_fetch_sorted[temp_grci_pointer_for_file_scan]
            if target_grci >= file_grci_end: break
            if target_grci >= file_grci_start:
                local_file_idx = int(target_grci - file_grci_start)
                grcis_in_current_file_data[local_file_idx] = target_grci
            elif target_grci < file_grci_start and target_grci not in results:
                 ic(f"Warning: Target GRCI {target_grci} < file start {file_grci_start}. File: {param_file_path}. Marking NaN.")
                 results[target_grci] = np.nan
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
    date_processing_str,
    run_flag, 
    config 
    ):
    """
    Adds a specific parameter to the events_dict.
    """
    ic(f"\nOrchestrating addition of '{parameter_name}' for date {date_str} on {date_processing_str}, flag {run_flag}")
    time_files_tmpl = config['time_files_template']
    # Derive EventID template from Time template
    eventid_files_tmpl = config['time_files_template'].replace("_Times", "_EventIDs")
    ext_cuts_tmpl = config['external_cuts_file_template']
    map_cache_tmpl = config['map_cache_template']
    param_files_tmpl = config['parameter_files_template']
    event_count_regex = re.compile(config['filename_event_count_regex_str'])
    
    checkpoint_file = None
    if 'checkpoint_path_template' in config and 'dataset_name_for_checkpoint' in config:
        checkpoint_file = config['checkpoint_path_template'].format(
            date_processing=date_processing_str, 
            flag=run_flag,
            dataset_name=config['dataset_name_for_checkpoint'],
            parameter_name=parameter_name
        )
        ic(f"Orchestrator will save progress to: {checkpoint_file}")

    unique_stations = {int(st_id) for event in events_dict.values() for st_id in event.get("stations", {}).keys()}
    sorted_unique_stations = sorted(list(unique_stations))
    if not sorted_unique_stations: ic("No stations in events_dict."); return events_dict
    ic(f"Processing {len(sorted_unique_stations)} stations: {sorted_unique_stations}")

    for station_id in sorted_unique_stations:
        ic(f"\n--- Processing Station {station_id} for parameter '{parameter_name}' ---")
        current_map_cache_path = map_cache_tmpl.format(date_processing=date_processing_str, flag=run_flag, station_id=station_id)
        current_ext_cuts_path = ext_cuts_tmpl.format(date=date_str, station_id=station_id)

        final_idx_to_grci_map = create_final_idx_to_grci_map(
            date_str, station_id, time_files_tmpl, eventid_files_tmpl,
            current_ext_cuts_path, current_map_cache_path
        )
        if not final_idx_to_grci_map:
            ic(f"Warning: No map for station {station_id}. Skipping.")
            continue

        grcis_to_fetch_for_station = []
        map_grci_to_event_updates = defaultdict(list)
        events_needing_param_count = 0

        for event_id, event_data_ref in events_dict.items():
            station_key_in_event = str(station_id) if str(station_id) in event_data_ref.get("stations", {}) else station_id if station_id in event_data_ref.get("stations", {}) else None
            
            if station_key_in_event:
                station_event_data = event_data_ref["stations"][station_key_in_event]
                indices_list = station_event_data.get("indices", [])
                if not indices_list: continue

                if parameter_name not in station_event_data or not isinstance(station_event_data[parameter_name], list) or len(station_event_data[parameter_name]) != len(indices_list):
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
            if checkpoint_file: _save_pickle_atomic(events_dict, checkpoint_file)
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
    # (The __main__ block remains unchanged as it sets up configuration and calls the orchestrator)
    import configparser
    config_parser = configparser.ConfigParser() 
    config_parser.read(os.path.join('HRAStationDataAnalysis', 'config.ini')) 
    date = config_parser['PARAMETERS']['date']
    date_cuts = config_parser['PARAMETERS']['date_cuts']
    date_processing = config_parser['PARAMETERS']['date_processing']
    ic(f"Running parameter addition for date {date} with processing date {date_processing}")

    data_path = "HRAStationDataAnalysis/StationData"
    cache_path = "HRAStationDataAnalysis/cache"
    
    os.makedirs(os.path.join(data_path, "nurFiles", date), exist_ok=True)
    os.makedirs(os.path.join(data_path, "cuts", date), exist_ok=True)

    CONFIG = {
        'time_files_template': os.path.join(data_path, "nurFiles", "{date}", "{date}_Station{station_id}_Times*.npy"),
        'external_cuts_file_template': os.path.join(data_path, "cuts", "{date}", "{date}_Station{station_id}_Cuts.npy"),
        'map_cache_template': os.path.join(cache_path, "{date_processing}", "{flag}", "maps", "st_{station_id}_final_to_grci.pkl"),
        'parameter_files_template': os.path.join(data_path, "nurFiles", "{date}", "{date}_Station{station_id}_{parameter_name}*_Xevts_*.npy").replace("_Xevts_", "_*evts_"),
        'filename_event_count_regex_str': r"_(\d+)evts_",
        'checkpoint_path_template': os.path.join(cache_path, "{date_processing}", "{flag}", "checkpoints", "{dataset_name}_{parameter_name}_events_dict.pkl"),
        'dataset_name_for_checkpoint': 'HRA_Events'
    }
    
    ic("Pre-calculating GRCI maps for all relevant stations...")
    stations = [13, 14, 15, 17, 18, 19, 30]
    map_creation_flag = "base"
    eventid_files_tmpl_main = CONFIG['time_files_template'].replace("_Times", "_EventIDs")
    for station_id in stations:
        map_cache_p = CONFIG['map_cache_template'].format(date_processing=date_processing, flag=map_creation_flag, station_id=station_id)
        create_final_idx_to_grci_map(
            date_str=date, station_id=station_id,
            time_files_template=CONFIG['time_files_template'],
            event_id_files_template=eventid_files_tmpl_main,
            external_cuts_file_path=CONFIG['external_cuts_file_template'].format(date=date, station_id=station_id),
            map_cache_file_path=map_cache_p
        )
    ic("Finished pre-calculating GRCI maps.")

    initial_events_data_path = os.path.join(data_path, 'processedNumpyData', date, f'{date_processing}_CoincidenceDatetimes.npy')
    if not os.path.exists(initial_events_data_path):
        ic(f"CRITICAL Error: Initial Coincidence events file not found at {initial_events_data_path}. Exiting.")
        exit()
        
    loaded_npy_data = np.load(initial_events_data_path, allow_pickle=True)
    initial_coincidence_events = loaded_npy_data[0]
    initial_coincidence_with_repeat_stations_events = loaded_npy_data[1]
    initial_coincidence_with_repeated_eventIDs = loaded_npy_data[2] if len(loaded_npy_data) > 2 else {}
    ic(f"Successfully loaded initial event dictionaries from {initial_events_data_path}")

    parameters_to_add = ['Traces', 'SNR', 'ChiRCR', 'Chi2016', 'ChiBad', 'Zen', 'Azi', 'Times']
    
    datasets_to_process = [
        {"name": "CoincidenceEvents", "data_dict": initial_coincidence_events, "run_flag": "base", "final_save_name": f'{date_processing}_CoincidenceDatetimes_with_all_params.pkl'},
        {"name": "CoincidenceEventsWithRepeat", "data_dict": initial_coincidence_with_repeat_stations_events, "run_flag": "with_repeat", "final_save_name": f'{date_processing}_CoincidenceRepeatStations_with_all_params.pkl'},
        {"name": "CoincidenceEventsWithRepeatedEventIDs", "data_dict": initial_coincidence_with_repeated_eventIDs, "run_flag": "with_repeated_eventIDs", "final_save_name": f'{date_processing}_CoincidenceRepeatEventIDs_with_all_params.pkl'}
    ]

    for dataset_info in datasets_to_process:
        dataset_label = dataset_info["name"]
        working_events_dict = dataset_info["data_dict"].copy()
        current_run_flag = dataset_info["run_flag"]
        final_save_filename = dataset_info["final_save_name"]
        
        ic(f"\nProcessing Dataset: {dataset_label} with run_flag: {current_run_flag}")
        if not working_events_dict:
            ic(f"Dataset '{dataset_label}' is empty. Skipping.")
            continue

        for param_idx, param_name in enumerate(parameters_to_add):
            ic(f"--- {dataset_label}: Adding Parameter '{param_name}' ({param_idx + 1}/{len(parameters_to_add)}) ---")

            current_param_checkpoint_path = CONFIG['checkpoint_path_template'].format(date_processing=date_processing, flag=current_run_flag, dataset_name=CONFIG['dataset_name_for_checkpoint'], parameter_name=param_name)
            
            # Simplified Resume Logic: Always start from the previous parameter's finished state, or initial if it's the first.
            # Resume from a parameter's own mid-run checkpoint if it exists.
            if os.path.exists(current_param_checkpoint_path):
                 ic(f"Resuming '{param_name}' from its own checkpoint: {current_param_checkpoint_path}")
                 working_events_dict = _load_pickle(current_param_checkpoint_path)
            elif param_idx > 0:
                prev_param_name = parameters_to_add[param_idx - 1]
                prev_param_checkpoint_path = CONFIG['checkpoint_path_template'].format(date_processing=date_processing, flag=current_run_flag, dataset_name=CONFIG['dataset_name_for_checkpoint'], parameter_name=prev_param_name)
                if os.path.exists(prev_param_checkpoint_path):
                    ic(f"Starting '{param_name}' from '{prev_param_name}'s completed checkpoint.")
                    working_events_dict = _load_pickle(prev_param_checkpoint_path)
                else: # Should not happen if run sequentially
                    ic(f"Warning: Checkpoint for prev param '{prev_param_name}' not found. Reverting to initial for this dataset.")
                    working_events_dict = dataset_info["data_dict"].copy()
            else: # It is the first parameter
                 ic(f"Starting first parameter '{param_name}' from initial data.")
                 working_events_dict = dataset_info["data_dict"].copy()
            
            if not working_events_dict: # If loading failed or initial was empty
                ic(f"Cannot proceed with '{param_name}', working dictionary is empty.")
                break # Break from parameters loop for this dataset

            add_parameter_orchestrator(
                events_dict=working_events_dict, parameter_name=param_name,
                date_str=date, date_processing_str=date_processing, run_flag=current_run_flag, config=CONFIG
            )
            ic(f"--- Finished processing '{param_name}' for '{dataset_label}'. ---")

        final_dataset_save_path = os.path.join(data_path, 'processedNumpyData', date, final_save_filename)
        _save_pickle_atomic(working_events_dict, final_dataset_save_path)
        ic(f"Final version of '{dataset_label}' with all parameters saved to: {final_dataset_save_path}")

    ic("All datasets and parameters processed.")