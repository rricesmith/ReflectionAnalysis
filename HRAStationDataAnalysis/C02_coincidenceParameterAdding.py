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
from HRAStationDataAnalysis.C_utils import getTimeEventMasks, load_station_events, build_station_cuts_path

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

    # Use unified loader to ensure identical selection to C01
    time_tmpl = time_files_template.format(date=date_str, station_id=station_id)
    evt_tmpl = event_id_files_template.format(date=date_str, station_id=station_id)
    loader = load_station_events(
        date_str=date_str,
        station_id=station_id,
        time_files_template=time_tmpl,
        event_id_files_template=evt_tmpl,
        external_cuts_file_path=external_cuts_file_path,
        apply_external_cuts=True,
    )

    final_grcis = loader['final_grcis']
    final_map_final_idx_to_grci = {i: grci for i, grci in enumerate(final_grcis)}
    
    ic(f"Station {station_id}: {len(final_map_final_idx_to_grci)} events passed all cuts.")
    _save_pickle_atomic(final_map_final_idx_to_grci, map_cache_file_path)
    ic(f"Saved final_idx_to_grci_map to cache: {map_cache_file_path}")
    return final_map_final_idx_to_grci

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
    # Cuts path will be computed via build_station_cuts_path using date_cuts
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
        current_ext_cuts_path = build_station_cuts_path(
            date_cuts=config['date_cuts'],
            date_str=date_str,
            station_id=station_id,
            station_data_root=config.get('station_data_root', 'HRAStationDataAnalysis/StationData')
        )

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
                    events_needing_param_count += 1

                for pos, final_idx in enumerate(indices_list):
                    if station_event_data[parameter_name][pos] is None:
                        if final_idx in final_idx_to_grci_map:
                            grci = final_idx_to_grci_map[final_idx]
                            grcis_to_fetch_for_station.append(grci)
                            map_grci_to_event_updates[grci].append((event_data_ref, station_key_in_event, pos))
                        else:
                            # Mark as NaN if index can't be mapped
                            station_event_data[parameter_name][pos] = np.nan        

        if not grcis_to_fetch_for_station:
            if events_needing_param_count == 0: ic(f"St {station_id}: No '{parameter_name}' values needed.")
            else: ic(f"St {station_id}: No GRCIs to fetch for '{parameter_name}' (all final_idx missing from map?).")
            if checkpoint_file: _save_pickle_atomic(events_dict, checkpoint_file)
            continue
        
        unique_grcis_to_fetch = sorted(list(set(grcis_to_fetch_for_station)))
        ic(f"St {station_id}: Fetching {len(unique_grcis_to_fetch)} unique GRCIs for '{parameter_name}'.")

        # Use unified loader to produce by_grci mapping for the parameter
        time_tmpl = time_files_tmpl
        evt_tmpl = eventid_files_tmpl
        cuts_path = build_station_cuts_path(
            date_cuts=config['date_cuts'],
            date_str=date_str,
            station_id=station_id,
            station_data_root=config.get('station_data_root', 'HRAStationDataAnalysis/StationData')
        )

        loader_for_param = load_station_events(
            date_str=date_str,
            station_id=station_id,
            time_files_template=time_tmpl,
            event_id_files_template=evt_tmpl,
            external_cuts_file_path=cuts_path,
            apply_external_cuts=True,
            parameter_name=parameter_name,
            parameter_files_template=param_files_tmpl,
            filename_event_count_regex=event_count_regex,
        )
        fetched_params_by_grci = loader_for_param.get('parameters', {}).get(parameter_name, {}).get('by_grci', {})
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
        'map_cache_template': os.path.join(cache_path, "{date_processing}", "{flag}", "maps", "st_{station_id}_final_to_grci.pkl"),
        'parameter_files_template': os.path.join(data_path, "nurFiles", "{date}", "{date}_Station{station_id}_{parameter_name}*_Xevts_*.npy").replace("_Xevts_", "_*evts_"),
        'filename_event_count_regex_str': r"_(\d+)evts_",
        'checkpoint_path_template': os.path.join(cache_path, "{date_processing}", "{flag}", "checkpoints", "{dataset_name}_{parameter_name}_events_dict.pkl"),
        'dataset_name_for_checkpoint': 'HRA_Events',
        'date_cuts': date_cuts,
        'station_data_root': data_path,
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
            external_cuts_file_path=build_station_cuts_path(
                date_cuts=CONFIG['date_cuts'],
                date_str=date,
                station_id=station_id,
                station_data_root=CONFIG['station_data_root']
            ),
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
    
    # datasets_to_process = [
    #     {"name": "CoincidenceEvents", "data_dict": initial_coincidence_events, "run_flag": "base", "final_save_name": f'{date_processing}_CoincidenceDatetimes_with_all_params.pkl'},
    #     {"name": "CoincidenceEventsWithRepeat", "data_dict": initial_coincidence_with_repeat_stations_events, "run_flag": "with_repeat", "final_save_name": f'{date_processing}_CoincidenceRepeatStations_with_all_params.pkl'},
    #     {"name": "CoincidenceEventsWithRepeatedEventIDs", "data_dict": initial_coincidence_with_repeated_eventIDs, "run_flag": "with_repeated_eventIDs", "final_save_name": f'{date_processing}_CoincidenceRepeatEventIDs_with_all_params.pkl'}
    # ]
    datasets_to_process = [
        {"name": "CoincidenceEvents", "data_dict": initial_coincidence_events, "run_flag": "base", "final_save_name": f'{date_processing}_CoincidenceDatetimes_with_all_params.pkl'}
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