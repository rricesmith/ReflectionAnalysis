import os
import glob
import numpy as np
import pickle
import datetime
import time # For logging/timestamps
import tempfile # For atomic saving
import argparse
import json # For blackout times
import configparser

from NuRadioReco.modules.io import NuRadioRecoio
import NuRadioReco.modules.correlationDirectionFitter
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units # Not directly used here, but often in NuRadioReco context

from icecream import ic
from HRAStationDataAnalysis.C_utils import load_station_events, build_station_cuts_path

# Attempt to import the user's function
try:
    from HRAStationDataAnalysis.batchHRADataConversion import loadStationNurFiles
    HAS_USER_NUR_LOADER = True
    ic("Successfully imported loadStationNurFiles from HRAStationDataAnalysis.batchHRADataConversion")
except ImportError:
    HAS_USER_NUR_LOADER = False
    ic("Warning: Could not import loadStationNurFiles from HRAStationDataAnalysis.batchHRADataConversion.")
    ic("Please ensure HRAStationDataAnalysis is in your PYTHONPATH or the script is in the correct directory.")
    # A fallback or error exit might be needed if this is critical and no template is provided.
    # For now, the script will check for this flag later.


# --- Helper for Caching (from C02_coincidenceParameterAdding.py) ---
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

# --- Blackout Times Helper (adapted from HRADataConvertToNpy.py) ---
def get_blackout_times(filepath):
    """Loads blackout times from a JSON file."""
    blackout_times_list = []
    if not filepath: return blackout_times_list
    try:
        with open(filepath, 'r') as f:
            blackout_data = json.load(f)
        for iB, tStart in enumerate(blackout_data.get('BlackoutCutStarts', [])):
            tEnd = blackout_data.get('BlackoutCutEnds', [])[iB]
            blackout_times_list.append([tStart, tEnd])
        ic(f"Loaded {len(blackout_times_list)} blackout intervals from {filepath}")
    except FileNotFoundError:
        ic(f"Warning: Blackout times file not found: {filepath}. Proceeding without blackout cuts.")
    except Exception as e:
        ic(f"Error loading blackout times from {filepath}: {e}. Proceeding without blackout cuts.")
    return blackout_times_list


# --- Main Recalculation Logic ---
def recalculate_zen_azi_for_events(events_dict, main_config, date_str, station_ids_to_process=None):
    """
    Recalculates Zenith and Azimuth for specified entries in the events_dict.
    Uses user-provided loadStationNurFiles if available.
    """
    ic("Starting Zenith/Azimuth recalculation process...")

    detector_file = main_config['DEFAULT']['detector_layout_file']
    # blackout_file = main_config['DEFAULT'].get('blackout_times_file') # Optional

    if not HAS_USER_NUR_LOADER:
        ic("CRITICAL: User's 'loadStationNurFiles' function is not available. This script cannot proceed without it.")
        ic("Please ensure 'HRAStationDataAnalysis.batchHRADataConversion' is accessible.")
        return events_dict


    try:
        det = detector.Detector(f"HRASimulation/HRAStationLayoutForCoREAS.json")
        # det = detector.Detector(detector_file_path=detector_file,
        #                         antenna_by_depth=False,
        #                         create_new=True)
        ic(f"Detector initialized from: {detector_file}")
    except Exception as e:
        ic(f"CRITICAL: Failed to initialize detector from {detector_file}: {e}")
        return events_dict

    fitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
    fitter.begin(debug=False)
    ic("Correlation direction fitter initialized.")

    # blackoutTimes = get_blackout_times(blackout_file) if blackout_file else []

    if station_ids_to_process is None:
        all_station_ids_in_dict = set()
        for event_data in events_dict.values():
            for st_id_key in event_data.get("stations", {}).keys():
                try:
                    all_station_ids_in_dict.add(int(st_id_key))
                except ValueError:
                    ic(f"Warning: Could not parse station ID '{st_id_key}' to int.")
        stations_to_iterate = sorted(list(all_station_ids_in_dict))
    else:
        stations_to_iterate = sorted(list(map(int, station_ids_to_process)))

    ic(f"Will process stations: {stations_to_iterate} for date {date_str}")

    # Prepare templates for uniform loading
    station_data_root = os.path.join('HRAStationDataAnalysis', 'StationData')
    time_files_template = os.path.join(station_data_root, 'nurFiles', '{date}', '{date}_Station{station_id}_Times*.npy')
    event_id_files_template = time_files_template.replace('_Times', '_EventIDs')
    date_cuts = main_config['PARAMETERS'].get('date_cuts', date_str)

    for station_id in stations_to_iterate:
        ic(f"\n--- Processing Station {station_id} ---")

        # Uniformly load station events to ensure Times/EventIDs match final indices
        cuts_path = build_station_cuts_path(date_cuts, date_str, station_id, station_data_root)
        loader = load_station_events(
            date_str=date_str,
            station_id=station_id,
            time_files_template=time_files_template,
            event_id_files_template=event_id_files_template,
            external_cuts_file_path=cuts_path,
            apply_external_cuts=True,
        )

        targets_for_station = []

        for event_id, event_data_ref in events_dict.items():
            station_key_in_event = None
            if station_id in event_data_ref.get("stations", {}):
                station_key_in_event = station_id
            elif str(station_id) in event_data_ref.get("stations", {}):
                station_key_in_event = str(station_id)

            if not station_key_in_event:
                continue

            station_event_data = event_data_ref["stations"][station_key_in_event]
            num_indices = len(station_event_data.get("indices", []))
            if num_indices == 0: continue

            azimuths = station_event_data.get("Azi")
            zeniths = station_event_data.get("Zen")
            event_ids_for_param = station_event_data.get("event_ids")
            times_for_param = station_event_data.get("Times")

            # Ensure uniform Times/EventIDs via loader if missing or length-mismatched
            if times_for_param is None or len(times_for_param) != num_indices or event_ids_for_param is None or len(event_ids_for_param) != num_indices:
                station_event_data["Times"] = [np.nan] * num_indices
                station_event_data["event_ids"] = [np.nan] * num_indices
                for k, final_idx in enumerate(station_event_data.get("indices", [])):
                    if final_idx is None or np.isnan(final_idx):
                        continue
                    final_idx_int = int(final_idx)
                    if 0 <= final_idx_int < len(loader['final_times']):
                        station_event_data["Times"][k] = loader['final_times'][final_idx_int]
                    if 0 <= final_idx_int < len(loader['final_event_ids']):
                        station_event_data["event_ids"][k] = loader['final_event_ids'][final_idx_int]
                event_ids_for_param = station_event_data.get("event_ids")
                times_for_param = station_event_data.get("Times")

            # ic(event_data_ref)
            # ic(f"Event {event_id}, St {station_id}: Found {num_indices} indices, "
            #    f"Azi={azimuths}, Zen={zeniths}, EventIDs={event_ids_for_param}, Times={times_for_param}")
            # ic(event_data_ref["stations"])
            # ic(event_data_ref["stations"][station_key_in_event])
            # ic(station_event_data)

            if not (event_ids_for_param and times_for_param and \
                    len(event_ids_for_param) == num_indices and \
                    len(times_for_param) == num_indices):
                ic(f"Event {event_id}, St {station_id}: Missing or mismatched EventIDs/Times. Cannot process.")
                ic(f"  EventIDs: {event_ids_for_param}, Times: {times_for_param}")
                ic(f"  Expected {num_indices} indices, but found {len(event_ids_for_param)} EventIDs and {len(times_for_param)} Times.")
                continue

            if azimuths is None or len(azimuths) != num_indices:
                azimuths = [np.nan] * num_indices
                station_event_data["Azi"] = azimuths
            if zeniths is None or len(zeniths) != num_indices:
                zeniths = [np.nan] * num_indices
                station_event_data["Zen"] = zeniths

            for k in range(num_indices):
                is_zero = (azimuths[k] == 0.0 and zeniths[k] == 0.0)
                is_nan = np.isnan(azimuths[k]) or np.isnan(zeniths[k])
                if is_zero or is_nan:
                    targets_for_station.append({
                        "event_id_tag": event_ids_for_param[k],
                        "station_time_tag": times_for_param[k],
                        "event_dict_ref": event_data_ref,
                        "station_key": station_key_in_event,
                        "list_pos": k,
                        "original_event_id_key": event_id
                    })
                else:
                    ic(f"Event {event_id}, St {station_id}, ListPos {k}: Azi={azimuths[k]:.4f}, Zen={zeniths[k]:.4f} - No recalculation needed.")

        if not targets_for_station:
            ic(f"Station {station_id}: No events require Zen/Azi recalculation.")
            continue

        ic(f"Station {station_id}: Found {len(targets_for_station)} entries needing Zen/Azi recalculation.")

        # Use the imported loadStationNurFiles function
        current_station_nur_files = loadStationNurFiles(station_id)
        ic(f"Station {station_id}: Retrieved {len(current_station_nur_files)} .nur files using 'loadStationNurFiles'.")


        if not current_station_nur_files:
            ic(f"Station {station_id}: No .nur files found by 'loadStationNurFiles'. Skipping recalculation for this station.")
            for target in targets_for_station: # Mark as NaN
                target["event_dict_ref"]["stations"][target["station_key"]]["Azi"][target["list_pos"]] = np.nan
                target["event_dict_ref"]["stations"][target["station_key"]]["Zen"][target["list_pos"]] = np.nan
            continue

        try:
            nur_reader = NuRadioRecoio.NuRadioRecoio(current_station_nur_files)
        except Exception as e:
            ic(f"Station {station_id}: Failed to create NuRadioRecoio reader: {e}. Skipping.")
            continue

        processed_targets_count = 0
        targets_for_station.sort(key=lambda t: t["station_time_tag"])

        for raw_evt_idx, raw_evt in enumerate(nur_reader.get_events()):
            if not targets_for_station:
                ic(f"Station {station_id}: All targets processed. Breaking from .nur file scan.")
                break

            if raw_evt_idx % 50000 == 0 and raw_evt_idx > 0: # Reduce log verbosity
                ic(f"Station {station_id}: Scanned {raw_evt_idx} raw events... {len(targets_for_station)} targets remaining.")

            current_raw_event_id = raw_evt.get_id()
            raw_station_obj = raw_evt.get_station(station_id)

            if not raw_station_obj: continue

            current_raw_station_time = raw_station_obj.get_station_time().unix
            current_raw_station_dt = raw_station_obj.get_station_time()

            for i in range(len(targets_for_station) - 1, -1, -1):
                target = targets_for_station[i]

                if current_raw_event_id == target["event_id_tag"] and \
                   abs(current_raw_station_time - target["station_time_tag"]) < 1.0: # Time tolerance of 1s

                    # Check if the event's date (from raw_station_obj.get_station_time()) matches date_str approximately
                    # This is an additional sanity check if loadStationNurFiles brings files from multiple dates.
                    event_date_str = current_raw_station_dt.strftime("%Y%m%d")
                    if event_date_str != date_str:
                        # This might happen if loadStationNurFiles gets all files and EventIDs are not unique across dates
                        # Or if the timestamp in events_dict is slightly off from the raw file's date boundary
                        # For now, we trust the EventID and precise timestamp from events_dict
                        # ic(f"  St {station_id}, RawEvID {current_raw_event_id}: Matched EvID/Time but raw date {event_date_str} differs from target date {date_str}. Proceeding cautiously.")
                        pass


                    ic(f"St {station_id}, CoincEvent {target['original_event_id_key']}, RawEvID {current_raw_event_id}: Match found. Recalculating...")

                    try:
                        det.update(current_raw_station_dt)
                    except LookupError:
                        ic(f"LookupError for St {station_id}, RawEvID {current_raw_event_id}, Time {current_raw_station_dt}. Using fallback detector update.")
                        det.update(datetime.datetime(2018, 12, 31, tzinfo=datetime.timezone.utc))

                    fitter.run(raw_evt, raw_station_obj, det, n_index=1.35)

                    new_azimuth = raw_station_obj.get_parameter(stnp.azimuth)
                    new_zenith = raw_station_obj.get_parameter(stnp.zenith)

                    target["event_dict_ref"]["stations"][target["station_key"]]["Azi"][target["list_pos"]] = new_azimuth
                    target["event_dict_ref"]["stations"][target["station_key"]]["Zen"][target["list_pos"]] = new_zenith

                    ic(f"  Updated St {station_id}, CoincEvent {target['original_event_id_key']}, ListPos {target['list_pos']}: Azi={new_azimuth:.4f}, Zen={new_zenith:.4f}")

                    targets_for_station.pop(i)
                    processed_targets_count += 1

        ic(f"Station {station_id}: Finished scanning .nur files. Processed {processed_targets_count} targets.")
        if targets_for_station:
            ic(f"Station {station_id}: Warning! {len(targets_for_station)} targets remain unprocessed.")
            for unfilled_target in targets_for_station:
                ic(f"  Unfilled for CoincEvent {unfilled_target['original_event_id_key']}, St {unfilled_target['station_key']}, RawEvID {unfilled_target['event_id_tag']}, Time {unfilled_target['station_time_tag']}")
                unfilled_target["event_dict_ref"]["stations"][unfilled_target["station_key"]]["Azi"][unfilled_target["list_pos"]] = np.nan
                unfilled_target["event_dict_ref"]["stations"][unfilled_target["station_key"]]["Zen"][unfilled_target["list_pos"]] = np.nan


    ic("Finished Zenith/Azimuth recalculation for all specified stations.")
    return events_dict

# --- Main Script Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Recalculate Zenith and Azimuth for coincidence events using raw .nur data."
    )
    parser.add_argument(
        "--stations",
        nargs='+',
        help="Optional list of station IDs to process. If not provided, all stations in the events_dict are processed."
    )

    import configparser
    main_config_parser = configparser.ConfigParser() 
    main_config_parser.read(os.path.join('HRAStationDataAnalysis', 'config.ini')) 
    date = main_config_parser['PARAMETERS']['date']
    date_processing = main_config_parser['PARAMETERS']['date_processing']
    ic(f"Running parameter addition for date {date} with processing date {date_processing}")

    input_events_file = f"HRAStationDataAnalysis/StationData/processedNumpyData/{date}/{date_processing}_CoincidenceDatetimes_with_all_params.pkl"
    output_events_file = f"HRAStationDataAnalysis/StationData/processedNumpyData/{date}/{date_processing}_CoincidenceDatetimes_with_all_params_recalcZenAzi.pkl"

    args = parser.parse_args()
    ic.enable()

    if not HAS_USER_NUR_LOADER:
        print("CRITICAL ERROR: The 'loadStationNurFiles' function could not be imported.")
        print("Please ensure 'HRAStationDataAnalysis.batchHRADataConversion' is in your Python path.")
        exit(1)

    if 'DEFAULT' not in main_config_parser or \
       'detector_layout_file' not in main_config_parser['DEFAULT']:
        ic("CRITICAL: 'detector_layout_file' not found in DEFAULT section of config file.")
        exit(1)

    ic(f"Loading initial events dictionary from: {input_events_file}")
    events_dictionary = _load_pickle(input_events_file)

    if events_dictionary is None:
        ic(f"CRITICAL: Failed to load events dictionary from {input_events_file}. Exiting.")
        exit(1)

    ic(f"Successfully loaded {len(events_dictionary)} events.")

    updated_events_dictionary = recalculate_zen_azi_for_events(
        events_dictionary,
        main_config_parser,
        date,
        station_ids_to_process=args.stations
    )

    ic(f"Saving updated events dictionary to: {output_events_file}")
    _save_pickle_atomic(updated_events_dictionary, output_events_file)

    ic("Processing complete.")