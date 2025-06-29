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
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units

from voltageToAnalyticEfieldConverter import voltageToAnalyticEfieldConverter

from icecream import ic

try:
    from HRAStationDataAnalysis.batchHRADataConversion import loadStationNurFiles
    HAS_USER_NUR_LOADER = True
    ic("Successfully imported loadStationNurFiles from HRAStationDataAnalysis.batchHRADataConversion")
except ImportError:
    HAS_USER_NUR_LOADER = False
    ic("Warning: Could not import loadStationNurFiles from HRAStationDataAnalysis.batchHRADataConversion.")
    ic("Please ensure HRAStationDataAnalysis is in your PYTHONPATH or the script is in the correct directory.")


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

# --- Main Polarization Calculation Logic ---
def calculate_polarization_for_events(events_dict, main_config, date_str, station_ids_to_process=None):
    """
    Calculates polarization angle for specified entries in the events_dict.
    """
    ic("Starting Polarization Angle calculation process...")

    detector_file = main_config['DEFAULT']['detector_layout_file']

    if not HAS_USER_NUR_LOADER:
        ic("CRITICAL: User's 'loadStationNurFiles' function is not available. This script cannot proceed without it.")
        return events_dict

    try:
        det = detector.Detector(detector_file_path=detector_file, antenna_by_depth=False, create_new=True)
        ic(f"Detector initialized from: {detector_file}")
    except Exception as e:
        ic(f"CRITICAL: Failed to initialize detector from {detector_file}: {e}")
        return events_dict

    eFieldConverter = voltageToAnalyticEfieldConverter()
    eFieldConverter.begin() 
    ic("voltageToAnalyticEfieldConverter initialized.")

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

    for station_id in stations_to_iterate:
        ic(f"\n--- Processing Station {station_id} for Polarization ---")

        targets_for_station = []
        for event_id, event_data_ref in events_dict.items():
            station_key_in_event = str(station_id) if str(station_id) in event_data_ref.get("stations", {}) else None
            if not station_key_in_event: continue

            station_event_data = event_data_ref["stations"][station_key_in_event]
            num_indices = len(station_event_data.get("indices", []))
            if num_indices == 0: continue
            
            station_event_data["PolAngle"] = [np.nan] * num_indices
            station_event_data["PolAngleErr"] = [np.nan] * num_indices
            station_event_data["ExpectedPolAngle"] = [np.nan] * num_indices

            for k in range(num_indices):
                targets_for_station.append({
                    "event_id_tag": station_event_data["event_ids"][k],
                    "station_time_tag": station_event_data["Times"][k],
                    "event_dict_ref": event_data_ref,
                    "station_key": station_key_in_event,
                    "list_pos": k,
                    "original_event_id_key": event_id
                })

        if not targets_for_station:
            ic(f"Station {station_id}: No events found for polarization calculation.")
            continue

        ic(f"Station {station_id}: Found {len(targets_for_station)} triggers to process.")

        current_station_nur_files = loadStationNurFiles(station_id)
        if not current_station_nur_files:
            ic(f"Station {station_id}: No .nur files found. Skipping.")
            continue
        
        ic(f"Station {station_id}: Retrieved {len(current_station_nur_files)} .nur files.")

        try:
            nur_reader = NuRadioRecoio.NuRadioRecoio(current_station_nur_files)
        except Exception as e:
            ic(f"Station {station_id}: Failed to create NuRadioRecoio reader: {e}. Skipping.")
            continue

        processed_targets_count = 0
        targets_for_station.sort(key=lambda t: t["station_time_tag"])

        for raw_evt in nur_reader.get_events():
            if not targets_for_station:
                ic(f"Station {station_id}: All targets processed. Breaking from .nur file scan.")
                break

            current_raw_event_id = raw_evt.get_id()
            raw_station_obj = raw_evt.get_station(station_id)

            if not raw_station_obj: continue

            current_raw_station_time = raw_station_obj.get_station_time().unix

            for i in range(len(targets_for_station) - 1, -1, -1):
                target = targets_for_station[i]

                if current_raw_event_id == target["event_id_tag"] and \
                   abs(current_raw_station_time - target["station_time_tag"]) < 1.0:

                    ic(f"St {station_id}, CoincEvent {target['original_event_id_key']}, RawEvID {current_raw_event_id}: Match found. Calculating polarization...")

                    try:
                        det.update(raw_station_obj.get_station_time())
                        
                        # NEW: Get the previously calculated zenith and azimuth from our events dictionary
                        station_data_from_dict = target["event_dict_ref"]["stations"][target["station_key"]]
                        list_pos = target['list_pos']
                        recalculated_zenith = station_data_from_dict["Zen"][list_pos]
                        recalculated_azimuth = station_data_from_dict["Azi"][list_pos]

                        # NEW: Check for valid angles and set them on the raw station object before running the calculation
                        if recalculated_zenith is not None and not np.isnan(recalculated_zenith) and \
                           recalculated_azimuth is not None and not np.isnan(recalculated_azimuth):
                            
                            # Set the parameters on the raw station object
                            raw_station_obj.set_parameter(stnp.zenith, recalculated_zenith)
                            raw_station_obj.set_parameter(stnp.azimuth, recalculated_azimuth)
                            ic(f"  Injecting Zen/Azi into raw_station_obj: {np.degrees(recalculated_zenith):.2f}° / {np.degrees(recalculated_azimuth):.2f}°")
                            
                            # Run the E-field conversion. It will now use the zenith/azimuth we just set.
                            eFieldConverter.run(raw_evt, raw_station_obj, det, use_channels=[0,1,2,3])
                            
                            efields = raw_station_obj.get_electric_fields()
                            if efields:
                                latest_efield = efields[-1] 
                                
                                pol_angle = latest_efield.get_parameter(efp.polarization_angle)
                                pol_angle_err = latest_efield.get_parameter(efp.polarization_angle_error)
                                expected_pol = latest_efield.get_parameter(efp.polarization_angle_expectation)

                                station_data_from_dict["PolAngle"][list_pos] = pol_angle
                                station_data_from_dict["PolAngleErr"][list_pos] = pol_angle_err
                                station_data_from_dict["ExpectedPolAngle"][list_pos] = expected_pol
                                
                                ic(f"  Success! PolAngle={np.degrees(pol_angle):.2f} ± {np.degrees(pol_angle_err):.2f}°")
                            else:
                                ic(f"  Warning: eFieldConverter ran but no electric field object was added to station {station_id}.")
                        else:
                            ic(f"  Skipping polarization calculation for trigger due to invalid Zen/Azi in input file.")

                    except Exception as e:
                        ic(f"  ERROR calculating polarization for St {station_id}, RawEvID {current_raw_event_id}: {e}")

                    targets_for_station.pop(i)
                    processed_targets_count += 1

        ic(f"Station {station_id}: Finished scanning .nur files. Processed {processed_targets_count} targets for polarization.")
        if targets_for_station:
            ic(f"Station {station_id}: Warning! {len(targets_for_station)} targets remain unprocessed.")

    eFieldConverter.end()
    ic("Finished polarization calculation for all specified stations.")
    return events_dict

# --- Main Script Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate polarization for coincidence events using the voltageToAnalyticEfieldConverter."
    )
    parser.add_argument(
        "--stations",
        nargs='+',
        help="Optional list of station IDs to process. If not provided, all stations are processed."
    )

    main_config_parser = configparser.ConfigParser()
    config_path = os.path.join('HRAStationDataAnalysis', 'config.ini')
    main_config_parser.read(config_path)
    
    date = main_config_parser['PARAMETERS']['date']
    date_processing = main_config_parser['PARAMETERS']['date_processing']
    ic(f"Running polarization calculation for date {date} with processing date {date_processing}")

    input_events_file = f"HRAStationDataAnalysis/StationData/processedNumpyData/{date}/{date_processing}_CoincidenceDatetimes_with_all_params_recalcZenAzi.pkl"
    output_events_file = f"HRAStationDataAnalysis/StationData/processedNumpyData/{date}/{date_processing}_CoincidenceDatetimes_with_all_params_recalcZenAzi_calcPol.pkl"

    args = parser.parse_args()
    ic.enable()

    if not os.path.exists(input_events_file):
        ic(f"CRITICAL ERROR: Input file not found: {input_events_file}")
        ic("Please run the C02A_coincidenceRecalcZenAzi.py script first.")
        exit(1)
        
    if not HAS_USER_NUR_LOADER:
        print("CRITICAL ERROR: The 'loadStationNurFiles' function could not be imported.")
        exit(1)

    if 'DEFAULT' not in main_config_parser or 'detector_layout_file' not in main_config_parser['DEFAULT']:
        ic(f"CRITICAL: 'detector_layout_file' not found in DEFAULT section of {config_path}.")
        exit(1)

    ic(f"Loading events dictionary from: {input_events_file}")
    events_dictionary = _load_pickle(input_events_file)

    if events_dictionary is None:
        ic(f"CRITICAL: Failed to load events dictionary. Exiting.")
        exit(1)

    ic(f"Successfully loaded {len(events_dictionary)} events.")

    updated_events_dictionary = calculate_polarization_for_events(
        events_dictionary,
        main_config_parser,
        date,
        station_ids_to_process=args.stations
    )

    ic(f"Saving updated events dictionary to: {output_events_file}")
    _save_pickle_atomic(updated_events_dictionary, output_events_file)

    ic("Polarization processing complete.")