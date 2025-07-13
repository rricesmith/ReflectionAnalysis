import datetime
import numpy as np
from NuRadioReco.modules.io import NuRadioRecoio
import NuRadioReco.modules.correlationDirectionFitter
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units
import os
import DeepLearning.D00_helperFunctions as D00_helperFunctions
from HRAStationDataAnalysis.calculateChi import getMaxAllChi
from HRAStationDataAnalysis.batchHRADataConversion import loadStationNurFiles
from icecream import ic

# --- Helper functions (from or adapted from your original script) ---

def getVrms(station_id):
    """This loads the Vrms for a given station from a text file."""
    try:
        with open(f'HRAStationDataAnalysis/vrms_per_station.txt', 'r') as f:
            for line in f:
                if line.startswith(f'{station_id}:'):
                    Vrms = float(line.split(':')[1].strip())
                    ic(f'Vrms for station {station_id} is {Vrms}')
                    return float(Vrms) * units.V
    except FileNotFoundError:
        ic(f'Vrms file not found. Could not get Vrms for station {station_id}.')
        return None
    ic(f'Vrms for station {station_id} not found in file.')
    return None

def calcSNR(traces, Vrms):
    """Calculate SNR from the average of the two highest traces."""
    if Vrms is None or Vrms == 0:
        return 0
    SNRs = []
    for trace in traces:
        p2p = np.max(trace) - np.min(trace)
        SNRs.append(p2p / (2 * Vrms))
    SNRs.sort(reverse=True)
    
    if len(SNRs) >= 2:
        return (SNRs[0] + SNRs[1]) / 2
    elif len(SNRs) == 1:
        return SNRs[0]
    return 0

def find_nur_file_for_run(station_id, run_number):
    """
    Finds the specific .nur file path for a given station and run number.
    It checks for common naming conventions like 'run101', 'R101', or '_101_'.
    """
    all_nur_files = loadStationNurFiles(station_id)
    search_pattern = str(run_number)

    for file_path in all_nur_files:
        # Check for patterns like /101/, _101_, run101, R101
        if (f"/{search_pattern}/" in file_path or
            f"_{search_pattern}_" in file_path or
            f"run{search_pattern}" in os.path.basename(file_path) or
            f"R{search_pattern}" in os.path.basename(file_path)):
            ic(f"Found matching file for run {search_pattern}: {file_path}")
            return file_path
            
    ic(f"No .nur file found for station {station_id} and run {run_number}")
    return None

# --- Main search and logging function ---

def find_and_log_events(events_to_find, output_filename="found_events.txt", delta_t=4.0):
    """
    Searches for specific events in .nur files and logs their parameters.

    Args:
        events_to_find (dict): Dict with station_ids as keys and event details as values.
        output_filename (str): The name of the text file to save results.
        delta_t (float): The time window in seconds to search around the specified time.
    """
    # Initialize detector and modules
    try:
        det = detector.Detector("HRASimulation/HRAStationLayoutForCoREAS.json")
    except FileNotFoundError:
        print("Error: Detector file 'HRASimulation/HRAStationLayoutForCoREAS.json' not found.")
        return
        
    correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
    correlationDirectionFitter.begin(debug=False)

    # Load templates
    try:
        templates_2016 = D00_helperFunctions.loadMultipleTemplates(100, date='2016')
        template_series_100 = D00_helperFunctions.loadMultipleTemplates(100)
        template_series_bad_100 = D00_helperFunctions.loadMultipleTemplates(100, bad=True)
        template_series_200 = D00_helperFunctions.loadMultipleTemplates(200)
        template_series_bad_200 = D00_helperFunctions.loadMultipleTemplates(200, bad=True)
        templates_loaded = True
    except Exception as e:
        print(f"Warning: Could not load templates due to error: {e}. Chi values will not be calculated.")
        templates_loaded = False

    stations_100s = [13, 15, 18, 32]
    stations_200s = [14, 17, 19, 30]
    save_channels = [0, 1, 2, 3]

    with open(output_filename, 'w') as f_out:
        for station_id, event_info in events_to_find.items():
            run_number = event_info['run_number']
            target_time = event_info['time']
            target_event_id = event_info['event_id']

            f_out.write(f"--- Searching in Station {station_id}, Run {run_number} ---\n")
            ic(f"Processing Station {station_id}, Run {run_number}")

            nur_file = find_nur_file_for_run(station_id, run_number)
            if not nur_file:
                f_out.write(f"ERROR: Could not find .nur file.\n\n")
                continue

            # Count total events in the file
            try:
                reader_for_count = NuRadioRecoio.NuRadioRecoio(nur_file)
                total_events = sum(1 for _ in reader_for_count.get_events())
                f_out.write(f"Total events in run file: {total_events}\n")
                print(f"Total events in '{os.path.basename(nur_file)}': {total_events}")
            except Exception as e:
                f_out.write(f"Could not count events in file. Error: {e}\n")
                continue

            # Get station-specific parameters
            Vrms = getVrms(station_id)
            if station_id in stations_100s:
                use_templates, use_templates_bad = (template_series_100, template_series_bad_100) if templates_loaded else (None, None)
            elif station_id in stations_200s:
                use_templates, use_templates_bad = (template_series_200, template_series_bad_200) if templates_loaded else (None, None)
            else:
                ic(f'Station {station_id} template group not defined.')
                use_templates, use_templates_bad = None, None

            found_count = 0
            file_reader = NuRadioRecoio.NuRadioRecoio(nur_file)
            for evt in file_reader.get_events():
                station = evt.get_station(station_id)
                if not station: continue

                current_event_id = evt.get_id()
                current_time = station.get_station_time().unix

                # Check for a match by event ID or time window
                if (current_event_id == target_event_id) or (abs(current_time - target_time) <= delta_t):
                    found_count += 1
                    f_out.write(f"\n>>> MATCH FOUND (Event ID: {current_event_id}, Time: {current_time}) <<<\n")
                    
                    det.update(station.get_station_time())
                    traces = [ch.get_trace() for ch in station.iter_channels(use_channels=save_channels)]
                    
                    # Log all parameters
                    f_out.write(f"  Unix Time: {current_time}\n")
                    f_out.write(f"  SNR: {calcSNR(traces, Vrms):.4f}\n")

                    if templates_loaded and use_templates:
                        f_out.write(f"  Chi (2016): {getMaxAllChi(traces, 2*units.GHz, templates_2016, 2*units.GHz):.4f}\n")
                        f_out.write(f"  Chi (RCR): {getMaxAllChi(traces, 2*units.GHz, use_templates, 2*units.GHz):.4f}\n")
                        f_out.write(f"  Chi (Bad RCR): {getMaxAllChi(traces, 2*units.GHz, use_templates_bad, 2*units.GHz):.4f}\n")
                    
                    try:
                        correlationDirectionFitter.run(evt, station, det, n_index=1.35)
                        f_out.write(f"  Azimuth (rad): {station.get_parameter(stnp.azimuth):.4f}\n")
                        f_out.write(f"  Zenith (rad): {station.get_parameter(stnp.zenith):.4f}\n")
                    except Exception as e:
                        f_out.write(f"  Azimuth/Zenith: Not calculated. Error: {e}\n")

            if found_count == 0:
                f_out.write("No matching event found.\n")
            f_out.write("\n")

if __name__ == "__main__":
    # --- How to Use ---
    # 1. Define the events to find in this dictionary.
    #    The script will search for an event if its ID matches OR its time is within the window.
    events_to_find_dict = {
        13: {
            'run_number': 114,
            'time': 1450734367,
            'event_id': 166218
        },
        15: {
            'run_number': 84,
            'time': 1450734371,
            'event_id': 166806
        },
        17: {
            'run_number': 120,
            'time': 1450734371,
            'event_id': 171308
        },
        18: {
            'run_number': 127,
            'time': 1450734371,
            'event_id': 202815
        }
    }

    # 2. Set the name for the output log file and the time search window in seconds.
    output_log_file = "HRADataAnalysis/specific_event_details.txt"
    time_search_window_seconds = 4.0

    # 3. Run the search function.
    print("Starting event search...")
    find_and_log_events(
        events_to_find=events_to_find_dict, 
        output_filename=output_log_file,
        delta_t=time_search_window_seconds
    )
    print(f"Search complete. Results are logged to '{output_log_file}'.")