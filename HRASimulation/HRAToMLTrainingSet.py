import os
import numpy as np
from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.modules import channelLengthAdjuster as cLA
from icecream import ic

def save_trace_batch(trace_list, output_folder, file_prefix, part_number, amp):
    """
    Saves a batch of traces to a .npy file if the list is not empty.

    Args:
        trace_list (list): The list of traces to save.
        output_folder (str): The directory to save the file in.
        file_prefix (str): The prefix for the output filename (e.g., 'all_traces_100s').
        part_number (int): The current file part number for this prefix.

    Returns:
        int: The next part number to use for this prefix.
    """
    if not trace_list:
        return part_number

    num_events = len(trace_list)
    ary = np.array(trace_list)
    
    # Construct filename with prefix, part number, and event count
    filename = f"{file_prefix}_part{part_number}_{num_events}events.npy"
    output_path = os.path.join(output_folder+f"{amp}/", filename)
    
    np.save(output_path, ary)
    ic(f"✅ Saved batch of {num_events} events to {output_path}")
    
    return part_number + 1

def multi_station_converter(input_folder, output_folder_BL, output_folder_RCR):
    """
    Loads .nur files, extracts traces from multiple stations, separates them
    into two groups, and saves them in batches to an output folder.
    """
    # --- Configuration ---
    MAX_EVENTS_PER_FILE = 50000  # Max events per output .npy file
    STATIONS_100S = {13, 15, 18}
    STATIONS_200S = {14, 17, 19, 30}
    STATION_100s_RCR = {113, 115, 118}
    STATION_200s_RCR = {114, 117, 119, 130}
    SAVE_CHANNELS = [0, 1, 2, 3]

    trigger_name = 'primary_LPDA_2of4_4.5sigma'

    # --- Initialization ---
    os.makedirs(output_folder_BL, exist_ok=True)
    os.makedirs(output_folder_RCR, exist_ok=True)
    os.makedirs(output_folder_BL + "100s/", exist_ok=True)
    os.makedirs(output_folder_BL + "200s/", exist_ok=True)
    os.makedirs(output_folder_RCR + "100s/", exist_ok=True)
    os.makedirs(output_folder_RCR + "200s/", exist_ok=True)

    traces_100s = []
    traces_200s = []
    traces_100s_RCR = []
    traces_200s_RCR = []
    part_100s = 0
    part_200s = 0
    part_100s_RCR = 0
    part_200s_RCR = 0

    channel_length_adjuster = cLA.channelLengthAdjuster()
    channel_length_adjuster.begin()

    ic(f"Searching for .nur files in: {input_folder}")
    nur_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.nur')]

    if not nur_files:
        ic("No .nur files found in the specified directory.")
        return

    ic(f"Found {len(nur_files)} file(s). Saving in batches of {MAX_EVENTS_PER_FILE} events.")

    # --- Main Loop ---
    for file_path in nur_files:
        ic(f"Processing file: {file_path}")
        template = NuRadioRecoio.NuRadioRecoio(file_path)

        for evt in template.get_events():
            # Iterate through all stations that are present in the current event
            for station in evt.get_stations():
                station_id = station.get_id()
                
                # Check if the station is one we care about
                if station_id not in STATIONS_100S and station_id not in STATIONS_200S and station_id not in STATION_100s_RCR and station_id not in STATION_200s_RCR:
                    continue

                if station.has_trigger(trigger_name): 
                    if not station.has_triggered(trigger_name=trigger_name):
                        continue
                else:
                    # No trigger even attempted, skip
                    continue

                event_traces = []

                for channel in station.iter_channels(use_channels=SAVE_CHANNELS):
                    trace = channel.get_trace()
                    ic(trace.shape, trace)
                    event_traces.append(trace)

                ic(len(event_traces), event_traces)
                quit()

                # Append to the correct list and check if it's time to save a batch
                if len(event_traces) == len(SAVE_CHANNELS):
                    if station_id in STATIONS_100S:
                        traces_100s.append(event_traces)
                        if len(traces_100s) >= MAX_EVENTS_PER_FILE:
                            part_100s = save_trace_batch(traces_100s, output_folder_BL, "all_traces_100s", part_100s, "100s")
                            traces_100s = []  # Reset for the next batch
                            
                    elif station_id in STATIONS_200S:
                        traces_200s.append(event_traces)
                        if len(traces_200s) >= MAX_EVENTS_PER_FILE:
                            part_200s = save_trace_batch(traces_200s, output_folder_BL, "all_traces_200s", part_200s, "200s")
                            traces_200s = []  # Reset for the next batch

                    elif station_id in STATION_100s_RCR:
                        traces_100s_RCR.append(event_traces)
                        if len(traces_100s_RCR) >= MAX_EVENTS_PER_FILE:
                            part_100s_RCR = save_trace_batch(traces_100s_RCR, output_folder_RCR, "all_traces_100s_RCR", part_100s_RCR, "100s")
                            traces_100s_RCR = []

                    elif station_id in STATION_200s_RCR:
                        traces_200s_RCR.append(event_traces)
                        if len(traces_200s_RCR) >= MAX_EVENTS_PER_FILE:
                            part_200s_RCR = save_trace_batch(traces_200s_RCR, output_folder_RCR, "all_traces_200s_RCR", part_200s_RCR, "200s")
                            traces_200s_RCR = []

    # --- Final Save ---
    ic("\nProcessing complete. Saving any remaining traces...")
    save_trace_batch(traces_100s, output_folder_BL, "all_traces_100s", part_100s, "100s")
    save_trace_batch(traces_200s, output_folder_BL, "all_traces_200s", part_200s, "200s")
    save_trace_batch(traces_100s_RCR, output_folder_RCR, "all_traces_100s_RCR", part_100s_RCR, "100s")
    save_trace_batch(traces_200s_RCR, output_folder_RCR, "all_traces_200s_RCR", part_200s_RCR, "200s")
    ic("✨ All done.")

if __name__ == "__main__":
    # Load the configuration
    import configparser
    config = configparser.ConfigParser()
    # Assuming the script is run from the root of the project
    config.read('HRASimulation/config.ini')

    date = config.get('SIMPARAMETERS', 'date')
    date_processing = config.get('SIMPARAMETERS', 'date_processing')

    input_folder = f"/dfs8/sbarwick_lab/ariannaproject/rricesmi/HRASimulations/{date}/"
    output_folder_RCR = f"/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/{date_processing}/"
    output_folder_BL = f"/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedBacklobe/{date_processing}/"

    multi_station_converter(input_folder, output_folder_BL, output_folder_RCR)