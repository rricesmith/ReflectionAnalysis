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

def multi_station_converter(input_folder, output_folder):
    """
    Loads .nur files, extracts traces from multiple stations, separates them
    into two groups, and saves them in batches to an output folder.
    """
    # --- Configuration ---
    MAX_EVENTS_PER_FILE = 50000  # Max events per output .npy file
    STATIONS_100S = {13, 15, 18}
    STATIONS_200S = {14, 17, 19, 30}
    SAVE_CHANNELS = [0, 1, 2, 3]

    # --- Initialization ---
    os.makedirs(output_folder, exist_ok=True)
    
    traces_100s = []
    traces_200s = []
    part_100s = 0
    part_200s = 0

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
            for station in evt.iter_stations():
                station_id = station.get_id()
                
                # Check if the station is one we care about
                if station_id not in STATIONS_100S and station_id not in STATIONS_200S:
                    continue

                event_traces = []
                channel_length_adjuster.run(evt, station, channel_ids=SAVE_CHANNELS)

                for channel in station.iter_channels(use_channels=SAVE_CHANNELS):
                    trace = channel.get_trace()
                    event_traces.append(trace)

                # Append to the correct list and check if it's time to save a batch
                if len(event_traces) == len(SAVE_CHANNELS):
                    if station_id in STATIONS_100S:
                        traces_100s.append(event_traces)
                        if len(traces_100s) >= MAX_EVENTS_PER_FILE:
                            part_100s = save_trace_batch(traces_100s, output_folder, "all_traces_100s", part_100s, "100s")
                            traces_100s = []  # Reset for the next batch
                            
                    elif station_id in STATIONS_200S:
                        traces_200s.append(event_traces)
                        if len(traces_200s) >= MAX_EVENTS_PER_FILE:
                            part_200s = save_trace_batch(traces_200s, output_folder, "all_traces_200s", part_200s, "200s")
                            traces_200s = []  # Reset for the next batch

    # --- Final Save ---
    ic("\nProcessing complete. Saving any remaining traces...")
    save_trace_batch(traces_100s, output_folder, "all_traces_100s", part_100s)
    save_trace_batch(traces_200s, output_folder, "all_traces_200s", part_200s)
    ic("✨ All done.")

if __name__ == "__main__":
    input_folder = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/HRASimulations/3.17.25/"
    output_folder = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/7.30.25/"

    multi_station_converter(input_folder, output_folder)