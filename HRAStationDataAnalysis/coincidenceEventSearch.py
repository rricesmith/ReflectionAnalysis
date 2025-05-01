import configparser
import numpy as np
import os
import datetime
from icecream import ic
import gc


# Commented is my manual code. Rest is openai code

# def findCoincidenceDatetimes(date):
#     # This function finds all coincidence datetimes between all stations, for further processing through other cuts

#     # Returns : a dictionary of coincidence datetimes for each coincidence number then listing the datetime, station, and station indices

#     station_data_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'

#     station_ids = [13, 14, 15, 17, 18, 19, 30]

#     # Load the data
#     station_data = {}
#     for station_id in station_ids:
#         station_data[station_id] = []
#         for file in os.listdir(station_data_folder):
#             if file.startswith(f'{date}_Station{station_id}_Times'):
#                 data = np.load(station_data_folder+file, allow_pickle=True)
#                 station_data[station_id].extend(data.tolist())
#                 del data
#                 gc.collect()    # Free memory just in case it's large

#         # Convert to numpy arrays
#         station_data[station_id] = np.array(station_data[station_id])

#     # Find coincidences by recursively checking each station's data against the others
#     coincidence_datetimes = {} 

#     return coincidence_datetimes       

# if __name__ == "__main__":
#     # This code goes through all station data, searching for coincidence events that pass certain cut criteria
#     # and saves them to a numpy file for later use


#     config = configparser.ConfigParser()
#     config.read('HRAStationDataAnalysis/config.ini')
#     date = config['PARAMETERS']['date']


#     # First check to see if datetimes have already been processed as first cut
#     # If so, load them and skip the rest
#     numpy_folder = f'HRAStationDataAnalysis/StationData/processedNumpyData/{date}/'
#     if os.path.exists(numpy_folder+f'{date}_CoincidenceDatetimes.npy'):
#         coincidence_datetimes = np.load(numpy_folder+f'{date}_CoincidenceDatetimes.npy', allow_pickle=True)
#     else:
#         coincidence_datetimes = findCoincidenceDatetimes(date)


def findCoincidenceDatetimes(date, cuts=True): 
    """ 
    Finds all coincidence events between stations within a one-second window.

    For each station data file in the corresponding date folder, this function loads the event
    timestamps (expected as Python datetime objects) and records the station and the index of the event.
    Instead of grouping events by exact timestamp, events are grouped if they occur within one second
    of the earliest event in the group.

    Each stored coincidence event is a dictionary with the following keys:
    - "numCoincidences": Number of events in the coincidence group.
    - "datetime": The representative event timestamp (the first event in the group).
    - "stations": List of station IDs in which the event occurred.
    - "indices": List of indices (positions in the station dataset) corresponding to the event.

    Args:
    date (str): The date folder to process, as read from the configuration.

    Returns:
    A dictionary where each key is an incrementing coincidence event number (0, 1, 2, ...) and 
    the value is the dictionary described above.
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
    # station_ids = [13, 14, 30]

    # Instead of grouping by exact time, collect all events in a list.
    # Each event is represented as a tuple: (timestamp, station_id, event_index)
    all_events = []

    # Load data for each station.
    for station_id in station_ids:
        # Process only files that start with the given pattern.
        file_list = sorted(glob.glob(station_data_folder + f'/{date}_Station{station_id}_Times*'))
        times = [np.load(f) for f in file_list]
        times = np.concatenate(times, axis=0)
        times = times.squeeze()
        times = np.array(times)
        # for file in os.listdir(station_data_folder):
        #     if file.startswith(f'{date}_Station{station_id}_Times'):
        #         file_path = os.path.join(station_data_folder, file)
        #         ic(f"Loading file: {file_path}")
        #         data = np.load(file_path, allow_pickle=True)
        #         data = data.flatten()  # Flatten the data to ensure it's a 1D array.
        #         station_events.extend(data.tolist())
        #         del data
        #         gc.collect()  # Free up memory if necessary

        # Filter out zero timestamps and pre-time events.
        zerotime_mask = times != 0
        times = times[zerotime_mask]
        pretime_mask = times >= datetime.datetime(2013, 1, 1).timestamp()
        times = times[pretime_mask]

        if cuts:
            # Load cuts data for the station.
            cuts_file = os.path.join(cuts_data_folder, f'{date}_Station{station_id}_Cuts.npy')
            if os.path.exists(cuts_file):
                cuts_data = np.load(cuts_file, allow_pickle=True)
            else:
                ic(f"Warning: Cuts file not found for station {station_id} on date {date}.")
                continue
            # Apply cuts
            ic(cuts_data)
            final_cuts = np.ones(len(times), dtype=bool)
            for cut in cuts_data:
                final_cuts &= cuts_data[cut]
            times = times[final_cuts]

        # Loop over events and add them to all_events list.
        for idx, event_time in enumerate(times):
            # Skip zero timestamps.
            if event_time == 0:
                continue
            ts = event_time   # Already a datetime object.
            all_events.append((ts, station_id, idx))

    # Sort all events by timestamp.
    # all_events.sort(key=lambda x: x[0]) # Already sorted - if there are unsorted events, they have bad times and so we don't care about them

    # Now group events - events are in a coincidence if they occur within one second of the
    # first event in the group.
    coincidence_datetimes = {}
    event_counter = 0

    n_events = len(all_events)
    i = 0
    one_second = datetime.timedelta(seconds=1)
    while i < n_events:
        current_group = [all_events[i]]
        j = i + 1
        # Include subsequent events only if their time is within one second of the first event in the group.
        while j < n_events and (all_events[j][0] - all_events[i][0]) <= one_second:
            current_group.append(all_events[j])
            j += 1
        
        # Only record a coincidence if at least 2 events are found.
        if len(current_group) > 1:
            stations = [event[1] for event in current_group]
            indices = [event[2] for event in current_group]
            # Use the first event's time as a representative time.
            coincidence_datetimes[event_counter] = {
                "numCoincidences": len(current_group),
                "datetime": all_events[i][0],
                "stations": stations,
                "indices": indices
            }
            event_counter += 1
            # Skip over the events already grouped.
            i = j
        else:
            i += 1

    return coincidence_datetimes



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
        coincidence_datetimes = np.load(output_file, allow_pickle=True).item()
        ic("Loaded processed coincidences", len(coincidence_datetimes))
    else:
        coincidence_datetimes = findCoincidenceDatetimes(date, cuts=True)
        np.save(output_file, coincidence_datetimes, allow_pickle=True)
        ic("Saved new coincidences", len(coincidence_datetimes))

    # Optional: print first few coincidences for verification.
    for key in list(coincidence_datetimes.keys()):
        ic(key, coincidence_datetimes[key])
