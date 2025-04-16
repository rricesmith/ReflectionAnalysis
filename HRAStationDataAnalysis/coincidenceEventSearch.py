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


def findCoincidenceDatetimes(date): 
    """ Finds all exact coincidence datetimes between all stations.
    For each station data file in the corresponding date folder, this function loads the event
    timestamps (expected as np.datetime64 types) and records the station and the index of the event.
    Events are grouped by exact timestamp. Only timestamps where at least two stations have events 
    (i.e. a coincidence) are stored.

    Each stored coincidence event is a dictionary with the following keys:
    - "numCoincidences": Number of events at that exact timestamp.
    - "datetime": The event timestamp.
    - "stations": List of station IDs in which the event occurred.
    - "indices": List of indices (positions in the station dataset) corresponding to the event.

    Args:
    date (str): The date folder to process, as read from the configuration.

    Returns:
    A dictionary where each key is an incrementing coincidence event number (0, 1, 2, ...) and the value 
    is the dictionary described above.
    """
    station_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'nurFiles', date)

    # station_ids = [13, 14, 15, 17, 18, 19, 30]
    station_ids = [13, 14, 30]

    # Dictionary keyed by event timestamp. Each key maps to a list of (station, index) tuples.
    events_by_time = {}

    # Load data for each station.
    for station_id in station_ids:
        station_events = []
        # Process only files that start with the given pattern.
        for file in os.listdir(station_data_folder):
            if file.startswith(f'{date}_Station{station_id}_Times'):
                file_path = os.path.join(station_data_folder, file)
                data = np.load(file_path, allow_pickle=True)
                station_events.extend(data.tolist())
                del data
                gc.collect()  # Free up memory if necessary
        
        # Convert to numpy array (if needed)
        station_events = np.array(station_events)
        # Loop over events and store them by their timestamp.
        for idx, event_time in enumerate(station_events):
            # Ensure the event time is a np.datetime64 type
            ts = event_time
            if ts not in events_by_time:
                events_by_time[ts] = []
            events_by_time[ts].append((station_id, idx))

    # Build the coincidences dictionary using only those timestamps with at least 2 events.
    coincidence_datetimes = {}
    event_counter = 0
    # Sort the timestamps for consistent ordering.
    for ts in sorted(events_by_time.keys()):
        event_list = events_by_time[ts]
        if len(event_list) > 1:
            stations = [item[0] for item in event_list]
            indices = [item[1] for item in event_list]
            coincidence_datetimes[event_counter] = {
                "numCoincidences": len(event_list),
                "datetime": ts,
                "stations": stations,
                "indices": indices
            }
            event_counter += 1

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
        coincidence_datetimes = findCoincidenceDatetimes(date)
        np.save(output_file, coincidence_datetimes, allow_pickle=True)
        ic("Saved new coincidences", len(coincidence_datetimes))

    # Optional: print first few coincidences for verification.
    for key in list(coincidence_datetimes.keys())[:5]:
        ic(key, coincidence_datetimes[key])
