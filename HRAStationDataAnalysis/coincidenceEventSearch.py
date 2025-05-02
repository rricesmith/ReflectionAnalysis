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
                ic(f"Loading cuts file: {cuts_file}")
                cuts_data = np.load(cuts_file, allow_pickle=True)
                cuts_data = cuts_data[()]
            else:
                ic(f"Warning: Cuts file not found for station {station_id} on date {date}.")
                continue
            # Apply cuts
            final_cuts = np.ones(len(times), dtype=bool)
            for cut in cuts_data.keys():
                ic(f"Applying cut: {cut}")
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
    all_events.sort(key=lambda x: x[0])

    # Now group events - events are in a coincidence if they occur within one second of the
    # first event in the group.
    coincidence_datetimes = {}
    coincidence_with_repeated_stations = {}
    valid_counter = 0
    duplicate_counter = 0

    n_events = len(all_events)
    i = 0
    # one_second = datetime.timedelta(seconds=1)
    one_second = 1
    while i < n_events:
        current_group = [all_events[i]]
        j = i + 1
        # Include subsequent events only if their time is within one second of the first event in the group.
        while j < n_events and (all_events[j][0] - all_events[i][0]) <= one_second:
            current_group.append(all_events[j])
            j += 1
        
        # Only record a coincidence if at least 2 events are found.
        if len(current_group) > 1:
            # Build list of station IDs.
            stations = [event[1] for event in current_group]
            # If all events are from the same station, skip this group.
            if len(set(stations)) == 1:
                i = j
                continue

            # Check if any station appears multiple times in the group.
            if len(set(stations)) < len(stations):
                target_dict = coincidence_with_repeated_stations
                idx_counter = duplicate_counter
                duplicate_counter += 1
            else:
                target_dict = coincidence_datetimes
                idx_counter = valid_counter
                valid_counter += 1


            indices = [event[2] for event in current_group]
            # Use the first event's time as a representative time.
            target_dict[idx_counter] = {
                "numCoincidences": len(current_group),
                "datetime": all_events[i][0],
                "stations": stations,
                "indices": indices
            }
            # Skip over the events already grouped.
            i = j
        else:
            i += 1

    return coincidence_datetimes, coincidence_with_repeated_stations

def analyze_coincidence_events(coincidence_datetimes, coincidence_with_repeated_stations):
    """
    Analyzes and ics the number of coincidence events for different coincidence numbers.

    Args:
        coincidence_datetimes (dict): Dictionary of coincidence events with unique stations.
        coincidence_with_repeated_stations (dict): Dictionary of coincidence events with potentially repeated stations.
    """

    # Analyze coincidence_datetimes (unique stations)
    coincidence_counts = {}
    for event_data in coincidence_datetimes.values():
        num_coincidences = event_data["numCoincidences"]
        coincidence_counts[num_coincidences] = coincidence_counts.get(num_coincidences, 0) + 1

    ic("Analysis of coincidence events with unique stations:")
    for num, count in sorted(coincidence_counts.items()):
        ic(f"Number of events with {num} coincidences: {count}")

    ic("\nAnalysis of coincidence events with potentially repeated stations:")
    repeated_coincidence_counts = {}
    for event_data in coincidence_with_repeated_stations.values():
        # Count only unique stations for this analysis
        unique_stations_count = len(set(event_data["stations"]))
        repeated_coincidence_counts[unique_stations_count] = repeated_coincidence_counts.get(unique_stations_count, 0) + 1

    for num, count in sorted(repeated_coincidence_counts.items()):
        ic(f"Number of events with {num} unique station coincidences: {count}")

import os
import numpy as np
import glob
from icecream import ic

def load_coincidence_event_data(date, coincidence_event, cuts=True):
    """
    Loads SNR and ChiRCR data for each station in a coincidence event.

    Args:
        date (str): The date folder to process.
        coincidence_event (dict): A single coincidence event dictionary
                                    from coincidence_datetimes or coincidence_with_repeated_stations.

    Returns:
        dict: A dictionary where keys are station IDs and values are dictionaries
              containing 'SNR' and 'ChiRCR' arrays for the corresponding events.
              Returns None if data loading fails for any station in the event.
    """
    station_data = {}
    station_ids = coincidence_event["stations"]
    indices = coincidence_event["indices"]

    station_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'nurFiles', date)

    for i, station_id in enumerate(station_ids):
        event_index = indices[i]
        snr_files = sorted(glob.glob(station_data_folder + f'/{date}_Station{station_id}_SNR*'))
        chir_files = sorted(glob.glob(station_data_folder + f'/{date}_Station{station_id}_ChiRCR*'))

        if not snr_files or not chir_files:
            ic(f"Warning: SNR or ChiRCR files not found for station {station_id} on date {date}.")
            return None

        try:
            snr_data = np.load(snr_files[0])  # Assuming only one SNR file per station per date
            chir_data = np.load(chir_files[0]) # Assuming only one ChiRCR file per station per date

            if event_index < len(snr_data) and event_index < len(chir_data):
                station_data[station_id] = {
                    "SNR": snr_data[event_index],
                    "ChiRCR": chir_data[event_index]
                }
            else:
                ic(f"Warning: Event index {event_index} out of bounds for station {station_id} on date {date}.")
                return None
        except Exception as e:
            ic(f"Error loading data for station {station_id} on date {date}: {e}")
            return None

    return station_data


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
        data = np.load(output_file, allow_pickle=True)
        coincidence_datetimes = data[0]
        coincidence_with_repeated_stations = data[1]
        ic("Loaded processed coincidences", len(coincidence_datetimes))
    else:
        coincidence_datetimes, coincidence_with_repeated_stations = findCoincidenceDatetimes(date, cuts=True)
        np.save(output_file, [coincidence_datetimes, coincidence_with_repeated_stations], allow_pickle=True)
        ic("Saved new coincidences", len(coincidence_datetimes))

    # Optional: ic first few coincidences for verification.
    for key in list(coincidence_datetimes.keys()):
        ic(key, coincidence_datetimes[key])

    for key in list(coincidence_with_repeated_stations.keys()):
        ic(key, coincidence_with_repeated_stations[key])

    # Analyze the coincidence events.
    analyze_coincidence_events(coincidence_datetimes, coincidence_with_repeated_stations)


    # Make plots of the coincidences
    import HRAStationDataAnalysis.loadHRAConvertedData as loadHRAConvertedData

    station_data = loadHRAConvertedData.loadHRAConvertedData(date, cuts=True, SNR='SNR', ChiRCR='ChiRCR', Chi2016='Chi2016', ChiBad='ChiBad')
    # Data is a dictionary with keys 'times', 'SNR', 'ChiRCR', 'Chi2016', and 'ChiRCR_bad'.
    # Each key contains a dictionary where keys are station IDs and values are the corresponding data arrays.

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    plot_folder = os.path.join('HRAStationDataAnalysis', 'plots', date, 'coincidence')
    os.makedirs(plot_folder, exist_ok=True)

    def plot_events(events, title_suffix, plot_folder):
        # Define Chi keys to plot against SNR.
        chi_keys = ['ChiRCR', 'Chi2016', 'ChiRCR_bad']
        # Create one figure with three side-by-side subplots.
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, chi in zip(axes, chi_keys):
            ax.set_xlabel(chi)
            ax.set_ylabel('SNR')
            ax.set_title(f'SNR vs {chi} ({title_suffix})')
            ax.set_xscale('log')
            ax.set_xlim(3, 100)
            ax.set_ylim(0, 1)

        num_events = len(events)
        # Generate a distinct color for each event using a colormap.
        colors = cm.jet(np.linspace(0, 1, num_events))

        # Loop over the events.
        for i, (event_id, event) in enumerate(events.items()):
            stations = event["stations"]
            indices = event["indices"]
            # For each Chi plot, collect the SNR and Chi values from station_data
            # station_data contains keys 'SNR', 'ChiRCR', 'Chi2016', 'ChiRCR_bad'
            event_snr = []
            event_chi = {chi: [] for chi in chi_keys}
            for j, station in enumerate(stations):
                idx = indices[j]
                # Extract SNR point if available.
                try:
                    snr_val = station_data['SNR'][station][idx]
                except (KeyError, IndexError):
                    snr_val = np.nan
                event_snr.append(snr_val)
                # Extract each Chi value; if not available, fill with nan.
                for chi in chi_keys:
                    try:
                        chi_val = station_data[chi][station][idx]
                    except (KeyError, IndexError):
                        chi_val = np.nan
                    event_chi[chi].append(chi_val)
            
            # For each of the three properties, plot scatter points and connect them.
            for ax, chi in zip(axes, chi_keys):
                ax.scatter(event_chi[chi], event_snr, color=colors[i])
                ax.plot(event_chi[chi], event_snr, color=colors[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f'coincidence_{title_suffix}.png'))
        plt.close(fig)

    # Plot scatter plots for coincidence events with unique stations.
    plot_events(coincidence_datetimes, "Unique Stations Coincidences", plot_folder)

    # Plot scatter plots for coincidence events (including repeated stations).
    plot_events(coincidence_with_repeated_stations, "Repeated Stations Coincidences", plot_folder)