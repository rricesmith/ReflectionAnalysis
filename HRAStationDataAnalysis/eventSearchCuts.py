import configparser
import os
import datetime
import argparse
from icecream import ic
import numpy as np
import glob



def cluster_cut(times, traces, amplitude_threshold, time_period, cut_frequency):
    """
    Creates a mask to remove events that occur in bursts.
    
    For a given array of event times and their corresponding amplitudes, this function
    returns a boolean mask (of the same length) that is True for events to be kept and
    False for events that fall in time windows where the number of events with amplitudes
    above 'amplitude_threshold' is greater than or equal to 'cut_frequency'. The time window
    is defined by 'time_period' (a datetime.timedelta).

    Args:
        times (array-like): Array or list of datetime.datetime objects.
        traces (array-like): Array or list of traces for each time (numerical values).
        amplitude_threshold (numeric): Only events with an amplitude greater than this value are considered.
        time_period (datetime.timedelta): The time range window, for example, 60 seconds.
        cut_frequency (float): The minimum number of events (above amplitude_threshold) within a time_period to trigger the cut in Hertz.

    Returns:
        numpy.ndarray: A boolean mask of the same length as times/amplitudes. True for events to keep, False for events to remove.
    """
    import numpy as np  # Ensure numpy is imported within the function if necessary.

    # Calculate the number of events in the time window.
    cut_frequency = cut_frequency * time_period.total_seconds()

    # Ensure times and amplitudes are numpy arrays
    times = np.array(times)
    traces = np.array(traces)
    n = len(times)
    mask = np.ones(n, dtype=bool)


    start = 0
    for end in range(n):
        # Advance the start index of the window until the time difference is less than time_period.
        while times[end] - times[start] >= time_period:
            start += 1
        # For each event in the window, check if any absolute amplitude exceeds the threshold.   
        window_events = traces[start:end+1]
        count = np.sum(np.any(np.abs(window_events) > amplitude_threshold, axis=(1, 2)))
        if count >= cut_frequency:
            # Mark all events within this window (at indices start to end)
            mask[start:end+1] = False
    return mask


if __name__ == "__main__":
    import gc
    # This function processes each station, running all cuts on the data, saving the masks of each cut to a numpy file
    # and returning the final mask to be used for the event search

    # Load the configuration file
    config = configparser.ConfigParser()
    config.read('HRAStationDataAnalysis/config.ini')
    date = config['PARAMETERS']['date']

    station_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'nurFiles', date)

    station_ids = [13, 14, 15, 17, 18, 19, 30]
    for station_id in station_ids:

        # Load the data for this station

        file_list = sorted(glob.glob(station_data_folder + f'/{date}_Station{station_id}_Times*'))
        times = [np.load(f) for f in file_list]
        times = np.concatenate(times, axis=0)
        file_list = sorted(glob.glob(station_data_folder + f'/{date}_Station{station_id}_Traces*'))
        traces = [np.load(f) for f in file_list]
        traces = np.concatenate(traces, axis=0)

        # for file in os.listdir(station_data_folder):
        #     if file.startswith(f'{date}_Station{station_id}_Times'):
        #         file_path = os.path.join(station_data_folder, file)
        #         ic(f"Loading file: {file_path}")
        #         data = np.load(file_path, allow_pickle=True)
        #         data = data.flatten()  # Flatten the data to ensure it's a 1D array
        #         times.extend(data.tolist())
        #         del data
        #     if file.startswith(f'{date}_Station{station_id}_Traces'):
        #         file_path = os.path.join(station_data_folder, file)
        #         ic(f"Loading file: {file_path}")
        #         data = np.load(file_path, allow_pickle=True)
        #         traces.extend(data.tolist())
        #         del data
        #     gc.collect()  # Free up memory if necessary
    
        # Convert to numpy arrays
        times = np.array(times)
        traces = np.array(traces)

        # Remove zero timestamps
        mask = times != 0
        times = times[mask]
        traces = traces[mask]

        # Cut times that have timestamps before stations exists
        mask = times >= datetime.datetime(2013, 1, 1).timestamp()
        times = times[mask]
        traces = traces[mask]


        # Apply first cluster cut for storms
        # Amplitude threshold is 300mV, time period is 3600s, cut frequency is 5Hz    
        storm_mask = cluster_cut(times, traces, 0.3, datetime.timedelta(seconds=3600), 5)
        # Apply second cluster cut for bursts
        # Amplitude threshold is 150mV, time period is 60s, cut frequency is 3Hz
        burst_mask = cluster_cut(times, traces, 0.15, datetime.timedelta(seconds=60), 3)

        ic(f"Storm mask: {sum(storm_mask)}, Burst mask: {sum(burst_mask)}, of total {len(times)}")
        ic(f"Storm mask % {sum(storm_mask)/len(times)}, Burst mask % {sum(burst_mask)/len(times)}")

        quit()
