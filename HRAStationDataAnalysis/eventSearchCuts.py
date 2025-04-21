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


    # Ensure times and amplitudes are numpy arrays
    times = np.array(times)
    traces = np.array(traces)
    n = len(times)
    mask = np.ones(n, dtype=bool)


    start = 0
    for end in range(n):
        # Advance the start index of the window until the time difference is less than time_period.
        while times[end] - times[start] >= time_period.total_seconds():
            start += 1
        # For each event in the window, check if any absolute amplitude exceeds the threshold.   
        window_events = traces[start:end+1]
        count = np.sum(np.any(np.abs(window_events) > amplitude_threshold, axis=(1, 2)))
        if count >= cut_frequency:
            # Mark all events within this window (at indices start to end)
            mask[start:end+1] = False
    return mask

def L1_cut(traces, power_cut=0.3):
    """
    Returns a boolean mask selecting events that pass the power cut.

    For each event (with trace shape (4, 256)), each channel is checked separately.
    If any channel has a bin value that exceeds power_cut * (the sum of that channel),
    then the entire event is removed. This means that an event passes (True) only if
    for every channel, no single bin exceeds the given fraction of that channel's total.

    Args:
        times (array-like): The event times (unused in this calculation, but kept for interface symmetry).
        traces (np.ndarray): Array of event traces with shape (n_events, 4, 256).
        power_cut (float): The fraction threshold (default 0.3) such that if any bin in a channel
                           is greater than power_cut times the channel's total, the event fails.

    Returns:
        np.ndarray: A boolean mask of shape (n_events,) where True indicates the event passes the cut.
    """
    import numpy as np
    from NuRadioReco.utilities.fft import time2freq

    n_events = traces.shape[0]
    mask = np.ones(n_events, dtype=bool)
    
    # Process each event.
    for i in range(n_events):
        # Check each channel independently.
        for channel in range(traces.shape[1]):
            trace_channel = time2freq(traces[i, channel, :], 2) # sampling rate of 2GHz
            total_power = np.sum(trace_channel)
            # Protect against a channel with zero total to avoid division-by-zero.
            if total_power == 0:
                continue
            # If any bin in this channel exceeds power_cut fraction of the total, flag the event.
            if np.any(trace_channel > power_cut * total_power):
                mask[i] = False
                break  # No need to check other channels for this event.
    return mask



def plot_cuts_amplitudes(times, traces, output_dir=".", **cuts):
    """
    Creates and saves scatter plots of events by season (October to April) for the years 2013-2020.
    
    For each seasonal period, the x-axis is the event time (converted from a raw unix timestamp 
    to a date formatted as 'MM/DD/YY'), and the y-axis is the maximum absolute amplitude from the trace.
    The function plots all events in light gray and overlays any specified cut masks with their labels.
    
    Args:
        times (array-like): Unix timestamps (one per event).
        traces (numpy.ndarray): Array of event traces of shape (n_events, 4, 256).
        output_dir (str): Directory in which to save the plots (default is current directory).
        **cuts: Arbitrary keyword arguments where each key is the cut name (for the legend) and each 
                value is a boolean array of shape (n_events,) tagging events of interest.
                
    Returns:
        None (each seasonal plot is saved as a PNG file).
    """
    import os
    import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import itertools
    
    # Create output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Convert raw unix times into datetime objects.
    dt_times = np.array([datetime.datetime.fromtimestamp(t) for t in times])
    
    # Compute maximum absolute amplitude for each trace.
    max_amps = np.max(np.abs(traces), axis=(1, 2))
    
    # Loop for each season between 2013 and 2020.
    for start_year in range(2013, 2020):
        # Reset the marker cycle for each season.
        markers = itertools.cycle(("v", "s", "*", "d"))

        # Define the seasonal period: October 1 of start_year to April 30 of start_year+1.
        season_start = datetime.datetime(start_year, 10, 1)
        season_end = datetime.datetime(start_year + 1, 4, 30, 23, 59, 59)
        season_mask = (dt_times >= season_start) & (dt_times <= season_end)
        
        # Skip this season if no events.
        if not np.any(season_mask):
            continue
        
        # Data for events in the season.
        season_times = dt_times[season_mask]
        season_amps = max_amps[season_mask]
        
        plt.figure(figsize=(10,6))
        plt.title(f"Season {start_year}-{start_year + 1} Activity")
        plt.xlabel("Time")
        plt.ylabel("Max Amplitude")
        
        # Plot all seasonal events in light gray.
        plt.scatter(season_times, season_amps, color="lightgray", s=3, label="All Events", facecolor="none", edgecolor="black")
        
        # Overlay each cut sequentially.
        cut_mask_sum = np.ones_like(season_mask, dtype=bool)
        for cut_name, cut_mask in cuts.items():
            # Ensure cut_mask is an array.
            cut_mask = np.array(cut_mask)
            cut_mask_sum = cut_mask_sum & cut_mask
            # Combine the cut mask with the seasonal mask.
            season_cut_mask = season_mask & cut_mask_sum
            if np.any(season_cut_mask):
                plt.scatter(dt_times[season_cut_mask],
                            max_amps[season_cut_mask], s=3, label=cut_name, marker=next(markers))
        
        # Format the x-axis to show dates as 'MM/DD/YY'.
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.tight_layout()
        
        # Save the figure.
        filename = os.path.join(output_dir, f"season_{start_year}_{start_year+1}.png")
        plt.savefig(filename)
        plt.close()
        ic(f"Saved plot for season {start_year}-{start_year+1} to {filename}")


def plot_cuts_rates(times, bin_size=30*60, output_dir=".", **cuts):
    """
    Creates and saves scatter plots of event rate versus time for seasonal periods.
    
    For each seasonal period (October 1 to April 30 between 2013 and 2020),
    events are binned over a period defined by bin_size (default is 30 minutes, i.e. 1800 seconds).
    The event rate in each bin is computed as:
          event_rate = number_of_events_in_bin / bin_size   (units: events/second)
    The function plots:
      • All events (in light gray) aggregated for that season, and
      • Overlays the event rate for any provided cuts (boolean arrays of shape [n_events])
        with the legend label specified by the keyword argument name.
        
    Args:
        times (array-like): Unix timestamps (one per event).
        traces (numpy.ndarray): Array of event traces with shape (n_events, 4, 256). (Note: this
                                parameter is not used in the rate calculation but is kept for interface symmetry.)
        bin_size (int or float): Bin width in seconds over which to average for the event rate.
                                 Default is 1800 seconds (30 minutes).
        output_dir (str): Directory in which to save the plots. Defaults to current directory.
        **cuts: Arbitrary keyword arguments where each key is the cut name (to appear in the legend)
                and each value is a boolean array of shape (n_events,) that flags events for that cut.
                
    Returns:
        None. A PNG file is saved for each seasonal period.
    """
    import os
    import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import itertools

    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert seasonal boundaries from datetime to unix timestamps.
    # We'll use the provided "times" (which are unix times) for the binning.
    # Loop from 2013 to 2019 to cover seasons 2013-2014 up to 2019-2020.
    for start_year in range(2013, 2020):
        # Reset the marker cycle for each season.
        markers = itertools.cycle(("v", "s", "*", "d"))

        # Define season: October 1 (start_year) to April 30 (start_year+1)
        season_start_dt = datetime.datetime(start_year, 10, 1)
        season_end_dt = datetime.datetime(start_year + 1, 4, 30, 23, 59, 59)
        
        season_start_unix = season_start_dt.timestamp()
        season_end_unix = season_end_dt.timestamp()
        
        # Mask for events in the seasonal window.
        season_mask = (np.array(times) >= season_start_unix) & (np.array(times) <= season_end_unix)
        if not np.any(season_mask):
            # No events in this season. Skip this season.
            continue
        
        # Get season events (using the original unix time stamps)
        season_times = np.array(times)[season_mask]
        
        # Create bins: each bin is bin_size seconds wide.
        # Ensure the final bin covers the full seasonal period.
        bins = np.arange(season_start_unix, season_end_unix + bin_size, bin_size)
        # Calculate bin centers to use as the x-axis value.
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Histogram for ALL events in this season.
        count_all, _ = np.histogram(season_times, bins=bins)
        rate_all = count_all / bin_size  # events per second
        
        # Prepare the figure.
        plt.figure(figsize=(10, 6))
        plt.title(f"Season {start_year}-{start_year + 1} Event Rate")
        plt.xlabel("Time")
        plt.ylabel("Event Rate (Hz)")
        
        # Plot the overall event rate as light gray scatter.
        # Convert bin centers to datetime objects.
        dt_bin_centers = [datetime.datetime.fromtimestamp(ts) for ts in bin_centers]
        plt.scatter(dt_bin_centers, rate_all, s=3, label="All Events", facecolor="none", edgecolor="black")
        
        # For each provided cut, compute and plot the binned event rate.
        for cut_name, cut_mask in cuts.items():
            # Ensure cut_mask is a numpy array.
            cut_mask = np.array(cut_mask)
            # Combine seasonal mask with the given cut mask.
            season_cut_mask = season_mask & cut_mask
            if not np.any(season_cut_mask):
                continue  # Skip if no events match the cut within this season.
            
            # Get the unix times for the cut events.
            season_cut_times = np.array(times)[season_cut_mask]
            count_cut, _ = np.histogram(season_cut_times, bins=bins)
            rate_cut = count_cut / bin_size
            
            plt.scatter(dt_bin_centers, rate_cut, s=3, label=cut_name, marker=next(markers))
        
        # Format the x-axis to display the date and time nicely.
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y %H:%M'))
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.tight_layout()
        
        # Save the plot instead of showing it.
        out_filename = os.path.join(output_dir, f"season_{start_year}_{start_year+1}_rate.png")
        plt.savefig(out_filename)
        plt.close()
        ic(f"Saved event rate plot for season {start_year}-{start_year+1} to {out_filename}")



if __name__ == "__main__":
    import gc
    # This function processes each station, running all cuts on the data, saving the masks of each cut to a numpy file
    # and returning the final mask to be used for the event search

    # Load the configuration file
    config = configparser.ConfigParser()
    config.read('HRAStationDataAnalysis/config.ini')
    date = config['PARAMETERS']['date']

    station_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'nurFiles', date)
    cuts_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'cuts', date)
    plot_folder = os.path.join('HRAStationDataAnalysis', 'plots', date)

    station_ids = [13, 14, 15, 17, 18, 19, 30]
    for station_id in station_ids:

        # Load the data for this station

        file_list = sorted(glob.glob(station_data_folder + f'/{date}_Station{station_id}_Times*'))
        times = [np.load(f) for f in file_list]
        times = np.concatenate(times, axis=0)
        times = times.squeeze()
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
        ic(times.shape, traces.shape)

        # Remove zero timestamps
        mask = times != 0
        times = times[mask]
        traces = traces[mask]

        # Cut times that have timestamps before stations exists
        mask = times >= datetime.datetime(2013, 1, 1).timestamp()
        times = times[mask]
        traces = traces[mask]


        # Check if cuts are already processed
        # If so load cuts, otherwise process cuts and save
        cut_file = os.path.join(cuts_data_folder, f'{date}_Station{station_id}_Cuts.npy')
        os.makedirs(cuts_data_folder, exist_ok=True)
        if os.path.exists(cut_file):
            ic(f"Cut file already exists: {cut_file}")
            cuts = np.load(cut_file, allow_pickle=True)
            cuts = cuts.item()
            L1_mask = cuts['L1_mask']
            storm_mask = cuts['storm_mask']
            burst_mask = cuts['burst_mask']
        else:
            # First cut is L1 cut
            # Amplitude threshold is 0.3, cut frequency is 2 in window
            L1_mask = L1_cut(traces, power_cut=0.3)
            # Amplitude threshold is 300mV, time period is 3600s, cut frequency is 2 in window
            storm_mask = cluster_cut(times, traces, 0.3, datetime.timedelta(seconds=3600), 2)
            # Apply second cluster cut for bursts
            # Amplitude threshold is 150mV, time period is 60s, cut frequency is 2 in window
            burst_mask = cluster_cut(times, traces, 0.15, datetime.timedelta(seconds=60), 2)
            # Save the cuts to a numpy file
            cuts = {
                'L1_mask': L1_mask,
                'storm_mask': storm_mask,
                'burst_mask': burst_mask
            }
            np.save(cut_file, cuts, allow_pickle=True)


        ic(f"\nL1 mask {L1_mask.shape}, Storm mask {storm_mask.shape}, Burst mask {burst_mask.shape}")
        ic(f"\nL1 mask: {sum(L1_mask)}, Storm mask: {sum(storm_mask)}, Burst mask: {sum(burst_mask)}, of total {len(times)}")
        ic(f"\nL1 mask % {100*sum(L1_mask)/len(times)}\nStorm mask % {100*sum(storm_mask)/len(times)}%\n Burst mask % {100*sum(burst_mask)/len(times)}%")

        ic(f"\nL1 + Storm + Burst: {sum(L1_mask & storm_mask & burst_mask)}, {sum(L1_mask & storm_mask & burst_mask)/len(times)}%")

        # Plot the cuts
        plot_folder_station = os.path.join(plot_folder, f'Station{station_id}')
        os.makedirs(plot_folder_station, exist_ok=True)
        plot_cuts_amplitudes(times, traces, plot_folder_station, L1_mask=L1_mask, storm_mask=storm_mask, burst_mask=burst_mask)

        plot_cuts_rates(times, output_dir=plot_folder_station, L1_mask=L1_mask,  storm_mask=storm_mask, burst_mask=burst_mask)

        quit()
