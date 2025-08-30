import numpy as np
import matplotlib.pyplot as plt
import os
import configparser
from icecream import ic
from NuRadioReco.utilities import fft as nu_fft
import glob
from HRAStationDataAnalysis.C_utils import getTimeEventMasks

def load_station_data(folder, date, station_id, data_name):
    """
    Loads and concatenates data files for a specific station and data type.
    """
    file_pattern = os.path.join(folder, f'{date}_Station{station_id}_{data_name}*')
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        ic(f"Warning: No files found for {data_name} with pattern: {file_pattern}")
        return np.array([])
    
    data_arrays = [np.load(f, allow_pickle=True) for f in file_list]
    data_arrays = [arr for arr in data_arrays if arr.size > 0]
    if not data_arrays:
        return np.array([])
        
    return np.concatenate(data_arrays, axis=0)


def plot_event_traces_and_ffts(event_id, traces, times, station_id, output_dir):
    """
    Plots time traces and their FFTs for a single event.
    """
    num_channels = traces.shape[0]
    fig, axs = plt.subplots(num_channels, 2, figsize=(14, 4 * num_channels), constrained_layout=True)
    if num_channels == 1:
        axs = np.array([axs])

    fig.suptitle(f'Station {station_id} - Event {event_id}', fontsize=16)

    for i in range(num_channels):
        # Plot Time Trace
        ax_trace = axs[i, 0]
        ax_trace.plot(times * 1e9, traces[i]) # time in ns
        ax_trace.set_xlabel('Time [ns]')
        ax_trace.set_ylabel('Amplitude [V]')
        ax_trace.set_title(f'Channel {i} Time Trace')
        ax_trace.grid(True)

        # Plot FFT
        ax_fft = axs[i, 1]
        dt = (times[1] - times[0]) if len(times) > 1 else 1
        the_fft = nu_fft.time2freq(traces[i], dt)
        freqs = np.fft.rfftfreq(len(traces[i]), dt)
        
        ax_fft.plot(freqs / 1e6, np.abs(the_fft)) # Freq in MHz
        ax_fft.set_xlabel('Frequency [MHz]')
        ax_fft.set_ylabel('Amplitude [V/Hz]')
        ax_fft.set_title(f'Channel {i} FFT')
        ax_fft.set_yscale('log')
        ax_fft.grid(True)

    plot_filename = os.path.join(output_dir, f'Station{station_id}_Event{event_id}.png')
    plt.savefig(plot_filename)
    ic(f"Saved plot: {plot_filename}")
    plt.close(fig)

def main(station_ids_to_process=[13, 14, 15, 17, 18, 19, 30]):
    ic.enable()
    ic.configureOutput(prefix='Plotter | ')

    config = configparser.ConfigParser()
    config_path = 'HRAStationDataAnalysis/config.ini'
    if not os.path.exists(config_path):
        ic(f"Error: Config file not found at {config_path}")
        return
    config.read(config_path)

    try:
        date = config['PARAMETERS']['date']
        date_processing = config['PARAMETERS']['date_processing']
    except KeyError as e:
        ic(f"Error: Missing parameter {e} in config file.")
        return

    plot_folder = f'HRAStationDataAnalysis/plots/{date_processing}/'
    station_data_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'
    output_plot_dir = os.path.join(plot_folder, 'events_passing_all_cuts')
    os.makedirs(output_plot_dir, exist_ok=True)

    for station_id in station_ids_to_process:
        ic(f"--- Processing Station {station_id} ---")
        
        passing_events_file = os.path.join(plot_folder, f'PassingEvents_Station{station_id}_{date}.npz')
        if not os.path.exists(passing_events_file):
            ic(f"Warning: Passing events file not found for station {station_id}, skipping.")
            ic(f"Checked path: {passing_events_file}")
            continue

        try:
            passing_events_data = np.load(passing_events_file, allow_pickle=True)
            if 'all_cuts' not in passing_events_data:
                ic(f"Warning: 'all_cuts' data not in {passing_events_file}. Skipping station.")
                continue
            
            passing_events = passing_events_data['all_cuts']
            if passing_events.size == 0:
                ic("No events passed all cuts for this station.")
                continue
            
            ic(f"Found {len(passing_events)} events passing all cuts.")

        except Exception as e:
            ic(f"Error loading passing events file for station {station_id}: {e}")
            continue

        # Load the full dataset for this station to find the traces
        ic("Loading raw data to find traces...")
        raw_event_ids = load_station_data(station_data_folder, date, station_id, 'EventID')
        raw_times = load_station_data(station_data_folder, date, station_id, 'Times')
        raw_traces = load_station_data(station_data_folder, date, station_id, 'Traces')
        
        if any(arr.size == 0 for arr in [raw_event_ids, raw_times, raw_traces]):
            ic("Missing one or more raw data files (EventID, Times, Traces). Skipping station.")
            continue

        # We need to map the 'unique_index' from the npz file back to the original raw data index.
        # The 'unique_index' was an index into a *masked* array.
        # We need to recreate that mask.
        raw_full_times = load_station_data(station_data_folder, date, station_id, 'Times')
        initial_mask, unique_indices_map = getTimeEventMasks(raw_full_times, raw_event_ids)
        
        # The indices in unique_indices_map should correspond to the unique_index in the npz
        masked_indices = np.where(initial_mask)[0]
        
        for passing_event in passing_events:
            event_id = passing_event['event_id']
            unique_idx = passing_event['unique_index']

            try:
                # Find the original index in the raw files
                original_raw_index = masked_indices[unique_indices_map[unique_idx]]
                
                # Verify we have the correct event
                if raw_event_ids[original_raw_index] != event_id:
                    ic(f"Mismatch! Event ID {event_id} not found at expected index. Searching...")
                    # Fallback search
                    possible_indices = np.where(raw_event_ids == event_id)[0]
                    if not possible_indices.any():
                        ic(f"Could not find event {event_id} at all. Skipping.")
                        continue
                    original_raw_index = possible_indices[0] # Take the first one
            
                trace = raw_traces[original_raw_index]
                # The times array for a single event is inside the raw_times array
                times_for_event = raw_times[original_raw_index][0] 

                plot_event_traces_and_ffts(event_id, trace, times_for_event, station_id, output_plot_dir)

            except IndexError:
                ic(f"Could not find original index for event {event_id} with unique_index {unique_idx}. Skipping.")
            except Exception as e:
                ic(f"An error occurred while plotting event {event_id}: {e}")


if __name__ == '__main__':
    # stations_to_plot = [13, 14, 15, 17, 18, 19, 30]
    stations_to_plot = [13, 15, 17, 18]
    main(stations_to_plot)
