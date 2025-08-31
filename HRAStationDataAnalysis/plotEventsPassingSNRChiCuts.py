import numpy as np
import matplotlib.pyplot as plt
import os
import configparser
from icecream import ic
from NuRadioReco.utilities import fft as nu_fft
from NuRadioReco.utilities import units
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


def plot_event_traces_and_ffts(event_id, traces, times, station_id, output_dir, event_info):
    """
    Plots time traces and their FFTs for a single event.
    """
    num_channels = traces.shape[0]
    fig, axs = plt.subplots(num_channels, 2, figsize=(16, 4 * num_channels), constrained_layout=True)
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

    # Add event info text
    info_text = (
        f"SNR: {event_info.get('snr', 'N/A'):.2f}\n"
        f"Azimuth: {np.rad2deg(event_info.get('azi', 0)):.2f}°\n"
        f"Zenith: {np.rad2deg(event_info.get('zen', 0)):.2f}°\n"
        f"Chi2016: {event_info.get('chi2016', 'N/A'):.3f}\n"
        f"ChiRCR: {event_info.get('chircr', 'N/A'):.3f}\n"
        f"ChiBad: {event_info.get('chibad', 'N/A'):.3f}"
    )
    fig.text(0.99, 0.5, info_text, transform=fig.transFigure, fontsize=12,
             verticalalignment='center', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    plt.subplots_adjust(right=0.85)

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
        raw_snr = load_station_data(station_data_folder, date, station_id, 'SNR')
        raw_chi2016 = load_station_data(station_data_folder, date, station_id, 'Chi2016')
        raw_chircr = load_station_data(station_data_folder, date, station_id, 'ChiRCR')
        raw_chibad = load_station_data(station_data_folder, date, station_id, 'ChiBad')
        raw_azi = load_station_data(station_data_folder, date, station_id, 'Azi')
        raw_zen = load_station_data(station_data_folder, date, station_id, 'Zen')
        
        if any(arr.size == 0 for arr in [raw_event_ids, raw_times, raw_traces]):
            ic("Missing one or more raw data files (EventID, Times, Traces). Skipping station.")
            continue

        initial_mask, unique_indices = getTimeEventMasks(raw_times, raw_event_ids)

        # Apply the mask and select unique events to get the final, cleaned arrays
        event_ids_clean = raw_event_ids[initial_mask][unique_indices]
        traces_clean = raw_traces[initial_mask][unique_indices]
        snr_clean = raw_snr[initial_mask][unique_indices] if raw_snr.size > 0 else np.array([])
        chi2016_clean = raw_chi2016[initial_mask][unique_indices] if raw_chi2016.size > 0 else np.array([])
        chircr_clean = raw_chircr[initial_mask][unique_indices] if raw_chircr.size > 0 else np.array([])
        chibad_clean = raw_chibad[initial_mask][unique_indices] if raw_chibad.size > 0 else np.array([])
        azi_clean = raw_azi[initial_mask][unique_indices] if raw_azi.size > 0 else np.array([])
        zen_clean = raw_zen[initial_mask][unique_indices] if raw_zen.size > 0 else np.array([])

        for passing_event in passing_events:
            event_id = passing_event['event_id']
            unique_idx = passing_event['unique_index']

            try:
                # The unique_idx from the passing_events file directly corresponds to the index
                # in the cleaned arrays (e.g., event_ids_clean, traces_clean).
                
                # Verify we have the correct event
                if event_ids_clean[unique_idx] != event_id:
                    ic(f"Mismatch! Event ID {event_id} from file does not match event ID {event_ids_clean[unique_idx]} at index {unique_idx}. Searching...")
                    # Fallback search in the cleaned array
                    possible_indices = np.where(event_ids_clean == event_id)[0]
                    if not possible_indices.any():
                        ic(f"Could not find event {event_id} in cleaned data. Skipping.")
                        continue
                    unique_idx = possible_indices[0] # Take the first match
                    ic(f"Found event {event_id} at new index {unique_idx} instead.")
                
                trace = traces_clean[unique_idx]
                # The times array for a single event is calculated from the sampling rate
                sampling_rate = 2 * units.GHz
                times_for_event = np.arange(trace.shape[1]) / sampling_rate

                event_info = {
                    'snr': snr_clean[unique_idx] if snr_clean.size > 0 else 'N/A',
                    'azi': azi_clean[unique_idx] if azi_clean.size > 0 else 0,
                    'zen': zen_clean[unique_idx] if zen_clean.size > 0 else 0,
                    'chi2016': chi2016_clean[unique_idx] if chi2016_clean.size > 0 else 'N/A',
                    'chircr': chircr_clean[unique_idx] if chircr_clean.size > 0 else 'N/A',
                    'chibad': chibad_clean[unique_idx] if chibad_clean.size > 0 else 'N/A'
                }

                plot_event_traces_and_ffts(event_id, trace, times_for_event, station_id, output_plot_dir, event_info)

            except IndexError:
                ic(f"Index error for event {event_id} with unique_index {unique_idx}. The index is out of bounds for the cleaned data arrays. Skipping.")
            except Exception as e:
                ic(f"An error occurred while plotting event {event_id}: {e}")


if __name__ == '__main__':
    stations_to_plot = [13, 14, 15, 17, 18, 19, 30]
    # stations_to_plot = [13, 15, 17, 18]
    main(stations_to_plot)
