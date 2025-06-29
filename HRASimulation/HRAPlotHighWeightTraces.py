import os
import numpy as np
import matplotlib.pyplot as plt
import configparser
import NuRadioReco.modules.io.eventReader
from NuRadioReco.utilities import units
from NuRadioReco.utilities import fft

# Assuming the HRAEventObject and HRANurToNpy files are in a location
# accessible by your Python path. If not, you might need to adjust the path.
# import sys
# sys.path.append('/path/to/your/modules')
from HRANurToNpy import loadHRAfromH5
# The HRAevent class is loaded via pickle, so we need it in the scope.
from HRAEventObject import HRAevent


def _plot_trace_and_fft(plot_folder, trace, times, channel_id, event_id, station_id, weight_name):
    """
    Helper function to generate and save a 1x2 plot containing the
    time-domain trace and its FFT spectrum.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        f'Event {event_id} - Station {station_id} - Channel {channel_id} (Weight: {weight_name})',
        fontsize=16
    )

    # --- 1. Plot Time-Domain Trace ---
    time_ax_ns = times / units.ns
    ax1.plot(time_ax_ns, trace / units.mV)
    ax1.set_xlabel('Time [ns]')
    ax1.set_ylabel('Voltage [mV]')
    ax1.set_title('Time Domain')
    ax1.grid(True, alpha=0.3)

    # --- 2. Calculate and Plot FFT ---
    if len(trace) > 1:
        sampling_rate_hz = 1 / (np.mean(np.diff(times))) # More robust than assuming 0.5ns
        spectrum = np.abs(fft.time2freq(trace, sampling_rate_hz))
        freq_ax_mhz = np.fft.rfftfreq(len(trace), d=1/sampling_rate_hz) / 1e6

        # Don't plot the DC component for better scaling
        if len(spectrum) > 0:
            ax2.plot(freq_ax_mhz[1:], spectrum[1:])

        ax2.set_xlabel('Frequency [MHz]')
        ax2.set_ylabel('Amplitude [ADC units / Hz]') # Check units if needed
        ax2.set_title('Frequency Domain (FFT)')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle

    plot_filename = os.path.join(plot_folder, f"trace_evt{event_id}_stn{station_id}_ch{channel_id}.png")
    plt.savefig(plot_filename)
    plt.close(fig)


def plot_highest_weight_events(
    hra_event_list,
    nur_file_dir,
    output_dir,
    weight_name,
    num_events_to_plot=100,
    sigma=4.5
):
    """
    Finds the highest weight events, plots their traces and FFTs, and caches the trace data.
    On subsequent runs, it replots from the cached data if available.
    """
    print(f"Processing weight name: '{weight_name}'...")

    # --- 1. Find the top N events based on the specified weight ---
    weighted_events = []
    for event in hra_event_list:
        if event.hasWeight(weight_name, sigma=sigma):
            weight = event.getWeight(weight_name, primary=True, sigma=sigma)
            if weight is not None and not np.isnan(weight):
                weighted_events.append((weight, event.getEventID()))

    # Diagnostic print to see how many events have a valid weight
    print(f"  Found {len(weighted_events)} events with valid weights for '{weight_name}' before filtering for top {num_events_to_plot}.")

    if not weighted_events:
        print(f"  Warning: No events found with weight '{weight_name}'. Skipping.")
        return

    # Sort events by weight in descending order
    weighted_events.sort(key=lambda x: x[0], reverse=True)

    # Get the top N events
    top_events = weighted_events[:num_events_to_plot]
    top_event_ids = {event_id for weight, event_id in top_events}

    if not top_event_ids:
        print(f"  No valid weighted events to plot for '{weight_name}'.")
        return

    print(f"  Identified top {len(top_event_ids)} events. Now searching .nur files or cache to plot traces...")

    # --- 2. Iterate through .nur files to find and plot these events ---
    eventReader = NuRadioReco.modules.io.eventReader.eventReader()
    nur_files = [os.path.join(nur_file_dir, f) for f in os.listdir(nur_file_dir) if f.endswith('.nur')]
    found_event_ids = set()

    for nur_file in nur_files:
        if not top_event_ids: break
        eventReader.begin(nur_file)
        print(f"  Scanning {os.path.basename(nur_file)}...")
        for event in eventReader.run():
            if not top_event_ids: break
            current_event_id = event.get_id()
            if current_event_id in top_event_ids:
                print(f"    Found event ID: {current_event_id}. Processing stations...")
                for station in event.get_stations():
                    if not station.has_triggered(): continue
                    station_id = station.get_id()
                    plot_folder = os.path.join(output_dir, weight_name, f"evt_{current_event_id}", f"stn_{station_id}")
                    os.makedirs(plot_folder, exist_ok=True)
                    trace_data_path = os.path.join(plot_folder, 'trace_data.npy')

                    if os.path.exists(trace_data_path):
                        print(f"      - Plotting from cached data for station {station_id}")
                        data = np.load(trace_data_path, allow_pickle=True).item()
                        traces, times, LPDA_channels = data['traces'], data['times'], data['channels']
                    else:
                        print(f"      - Processing from .nur file for station {station_id}")
                        LPDA_channels = [4, 5, 6, 7] if station_id == 52 else [0, 1, 2, 3]
                        traces, times = [], []
                        for channel in station.iter_channels(use_channels=LPDA_channels):
                            traces.append(channel.get_trace())
                            times.append(channel.get_times())
                        data_to_save = {'traces': traces, 'times': times, 'channels': LPDA_channels}
                        np.save(trace_data_path, data_to_save)

                    # --- Plotting loop for each channel ---
                    for i, trace in enumerate(traces):
                        _plot_trace_and_fft(
                            plot_folder, trace, times[i], LPDA_channels[i],
                            current_event_id, station_id, weight_name
                        )

                found_event_ids.add(current_event_id)
                top_event_ids.remove(current_event_id)
        eventReader.end()

    print(f"\nFinished processing for '{weight_name}'.")
    print(f"Plotted {len(found_event_ids)} out of {min(len(weighted_events), num_events_to_plot)} requested top events.")
    if len(top_event_ids) > 0:
        print(f"Warning: Could not find the following event IDs: {sorted(list(top_event_ids))}")

if __name__ == "__main__":
    config_path = 'HRASimulation/config.ini'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}.")

    config = configparser.ConfigParser()
    config.read(config_path)

    sim_folder = config['FOLDERS']['sim_folder']
    numpy_folder = config['FOLDERS']['numpy_folder']
    save_folder = config['FOLDERS']['save_folder']
    plot_output_folder = os.path.join(save_folder, 'high_weight_event_plots')
    os.makedirs(plot_output_folder, exist_ok=True)

    h5_event_file = os.path.join(numpy_folder, 'HRAeventList.h5')
    if not os.path.exists(h5_event_file):
        raise FileNotFoundError(f"HDF5 event file not found at {h5_event_file}.")

    print(f"Loading HRA event list from {h5_event_file}...")
    HRAeventList = loadHRAfromH5(h5_event_file)
    print(f"Loaded {len(HRAeventList)} events.")

    weights_to_process = [
        '1_coincidence_wrefl', '2_coincidence_wrefl', 'combined_direct',
        'combined_reflected', '100s_direct', '100s_reflected',
        '200s_direct', '200s_reflected'
    ]

    for weight in weights_to_process:
        plot_highest_weight_events(
            hra_event_list=HRAeventList, nur_file_dir=sim_folder,
            output_dir=plot_output_folder, weight_name=weight,
            num_events_to_plot=100
        )
        print("-" * 50)

    print("All plotting tasks complete.")
