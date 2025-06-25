import os
import numpy as np
import matplotlib.pyplot as plt
import configparser
import NuRadioReco.modules.io.eventReader
from NuRadioReco.utilities import units

# Assuming the HRAEventObject and HRANurToNpy files are in a location
# accessible by your Python path. If not, you might need to adjust the path.
# For example, by adding:
# import sys
# sys.path.append('/path/to/your/modules')
from HRANurToNpy import loadHRAfromH5
# The HRAevent class is loaded via pickle, so we need it in the scope.
from HRAEventObject import HRAevent


def _plot_traces(plot_folder, times, traces, lpda_channels, event_id, station_id, weight_name):
    """
    Helper function to generate and save the plot from trace data.
    """
    fig, axs = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
    fig.suptitle(f'Event {event_id} - Station {station_id} - Weight Name: {weight_name}', fontsize=16)

    for i in range(4):
        if i < len(traces):
            axs[i].plot(times[i] / units.ns, traces[i] / units.mV)
            axs[i].set_title(f'Channel {lpda_channels[i]}')
            axs[i].set_xlabel('Time [ns]')
        else:
            # Handle cases where a channel might be missing
            axs[i].set_title(f'Channel {lpda_channels[i]} (No Data)')
            axs[i].axis('off')

    axs[0].set_ylabel('Voltage [mV]')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle

    plot_filename = os.path.join(plot_folder, f"traces_evt{event_id}_stn{station_id}.png")
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
    Finds the highest weight events, plots their traces, and caches the trace data.
    On subsequent runs, it replots from the cached data if available.

    Args:
        hra_event_list (list): A list of HRAevent objects.
        nur_file_dir (str): The directory containing the original .nur files.
        output_dir (str): The base directory where plots will be saved.
        weight_name (str): The name of the weight to use for sorting events.
        num_events_to_plot (int, optional): The number of top events to plot.
                                            Defaults to 100.
        sigma (float, optional): The sigma value for getting the weight.
                                 Defaults to 4.5.
    """
    print(f"Processing weight name: '{weight_name}'...")

    # --- 1. Find the top N events based on the specified weight ---
    weighted_events = []
    for event in hra_event_list:
        if event.hasWeight(weight_name, sigma=sigma):
            # We get the primary weight, assuming that's the one of interest.
            weight = event.getWeight(weight_name, primary=True, sigma=sigma)
            if weight is not None and not np.isnan(weight):
                weighted_events.append((weight, event.getEventID()))

    if not weighted_events:
        print(f"Warning: No events found with weight '{weight_name}'. Skipping.")
        return

    # Sort events by weight in descending order
    weighted_events.sort(key=lambda x: x[0], reverse=True)

    # Get the top N events
    top_events = weighted_events[:num_events_to_plot]
    top_event_ids = {event_id for weight, event_id in top_events}

    if not top_event_ids:
        print(f"No valid weighted events to plot for '{weight_name}'.")
        return

    print(f"Found {len(top_event_ids)} events. Now searching .nur files or cache to plot traces...")

    # --- 2. Iterate through .nur files to find and plot these events ---
    eventReader = NuRadioReco.modules.io.eventReader.eventReader()
    nur_files = [os.path.join(nur_file_dir, f) for f in os.listdir(nur_file_dir) if f.endswith('.nur')]

    found_event_ids = set()

    for nur_file in nur_files:
        if not top_event_ids:  # Stop if we've found all events
            break

        eventReader.begin(nur_file)
        print(f"Scanning {os.path.basename(nur_file)}...")

        for i, event in enumerate(eventReader.run()):
            current_event_id = event.get_id()

            if current_event_id in top_event_ids:
                print(f"  Found event ID: {current_event_id}. Plotting traces...")

                for station in event.get_stations():
                    if not station.has_triggered():
                        continue

                    station_id = station.get_id()

                    # --- Create Directories for Plots ---
                    plot_folder = os.path.join(
                        output_dir,
                        weight_name,
                        f"evt_{current_event_id}",
                        f"stn_{station_id}"
                    )
                    os.makedirs(plot_folder, exist_ok=True)

                    trace_data_path = os.path.join(plot_folder, 'trace_data.npy')

                    # Check for cached trace data
                    if os.path.exists(trace_data_path):
                        print(f"    - Plotting from cached data for station {station_id}")
                        data = np.load(trace_data_path, allow_pickle=True).item()
                        traces = data['traces']
                        times = data['times']
                        LPDA_channels = data['channels']
                    else:
                        print(f"    - Processing from .nur file for station {station_id}")
                        # --- Get Traces for the 4 LPDA channels ---
                        if station_id == 52:
                            LPDA_channels = [4, 5, 6, 7]
                        else:
                            LPDA_channels = [0, 1, 2, 3]

                        traces = []
                        times = []
                        for channel in station.iter_channels(use_channels=LPDA_channels):
                            traces.append(channel.get_trace())
                            times.append(channel.get_times())

                        # Save the extracted data to the cache file
                        data_to_save = {
                            'traces': traces,
                            'times': times,
                            'channels': LPDA_channels
                        }
                        np.save(trace_data_path, data_to_save)

                    # --- Plotting ---
                    _plot_traces(
                        plot_folder, times, traces, LPDA_channels,
                        current_event_id, station_id, weight_name
                    )

                # Mark this event as found and remove from search set
                found_event_ids.add(current_event_id)
                top_event_ids.remove(current_event_id)

                # Stop iterating through this file if we've found all we need from it
                if not top_event_ids:
                    break

        eventReader.end()

    print(f"\nFinished processing for '{weight_name}'.")
    print(f"Plotted {len(found_event_ids)} out of {min(len(found_event_ids), num_events_to_plot)} requested events.")
    if len(top_event_ids) > 0:
        print(f"Warning: Could not find the following event IDs: {sorted(list(top_event_ids))}")


if __name__ == "__main__":
    # --- Configuration ---
    # This assumes a 'config.ini' file is in the specified path
    # relative to where you run the script.
    config_path = 'HRASimulation/config.ini'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}. Please ensure the path is correct.")

    config = configparser.ConfigParser()
    config.read(config_path)

    sim_folder = config['FOLDERS']['sim_folder']      # .nur file directory
    numpy_folder = config['FOLDERS']['numpy_folder']  # .h5 file directory
    save_folder = config['FOLDERS']['save_folder']    # Base output folder

    # Define an output folder specifically for these plots
    plot_output_folder = os.path.join(save_folder, 'high_weight_event_plots')
    os.makedirs(plot_output_folder, exist_ok=True)

    # --- Load HRA Event List ---
    # This file is created by HRANurToNpy.py and contains all event metadata and weights
    h5_event_file = os.path.join(numpy_folder, 'HRAeventList.h5')
    if not os.path.exists(h5_event_file):
        raise FileNotFoundError(
            f"The HDF5 event file was not found at {h5_event_file}. "
            "Please run HRANurToNpy.py first to generate it."
        )

    print(f"Loading HRA event list from {h5_event_file}...")
    HRAeventList = loadHRAfromH5(h5_event_file)
    print(f"Loaded {len(HRAeventList)} events.")

    # --- Define Weights to Process ---
    # Add all the weight names you want to generate plots for
    weights_to_process = [
        # '1_coincidence_wrefl',
        # '2_coincidence_wrefl',
        'combined_direct',
        'combined_reflected',
        # '100s_direct',
        # '100s_reflected',
        # '200s_direct',
        # '200s_reflected'
        # Add any other weight names you defined in HRANurToNpy.py
    ]

    # --- Run the Plotting Function for Each Weight ---
    for weight in weights_to_process:
        plot_highest_weight_events(
            hra_event_list=HRAeventList,
            nur_file_dir=sim_folder,
            output_dir=plot_output_folder,
            weight_name=weight,
            num_events_to_plot=100
        )
        print("-" * 50)

    print("All plotting tasks complete.")
