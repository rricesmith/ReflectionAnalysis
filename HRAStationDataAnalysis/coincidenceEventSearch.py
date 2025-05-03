import configparser
import numpy as np
import os
import datetime
from icecream import ic
import gc
import glob



def findCoincidenceDatetimes(date, cuts=True): 
    """ 
    Finds all coincidence events between stations within a one-second window.
    
    For each station data file in the corresponding date folder, this function loads the event
    timestamps and records the station and the index of the event.
    Instead of grouping events by exact timestamp, events are grouped if they occur within one second
    of the earliest event in the group.
    
    Each stored coincidence event is a dictionary with the following keys:
    - "numCoincidences": Number of events in the coincidence group.
    - "datetime": The representative event timestamp (the first event in the group).
    - "stations": Dictionary where each key is a station number, and its value is another dictionary.
                  Initially, this inner dictionary contains the key 'indices' with a list of indices 
                  corresponding to the events for that station.
    
    Args:
      date (str): The date folder to process, as read from the configuration.
    
    Returns:
      A tuple of two dictionaries: (coincidence_datetimes, coincidence_with_repeated_stations)
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

    # Each event is represented as a tuple: (timestamp, station_id, event_index)
    all_events = []

    # Load data for each station.
    for station_id in station_ids:
        file_list = sorted(glob.glob(station_data_folder + f'/{date}_Station{station_id}_Times*'))
        times = [np.load(f) for f in file_list]
        times = np.concatenate(times, axis=0)
        times = times.squeeze()
        times = np.array(times)
        # Filter out zero timestamps and pre-time events.
        zerotime_mask = times != 0
        times = times[zerotime_mask]
        pretime_mask = times >= datetime.datetime(2013, 1, 1).timestamp()
        times = times[pretime_mask]

        if cuts:
            cuts_file = os.path.join(cuts_data_folder, f'{date}_Station{station_id}_Cuts.npy')
            if os.path.exists(cuts_file):
                ic(f"Loading cuts file: {cuts_file}")
                cuts_data = np.load(cuts_file, allow_pickle=True)
                cuts_data = cuts_data[()]
            else:
                ic(f"Warning: Cuts file not found for station {station_id} on date {date}.")
                continue
            final_cuts = np.ones(len(times), dtype=bool)
            for cut in cuts_data.keys():
                ic(f"Applying cut: {cut}")
                final_cuts &= cuts_data[cut]
            times = times[final_cuts]

        for idx, event_time in enumerate(times):
            if event_time == 0:
                continue
            ts = event_time  
            all_events.append((ts, station_id, idx))

    all_events.sort(key=lambda x: x[0])

    coincidence_datetimes = {}
    coincidence_with_repeated_stations = {}
    valid_counter = 0
    duplicate_counter = 0

    n_events = len(all_events)
    i = 0
    one_second = 1
    while i < n_events:
        current_group = [all_events[i]]
        j = i + 1
        while j < n_events and (all_events[j][0] - all_events[i][0]) <= one_second:
            current_group.append(all_events[j])
            j += 1

        if len(current_group) > 1:
            # Build a dictionary that separates station information.
            stations_info = {}
            for ts, station_id, idx in current_group:
                if station_id not in stations_info:
                    stations_info[station_id] = {"indices": []}
                stations_info[station_id]["indices"].append(idx)
            
            # Skip groups where all events come from the same station.
            if len(stations_info) == 1:
                i = j
                continue

            # Determine which dictionary to add this event to.
            if any(len(info["indices"]) > 1 for info in stations_info.values()):
                target_dict = coincidence_with_repeated_stations
                idx_counter = duplicate_counter
                duplicate_counter += 1
            else:
                target_dict = coincidence_datetimes
                idx_counter = valid_counter
                valid_counter += 1

            target_dict[idx_counter] = {
                "numCoincidences": len(current_group),
                "datetime": all_events[i][0],
                "stations": stations_info
            }
            i = j
        else:
            i += 1

    return coincidence_datetimes, coincidence_with_repeated_stations


def analyze_coincidence_events(coincidence_datetimes, coincidence_with_repeated_stations):
        """
        Analyzes and logs the number of coincidence events for different coincidence numbers,
        and counts unique stations while taking into account that each 'stations' entry may have multiple keys.

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
            # Here the 'stations' dictionary may contain multiple keys (e.g., 'indices', 'metadata', etc.)
            # We count unique station IDs by iterating over its keys.
            unique_station_ids = list(event_data["stations"].keys())
            unique_stations_count = len(unique_station_ids)
            repeated_coincidence_counts[unique_stations_count] = repeated_coincidence_counts.get(unique_stations_count, 0) + 1

        for num, count in sorted(repeated_coincidence_counts.items()):
            ic(f"Number of events with {num} unique station coincidences: {count}")

def add_parameter_to_events(events_dict, parameter_name, date, cuts=True):
    """
    Loads the given parameter (e.g., 'SNR') for each station present in the coincidence events,
    but now loads and adds the parameter value even when it was processed before.
    After processing each station, the updated events_dict is saved to a temporary file.
    This function also takes into account that some stations may appear multiple times in the
    same coincidence, so the number of parameter values must match the number of indices per station.
    """
    station_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'nurFiles', date)
    cuts_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'cuts', date)
    threshold = datetime.datetime(2013, 1, 1).timestamp()

    # Collect unique station IDs from events.
    unique_stations = set()
    for event in events_dict.values():
        for st in event["stations"].keys():
            unique_stations.add(int(st))
    unique_stations = list(unique_stations)

    # Define temporary folder where updated events are stored.
    temp_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'processedNumpyData', date)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    for station in unique_stations:
        temp_save_file = os.path.join(temp_folder, f'temp_param_{parameter_name}_Station{station}.npy')
        if os.path.exists(temp_save_file):
            ic(f"Parameter {parameter_name} for station {station} already processed. Loading saved values.")
            processed_events = np.load(temp_save_file, allow_pickle=True).item()
            # Update the current events_dict for this station.
            for event_id, event in events_dict.items():
                st_key = None
                if station in event["stations"]:
                    ic(station, event["stations"])
                    st_key = station
                elif str(station) in event["stations"]:
                    ic(station, event["stations"])
                    st_key = str(station)
                ic(st_key, event_id, processed_events[event_id], processed_events[event_id]["stations"])
                if st_key is not None and parameter_name in processed_events[event_id]["stations"][st_key]:
                    event["stations"][st_key][parameter_name] = processed_events[event_id]["stations"][st_key][parameter_name]
            continue

        ic(f"Processing station {station} for parameter {parameter_name}")

        # Get all times files for this station.
        time_pattern = os.path.join(station_data_folder, f'{date}_Station{station}_Times*')
        times_files = sorted(glob.glob(time_pattern))
        if not times_files:
            continue

        # First pass: Get the number of valid entries per file (after applying the valid mask).
        file_counts = []
        total_valid = 0
        for tfile in times_files:
            with open(tfile, 'rb') as f:
                f_times = np.load(f)
            f_times = np.array(f_times).squeeze()
            valid_mask = (f_times != 0) & (f_times >= threshold)
            count = int(np.sum(valid_mask))
            file_counts.append(count)
            total_valid += count

        # Load and apply cuts at the station level (if available).
        final_cuts = None
        cuts_file = os.path.join(cuts_data_folder, f'{date}_Station{station}_Cuts.npy')
        if cuts and os.path.exists(cuts_file):
            with open(cuts_file, 'rb') as f:
                cuts_data = np.load(f, allow_pickle=True)[()]
            final_cuts = np.ones(total_valid, dtype=bool)
            for cut in cuts_data.keys():
                final_cuts &= cuts_data[cut]

        # Second pass: Build mapping of file info.
        file_info_list = []
        current_global = 0  # running count of processed (valid+cut) entries
        cumulative_valid = 0  # running count of valid entries (before applying cuts)
        for idx, tfile in enumerate(times_files):
            with open(tfile, 'rb') as f:
                f_times = np.load(f)
            f_times = np.array(f_times).squeeze()
            valid_mask = (f_times != 0) & (f_times >= threshold)
            valid_indices = np.nonzero(valid_mask)[0]
            count_in_file = len(valid_indices)

            # Apply cuts to this file's portion.
            if cuts and final_cuts is not None:
                file_cut_mask = final_cuts[cumulative_valid: cumulative_valid + count_in_file]
            else:
                file_cut_mask = np.ones(count_in_file, dtype=bool)
            local_surviving = np.nonzero(file_cut_mask)[0]
            processed_count = len(local_surviving)

            file_info_list.append({
                'file': tfile,
                'global_start': current_global,
                'global_end': current_global + processed_count,
                'local_surviving': local_surviving,  # maps processed order to local index in the file
            })
            current_global += processed_count
            cumulative_valid += count_in_file

        # Group, for this station, the events by which file their global index falls into.
        # Only add the parameter value for entries where it is missing.
        file_updates = {}
        for event in events_dict.values():
            st_key = None
            if station in event["stations"]:
                st_key = station
            elif str(station) in event["stations"]:
                st_key = str(station)
            else:
                continue
            indices_list = event["stations"][st_key].get("indices", [])
            # Initialize the parameter list if it is missing or if its length is insufficient.
            if (parameter_name not in event["stations"][st_key] or
                    len(event["stations"][st_key][parameter_name]) < len(indices_list)):
                event["stations"][st_key][parameter_name] = [None] * len(indices_list)
            for pos, global_idx in enumerate(indices_list):
                if event["stations"][st_key][parameter_name][pos] is not None:
                    # Parameter already added for this occurrence; skip.
                    continue
                for finfo in file_info_list:
                    if finfo['global_start'] <= global_idx < finfo['global_end']:
                        local_pos = global_idx - finfo['global_start']
                        local_index = int(finfo['local_surviving'][local_pos])
                        file_updates.setdefault(finfo['file'], []).append(
                            (event, st_key, pos, local_index)
                        )
                        break

        # For each times file with events needing parameter values,
        # load the corresponding parameter file (matched by replacing "Times" with parameter_name)
        for tfile, updates in file_updates.items():
            param_file = tfile.replace('Times', parameter_name)
            if not os.path.exists(param_file):
                ic(f"Parameter file {param_file} not found for station {station} and parameter {parameter_name}.")
                continue
            with open(param_file, 'rb') as pf:
                param_array = np.load(pf)
            param_array = np.array(param_array)
            if parameter_name != 'Traces':
                param_array = param_array.squeeze()
            for (event, st_key, pos, local_index) in updates:
                try:
                    value = param_array[local_index]
                except IndexError:
                    value = np.nan
                event["stations"][st_key][parameter_name][pos] = value

        # Save the updated events_dict for this station.
        np.save(temp_save_file, events_dict, allow_pickle=True)
        ic(f"Saved parameter {parameter_name} for station {station} to {temp_save_file}")

    gc.collect()


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


    # Add parameters to events.
    parameters_to_add = ['Traces', 'SNR', 'ChiRCR', 'Chi2016', 'ChiBad', 'Zen', 'Azi']
    for param in parameters_to_add:
        ic(f"Adding parameter: {param}")
        add_parameter_to_events(coincidence_datetimes, param, date, cuts=True)
        add_parameter_to_events(coincidence_with_repeated_stations, param, date, cuts=True)
    # Optional: ic first few coincidences for verification.
    for key in list(coincidence_datetimes.keys()):
        ic(key, coincidence_datetimes[key])
        if isinstance(coincidence_datetimes[key], dict):
            ic(coincidence_datetimes[key].keys())
    for key in list(coincidence_with_repeated_stations.keys()):
        ic(key, coincidence_with_repeated_stations[key])
        if isinstance(coincidence_with_repeated_stations[key], dict):
            ic(coincidence_with_repeated_stations[key].keys()) 

    quit()
    # Save the updated events with parameters.
    np.save(output_file, [coincidence_datetimes, coincidence_with_repeated_stations], allow_pickle=True)
    ic("Updated events with parameters and saved.")



    # Make plots of the coincidences
    import HRAStationDataAnalysis.loadHRAConvertedData as loadHRAConvertedData

    station_data = loadHRAConvertedData.loadHRAConvertedData(date, cuts=True, SNR='SNR', ChiRCR='ChiRCR', Chi2016='Chi2016', ChiBad='ChiBad', Zen='Zen', Azi='Azi', Trace='Trace')
    # Data is a dictionary with keys 'times', 'SNR', 'ChiRCR', 'Chi2016', and 'ChiRCR_bad'.
    # Each key contains a dictionary where keys are station IDs and values are the corresponding data arrays.

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    plot_folder = os.path.join('HRAStationDataAnalysis', 'plots', date, 'coincidence')
    os.makedirs(plot_folder, exist_ok=True)

    def plot_events(events, title_suffix, plot_folder):
        # Define Chi keys to plot against SNR.
        chi_keys = ['ChiRCR', 'Chi2016', 'ChiBad']
        # Create one figure with three side-by-side subplots.
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, chi in zip(axes, chi_keys):
            ax.set_xlabel('SNR')
            ax.set_ylabel(chi)
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
                ax.scatter(event_snr,event_chi[chi], color=colors[i])
                ax.plot(event_snr, event_chi[chi], color=colors[i])
        
        plt.tight_layout()
        ic(f"Saving plot for {title_suffix}")
        plt.savefig(os.path.join(plot_folder, f'coincidence_{title_suffix}.png'))
        plt.close(fig)

    # Plot scatter plots for coincidence events with unique stations.
    plot_events(coincidence_datetimes, "Unique Stations Coincidences", plot_folder)

    # Plot scatter plots for coincidence events (including repeated stations).
    plot_events(coincidence_with_repeated_stations, "Repeated Stations Coincidences", plot_folder)

    def plot_master_events(events, station_data, plot_folder):
        import matplotlib.pyplot as plt

        master_folder = os.path.join(plot_folder, "master")
        os.makedirs(master_folder, exist_ok=True)

        # Define fixed color mapping for each station.
        # Stations are [13, 14, 15, 17, 18, 30].
        color_map = {13: 'tab:blue',
                     14: 'tab:orange',
                     15: 'tab:green',
                     17: 'tab:red',
                     18: 'tab:purple',
                     30: 'tab:brown'}

        # List of marker styles to cycle through if a station appears more than once.
        marker_list = ['o', 's', 'D', '^', 'v', '>', '<', 'p']

        # Iterate over each event.
        for event_id, event in events.items():
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

            # Compute the average time from the event's 'Times' data.
            try:
                times = event["datetime"]  # 'Times' is a list/array with time stamps from all stations.
                ic(times)
                # avg_time = sum(times) / len(times)
                # event_time = datetime.datetime.fromtimestamp(avg_time)
                event_time = datetime.datetime.fromtimestamp(times)  # Use the first time as representative.
            except Exception:
                event_time = "Unknown Time"
            fig.suptitle(f"Master Event {event_id} - Average Time: {event_time}", fontsize=16)

            # Upper left: SNR vs Chi scatter
            ax_scatter = axs[0, 0]
            # Upper right: Polar plot
            ax_polar = plt.subplot(2, 2, 2, polar=True)
            # Bottom left: Time trace
            ax_trace = axs[1, 0]
            # Bottom right: Dummy plot
            ax_dummy = axs[1, 1]

            # To track repetition for each station in this event.
            occurrence_counter = {}
            # To build the super legend: one entry per station in the event.
            legend_entries = {}

            # Loop through each station in this event.
            for j, station in enumerate(event["stations"]):
                idx = event["indices"][j]
                # Setup repetition counter for marker selection.
                occurrence_counter.setdefault(station, 0)
                marker = marker_list[occurrence_counter[station] % len(marker_list)]
                occurrence_counter[station] += 1

                try:
                    snr_val      = station_data['SNR'][station][idx]
                    chi_rcr_val  = station_data['ChiRCR'][station][idx]
                    chi_2016_val = station_data['Chi2016'][station][idx]
                    zen_val      = station_data['Zen'][station][idx]
                    azi_val      = station_data['Azi'][station][idx]
                    trace_val    = station_data['Trace'][station][idx]
                except (KeyError, IndexError):
                    continue

                # Get the color for this station.
                color = color_map.get(station, 'black')

                # Upper left: Plot Chi2016 and ChiRCR, with arrow from Chi2016 to ChiRCR.
                ax_scatter.scatter(snr_val, chi_2016_val, color=color, marker=marker)
                ax_scatter.scatter(snr_val, chi_rcr_val, color=color, marker=marker)
                ax_scatter.annotate("",
                                    xy=(snr_val, chi_rcr_val),
                                    xytext=(snr_val, chi_2016_val),
                                    arrowprops=dict(arrowstyle="->", color=color))

                # Upper right: Polar plot using azi as the angle and zen as the radial coordinate.
                ax_polar.scatter(azi_val, zen_val, color=color, marker=marker)

                # Bottom left: Plot the time trace; x-axis: 256 point array spanning 0 to 128.
                x_vals = np.linspace(0, 128, 256)
                ax_trace.plot(x_vals, trace_val, color=color, marker=marker)

                # Build the legend entry if not already added.
                if station not in legend_entries:
                    legend_entries[station] = Line2D([0], [0], marker='o', color=color, linestyle='None',
                                                     markersize=8, label=f"Station {station}")

            ax_scatter.set_xlabel("SNR")
            ax_scatter.set_ylabel("Chi")
            ax_scatter.set_title("SNR vs Chi (Arrow: Chi2016 -> ChiRCR)")

            ax_polar.set_title("Polar: Zenith vs Azimuth")

            ax_trace.set_xlabel("Time")
            ax_trace.set_ylabel("Trace")
            ax_trace.set_title("Time Trace")

            ax_dummy.set_title("Dummy Plot")
            ax_dummy.text(0.5, 0.5, "Dummy Plot", horizontalalignment='center', verticalalignment='center')

            # Create a single super legend (only for stations present in this event)
            if legend_entries:
                handles = list(legend_entries.values())
                ax_scatter.legend(handles=handles, loc='best', title="Stations")

            plt.tight_layout()
            master_filename = os.path.join(master_folder, f'master_event_{event_id}.png')
            plt.savefig(master_filename)
            plt.close(fig)

    # Plot master plots for each individual coincidence event (unique stations)
    plot_master_events(coincidence_datetimes, station_data, plot_folder)
