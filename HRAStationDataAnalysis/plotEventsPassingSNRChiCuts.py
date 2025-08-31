import numpy as np
import matplotlib.pyplot as plt
import os
import configparser
from icecream import ic
from NuRadioReco.utilities import fft as nu_fft
from NuRadioReco.utilities import units
import glob
from HRAStationDataAnalysis.C_utils import getTimeEventMasks
import matplotlib.dates as mdates
from datetime import datetime

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

def plot_time_series_events(event_data, station_id, output_dir, event_type):
    """
    Creates a time series plot showing events over time.
    """
    if len(event_data) == 0:
        ic(f"No {event_type} events to plot for station {station_id}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert timestamps to datetime objects
    timestamps = [datetime.fromtimestamp(ts) for ts in event_data['times']]
    event_ids = event_data['event_ids']
    
    # Create scatter plot
    ax.scatter(timestamps, event_ids, alpha=0.7, s=50)
    ax.set_xlabel('Time')
    ax.set_ylabel('Event ID')
    ax.set_title(f'Station {station_id} - {event_type} Events Over Time')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, f'Station{station_id}_{event_type}_TimeSeries.png')
    plt.savefig(plot_filename)
    ic(f"Saved time series plot: {plot_filename}")
    plt.close(fig)

def plot_polar_events(event_data, station_id, output_dir, event_type):
    """
    Creates a polar plot showing azimuth vs zenith for events.
    """
    if len(event_data) == 0:
        ic(f"No {event_type} events to plot for station {station_id}")
        return
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
    
    azimuths = event_data['azimuths']
    zeniths = event_data['zeniths']
    
    # Convert zenith to polar angle (0 at zenith, pi at horizon)
    polar_angles = zeniths
    
    # Create scatter plot
    scatter = ax.scatter(azimuths, polar_angles, alpha=0.7, s=50, c=event_data['snrs'], cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('SNR')
    
    ax.set_title(f'Station {station_id} - {event_type} Events (Azimuth vs Zenith)')
    ax.set_theta_zero_location('N')  # North at top
    ax.set_theta_direction(-1)      # Clockwise from North
    
    # Set radial limits (zenith to horizon)
    ax.set_rlim(0, np.pi/2)
    ax.set_rticks([0, np.pi/6, np.pi/3, np.pi/2])
    ax.set_yticklabels(['90°', '60°', '30°', '0°'])
    
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, f'Station{station_id}_{event_type}_Polar.png')
    plt.savefig(plot_filename)
    ic(f"Saved polar plot: {plot_filename}")
    plt.close(fig)

def plot_combined_events(rcr_data, backlobe_data, station_id, output_dir):
    """
    Creates combined plots showing both RCR and backlobe events together.
    """
    # Combined time series
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if len(rcr_data) > 0:
        rcr_timestamps = [datetime.fromtimestamp(ts) for ts in rcr_data['times']]
        ax.scatter(rcr_timestamps, rcr_data['event_ids'], alpha=0.7, s=50, label='RCR Events', color='blue')
    
    if len(backlobe_data) > 0:
        backlobe_timestamps = [datetime.fromtimestamp(ts) for ts in backlobe_data['times']]
        ax.scatter(backlobe_timestamps, backlobe_data['event_ids'], alpha=0.7, s=50, label='Backlobe Events', color='red')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Event ID')
    ax.set_title(f'Station {station_id} - All Events Over Time')
    ax.legend()
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, f'Station{station_id}_Combined_TimeSeries.png')
    plt.savefig(plot_filename)
    ic(f"Saved combined time series plot: {plot_filename}")
    plt.close(fig)
    
    # Combined polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 8))
    
    if len(rcr_data) > 0:
        rcr_azimuths = rcr_data['azimuths']
        rcr_zeniths = rcr_data['zeniths']
        ax.scatter(rcr_azimuths, rcr_zeniths, alpha=0.7, s=50, label='RCR Events', color='blue')
    
    if len(backlobe_data) > 0:
        backlobe_azimuths = backlobe_data['azimuths']
        backlobe_zeniths = backlobe_data['zeniths']
        ax.scatter(backlobe_azimuths, backlobe_zeniths, alpha=0.7, s=50, label='Backlobe Events', color='red')
    
    ax.set_title(f'Station {station_id} - All Events (Azimuth vs Zenith)')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim(0, np.pi/2)
    ax.set_rticks([0, np.pi/6, np.pi/3, np.pi/2])
    ax.set_yticklabels(['90°', '60°', '30°', '0°'])
    ax.legend()
    
    plt.tight_layout()
    
    plot_filename = os.path.join(output_dir, f'Station{station_id}_Combined_Polar.png')
    plt.savefig(plot_filename)
    ic(f"Saved combined polar plot: {plot_filename}")
    plt.close(fig)

def process_events_for_station(station_id, plot_folder, date, station_data_folder, event_type='rcr'):
    """
    Process events for a specific station and event type (RCR or backlobe).
    """
    ic(f"--- Processing {event_type.upper()} events for Station {station_id} ---")
    
    # Determine which cut type to load
    if event_type == 'rcr':
        cut_key = 'all_cuts'
    else:  # backlobe
        cut_key = 'backlobe_all_cuts'
    
    passing_events_file = os.path.join(plot_folder, f'PassingEvents_Station{station_id}_{date}.npz')
    if not os.path.exists(passing_events_file):
        ic(f"Warning: Passing events file not found for station {station_id}, skipping.")
        return None

    try:
        passing_events_data = np.load(passing_events_file, allow_pickle=True)
        if cut_key not in passing_events_data:
            ic(f"Warning: '{cut_key}' data not in {passing_events_file}. Skipping station.")
            return None
        
        passing_events = passing_events_data[cut_key]
        if passing_events.size == 0:
            ic(f"No {event_type} events passed all cuts for this station.")
            return None
        
        ic(f"Found {len(passing_events)} {event_type} events passing all cuts.")

    except Exception as e:
        ic(f"Error loading passing events file for station {station_id}: {e}")
        return None

    # Load the full dataset for this station
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
        ic("Missing one or more raw data files. Skipping station.")
        return None

    initial_mask, unique_indices = getTimeEventMasks(raw_times, raw_event_ids)

    # Apply the mask and select unique events
    event_ids_clean = raw_event_ids[initial_mask][unique_indices]
    traces_clean = raw_traces[initial_mask][unique_indices]
    times_clean = raw_times[initial_mask][unique_indices]
    snr_clean = raw_snr[initial_mask][unique_indices] if raw_snr.size > 0 else np.array([])
    chi2016_clean = raw_chi2016[initial_mask][unique_indices] if raw_chi2016.size > 0 else np.array([])
    chircr_clean = raw_chircr[initial_mask][unique_indices] if raw_chircr.size > 0 else np.array([])
    chibad_clean = raw_chibad[initial_mask][unique_indices] if raw_chibad.size > 0 else np.array([])
    azi_clean = raw_azi[initial_mask][unique_indices] if raw_azi.size > 0 else np.array([])
    zen_clean = raw_zen[initial_mask][unique_indices] if raw_zen.size > 0 else np.array([])

    # Prepare data for plotting
    event_data = {
        'times': [],
        'event_ids': [],
        'azimuths': [],
        'zeniths': [],
        'snrs': []
    }

    for passing_event in passing_events:
        event_id = passing_event['event_id']
        unique_idx = passing_event['unique_index']

        try:
            # Verify we have the correct event
            if event_ids_clean[unique_idx] != event_id:
                ic(f"Mismatch! Event ID {event_id} from file does not match event ID {event_ids_clean[unique_idx]} at index {unique_idx}. Searching...")
                possible_indices = np.where(event_ids_clean == event_id)[0]
                if not possible_indices.any():
                    ic(f"Could not find event {event_id} in cleaned data. Skipping.")
                    continue
                unique_idx = possible_indices[0]
                ic(f"Found event {event_id} at new index {unique_idx} instead.")
            
            # Store event data for summary plots
            event_data['times'].append(times_clean[unique_idx])
            event_data['event_ids'].append(event_id)
            event_data['azimuths'].append(azi_clean[unique_idx] if azi_clean.size > 0 else 0)
            event_data['zeniths'].append(zen_clean[unique_idx] if zen_clean.size > 0 else 0)
            event_data['snrs'].append(snr_clean[unique_idx] if snr_clean.size > 0 else 0)
            
            # Plot individual event traces
            trace = traces_clean[unique_idx]
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

            plot_event_traces_and_ffts(event_id, trace, times_for_event, station_id, output_dir, event_info)

        except IndexError:
            ic(f"Index error for event {event_id} with unique_index {unique_idx}. Skipping.")
        except Exception as e:
            ic(f"An error occurred while plotting event {event_id}: {e}")

    return event_data

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
    
    # Create separate output directories for RCR and backlobe events
    rcr_output_dir = os.path.join(plot_folder, 'rcr_events_passing_cuts')
    backlobe_output_dir = os.path.join(plot_folder, 'backlobe_events_passing_cuts')
    combined_output_dir = os.path.join(plot_folder, 'combined_events')
    
    os.makedirs(rcr_output_dir, exist_ok=True)
    os.makedirs(backlobe_output_dir, exist_ok=True)
    os.makedirs(combined_output_dir, exist_ok=True)

    for station_id in station_ids_to_process:
        ic(f"--- Processing Station {station_id} ---")
        
        # Process RCR events
        rcr_data = process_events_for_station(station_id, plot_folder, date, station_data_folder, 'rcr')
        
        # Process backlobe events
        backlobe_data = process_events_for_station(station_id, plot_folder, date, station_data_folder, 'backlobe')
        
        # Create regular plots for RCR events
        if rcr_data and len(rcr_data['times']) > 0:
            plot_time_series_events(rcr_data, station_id, rcr_output_dir, 'RCR')
            plot_polar_events(rcr_data, station_id, rcr_output_dir, 'RCR')
            plot_waveform_examples(rcr_data, station_id, rcr_output_dir, 'RCR', station_data_folder, date)
            
            # Create seasonal plots for RCR events
            plot_time_series_events_seasonal(rcr_data, station_id, rcr_output_dir, 'RCR')
            plot_polar_events_seasonal(rcr_data, station_id, rcr_output_dir, 'RCR')
        
        # Create regular plots for backlobe events
        if backlobe_data and len(backlobe_data['times']) > 0:
            plot_time_series_events(backlobe_data, station_id, backlobe_output_dir, 'Backlobe')
            plot_polar_events(backlobe_data, station_id, backlobe_output_dir, 'Backlobe')
            plot_waveform_examples(backlobe_data, station_id, backlobe_output_dir, 'Backlobe', station_data_folder, date)
            
            # Create seasonal plots for backlobe events
            plot_time_series_events_seasonal(backlobe_data, station_id, backlobe_output_dir, 'Backlobe')
            plot_polar_events_seasonal(backlobe_data, station_id, backlobe_output_dir, 'Backlobe')
        
        # Create combined plots
        if (rcr_data and len(rcr_data['times']) > 0) or (backlobe_data and len(backlobe_data['times']) > 0):
            plot_combined_events(rcr_data or {'times': [], 'event_ids': [], 'azimuths': [], 'zeniths': [], 'snrs': []}, 
                               backlobe_data or {'times': [], 'event_ids': [], 'azimuths': [], 'zeniths': [], 'snrs': []}, 
                               station_id, combined_output_dir)
            
            # Create seasonal combined plots
            plot_combined_events_seasonal(rcr_data, backlobe_data, station_id, combined_output_dir)
            
            # Create seasonal summary
            plot_seasonal_summary(rcr_data, backlobe_data, station_id, combined_output_dir)

if __name__ == '__main__':
    stations_to_plot = [13, 14, 15, 17, 18, 19, 30]
    # stations_to_plot = [13, 15, 17, 18]
    main(stations_to_plot)
