import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from icecream import ic
import sys

# Add paths to helper functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from DeepLearning import D00_helperFunctions
from HRAStationDataAnalysis.calculateChi import getMaxAllChi
from NuRadioReco.utilities import units

def plot_trace_and_fft(time_series, trace, title, save_path):
    """Plots time trace and its FFT."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time trace
    axs[0].plot(time_series, trace)
    axs[0].set_xlabel('Time (ns)')
    axs[0].set_ylabel('Amplitude (ADC)')
    axs[0].set_title(f'Time Trace - {title}')
    axs[0].grid(True)
    
    # FFT
    sampling_rate = 1 / (time_series[1] - time_series[0])
    n = len(trace)
    freq = np.fft.rfftfreq(n, 1/sampling_rate)
    fft_val = np.abs(np.fft.rfft(trace))
    
    axs[1].plot(freq / units.MHz, fft_val)
    axs[1].set_xlabel('Frequency (MHz)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title(f'Frequency Spectrum - {title}')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def main():
    # Load templates
    templates = {
        '2016': D00_helperFunctions.loadMultipleTemplates(100, date='2016'),
        'RCR_100': D00_helperFunctions.loadMultipleTemplates(100),
        'RCR_bad_100': D00_helperFunctions.loadMultipleTemplates(100, bad=True),
        'RCR_200': D00_helperFunctions.loadMultipleTemplates(200),
        'RCR_bad_200': D00_helperFunctions.loadMultipleTemplates(200, bad=True)
    }
    ic("Templates loaded.")

    # Plot templates
    plots_base_dir = "HRAStationDataAnalysis/plots/checkData"
    time = np.arange(0, 256 * 0.5, 0.5) # Assuming 256 samples at 2GS/s (0.5ns sampling)

    for name, template_set in templates.items():
        template_dir = os.path.join(plots_base_dir, name)
        os.makedirs(template_dir, exist_ok=True)
        for i, template_name in enumerate(template_set):
            template_trace = template_set[template_name]
            title = f"Template {name} #{i+1}, {template_name}"
            save_path = os.path.join(template_dir, f"template_{i+1}.png")
            ic(time, template_trace, title, save_path)
            plot_trace_and_fft(time, template_trace, title, save_path)
    ic("Templates plotted.")

    # Process station data
    station_data_dir = "HRAStationDataAnalysis/StationData/nurFiles/*/"
    station_files = glob.glob(os.path.join(station_data_dir, "*_Traces_*.npy"))
    
    unique_stations = {}
    for f in station_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        station_id = parts[1]
        if station_id not in unique_stations:
            unique_stations[station_id] = []
        unique_stations[station_id].append(f)

    for station_id, files in unique_stations.items():
        ic(f"Processing {station_id}")
        
        # Create directory for this station's plots
        station_dir = os.path.join(plots_base_dir, station_id)
        os.makedirs(station_dir, exist_ok=True)
        
        # Select up to 10 random files for this station (or all files if less than 10)
        num_files = min(len(files), 10)
        if num_files == 0:
            ic(f"No files found for {station_id}")
            continue
            
        selected_files = np.random.choice(files, num_files, replace=False)
        events_processed = 0
        
        # Process one file at a time to conserve memory
        for file_idx, file_path in enumerate(selected_files):
            ic(f"Processing file {file_idx+1}/{num_files}: {os.path.basename(file_path)}")
            
            # Load the file
            traces = np.load(file_path)
            
            if len(traces) == 0:
                ic(f"No events in file {file_path}")
                continue
                
            # Select one random event from this file
            event_idx = np.random.randint(0, len(traces))
            event_traces = traces[event_idx]
            
            # Get a unique identifier for this event (combining file index and event index)
            event_id = f"{file_idx}_{event_idx}"
            ic(f"  Selected event {event_idx} from file {file_idx+1}")
            
            # Plot event traces
            for ch_idx, trace in enumerate(event_traces):
                title = f"{station_id} File {file_idx+1} Event {event_idx} Channel {ch_idx}"
                save_path = os.path.join(station_dir, f"event_{event_id}_ch_{ch_idx}.png")
                plot_trace_and_fft(time, trace, title, save_path)
            
            # Calculate and print Chi
            print(f"  Chi-squared for {station_id} File {file_idx+1} Event {event_idx}:")
            for name, template_set in templates.items():
                chi = getMaxAllChi(event_traces, 2*units.GHz, template_set, 2*units.GHz)
                print(f"    vs {name}: {chi:.4f}")
                
            events_processed += 1
            
            # Clear memory
            del traces
            
        ic(f"Finished processing {station_id} - {events_processed} events processed")

if __name__ == "__main__":
    main()
