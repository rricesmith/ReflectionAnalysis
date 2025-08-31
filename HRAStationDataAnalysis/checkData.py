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
            template_trace = template_set[template_trace]
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
        all_traces = []
        for f in files:
            all_traces.append(np.load(f))
        
        if not all_traces:
            ic(f"No traces found for {station_id}")
            continue

        all_traces = np.concatenate(all_traces, axis=0)
        
        if len(all_traces) == 0:
            ic(f"No events in concatenated traces for {station_id}")
            continue

        num_events = len(all_traces)
        num_samples = min(num_events, 10)
        
        if num_samples == 0:
            ic(f"No events to sample for {station_id}")
            continue
            
        random_indices = np.random.choice(num_events, num_samples, replace=False)
        
        station_dir = os.path.join(plots_base_dir, station_id)
        os.makedirs(station_dir, exist_ok=True)

        for i, event_idx in enumerate(random_indices):
            event_traces = all_traces[event_idx]
            ic(f"  Event {i+1}/{num_samples} (index {event_idx})")

            # Plot event traces
            for ch_idx, trace in enumerate(event_traces):
                title = f"{station_id} Event {event_idx} Channel {ch_idx}"
                save_path = os.path.join(station_dir, f"event_{event_idx}_ch_{ch_idx}.png")
                plot_trace_and_fft(time, trace, title, save_path)

            # Calculate and print Chi
            print(f"  Chi-squared for Event {event_idx}:")
            for name, template_set in templates.items():
                chi = getMaxAllChi(event_traces, 2*units.GHz, template_set, 2*units.GHz)
                print(f"    vs {name}: {chi:.4f}")
        ic(f"Finished processing {station_id}")

if __name__ == "__main__":
    main()
