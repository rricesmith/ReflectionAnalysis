import os
import numpy as np
from NuRadioReco.utilities import fft, units
from scipy import signal

# Input and output directories
input_dir = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/5.20.25/"
output_dir = os.path.join(input_dir, "testFiltering/")

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the sampling rate
sampling_rate_hz = 2 * units.MHz

# Get a list of all files in the input directory that contain "traces"
file_list = [f for f in os.listdir(input_dir) if "traces" in f and f.endswith('.npy')]

# Iterate over each file
for filename in file_list:
    print(f"Processing {filename}...")
    file_path = os.path.join(input_dir, filename)
    
    # Load the numpy array from the file
    data = np.load(file_path)
    
    # Create an empty list to store the filtered traces
    filtered_traces = []
    
    # Iterate over each event in the data
    for event_data in data:
        filtered_event = []
        # Iterate over each channel in the event
        for trace_ch_data_arr in event_data:
            # Get the frequency spectrum
            freqs = np.fft.rfftfreq(len(trace_ch_data_arr), d=1/sampling_rate_hz)
            spectrum = fft.time2freq(trace_ch_data_arr, sampling_rate_hz)
            
            # Define the butterworth filter parameters
            passband = [50 * units.MHz, 1000 * units.MHz] 
            order = 2
            
            # Create the butterworth filter
            b, a = signal.butter(order, passband, btype='band', fs=sampling_rate_hz)
            
            # Apply the filter to the frequency spectrum
            w, h = signal.freqs(b, a, worN=freqs)
            filtered_spectrum = spectrum * h
            
            # Convert the filtered spectrum back to the time domain
            filtered_trace = fft.freq2time(filtered_spectrum, sampling_rate_hz)
            filtered_event.append(filtered_trace.real)
            
        filtered_traces.append(filtered_event)
        
    # Convert the list of filtered traces to a numpy array
    filtered_data = np.array(filtered_traces)
    
    # Create the new filename
    new_filename = filename.replace(".npy", "_filtered.npy")
    output_path = os.path.join(output_dir, new_filename)
    
    # Save the filtered data to the new file
    np.save(output_path, filtered_data)
    print(f"Saved filtered data to {output_path}")

print("All files processed successfully!")
