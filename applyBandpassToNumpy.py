import os
import numpy as np
from NuRadioReco.utilities import fft, units
from scipy import signal

def apply_butterworth(spectrum, frequencies, passband, order=8):
    """
    Calculates the response from a Butterworth filter and applies it to the
    input spectrum

    Parameters
    ----------
    spectrum: array of complex
        Fourier spectrum to be filtere
    frequencies: array of floats
        Frequencies of the input spectrum
    passband: (float, float) tuple
        Tuple indicating the cutoff frequencies
    order: integer
        Filter order

    Returns
    -------
    filtered_spectrum: array of complex
        The filtered spectrum
    """

    f = np.zeros_like(frequencies, dtype=complex)
    mask = frequencies > 0
    b, a = signal.butter(order, passband, "bandpass", analog=True)
    w, h = signal.freqs(b, a, frequencies[mask])
    f[mask] = h

    filtered_spectrum = f * spectrum

    return filtered_spectrum

def butterworth_filter_trace(trace, sampling_frequency, passband, order=8):
    """
    Filters a trace using a Butterworth filter.

    Parameters
    ----------
    trace: array of floats
        Trace to be filtered
    sampling_frequency: float
        Sampling frequency
    passband: (float, float) tuple
        Tuple indicating the cutoff frequencies
    order: integer
        Filter order

    Returns
    -------

    filtered_trace: array of floats
        The filtered trace
    """

    n_samples = len(trace)

    spectrum = fft.time2freq(trace, sampling_frequency)
    frequencies = np.fft.rfftfreq(n_samples, d = 1 / sampling_frequency)
    frequencies = fft.freqs(n_samples, sampling_frequency)

    filtered_spectrum = apply_butterworth(spectrum, frequencies, passband, order)
    filtered_trace = fft.freq2time(filtered_spectrum, sampling_frequency)

    return filtered_trace

# Input and output directories
input_dir = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/5.20.25/"
output_dir = os.path.join(input_dir, "testFiltering/")

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the sampling rate
sampling_rate_hz = 2 * units.MHz

# Get a list of all files in the input directory that contain "traces"
file_list = [f for f in os.listdir(input_dir) if "Traces" in f and f.endswith('.npy')]

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
            # Define the butterworth filter parameters
            passband = [50 * units.MHz, 1000 * units.MHz] 
            order = 2
            
            # Apply the butterworth filter
            filtered_trace = butterworth_filter_trace(trace_ch_data_arr, sampling_rate_hz, passband, order)
            
            filtered_event.append(filtered_trace)
            
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
