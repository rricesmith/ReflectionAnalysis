import os
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import fft, units
from icecream import ic

def plot_trace_and_spectrum(trace, output_path):
    """
    Generates and saves a 1x2 plot containing a time-domain trace
    and its frequency spectrum.

    Args:
        trace (np.ndarray): The array containing the trace data.
        output_path (str): The full path where the output image will be saved.
    """
    # --- 1. Create a 1x2 plot for the trace and its FFT ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # --- 2. Plot the Trace (Time Domain) ---
    ax1.plot(trace)
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 3. Calculate and Plot the FFT (Frequency Domain) ---
    # Ensure there's data to process
    if trace.size > 1:
        # Calculate the real FFT and the corresponding frequency axis
        sampling_rate_hz = 2e9 
        freqs = np.fft.rfftfreq(len(trace), d=1/sampling_rate_hz) / 1e6 
        spectrum = np.abs(fft.time2freq(trace, sampling_rate_hz))

        ic(freqs.shape, spectrum.shape)
        ic(freqs, spectrum)

        # Plot the spectrum, skipping the DC component (index 0) for better scaling
        ax2.plot(freqs[1:], spectrum[1:])
        ax2.set_xlabel("Normalized Frequency [cycles/sample]")
        ax2.set_ylabel("Spectral Amplitude")
        ax2.grid(True, linestyle='--', alpha=0.6)

    # --- 4. Finalize and Save the Plot ---
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig) # Close the figure to free up memory


def process_folders(input_dirs, base_output_dir):
    """
    Processes all .npy files in a list of input directories and saves plots.

    Args:
        input_dirs (list): A list of paths to directories containing .npy files.
        base_output_dir (str): The path to the main folder where output
                               subdirectories and plots will be saved.
    """

    for input_dir, subfolder_out in input_dirs:
        output_dir = os.path.join(base_output_dir, subfolder_out)
        # Ensure the input directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create a corresponding subfolder in the output directory

        print(f"\nProcessing files in: {input_dir}")
        print(f"Saving plots to:   {output_dir}")

        # Find and process every .npy file in the directory
        found_files = 0
        for filename in os.listdir(input_dir):
            if filename.endswith('.npy'):
                found_files += 1
                npy_path = os.path.join(input_dir, filename)
                
                # Load the trace data from the .npy file
                try:
                    trace_data = np.load(npy_path)
                except Exception as e:
                    print(f"  - Could not load {filename}. Error: {e}")
                    continue

                # Define the output filename and full path
                plot_filename_base = os.path.splitext(filename)[0]
                if len(trace_data) == 4:
                    # All 4 traces, so run 4 times
                    for i in range(4):
                        plot_output_path = os.path.join(output_dir, f"{plot_filename_base}_trace{i}.png")
                        ic(trace_data[i].shape)
                        plot_trace_and_spectrum(trace_data[i], plot_output_path)
                else:
                    plot_output_path = os.path.join(output_dir, f"{plot_filename_base}.png")

                    # Generate and save the plot
                    ic(trace_data, trace_data.shape)
                    plot_trace_and_spectrum(trace_data, plot_output_path)
        
        if found_files == 0:
            print("  - No .npy files found in this directory.")

if __name__ == '__main__':
    # --- Configuration ---
    # Define the main folder where all plots will be saved.
    BASE_PLOT_FOLDER = "trace_plots"

    # List the folders containing the .npy trace files you want to process.
    INPUT_FOLDERS = [["StationDataAnalysis/templates/confirmed2016Templates/", "2016Templates"], ["DeepLearning/templates/RCR/3.29.25/", "RCRs"]]
    

    # --- Run the processing function ---
    process_folders(input_dirs=INPUT_FOLDERS, base_output_dir=BASE_PLOT_FOLDER)

    print("\nProcessing complete.")