import numpy as np
import os
import glob
from icecream import ic

def getMaxAllChi(traces, templates):
    """
    Calculates the maximum normalized cross-correlation for a trace against a set of templates.

    Args:
        traces (list of np.ndarray): A list containing the trace(s) to be analyzed.
                                     For this script, it will be a list with a single trace.
        templates (list of np.ndarray): A list of traces to be used as templates.

    Returns:
        float: The maximum cross-correlation value found.
    """
    max_chi_overall = -np.inf

    for trace in traces:
        # Normalize the primary trace
        trace_norm = (trace - np.mean(trace)) / np.std(trace)
        
        max_chi_for_trace = -np.inf

        for template in templates:
            # Normalize the template trace
            template_norm = (template - np.mean(template)) / np.std(template)

            # Ensure templates are not flat lines (std dev is not zero)
            if np.std(template_norm) == 0 or np.std(trace_norm) == 0:
                continue
            
            # Calculate cross-correlation
            correlation = np.correlate(trace_norm, template_norm, mode='full')
            
            # The maximum of the absolute correlation is the chi value
            current_chi = np.max(np.abs(correlation)) / len(trace)

            if current_chi > max_chi_for_trace:
                max_chi_for_trace = current_chi
        
        if max_chi_for_trace > max_chi_overall:
            max_chi_overall = max_chi_for_trace

    return max_chi_overall

def process_group(file_paths, group_name, output_file):
    """
    Loads a group of traces, performs leave-one-out Chi analysis,
    and writes the results to a file.

    Args:
        file_paths (list): List of file paths for the numpy arrays in the group.
        group_name (str): The name of the group (e.g., "100s").
        output_file (file object): The file to write results to.
    """
    ic(f"Processing group: {group_name}")
    output_file.write(f"--- Results for Group: {group_name} ---\n")

    # Load all traces from the files
    all_traces = [np.load(f) for f in file_paths]
    num_traces = len(all_traces)

    if num_traces < 2:
        ic(f"Skipping group {group_name}: Not enough traces to perform leave-one-out analysis.")
        output_file.write("Not enough traces to perform analysis (< 2).\n\n")
        return

    chi_values = []
    for i in range(num_traces):
        # The current trace to be tested
        current_trace = all_traces[i]
        
        # The templates are all other traces in the group
        templates = all_traces[:i] + all_traces[i+1:]
        
        # The user specified updating the single trace to [[256,]] to work with the method
        # We will pass it as a list containing one trace.
        chi_result = getMaxAllChi([current_trace], templates)
        
        chi_values.append(chi_result)
        
        original_filename = os.path.basename(file_paths[i])
        output_file.write(f"File: {original_filename}, Chi: {chi_result:.6f}\n")
        
        if (i + 1) % 50 == 0:
            ic(f"Processed {i+1}/{num_traces} traces for group {group_name}")

    # Calculate statistics
    mean_chi = np.mean(chi_values)
    std_chi = np.std(chi_values)
    
    ic(f"Group {group_name} Mean Chi: {mean_chi}")
    ic(f"Group {group_name} Std Dev Chi: {std_chi}")

    # Write statistics to file
    output_file.write("\n--- Statistics for Group: {} ---\n".format(group_name))
    output_file.write(f"Total Traces: {num_traces}\n")
    output_file.write(f"Mean Chi: {mean_chi:.6f}\n")
    output_file.write(f"Standard Deviation of Chi: {std_chi:.6f}\n")
    output_file.write("\n" + "="*40 + "\n\n")


if __name__ == "__main__":
    # --- USER SETTINGS ---
    # !!! IMPORTANT: Update this path to the folder containing your .npy files !!!
    data_folder = 'DeepLearning/templates/RCR/3.29.25/' # Assumes files are in the same directory as the script
    # --- END USER SETTINGS ---

    ic.enable()
    ic(f"Searching for .npy files in: {os.path.abspath(data_folder)}")

    # Find all .npy files in the specified folder
    all_files = glob.glob(os.path.join(data_folder, '*.npy'))

    # Separate files into "100s" and "200s" groups
    files_100s = [f for f in all_files if '100s' in os.path.basename(f)]
    files_200s = [f for f in all_files if '200s' in os.path.basename(f)]

    ic(f"Found {len(files_100s)} files for '100s' group.")
    ic(f"Found {len(files_200s)} files for '200s' group.")

    output_filename = "chi_results.txt"

    with open(output_filename, 'w') as f:
        # Process the "100s" group
        if files_100s:
            process_group(files_100s, "100s", f)
        else:
            ic("No files found for '100s' group.")

        # Process the "200s" group
        if files_200s:
            process_group(files_200s, "200s", f)
        else:
            ic("No files found for '200s' group.")

    ic(f"Analysis complete. Results have been saved to '{output_filename}'.")