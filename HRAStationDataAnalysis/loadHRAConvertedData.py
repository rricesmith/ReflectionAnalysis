import os
import numpy as np
import datetime
from icecream import ic
import glob
import pickle
import gc

def loadHRAConvertedData(date, cuts=True, **data_kwargs):
    """
    Loads HRA converted data (times and specified parameters) for a given date and applies cuts.
    If the processed data is already saved, it is loaded and returned.

    Args:
        date (str): The date folder to process.
        cuts (bool, optional): Whether to apply cuts. Defaults to True.
        **data_kwargs: Keyword arguments where keys are the names you want to use
                       for the output dictionaries (e.g., 'snr', 'chi2016') and
                       values are the corresponding file name prefixes (e.g., 'SNR', 'Chi2016').
                       Example: loadHRAConvertedData('2023-01-01', snr='SNR', chir='ChiRCR').

    Returns:
        tuple: A tuple containing:
            - 'Times': A dictionary where keys are station IDs and values are the
                     NumPy arrays of event times for that station (after applying cuts).
            - A dictionary for each parameter specified in data_kwargs. The keys of these
              dictionaries are station IDs, and the values are the NumPy arrays of the
              corresponding data (after applying cuts and ensuring alignment with times).
              Returns None for a parameter if no corresponding files are found for any station.
    """
    # Define the path where processed data is stored
    processed_folder = os.path.join('HRAStationDataAnalysis', 'ProcessedData')
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    processed_file = os.path.join(processed_folder, f"{date}_converted_data_cuts_{cuts}.pkl")

    # If already processed, load and return the data.
    if os.path.exists(processed_file):
        ic(f"Loading processed data from {processed_file}")
        with open(processed_file, 'rb') as f:
            return pickle.load(f)

    station_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'nurFiles', date)
    cuts_data_folder = os.path.join('HRAStationDataAnalysis', 'StationData', 'cuts', date)
    station_ids = [13, 14, 15, 17, 18, 19, 30]
    all_station_data = {'Times': {}}
    for param_name, param in data_kwargs.items():
        all_station_data[param] = {}

    for station_id in station_ids:
        # Load times and apply initial filters
        time_files = sorted(glob.glob(os.path.join(station_data_folder, f'{date}_Station{station_id}_Times*')))
        if not time_files:
            ic(f"Warning: Time files not found for station {station_id} on date {date}.")
            continue
        times = [np.load(f) for f in time_files]
        times = np.concatenate(times, axis=0)
        times = times.squeeze()
        times = np.array(times)
        zerotime_mask = times != 0
        times = times[zerotime_mask]
        pretime_mask = times >= datetime.datetime(2013, 1, 1).timestamp()
        times = times[pretime_mask]
        cut_mask = np.ones(len(times), dtype=bool)  # Initialize cut mask

        if cuts:
            cuts_file = os.path.join(cuts_data_folder, f'{date}_Station{station_id}_Cuts.npy')
            if os.path.exists(cuts_file):
                ic(f"Loading cuts file: {cuts_file}")
                cuts_data = np.load(cuts_file, allow_pickle=True)[()]
                for cut in cuts_data:
                    ic(f"Applying cut: {cut}")
                    cut_mask &= cuts_data[cut]
                times = times[cut_mask]
            else:
                ic(f"Warning: Cuts file not found for station {station_id} on date {date}.")

        all_station_data['Times'][station_id] = times
        del times  # Free memory
        gc.collect()  # Force garbage collection

        # Load other parameters
        for param_name, file_prefix in data_kwargs.items():
            param_files = sorted(glob.glob(os.path.join(station_data_folder, f'{date}_Station{station_id}_{file_prefix}*')))
            if param_files:
                param_data = [np.load(f) for f in param_files]
                param_data = np.concatenate(param_data, axis=0).squeeze()
                param_data = np.array(param_data)
                param_data = param_data[zerotime_mask]  # Apply the zerotime mask
                param_data = param_data[pretime_mask]
                if cuts and os.path.exists(cuts_file):
                    param_data = param_data[cut_mask] # Apply the same cuts
                all_station_data[param_name][station_id] = param_data
                del param_data
                gc.collect()
            else:
                ic(f"Warning: Files for {file_prefix} not found for station {station_id} on date {date}.")
                if station_id in all_station_data[param_name]:
                    del all_station_data[param_name][station_id] # Remove if no data

    # Check if any data was loaded for each parameter
    for param_name in data_kwargs:
        if not all_station_data[param_name]:
            ic(f"Warning: No data loaded for parameter '{param_name}'.")
            all_station_data[param_name] = None

    # Save the processed data before returning.
    with open(processed_file, 'wb') as f:
        pickle.dump(all_station_data, f)
        ic(f"Processed data saved to {processed_file}")

    return all_station_data





if __name__ == "__main__":
    import configparser

    # Read configuration to get the date
    config = configparser.ConfigParser() 
    config.read(os.path.join('HRAStationDataAnalysis', 'config.ini')) 
    date = config['PARAMETERS']['date']

    # Process the data with the specified parameters (example: 'snr' and 'chir')
    station_data = loadHRAConvertedData(date, cuts=True, SNR='SNR', ChiRCR='ChiRCR', Chi2016='Chi2016', ChiBad='ChiBad', Zen='Zen', Azi='Azi', Trace='Trace')

    # Optionally, print a message indicating successful processing
    print(f"Data for {date} has been processed and saved for future use.")