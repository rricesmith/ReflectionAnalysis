import numpy as np
import matplotlib.pyplot as plt
import os
import argparse



if __name__ == "__main__":
    # This code makes the following plots
    # SNR vs 2016_Chi
    # SNR vs RCR_Chi
    # SNR vs RCR_Chi_bad
    # A plot with 3 subplots of the Chi's vs each other
    # A SNR vs (RCR_Chi - 2016_Chi) plot
    # A SNR vs (RCR_Chi_bad - 2016_Chi) plot
    # A SNR vs (RCR_Chi - RCR_Chi_bad) plot


    parser = argparse.ArgumentParser(description='Convert HRA Nur files to numpy files')
    parser.add_argument('stnID', type=int)
    parser.add_argument('date', type=str)

    args = parser.parse_args()
    station_id = args.stnID
    date = args.date

    data_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'
    plot_folder = f'HRAStationDataAnalysis/plots/{date}/'
    os.makedirs(plot_folder, exist_ok=True)

    SNR_array = []
    chi_2016_array = []
    chi_RCR_array = []
    chi_RCR_bad_array = []    

    # Load the data
    for file in data_folder:
        if file.startswith(f'{date}_Station{station_id}_SNR'):
            data = np.load(data_folder+file, allow_pickle=True)
            SNR_array.append(data.tolist())

        if file.startswith(f'{date}_Station{station_id}_Chi2016'):
            data = np.load(data_folder+file, allow_pickle=True)
            chi_2016_array.append(data.tolist())

        if file.startswith(f'{date}_Station{station_id}_ChiRCR'):
            data = np.load(data_folder+file, allow_pickle=True)
            chi_RCR_array.append(data.tolist())

        if file.startswith(f'{date}_Station{station_id}_ChiBad'):
            data = np.load(data_folder+file, allow_pickle=True)
            chi_RCR_bad_array.append(data.tolist())


    # Convert to numpy arrays
    SNR_array = np.array(SNR_array)
    chi_2016_array = np.array(chi_2016_array)
    chi_RCR_array = np.array(chi_RCR_array)
    chi_RCR_bad_array = np.array(chi_RCR_bad_array)


    # Make the plots
