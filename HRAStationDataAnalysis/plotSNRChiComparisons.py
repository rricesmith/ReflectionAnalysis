import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import gc
from icecream import ic
import configparser

def setSNRChiPlot(ax, ylabel_add=None, ylabel_add_color='black', diff=False):
    ax.set_xlabel('SNR')
    ax.set_xscale('log')
    if ylabel_add is not None:
        ax.set_ylabel(f' {ylabel_add} Chi', color=ylabel_add_color)
    else:
        ax.set_ylabel('Chi')
    if diff:
        ax.set_ylim(-1, 1)
    else:
        ax.set_ylim(0, 1)
    ax.set_xlim(3, 100)
    ax.grid(visible=True, which='major', axis='both')

if __name__ == "__main__":
    # This code makes the following plots
    # SNR vs 2016_Chi
    # SNR vs RCR_Chi
    # SNR vs RCR_Chi_bad
    # A plot with 3 subplots of the Chi's vs each other
    # A SNR vs (RCR_Chi - 2016_Chi) plot
    # A SNR vs (RCR_Chi_bad - 2016_Chi) plot
    # A SNR vs (RCR_Chi - RCR_Chi_bad) plot

    config = configparser.ConfigParser()
    config.read('HRAStationDataAnalysis/config.ini')
    date = config['PARAMETERS']['date']


    # parser = argparse.ArgumentParser(description='Convert HRA Nur files to numpy files')
    # parser.add_argument('stnID', type=int)
    # parser.add_argument('date', type=str)

    # args = parser.parse_args()
    # station_id = args.stnID
    # date = args.date
    station_id = 13

    data_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'
    plot_folder = f'HRAStationDataAnalysis/plots/{date}/'
    os.makedirs(plot_folder, exist_ok=True)

    SNR_array = []
    chi_2016_array = []
    chi_RCR_array = []
    chi_RCR_bad_array = []    

    # Load the data
    for file in os.listdir(data_folder):
        if file.startswith(f'{date}_Station{station_id}_SNR'):
            data = np.load(data_folder+file, allow_pickle=True)
            SNR_array.extend(data.tolist())
            del data

        if file.startswith(f'{date}_Station{station_id}_Chi2016'):
            data = np.load(data_folder+file, allow_pickle=True)
            chi_2016_array.extend(data.tolist())
            del data

        if file.startswith(f'{date}_Station{station_id}_ChiRCR'):
            data = np.load(data_folder+file, allow_pickle=True)
            chi_RCR_array.extend(data.tolist())
            del data

        if file.startswith(f'{date}_Station{station_id}_ChiBad'):
            data = np.load(data_folder+file, allow_pickle=True)
            chi_RCR_bad_array.extend(data.tolist())
            del data

        gc.collect()    # Free memory just in case it's large

    # Convert to numpy arrays
    SNR_array = np.array(SNR_array)
    chi_2016_array = np.array(chi_2016_array)
    chi_RCR_array = np.array(chi_RCR_array)
    chi_RCR_bad_array = np.array(chi_RCR_bad_array)


    # Make the plots
    # Since 3 sets of data, will plot in a triangular matrix with the SNR-Chi plots on the diagonal

    # 2016 is 0 in both dims
    # RCR is 1 in both dims
    # RCR Bad is 2 in both dims
    colors = {'2016': 'blue', 'RCR': 'green', 'RCR Bad': 'red'}

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'SNR and Chi for station {station_id} on {date}', fontsize=20)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # Upper left is SNR vs 2016 Chi
    axs[0, 0].scatter(SNR_array, chi_2016_array)
    axs[0, 0].plot([0, 100], [0, 1], linestyle='--', color='red')
    axs[0, 0].set_title('SNR vs 2016 Chi')
    setSNRChiPlot(axs[0, 0], ylabel_add='2016', ylabel_add_color=colors['2016'])
    # Middle left is Chi 2016 vs RCR Chi
    axs[1, 0].scatter(chi_2016_array, chi_RCR_array)
    axs[1, 0].plot([0, 1], [0, 1], linestyle='--', color='red')
    axs[1, 0].set_title('2016 Chi vs RCR Chi')
    axs[1, 0].set_xlabel('2016 Chi', color=colors['2016'])
    axs[1, 0].set_ylabel('RCR Chi', color=colors['RCR'])
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].set_xlim(0, 1)
    # Bottom left is Chi 2016 vs RCR Bad Chi
    axs[2, 0].scatter(chi_2016_array, chi_RCR_bad_array)
    axs[2, 0].plot([0, 1], [0, 1], linestyle='--', color='red')
    axs[2, 0].set_title('2016 Chi vs RCR Bad Chi')
    axs[2, 0].set_xlabel('2016 Chi', color=colors['2016'])
    axs[2, 0].set_ylabel('RCR Bad Chi', color=colors['RCR Bad'])
    axs[2, 0].set_ylim(0, 1)
    axs[2, 0].set_xlim(0, 1)

    # Middle plot is SNR vs RCR Chi
    axs[1, 1].scatter(SNR_array, chi_RCR_array)
    axs[1, 1].set_title('SNR vs RCR Chi')
    setSNRChiPlot(axs[1, 1], ylabel_add='RCR', ylabel_add_color=colors['RCR'])
    # Bottom middle is Chi RCR Chi vs RCR Bad Chi
    axs[2, 1].scatter(chi_RCR_array, chi_RCR_bad_array)
    axs[2, 1].set_title('RCR Chi vs RCR Bad Chi')
    axs[2, 1].set_xlabel('RCR Chi', color=colors['RCR'])
    axs[2, 1].set_ylabel('RCR Bad Chi', color=colors['RCR Bad'])
    axs[2, 1].set_ylim(0, 1)
    axs[2, 1].set_xlim(0, 1)

    # Bottom right is SNR vs RCR Bad Chi
    axs[2, 2].scatter(SNR_array, chi_RCR_bad_array)
    axs[2, 2].set_title('SNR vs RCR Bad Chi')
    setSNRChiPlot(axs[2, 2], ylabel_add='RCR Bad', ylabel_add_color=colors['RCR Bad'])

    # Do the Chi differences on upper triangular matrix
    # Upper middle is SNR vs (RCR Chi - 2016 Chi)
    fsize = 6
    axs[0, 1].scatter(SNR_array, chi_RCR_array - chi_2016_array)
    axs[0, 1].set_title('SNR vs (RCR Chi - 2016 Chi)')
    axs[0, 1].text(0.35, 0.35, 'More RCR', fontsize=6, color=colors['RCR'])
    axs[0, 1].text(-0.35, -0.35, 'More 2016', fontsize=fsize, color=colors['2016'])
    setSNRChiPlot(axs[0, 1], diff=True)
    # Upper right is SNR vs (RCR Bad Chi - 2016 Chi)
    axs[0, 2].scatter(SNR_array, chi_RCR_bad_array - chi_2016_array)
    axs[0, 2].set_title('SNR vs (RCR Bad Chi - 2016 Chi)')
    axs[0, 2].text(0.35, 0.35, 'More RCR Bad', fontsize=fsize, color=colors['RCR Bad'])
    axs[0, 2].text(-0.35, -0.35, 'More 2016', fontsize=fsize, color=colors['2016'])
    setSNRChiPlot(axs[0, 2], diff=True)
    # Middle right is SNR vs (RCR Chi - RCR Bad Chi)
    axs[1, 2].scatter(SNR_array, chi_RCR_array - chi_RCR_bad_array)
    axs[1, 2].set_title('SNR vs (RCR Chi - RCR Bad Chi)')
    axs[1, 2].text(0.35, 0.35, 'More RCR', fontsize=fsize, color=colors['RCR'])
    axs[1, 2].text(-0.35, -0.35, 'More RCR Bad', fontsize=fsize, color=colors['RCR Bad'])
    setSNRChiPlot(axs[1, 2], diff=True)


    # Save the plots
    savename = f'{plot_folder}SNR_Chi_Station{station_id}_{date}.png'
    plt.savefig(savename)
    ic(f'Saved plot to {savename}')
    plt.close(fig)
    # Clear the figure
    fig.clf()
    plt.close(fig)

