import numpy as np
from glob import glob
import os
from icecream import ic
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import configparser
import DeepLearning.D00_helperFunctions as D00_helperFunctions
from DeepLearning.D04B_reprocessNurPassingCut import getMaxAllChi, pT
from NuRadioReco.utilities import units
from S01_StationDataAndDeeplearnPlot import set_CHI_SNR_axis

if __name__ == "__main__":



    config = configparser.ConfigParser()
    config.read('StationDataAnalysis/config.ini')
    datapass = config['BASEFOLDER']['base_folder']
    # template_date = config['TEMPLATE']['template']

    # stations_100s = [13, 15, 18, 32]  # Station 32 data not compiled yet
    stations_100s = [13, 15, 18]
    stations_200s = [14, 17, 19, 30]
    stations = {100: stations_100s, 200: stations_200s}

    # Dictionary for finding coincidence events
    times_dict = {}
    data_dict = {}



    plotfolder = f'StationDataAnalysis/plots/{datapass}/TemplateDifferences'

    all_2016_SNRs = []
    all_2016_RCR_Chi = []
    all_2016_Chi = []

    template_series_RCR = D00_helperFunctions.loadMultipleTemplates('200') # Hack b/c no 100s templates yet

    for series in stations.keys():
        # template_series_RCR = D00_helperFunctions.loadMultipleTemplates(series)
        template_series_2016 = D00_helperFunctions.loadMultipleTemplates(series, date='2016')
        for station_id in stations[series]:
            times_dict[station_id] = []
            data_dict[station_id] = []

            station_data_folder = f'DeepLearning/data/{datapass}/Station{station_id}'

            # data = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_SNR_Chi.npy', allow_pickle=True)
            # data_SnrChiCut = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_SnrChiCut.npy', allow_pickle=True)
            data_in2016 = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_In2016.npy', allow_pickle=True)
            # times = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_Times.npy', allow_pickle=True)


            # Load all the data
            in2016_SNRs, in2016_RCR_Chi, in2016_Azi, in2016_Zen, in2016_Traces, in2016_datetimes = data_in2016


            if not len(in2016_datetimes) == 0:
                in2016_datetimes = np.vectorize(datetime.datetime.fromtimestamp)(in2016_datetimes)

            RCR_Chi = []
            for iT, traces in enumerate(in2016_Traces):
                RCR_Chi.append(getMaxAllChi(traces, 2*units.GHz, template_series_RCR, 2*units.GHz))

            Chi_2016 = []
            for iT, traces in enumerate(in2016_Traces):
                Chi_2016.append(getMaxAllChi(traces, 2*units.GHz, template_series_2016, 2*units.GHz), exclude_match=in2016_datetimes[iT].timestamp())

            for i, chi16 in enumerate(Chi_2016):
                ic(RCR_Chi[i], chi16)
                pT(in2016_Traces[i], in2016_datetimes[i].strftime('%Y-%m-%d %H:%M:%S') + f', {in2016_Azi[i]:.1f}deg Azi, {in2016_Zen[i]:.1f}deg Zen,\n {in2016_SNRs[i]:.1f} SNR, {chi16:.2f} BL Chi -> {RCR_Chi[i]:.2f} RCR Chi',
                    f'{plotfolder}/Station{station_id}_SNR{in2016_SNRs[i]:.2f}_{in2016_datetimes[i]}.png')

            all_2016_SNRs.extend(in2016_SNRs)
            all_2016_Chi.extend(Chi_2016)
            all_2016_RCR_Chi.extend(RCR_Chi)
    
    fig, ax = plt.subplots()

    ax.plot(all_2016_SNRs, all_2016_Chi, 'o', label='2016 Templates')
    ax.plot(all_2016_SNRs, all_2016_RCR_Chi, 'o', label='RCR Templates')
    for i in range(len(all_2016_SNRs)):
        ax.arrow(all_2016_SNRs[i], all_2016_Chi[i], 0, all_2016_RCR_Chi[i] - all_2016_Chi[i], color='black')
    ax = set_CHI_SNR_axis(ax, 'All Stations')

    savename = f'{plotfolder}/TemplateDifferences.png'
    fig.savefig(savename)
    ic(f'Saved {savename}')