import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from NuRadioReco.utilities import units
import os
from DeepLearning.D04B_reprocessNurPassingCut import plotSimSNRChi, pT
import DeepLearning.D00_helperFunctions as D00_helperFunctions
from icecream import ic
import DeepLearning.D04C_CutInBacklobeRCR as D04C_CutInBacklobeRCR
from StationDataAnalysis.S00_FoundEventsSearchUtil import inStation2016


def set_CHI_SNR_axis(ax, station_id):

    ax.set_xlim((3, 100))
    ax.set_ylim((0, 1))
    ax.set_xlabel('SNR')
    ax.set_xscale('log')
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax.grid(visible=True, which='both', axis='both')
    ax.set_title(f'Station {station_id}')
    ax.legend()
    ax.set_ylabel('Avg Chi Highest Parallel Channels')
    return ax

datapass = '7thpass'
stations_100s = [13, 15, 18, 32]
stations_200s = [14, 17, 19, 30]
stations = {100: stations_100s, 200: stations_200s}

station_id = 30
series = 200


for series in stations.keys():
    if series == 200:
        noiseRMS = 22.53 * units.mV
    elif series == 100:
        # Look up
        noiseRMS = 20.0 * units.mV
    for station_id in stations[series]:


        station_data_folder = f'DeepLearning/data/{datapass}/Station{station_id}'

        data = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_SNR_Chi.npy', allow_pickle=True)
        data_SnrChiCut = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_SnrChiCut.npy', allow_pickle=True)
        data_in2016 = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_In2016.npy', allow_pickle=True)

        plotfolder = f'StationDataAnalysis/plots/Station_{station_id}'
        if not os.path.exists(plotfolder):
            os.makedirs(plotfolder)



        # Load data and set plot bins
        SNRbins = np.logspace(0.477, 2, num=80)
        maxCorrBins = np.arange(0, 1.0001, 0.01)

        All_SNRs, All_RCR_Chi, PassingCut_SNRs, PassingCut_RCR_Chi, PassingCut_Azi, PassingCut_Zen = data
        in2016_SNRs, in2016_RCR_Chi, in2016_Azi, in2016_Zen, in2016_Traces, in2016_Times = data_in2016


        templates_RCR = D00_helperFunctions.loadMultipleTemplates(series)
        templates_RCR.append(D00_helperFunctions.loadSingleTemplate(series))

        ############################################################################################
        # First plot is only station data, Chi v SNR
        fig, axs = plt.subplots(1, 1, sharey=True, facecolor='w')
        axs = np.atleast_1d(axs)

        axs[0].hist2d(All_SNRs, All_RCR_Chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm(), label=f'{np.sum(D04C_CutInBacklobeRCR.RCRChiSNRCutMask(All_SNRs, All_RCR_Chi))} Events Passing Cut')        
        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs[0], label=True)
        set_CHI_SNR_axis(axs[0], station_id)

        savename = f'{plotfolder}/ChiSNR_Stn{station_id}.png'
        fig.savefig(savename)
        print(f'Saved {savename}')
        # quit()


        ############################################################################################
        # Second plot is station data with simulated air showers
        fig2, axs2 = plt.subplots(1, 1, sharey=True, facecolor='w')
        axs2 = np.atleast_1d(axs2)

        plotSimSNRChi(templates_RCR, noiseRMS, ax=axs2[0], cut=True, path=f'DeepLearning/data/3rdpass/')
        plotSimSNRChi(templates_RCR, noiseRMS, ax=axs[0], cut=True, path=f'DeepLearning/data/3rdpass/')

        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs[0], label=False)
        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs2[0], label=True)

        set_CHI_SNR_axis(axs2[0], station_id)
        axs[0].legend()

        savename = f'{plotfolder}/ChiSNR_wRCR_Stn{station_id}.png'
        fig.savefig(savename)
        print(f'Saved {savename}')
        savename = f'{plotfolder}/ChiSNR_RCR_Stn{station_id}.png'
        fig2.savefig(savename)
        print(f'Saved {savename}')


        ############################################################################################
        # Third plot adds backlobe onto total plot, and does separate of backlobe only
        fig2, axs2 = plt.subplots(1, 1, sharey=True, facecolor='w')
        axs2 = np.atleast_1d(axs2)
                
        plotSimSNRChi(templates_RCR, noiseRMS, type='Backlobe', ax=axs[0], cut=True, path=f'DeepLearning/data/3rdpass/')
        plotSimSNRChi(templates_RCR, noiseRMS, type='Backlobe', ax=axs2[0], cut=True, path=f'DeepLearning/data/3rdpass/')

        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs[0], label=False)
        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs2[0], label=True)

        set_CHI_SNR_axis(axs2[0], station_id)

        savename = f'{plotfolder}/ChiSNR_wRCR_wBacklobe_Stn{station_id}.png'
        fig.savefig(savename)
        print(f'Saved {savename}')
        savename = f'{plotfolder}/ChiSNR_Backlobe_Stn{station_id}.png'
        fig2.savefig(savename)
        print(f'Saved {savename}')


        ############################################################################################
        # Fourth plot shows the selected events from Time-ML cut
        fig2, axs2 = plt.subplots(1, 1, sharey=True, facecolor='w')
        axs2 = np.atleast_1d(axs2)

        axs[0].scatter(PassingCut_SNRs, PassingCut_RCR_Chi, label=f'{len(PassingCut_RCR_Chi)} Events Passing Time-ML Cut', facecolor='none', edgecolor='black')
        axs2[0].scatter(PassingCut_SNRs, PassingCut_RCR_Chi, label=f'{len(PassingCut_RCR_Chi)} Events Passing Time-ML Cut', facecolor='none', edgecolor='black')

        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs[0], label=False)
        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs2[0], label=True)

        set_CHI_SNR_axis(axs2[0], station_id)


        savename = f'{plotfolder}/ChiSNR_wRCR_wBacklobe_wSelected_Stn{station_id}.png'
        fig.savefig(savename)
        print(f'Saved {savename}')

        savename = f'{plotfolder}/ChiSNR_Selected_Stn{station_id}.png'
        fig2.savefig(savename)
        print(f'Saved {savename}')


        ############################################################################################
        # Fifth plot shows events from 2016
        fig2, axs2 = plt.subplots(1, 1, sharey=True, facecolor='w')
        axs2 = np.atleast_1d(axs2)

        axs[0].scatter(in2016_SNRs, in2016_RCR_Chi, label=f'{len(in2016_RCR_Chi)} Events from 2016', marker='*', facecolor='red', edgecolor='black')
        axs2[0].scatter(in2016_SNRs, in2016_RCR_Chi, label=f'{len(in2016_RCR_Chi)} Events from 2016', marker='*', facecolor='red', edgecolor='black')

        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs[0], label=False)
        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs2[0], label=True)

        set_CHI_SNR_axis(axs2[0], station_id)

        savename = f'{plotfolder}/ChiSNR_wRCR_wBacklobe_wSelected_w2016_Stn{station_id}.png'
        fig.savefig(savename)
        print(f'Saved {savename}')

        savename = f'{plotfolder}/ChiSNR_Selected_w2016_Stn{station_id}.png'
        fig2.savefig(savename)
        print(f'Saved {savename}')


        plt.close(fig)
        plt.close(fig2)
        ############################################################################################
        # Plot all the traces found from 2016 and save the numpy traces for use as templates
        for iT, traces in enumerate(in2016_Traces):
            stationtime = in2016_Times[iT]
            azi = in2016_Azi[iT]
            zen = in2016_Zen[iT]
            chi = in2016_RCR_Chi[iT]
            SNR = in2016_SNRs[iT]


            plotfolder = f'StationDataAnalysis/plots/Station_{station_id}/Events2016'
            if not os.path.exists(plotfolder):
                os.makedirs(plotfolder)
            pT(traces, datetime.datetime.fromtimestamp(stationtime).strftime("%m-%d-%Y, %H:%M:%S") + f' Chi {chi:.2f} SNR {SNR:.2f}, {zen:.1f}Deg Zen {azi:.1f}Deg Azi', 
                f'{plotfolder}/Event2016_{stationtime}_Chi{chi:.2f}_SNR{SNR:.2f}.png')
            tracefolder = f'StationDataAnalysis/templates/Station_{station_id}'
            np.save(f'{tracefolder}/Event2016_{stationtime}_Chi{chi:.2f}_SNR{SNR:.2f}.npy', traces)


    plt.close('all')