import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from NuRadioReco.utilities import units

from DeepLearning.D04B_reprocessNurPassingCut import plotSimSNRChi
import DeepLearning.D00_helperFunctions as D00_helperFunctions

from icecream import ic
import DeepLearning.D04C_CutInBacklobeRCR as D04C_CutInBacklobeRCR

station_id = 30
series = 200

if series == 200:
    noiseRMS = 22.53 * units.mV
elif series == 100:
    # Look up
    # noiseRMS = 15.8
    i = 0

data_to_plot = []

# data = np.load(f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/Station{station_id}_SNR_Chi.npy', allow_pickle=True)
# data_to_plot.append(data)
# data = np.load(f'DeepLearning/data/5thpass/Station{station_id}_SNR_Chi.npy', allow_pickle=True)
# data_to_plot.append(data)
data_to_plot.append(f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/Station{station_id}_SNR_Chi.npy')
data_to_plot.append(f'DeepLearning/data/5thpass/Station{station_id}_SNR_Chi.npy')


# First plot is just station data, old vs new chi
fig, axs = plt.subplots(1, len(data_to_plot), sharey=True, facecolor='w')

for iD, data in enumerate(data_to_plot):
    data = np.load(data, allow_pickle=True)

    All_SNRs, All_RCR_Chi, All_Azi, All_Zen, PassingCut_SNRs, PassingCut_RCR_Chi, PassingCut_Azi, PassingCut_Zen = data

    SNRbins = np.logspace(0.477, 2, num=80)
    maxCorrBins = np.arange(0, 1.0001, 0.01)


    if iD == 0:
        templates_RCR = D00_helperFunctions.loadSingleTemplate(series)
        templates_RCR = [templates_RCR]
        axs[iD].set_title('Single Template')
    elif iD == 1:
        templates_RCR = D00_helperFunctions.loadMultipleTemplates(series)
        templates_RCR.append(D00_helperFunctions.loadSingleTemplate(series))
        axs[iD].set_title('Multiple Templates')

    if iD == 0:
        axs[iD].hist2d(All_SNRs, All_RCR_Chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
    if iD == 1:
        axs[iD].hist2d(All_SNRs, All_RCR_Chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm(), label=f'{np.sum(D04C_CutInBacklobeRCR.RCRChiSNRCutMask(All_SNRs, All_RCR_Chi))} Events Passing Cut')        
        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs[iD], label=True)
        axs[iD].legend()

    axs[iD].set_xlim((3, 100))
    axs[iD].set_ylim((0, 1))
    axs[iD].set_xlabel('SNR')
    axs[iD].set_xscale('log')
    axs[iD].tick_params(axis='x', which='minor', bottom=True)
    axs[iD].grid(visible=True, which='both', axis='both')
    # axs[iD].title(f'Station {station_id}')
    # axs[iD].legend()

axs[0].set_ylabel('Avg Chi Highest Parallel Channels')

savename = f'DeepLearning/plots/Station_{station_id}/ChiSNR_Stn{station_id}.png'
fig.savefig(savename)
print(f'Saved {savename}')
quit()

# Second plot is station data with simulated air showers
fig2, axs2 = plt.subplots(1, len(data_to_plot), sharey=True, facecolor='w')

for iD, data in enumerate(data_to_plot):
    data = np.load(data, allow_pickle=True)

    All_SNRs, All_RCR_Chi, All_Azi, All_Zen, PassingCut_SNRs, PassingCut_RCR_Chi, PassingCut_Azi, PassingCut_Zen = data

    SNRbins = np.logspace(0.477, 2, num=80)
    maxCorrBins = np.arange(0, 1.0001, 0.01)


    if iD == 0:
        templates_RCR = D00_helperFunctions.loadSingleTemplate(series)
        templates_RCR = [templates_RCR]
        axs2[iD].set_title('Single Template')
    elif iD == 1:
        templates_RCR = D00_helperFunctions.loadMultipleTemplates(series)
        templates_RCR.append(D00_helperFunctions.loadSingleTemplate(series))
        axs2[iD].set_title('Multiple Templates')


    if iD == 0:
        axs[iD].scatter([], [], color='red', label='Simulated Air Showers')
        axs2[iD].scatter([], [], color='red', label='Simulated Air Showers')
        plotSimSNRChi(templates_RCR, noiseRMS, ax=axs2[iD])
        plotSimSNRChi(templates_RCR, noiseRMS, ax=axs[iD])
    elif iD == 1:
        plotSimSNRChi(templates_RCR, noiseRMS, ax=axs2[iD], cut=True)
        plotSimSNRChi(templates_RCR, noiseRMS, ax=axs[iD], cut=True)

        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs[iD], label=False)
        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs2[iD], label=True)


    axs2[iD].set_xlim((3, 100))
    axs2[iD].set_ylim((0, 1))
    axs2[iD].set_xlabel('SNR')
    axs2[iD].set_xscale('log')
    axs2[iD].tick_params(axis='x', which='minor', bottom=True)
    axs2[iD].grid(visible=True, which='both', axis='both')
    # axs[iD].title(f'Station {station_id}')
    axs2[iD].legend()

    axs[iD].legend()

axs2[0].set_ylabel('Avg Chi Highest Parallel Channels')


savename = f'DeepLearning/plots/Station_{station_id}/ChiSNR_wRCR_Stn{station_id}.png'
fig.savefig(savename)
print(f'Saved {savename}')

savename = f'DeepLearning/plots/Station_{station_id}/ChiSNR_RCR_Stn{station_id}.png'
fig2.savefig(savename)
print(f'Saved {savename}')

# Third plot adds backlobe onto total plot, and does separate of backlobe only
fig2, axs2 = plt.subplots(1, len(data_to_plot), sharey=True, facecolor='w')


for iD, data in enumerate(data_to_plot):
    data = np.load(data, allow_pickle=True)

    All_SNRs, All_RCR_Chi, All_Azi, All_Zen, PassingCut_SNRs, PassingCut_RCR_Chi, PassingCut_Azi, PassingCut_Zen = data

    SNRbins = np.logspace(0.477, 2, num=80)
    maxCorrBins = np.arange(0, 1.0001, 0.01)


    if iD == 0:
        templates_RCR = D00_helperFunctions.loadSingleTemplate(series)
        templates_RCR = [templates_RCR]
        axs2[iD].set_title('Single Template')
    elif iD == 1:
        templates_RCR = D00_helperFunctions.loadMultipleTemplates(series)
        templates_RCR.append(D00_helperFunctions.loadSingleTemplate(series))
        axs2[iD].set_title('Multiple Templates')
        
    if iD == 0:
        axs[iD].scatter([], [], color='green', label='Simulated Backlobe')
        axs2[iD].scatter([], [], color='green', label='Simulated Backlobe')        
        plotSimSNRChi(templates_RCR, noiseRMS, type='Backlobe', ax=axs[iD])    
        plotSimSNRChi(templates_RCR, noiseRMS, type='Backlobe', ax=axs2[iD])    
    elif iD == 1:
        plotSimSNRChi(templates_RCR, noiseRMS, type='Backlobe', ax=axs[iD], cut=True)
        plotSimSNRChi(templates_RCR, noiseRMS, type='Backlobe', ax=axs2[iD], cut=True)

        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs[iD], label=False)
        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs2[iD], label=True)


    axs2[iD].set_xlim((3, 100))
    axs2[iD].set_ylim((0, 1))
    axs2[iD].set_xlabel('SNR')
    axs2[iD].set_xscale('log')
    axs2[iD].tick_params(axis='x', which='minor', bottom=True)
    axs2[iD].grid(visible=True, which='both', axis='both')
    # axs[iD].title(f'Station {station_id}')
    axs2[iD].legend()

axs2[0].set_ylabel('Avg Chi Highest Parallel Channels')


savename = f'DeepLearning/plots/Station_{station_id}/ChiSNR_wRCR_wBacklobe_Stn{station_id}.png'
fig.savefig(savename)
print(f'Saved {savename}')

savename = f'DeepLearning/plots/Station_{station_id}/ChiSNR_Backlobe_Stn{station_id}.png'
fig2.savefig(savename)
print(f'Saved {savename}')

# Fourth plot shows the selected events from Time-ML cut
fig2, axs2 = plt.subplots(1, len(data_to_plot), sharey=True, facecolor='w')


for iD, data in enumerate(data_to_plot):
    data = np.load(data, allow_pickle=True)

    All_SNRs, All_RCR_Chi, All_Azi, All_Zen, PassingCut_SNRs, PassingCut_RCR_Chi, PassingCut_Azi, PassingCut_Zen = data

    SNRbins = np.logspace(0.477, 2, num=80)
    maxCorrBins = np.arange(0, 1.0001, 0.01)


    if iD == 0:
        templates_RCR = D00_helperFunctions.loadSingleTemplate(series)
        templates_RCR = [templates_RCR]
        axs[iD].set_title('Single Template')
    elif iD == 1:
        templates_RCR = D00_helperFunctions.loadMultipleTemplates(series)
        templates_RCR.append(D00_helperFunctions.loadSingleTemplate(series))
        axs[iD].set_title('Multiple Templates')


    if iD == 0:
        axs[iD].scatter(PassingCut_SNRs, PassingCut_RCR_Chi, label=f'{len(PassingCut_RCR_Chi)} Events Passing Time-ML Cut', facecolor='none', edgecolor='black')
        axs2[iD].scatter(PassingCut_SNRs, PassingCut_RCR_Chi, label=f'{len(PassingCut_RCR_Chi)} Events Passing Time-ML Cut', facecolor='none', edgecolor='black')
    elif iD == 1:
        axs[iD].scatter(PassingCut_SNRs, PassingCut_RCR_Chi, label=f'{len(PassingCut_RCR_Chi)} Events Passing Time-ML Cut', facecolor='none', edgecolor='black')
        axs2[iD].scatter(PassingCut_SNRs, PassingCut_RCR_Chi, label=f'{len(PassingCut_RCR_Chi)} Events Passing Time-ML Cut', facecolor='none', edgecolor='black')

        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs[iD], label=False)
        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs2[iD], label=True)


    axs2[iD].set_xlim((3, 100))
    axs2[iD].set_ylim((0, 1))
    axs2[iD].set_xlabel('SNR')
    axs2[iD].set_xscale('log')
    axs2[iD].tick_params(axis='x', which='minor', bottom=True)
    axs2[iD].grid(visible=True, which='both', axis='both')
    # axs[iD].title(f'Station {station_id}')
    axs2[iD].legend()

axs2[0].set_ylabel('Avg Chi Highest Parallel Channels')


savename = f'DeepLearning/plots/Station_{station_id}/ChiSNR_wRCR_wBacklobe_wSelected_Stn{station_id}.png'
fig.savefig(savename)
print(f'Saved {savename}')

savename = f'DeepLearning/plots/Station_{station_id}/ChiSNR_Selected_Stn{station_id}.png'
fig2.savefig(savename)
print(f'Saved {savename}')

fig2, axs2 = plt.subplots(1, len(data_to_plot), sharey=True, facecolor='w')

# fig.colorbar()
