import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from NuRadioReco.utilities import units
import os
from DeepLearning.D04B_reprocessNurPassingCut import plotSimSNRChi
import DeepLearning.D00_helperFunctions as D00_helperFunctions

from icecream import ic
import DeepLearning.D04C_CutInBacklobeRCR as D04C_CutInBacklobeRCR
from StationDataAnalysis.S00_FoundEventsSearchUtil import inStation2016


datapass = '7thpass'
stations_100s = [13, 15, 18, 32]
stations_200s = [14, 17, 19, 30]

station_id = 30
series = 200

if series == 200:
    noiseRMS = 22.53 * units.mV
elif series == 100:
    # Look up
    # noiseRMS = 15.8
    i = 0

data_to_plot = []

data_to_plot.append(f'DeepLearning/data/{datapass}/Station{station_id}/FilteredStation{station_id}_Data_SNR_Chi.npy')
plotfolder = f'StationDataAnalysis/plots/Station_{station_id}'
if not os.path.exists(plotfolder):
    os.makedirs(plotfolder)


# First plot is just station data, old vs new chi
fig, axs = plt.subplots(1, len(data_to_plot), sharey=True, facecolor='w')
axs = np.atleast_1d(axs)

data = np.load(data_to_plot[0], allow_pickle=True)

All_SNRs, All_RCR_Chi, PassingCut_SNRs, PassingCut_RCR_Chi, PassingCut_Azi, PassingCut_Zen = data

SNRbins = np.logspace(0.477, 2, num=80)
maxCorrBins = np.arange(0, 1.0001, 0.01)


templates_RCR = D00_helperFunctions.loadMultipleTemplates(series)
templates_RCR.append(D00_helperFunctions.loadSingleTemplate(series))
# templates_RCR = [templates_RCR]

axs[0].hist2d(All_SNRs, All_RCR_Chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm(), label=f'{np.sum(D04C_CutInBacklobeRCR.RCRChiSNRCutMask(All_SNRs, All_RCR_Chi))} Events Passing Cut')        
D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs[0], label=True)

axs[0].set_xlim((3, 100))
axs[0].set_ylim((0, 1))
axs[0].set_xlabel('SNR')
axs[0].set_xscale('log')
axs[0].tick_params(axis='x', which='minor', bottom=True)
axs[0].grid(visible=True, which='both', axis='both')
axs[0].set_title(f'Station {station_id}')
axs[0].legend()

axs[0].set_ylabel('Avg Chi Highest Parallel Channels')

savename = f'{plotfolder}/ChiSNR_Stn{station_id}.png'
fig.savefig(savename)
print(f'Saved {savename}')
# quit()

# Second plot is station data with simulated air showers
fig2, axs2 = plt.subplots(1, len(data_to_plot), sharey=True, facecolor='w')
axs2 = np.atleast_1d(axs2)



# templates_RCR = D00_helperFunctions.loadMultipleTemplates(series)
# templates_RCR.append(D00_helperFunctions.loadSingleTemplate(series))


plotSimSNRChi(templates_RCR, noiseRMS, ax=axs2[0], cut=True, path=f'DeepLearning/data/5thpass/')
plotSimSNRChi(templates_RCR, noiseRMS, ax=axs[0], cut=True, path=f'DeepLearning/data/5thpass/')

D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs[0], label=False)
D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs2[0], label=True)


axs2[0].set_xlim((3, 100))
axs2[0].set_ylim((0, 1))
axs2[0].set_xlabel('SNR')
axs2[0].set_xscale('log')
axs2[0].tick_params(axis='x', which='minor', bottom=True)
axs2[0].grid(visible=True, which='both', axis='both')
axs[0].set_title(f'Station {station_id}')
axs2[0].legend()

axs[0].legend()

axs2[0].set_ylabel('Avg Chi Highest Parallel Channels')


savename = f'{plotfolder}/ChiSNR_wRCR_Stn{station_id}.png'
fig.savefig(savename)
print(f'Saved {savename}')

savename = f'{plotfolder}/ChiSNR_RCR_Stn{station_id}.png'
fig2.savefig(savename)
print(f'Saved {savename}')

# Third plot adds backlobe onto total plot, and does separate of backlobe only
fig2, axs2 = plt.subplots(1, len(data_to_plot), sharey=True, facecolor='w')
axs2 = np.atleast_1d(axs2)


# templates_RCR = D00_helperFunctions.loadMultipleTemplates(series)
# templates_RCR.append(D00_helperFunctions.loadSingleTemplate(series))
        
plotSimSNRChi(templates_RCR, noiseRMS, type='Backlobe', ax=axs[0], cut=True, path=f'DeepLearning/data/5thpass/')
plotSimSNRChi(templates_RCR, noiseRMS, type='Backlobe', ax=axs2[0], cut=True, path=f'DeepLearning/data/5thpass/')

D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs[0], label=False)
D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs2[0], label=True)


axs2[0].set_xlim((3, 100))
axs2[0].set_ylim((0, 1))
axs2[0].set_xlabel('SNR')
axs2[0].set_xscale('log')
axs2[0].tick_params(axis='x', which='minor', bottom=True)
axs2[0].grid(visible=True, which='both', axis='both')
axs[0].set_title(f'Station {station_id}')
axs2[0].legend()

axs2[0].set_ylabel('Avg Chi Highest Parallel Channels')


savename = f'{plotfolder}/ChiSNR_wRCR_wBacklobe_Stn{station_id}.png'
fig.savefig(savename)
print(f'Saved {savename}')

savename = f'{plotfolder}/ChiSNR_Backlobe_Stn{station_id}.png'
fig2.savefig(savename)
print(f'Saved {savename}')

# Fourth plot shows the selected events from Time-ML cut
fig2, axs2 = plt.subplots(1, len(data_to_plot), sharey=True, facecolor='w')
axs2 = np.atleast_1d(axs2)


# templates_RCR = D00_helperFunctions.loadMultipleTemplates(series)
# templates_RCR.append(D00_helperFunctions.loadSingleTemplate(series))

axs[0].scatter(PassingCut_SNRs, PassingCut_RCR_Chi, label=f'{len(PassingCut_RCR_Chi)} Events Passing Time-ML Cut', facecolor='none', edgecolor='black')
axs2[0].scatter(PassingCut_SNRs, PassingCut_RCR_Chi, label=f'{len(PassingCut_RCR_Chi)} Events Passing Time-ML Cut', facecolor='none', edgecolor='black')

D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs[0], label=False)
D04C_CutInBacklobeRCR.plotRCRChiSNRCut(ax=axs2[0], label=True)


axs2[0].set_xlim((3, 100))
axs2[0].set_ylim((0, 1))
axs2[0].set_xlabel('SNR')
axs2[0].set_xscale('log')
axs2[0].tick_params(axis='x', which='minor', bottom=True)
axs2[0].grid(visible=True, which='both', axis='both')
axs[0].set_title(f'Station {station_id}')
axs2[0].legend()

axs2[0].set_ylabel('Avg Chi Highest Parallel Channels')


savename = f'{plotfolder}/ChiSNR_wRCR_wBacklobe_wSelected_Stn{station_id}.png'
fig.savefig(savename)
print(f'Saved {savename}')

savename = f'{plotfolder}/ChiSNR_Selected_Stn{station_id}.png'
fig2.savefig(savename)
print(f'Saved {savename}')

fig2, axs2 = plt.subplots(1, len(data_to_plot), sharey=True, facecolor='w')

# fig.colorbar()
