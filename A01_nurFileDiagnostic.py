import os
import numpy as np
from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.framework.parameters import stationParameters as stnp

import matplotlib.pyplot as plt

#Diagnostic settings
# nurPath = '../../../ariannaproject/rricesmi/simulatedRCRs/200s/'
# nurPath = '../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/simStn51/3.11.24/'
nurPath = 'SimpleFootprintSimulation/output/Backlobe/5.7.24/100s/'
saveFolder = 'plots/diagnostics/Backlobe'
channel_ids = [0, 1, 2, 3]
# channel_ids = [4, 5, 6]

maxSave = 100
saveIter = 500












#Get files
nurFiles = []
for filename in os.listdir(nurPath):
    if filename.endswith('.nur'):
        nurFiles.append(os.path.join(nurPath, filename))

#Begin diagnostic
eventReader = NuRadioRecoio.NuRadioRecoio(nurFiles)
for i, evt in enumerate(eventReader.get_events()):
    station_ids = evt.get_station_ids()

    for stn_id in station_ids:
        station = evt.get_station(stn_id)
        if not station.has_triggered():
            continue
        if not i % saveIter == 0:
            continue
        print(f'printing {i}')            
        fig, axs = plt.subplots(len(channel_ids), sharex=True)
        for ChId, channel in enumerate(station.iter_channels(use_channels=channel_ids)):

            y = channel.get_trace()
            t = channel.get_times()

            axs[ChId].plot(t, y)

        axs[ChId].set_xlabel('time [ns]', fontsize=16)
        for iC, ch in enumerate(channel_ids):
            axs[iC].set_ylabel(f'ch{ch}',labelpad=10,rotation=0,fontsize=13)
            # axs[i].set_ylim(-250,250)
#            axs[c].set_xlim(-3,260)
            axs[iC].tick_params(labelsize=13)
        axs[0].tick_params(labelsize=13)
        axs[0].set_ylabel(f'ch{0}',labelpad=3,rotation=0,fontsize=13)
#        axs[c].set_xlim(-3,260)
        # axs[3].set_ylabel('mV',labelpad=0)
        fig.text(0.03, 0.5, 'voltage [V]', ha='center', va='center', rotation='vertical',fontsize=18)
        plt.xticks(size=13)
        # plt.yticks(size=15)
    #    plt.show()
        plt.savefig(f'{saveFolder}/trace{i}.png')
        plt.clf()
        plt.close()

print(f'Done!')
