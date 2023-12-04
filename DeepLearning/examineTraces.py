
import os
import glob
from matplotlib import pyplot as plt
import numpy as np
import math
import json
from NuRadioReco.utilities import units, fft

#path = "/Users/astrid/Desktop/DL_on_series100-200/st18"
#signal = np.load(os.path.join(path, "signal/Station18_SimNu_forML_shuffSC_centered.npy"))
#noise = np.load(os.path.join(path, "noise/traces_station_18_run_all_shuffNC_centered.npy"))

#signal = np.load('DeepLearning/data/1stpass/NuFilter_100000events.npy')
#label = 'Nu'
#signal = np.load('DeepLearning/data/1stpass/RCR200s_5730events.npy')
#label = 'RCR'
signal = np.load('DeepLearning/data/1stpass/Stn19Data_3895events.npy')
label = 'Stn19'

data = signal

sampling_rate = 2 #In GHz

x = np.linspace(1, int(256 / sampling_rate), num=256)
x_freq = np.fft.rfftfreq(len(x), d=(1 / sampling_rate*units.GHz)) / units.MHz
print(f'len xfreq {len(x_freq)}')


#for i in range(len(data)):
#    highV = data[i] * 1000 > 150
#    if not np.any(highV):
#        continue
for i in range(100):
    i = i * 30
    fig, axs = plt.subplots(nrows=4, ncols=2, sharex=False)

    axs[0][0].plot(x, 1000*data[i, 0])
    axs[1][0].plot(x, 1000*data[i, 1])
    axs[2][0].plot(x, 1000*data[i, 2])
    axs[3][0].plot(x, 1000*data[i, 3])
    
    axs[3][0].set_xlabel('time [ns]',fontsize=18)

    test = np.abs(fft.time2freq(data[i, 0], sampling_rate*units.GHz))
#    print(f'len test {len(test)} and len x_freq {len(x_freq)}')
#    print(f'len data {len(data[i, 0])}')
    result = np.abs(fft.time2freq(data[i, 0], sampling_rate*units.GHz))
#    print(f'result {result}')
    axs[0][1].plot(x_freq, np.abs(fft.time2freq(data[i, 0], sampling_rate*units.GHz)))
    axs[1][1].plot(x_freq, np.abs(fft.time2freq(data[i, 1], sampling_rate*units.GHz)))
    axs[2][1].plot(x_freq, np.abs(fft.time2freq(data[i, 2], sampling_rate*units.GHz)))
    axs[3][1].plot(x_freq, np.abs(fft.time2freq(data[i, 3], sampling_rate*units.GHz)))

    axs[3][1].set_xlabel('Frequency [MHz]',fontsize=18)

    for c in [0,1,2,3]:
        axs[c][0].axhline(y=4.4*23, color='red', linestyle='--')
        axs[c][0].axhline(y=-4.4*23, color='red', linestyle='--')

        axs[c][0].set_ylabel(f'ch{c}',labelpad=10,rotation=0,fontsize=13)
        # axs[i].set_ylim(-250,250)
        axs[c][0].set_xlim(-3,260 / sampling_rate)
        axs[c][1].set_xlim(-3, 500)
        axs[c][0].tick_params(labelsize=13)
        axs[c][1].tick_params(labelsize=13)
    axs[0][0].tick_params(labelsize=13)
    axs[0][1].tick_params(labelsize=13)
    axs[0][0].set_ylabel(f'ch{0}',labelpad=3,rotation=0,fontsize=13)
    axs[c][0].set_xlim(-3,260 / sampling_rate)
    axs[c][1].set_xlim(-3, 500)
    # axs[3].set_ylabel('mV',labelpad=0)
    fig.text(0.03, 0.5, 'voltage [mV]', ha='center', va='center', rotation='vertical',fontsize=18)
    plt.xticks(size=13)
    # plt.yticks(size=15)
#    plt.show()
    plt.savefig(f'DeepLearning/plots/testTraces/{label}_trace{i}.png')
    plt.clf()
    plt.close()

print(f'Done!')