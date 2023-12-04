import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.modules.io import NuRadioRecoio
import numpy as np
from NuRadioReco.utilities import units, fft
import argparse
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt








def calcAverageFFT(fft_traces):
    #FFT traces should be a list/array, each of 4 ffts, 1 for each channel

    average_ffts = np.zeros_like(fft_traces[0])
    channels = len(fft_traces[0])
    num_traces = len(fft_traces)

    for fft in fft_traces:
        for i in range(channels):
            average_ffts[i] += fft[i]


    average_per_channel = average_ffts / num_traces
    average_overall = np.zeros_like(average_per_channel[0])
    for i in range(channels):
        average_overall += average_per_channel[i]
    average_overall = average_overall / channels

    return average_per_channel, average_overall


def plotAverageFFT(average_overall, station_id, average_per_channel=None, sampling_rate=2):
    x = np.linspace(1, int(256 / sampling_rate), num=256)
    x_freq = np.fft.rfftfreq(len(x), d=(1 / sampling_rate*units.GHz)) / units.MHz
    x_freq = x_freq/1000

    plt.plot(x_freq, average_overall, color='black', label='Average All Chs')
    plt.xlabel('Frequency [GHz]', fontsize=18)
    plt.ylabel('Amplitude')
    plt.xlim(-0.003, 1.050)
    plt.xticks(size=13)
    plt.title(f'Noise FFT Station {station_id}')
    plt.savefig(f'DeepLearning/plots/Station_{station_id}/averageNoiseFFT_Stn{station_id}.png')

    if not len(average_per_channel) == 4:
        plt.close()
        return

    plt.plot(x_freq, average_per_channel[0], color='orange', linestyle='--', label='Ch0')
    plt.plot(x_freq, average_per_channel[1], color='blue', linestyle='--', label='Ch1')
    plt.plot(x_freq, average_per_channel[2], color='purple', linestyle='--', label='Ch2')
    plt.plot(x_freq, average_per_channel[3], color='green', linestyle='--', label='Ch3')
    plt.legend()
    plt.title(f'Noise FFT Station {station_id}')
    plt.savefig(f'DeepLearning/plots/Station_{station_id}/averageNoiseFFTAllChs_Stn{station_id}.png')
    plt.close()

    return




if __name__ == "__main__":
    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
    
    parser = argparse.ArgumentParser(description='Average FFT calc on particular station')
    parser.add_argument('station', type=int, default=19, help='Station to run on')
    args = parser.parse_args()
    station_id = args.station

    station_path = f"../../../ariannaproject/station_nur/station_{station_id}/"

    DataFiles = []
    for filename in os.listdir(station_path):
        if filename.endswith('_statDatPak.root.nur'):
            continue
        else:
            DataFiles.append(os.path.join(station_path, filename))

    saveChannels = [0, 1, 2, 3]

    forced_ffts = []

    template = NuRadioRecoio.NuRadioRecoio(DataFiles)
    for i, evt in enumerate(template.get_events()):
        station = evt.get_station(station_id)
        if station.has_triggered():
            #Actual trigger, skip
            continue

        ffts = []

        channelBandPassFilter.run(evt, station, det=None, passband=[1*units.Hz, 1000*units.MHz], filter_type='butter', order=10)
        for ChId, channel in enumerate(station.iter_channels(use_channels=saveChannels)):
            trace = channel.get_trace()
            ffts.append(np.abs(fft.time2freq(trace, 2*units.GHz)))

        forced_ffts.append(ffts)

    average_per_channel, average_overall = calcAverageFFT(forced_ffts)
    plotAverageFFT(average_overall, station_id, average_per_channel)

    np.save(f'DeepLearning/data/Station{station_id}_NoiseFFT_Filtered.npy', [average_overall, average_per_channel])