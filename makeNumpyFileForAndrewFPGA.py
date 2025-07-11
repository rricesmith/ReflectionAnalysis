from NuRadioReco.modules.io import NuRadioRecoio
from icecream import ic
from NuRadioReco.modules.channelLengthAdjuster import channelLengthAdjuster
import numpy as np

# file = 'NeutrinoAnalysis/output/MJob/400/SP/MJob_SP_Allsigma_1e+20_n10000.0.nur'
# savename = 'data/AndrewFPGA_Neutrino.npy'
# file = 'NeutrinoAnalysis/output/MJob/400/SP/MJob_SP_Allsigma_1e+17_n1000000.0_part0001.nur'

# Note this file is a bit wacky. The station 62 config online is not the same as Steve said it would be
# Also I had to make some changes to the config that I'm not sure matched it
# The downward facing LPDAs should be correct, but maybe not the others
# Should be fine, but if need new ones, check configurations/station61.json before running NeutrinoAnalysis/M02a_SubmitJob.py again
# file = 'NeutrinoAnalysis/output/MJob/300/SP/MJob_SP_Allsigma_1e+19_n10000.0_EventForAndrew.nur' 
# file = 'NeutrinoAnalysis/output/MJob/300/SP/MJob_SP_Allsigma_4.281332398719396e+17_n1000000.0_EventForAndrew_part0018.nur'
file = 'NeutrinoAnalysis/output/MJob/300/SP/MJob_SP_Allsigma_4.281332398719396e+17_n1000000.0_EventForAndrewRedo_part0017.nur'
savename = 'data/AndrewFPGA_300s_Noise_1e17.npy'

template = NuRadioRecoio.NuRadioRecoio(file)

channelLengthAdjuster = channelLengthAdjuster()
channelLengthAdjuster.begin()

max = 100
saveTrace = np.zeros((max, 8, 256))

n=0
for i, evt in enumerate(template.get_events()):
    station = evt.get_station(61)
    if not station.has_triggered('LPDA_2of4_2sigma'):
        continue
    channelLengthAdjuster.run(evt, station)
    for ChID, channel in enumerate(station.iter_channels()):
        ic(ChID)
        trace = channel.get_trace()
        ic(len(trace))
        ic(channel.get_sampling_rate())
        # if not ChID == 4:
        #     saveTrace[i][ChID] = trace
        #     saveTrace[i][ChID+4] = trace * 0.1
        # else:
        #     saveTrace[i][-1] = trace
        saveTrace[n][ChID] = trace
    n += 1
    if n >= max:
        break
if n == 0:
    ic('No events found')
    exit(0)
ic(saveTrace)
ic(saveTrace.shape)


if True:
    for n in range(100):
        if not np.any(saveTrace[n] > 0):
            continue
        # Plot the noise
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(8, 1, figsize=(20, 10), sharex=True, sharey=True)
        for ch in range(8):
            ax[ch].plot(saveTrace[n][ch])
            ax[ch].set_title(f'Channel {ch}')
            ax[ch].set_ylabel('Amplitude (V)')
        ax[-1].set_xlabel('time (ns)')
        fig.suptitle('300s sample trace')
        plt.grid()
        plt.savefig(f'SimpleFootprintSimulation/plots/300s_trace_FPGA_Andrew_REDO_{n}.png')
        print(f'Saved {n}')
        plt.close()



np.save(savename, saveTrace)
ic(f'Saved {savename}')

