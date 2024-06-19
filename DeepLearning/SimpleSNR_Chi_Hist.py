import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from icecream import ic
from NuRadioReco.utilities import units
from NuRadioReco.utilities.io_utilities import read_pickle
import os
import templateCrossCorr as txc

def getMaxChi(traces, sampling_rate, template_trace, template_sampling_rate, parallelChannels=[[0, 2], [1, 3]]):
    #Parallel channels should be index corresponding to the channel in traces


    maxCorr = []
    for parChans in parallelChannels:
        parCorr = 0
        for chan in parChans:
            xCorr = txc.get_xcorr_for_channel(traces[chan], template_trace, sampling_rate, template_sampling_rate)
            parCorr += np.abs(xCorr)
        maxCorr.append(parCorr / len(parChans))

    return max(maxCorr)

def getMaxSNR(traces, noiseRMS=22.53 * units.mV):

    SNRs = []
    for trace in traces:
        p2p = (np.max(trace) + np.abs(np.min(trace))) * units.V
        SNRs.append(p2p / (2*noiseRMS))

    return max(SNRs)

def loadTemplate(type='RCR', amp='200s'):
    if type == 'RCR':
        if amp == '200s':
                templates_RCR = '../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/reflectedCR_template_200series.pkl'
                templates_RCR = read_pickle(templates_RCR)
                for key in templates_RCR:
                    temp = templates_RCR[key]
                templates_RCR = temp
                return templates_RCR
        elif amp == '100s':
                templates_RCR = '../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/reflectedCR_template_100series.pkl'
                templates_RCR = read_pickle(templates_RCR)
                for key in templates_RCR:
                    temp = templates_RCR[key]
                templates_RCR = temp
                return templates_RCR

    print(f'{type} {amp} not implemented')
    quit()


def plotSimSNRChi(templates_RCR, noiseRMS, amp='200s', type='RCR'):

    # path = 'DeepLearning/data/3rdpass/'
    path = '../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'
    simulation_date = '2.9.24'

    RCR_files = []
    if type == 'RCR':
        path += f'simulatedRCRs/{amp}_{simulation_date}/'
        for filename in os.listdir(path):
            if filename.startswith(f'FilteredSimRCR'):
                RCR_files.append(os.path.join(path, filename))
            if filename.startswith(f'SimWeights'):
                RCR_weights_file = os.path.join(path, filename)
    elif type == 'Backlobe':
        path += f'simulatedBacklobes/{amp}_{simulation_date}/'
        for filename in os.listdir(path):
            if filename.startswith(f'Backlobe'):
                RCR_files.append(os.path.join(path, filename))
            if filename.startswith(f'SimWeights'):
                RCR_weights_file = os.path.join(path, filename)

    ic(RCR_files, RCR_weights_file)

    for file in RCR_files:
        RCR_sim = np.load(file)
    RCR_weights = np.load(RCR_weights_file)    

    sim_SNRs = []
    sim_Chi = []
    sim_weights = []
    for iR, RCR in enumerate(RCR_sim):
            
        traces = []
        for trace in RCR:
            traces.append(trace * units.V)
        sim_SNRs.append(getMaxSNR(traces, noiseRMS=noiseRMS))
        sim_Chi.append(getMaxChi(traces, 2*units.GHz, templates_RCR, 2*units.GHz))

        sim_weights.append(RCR_weights[iR])
        # if iR % 1000 == 0:
        #     pT(traces, f'Backlobe SNR {sim_SNRs[-1]:.1f} Chi {sim_Chi[-1]:.2f}', f'DeepLearning/plots/Backlobe/SimBacklobe_SNR{sim_SNRs[-1]:.1f}_Chi{sim_Chi[-1]:.2f}_{iR}.png')

    if False:
        SNRbins = np.logspace(0.477, 2, num=80)
        maxCorrBins = np.arange(0, 1.0001, 0.01)

        plt.hist(sim_Chi, weights=sim_weights, bins=maxCorrBins, density=True)
        plt.xlabel('Chi')
        plt.savefig('DeepLearning/plots/200s_Chi_hist.png')
        plt.clf()

        plt.hist(sim_SNRs, weights=sim_weights, bins=SNRbins, density=True)
        plt.xlabel('SNR')
        plt.xscale('log')
        plt.savefig('DeepLearning/plots/200s_SNR_hist.png')
        plt.clf()

        quit()


    SNRbins = np.logspace(0.477, 2, num=80)
    maxCorrBins = np.arange(0, 1.0001, 0.01)

    print(f'len sim snr {len(sim_SNRs)} and chi {len(sim_Chi)}')
    # print(f'weights {sim_weights}')

    sim_weights = np.array(sim_weights)
    sim_SNRs = np.array(sim_SNRs)
    sim_Chi = np.array(sim_Chi)

    sort_order = sim_weights.argsort()
#    sim_SNRs = sim_SNRs[sort_order[::-1]]
#    sim_Chi = sim_Chi[sort_order[::-1]]
#    sim_weights = sim_weights[sort_order[::-1]]
    sim_SNRs = sim_SNRs[sort_order]
    sim_Chi = sim_Chi[sort_order]
    sim_weights = sim_weights[sort_order]

    if type == 'RCR':
        cmap = 'seismic'
    else:
        cmap = 'PiYG'
    # plt.scatter(sim_SNRs, sim_Chi, c=sim_weights, label=f'Simulated {type}', cmap=cmap, alpha=0.9, norm=matplotlib.colors.LogNorm())
    plt.scatter(sim_SNRs, sim_Chi, c=sim_weights, cmap=cmap, alpha=0.9, norm=matplotlib.colors.LogNorm())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Analysis on particular station')
    parser.add_argument('station', type=int, default=19, help='Station to run on')
    args = parser.parse_args()
    station_id = args.station

    data = np.load(f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/Station{station_id}_SNR_Chi.npy', allow_pickle=True)
    plot_folder = f'plots/DeepLearning/simplePlots/Station_{station_id}'
    Path(plot_folder).mkdir(parents=True, exist_ok=True)


    if station_id in [14, 17, 19, 30]:
        amp_type = '200s'
        noiseRMS = 22.53 * units.mV
    elif station_id in [13, 15, 18]:
        amp_type = '100s'
        noiseRMS = None #Need to add
    templates_RCR = loadTemplate(type='RCR', amp=amp_type)


    All_SNRs = data[0]
    All_RCR_Chi = data[1]
    All_Azi = data[2]
    All_Zen = data[3]
    PassingCut_SNRs = data[4]
    PassingCut_RCR_Chi = data[5]
    PassingCut_Azi = data[6]
    PassingCut_Zen = data[7]


    SNRbins = np.logspace(0.477, 2, num=80)
    maxCorrBins = np.arange(0, 1.0001, 0.01)

    #Plot of all events in Chi-SNR space
    plt.hist2d(All_SNRs, All_RCR_Chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.xlim((3, 100))
    plt.ylim((0, 1))
    plt.xlabel('SNR')
    plt.ylabel('Avg Chi Highest Parallel Channels')
    # plt.legend()
    plt.xscale('log')
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.grid(visible=True, which='both', axis='both')
    plt.title(f'Station {station_id}')
    print(f'Saving {plot_folder}/ChiSNR_All_Stnd{station_id}.png')
    plt.savefig(f'{plot_folder}/ChiSNR_All_Stnd{station_id}.png')


    #Plot of sim overlayed on top of all events
    plotSimSNRChi(templates_RCR, noiseRMS)
    plt.scatter([], [], color='red', label='Simulated Air Showers')
    plt.legend()
    print(f'Saving {plot_folder}/ChiSNR_wSim_Stnd{station_id}.png')
    plt.savefig(f'{plot_folder}/ChiSNR_wSim_Stnd{station_id}.png')

    #Plot of station & sim, with events passing cuts circled
    plt.scatter(PassingCut_SNRs, PassingCut_RCR_Chi, label=f'{len(PassingCut_RCR_Chi)} Events Passing Cuts', facecolor='none', edgecolor='black')
    plt.xlim((3, 100))
    plt.ylim((0, 1))
    plt.xlabel('SNR')
    plt.ylabel('Avg Chi Highest Parallel Channels')
    plt.legend()
    plt.xscale('log')
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.grid(visible=True, which='both', axis='both')
    plt.title(f'Station {station_id} RCR SNR-Chi')
    print(f'Saving {plot_folder}/ChiSNR_PassedCuts_Stnd{station_id}.png')
    plt.savefig(f'{plot_folder}/ChiSNR_PassedCuts_Stnd{station_id}.png')
    plt.clf()
    plt.close()

    # Redoing above but adding simulated backlobes on top of simulated air showers
    plt.hist2d(All_SNRs, All_RCR_Chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
    plotSimSNRChi(templates_RCR, noiseRMS)
    plotSimSNRChi(templates_RCR, noiseRMS, type='Backlobe')    
    plt.scatter([], [], color='red', label='Simulated Air Showers')
    plt.scatter([], [], color='green', label='Simulated Backlobe')
    plt.scatter(PassingCut_SNRs, PassingCut_RCR_Chi, label=f'{len(PassingCut_RCR_Chi)} Events Passing Cuts', facecolor='none', edgecolor='black')

    plt.colorbar()
    plt.xlim((3, 100))
    plt.ylim((0, 1))
    plt.xlabel('SNR')
    plt.ylabel('Avg Chi Highest Parallel Channels')
    plt.xscale('log')
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.grid(visible=True, which='both', axis='both')
    plt.title(f'Station {station_id}')
    plt.legend()
    print(f'Saving {plot_folder}/ChiSNR_wBacklobe_PassedCuts_Stnd{station_id}.png')
    plt.savefig(f'{plot_folder}/ChiSNR_wBacklobe_PassedCuts_Stnd{station_id}.png')
    plt.clf()
    plt.close()

    print(f'Done!')