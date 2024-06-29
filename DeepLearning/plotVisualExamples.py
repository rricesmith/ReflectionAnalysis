import os
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units, fft


def plot_trace(traces, title, saveLoc, sampling_rate=2, show=False, average_fft_per_channel=[]):
    #Sampling rate should be in GHz
    print(f'printing')
    x = np.linspace(1, int(256 / sampling_rate), num=256)
    x_freq = np.fft.rfftfreq(len(x), d=(1 / sampling_rate*units.GHz)) / units.MHz

    """ #Method for plotting a single plot
    #fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False)
    plt.plot(x, traces[0]*100, color='orange')
    plt.plot(x, traces[1]*100, color='blue')
    plt.plot(x, traces[2]*100, color='purple')
    plt.plot(x, traces[3]*100, color='green')
    plt.xlabel('time [ns]',fontsize=18)
    plt.ylabel('Amplitude (mV)')
    plt.xlim(-3,260 / sampling_rate)
    plt.title(title)
    plt.savefig(saveLoc + '_Traces.png', format='png')
    plt.clf()
    plt.close()


    freqs = []
    for trace in traces:
        freqtrace = np.abs(fft.time2freq(trace, sampling_rate*units.GHz))
        freqs.append(freqtrace)
    plt.plot(x_freq/1000, freqs[0], color='orange', label='Channel 0')
    plt.plot(x_freq/1000, freqs[1], color='blue', label='Channel 1')
    plt.plot(x_freq/1000, freqs[2], color='purple', label='Channel 2')
    plt.plot(x_freq/1000, freqs[3], color='green', label='Channel 3')
    plt.xlabel('Frequency [GHz]',fontsize=18)
    plt.ylabel('Amplitude')
#    axs[0][1].set_ylabel('Amplitude')
    plt.xlim(-0.003, 1.050)
    plt.xticks(size=13)
    plt.title(title)
    plt.savefig(saveLoc + '_Freqs.png', format='png')
#    plt.savefig(f'DeepLearning/plots/Station_19/GoldenDay/NuSearchFreqs_{title}.png', format='png')
    plt.clf()
    plt.close()
    return
    """

#    print(f'shape traces {np.shape(traces)}')

    fig, axs = plt.subplots(nrows=4, ncols=2, sharex=False)
    fmax = 0
    vmax = 0
    for chID, trace in enumerate(traces):
        trace = trace.reshape(len(trace))
        freqtrace = np.abs(fft.time2freq(trace, sampling_rate*units.GHz))
        axs[chID][0].plot(x, trace)
#        print(f'shape trace {np.shape(trace)}')
#        print(f'shape fft trace {np.shape(np.abs(fft.time2freq(trace, sampling_rate*units.GHz)))}')
#        print(f'trace {trace}')
#        print(f'fft {np.abs(fft.time2freq(trace, sampling_rate*units.GHz))}')
        if len(average_fft_per_channel) > 0:
            axs[chID][1].plot(x_freq, average_fft_per_channel[chID], color='gray', linestyle='--')
        axs[chID][1].plot(x_freq, freqtrace)
        fmax = max(fmax, max(freqtrace))
        vmax = max(vmax, max(trace))

    axs[3][0].set_xlabel('time [ns]',fontsize=18)
    axs[3][1].set_xlabel('Frequency [MHz]',fontsize=18)

    for chID, trace in enumerate(traces):
        axs[chID][0].set_ylabel(f'ch{chID}',labelpad=10,rotation=0,fontsize=13)
        # axs[i].set_ylim(-250,250)
        axs[chID][0].set_xlim(-3,260 / sampling_rate)
        axs[chID][1].set_xlim(-3, 1000)
        axs[chID][0].tick_params(labelsize=13)
        axs[chID][1].tick_params(labelsize=13)

        axs[chID][0].set_ylim(-vmax * 1.1, vmax * 1.1)
        axs[chID][1].set_ylim(-0.05, fmax * 1.1)

    axs[0][0].tick_params(labelsize=13)
    axs[0][1].tick_params(labelsize=13)
    axs[0][0].set_ylabel(f'ch{0}',labelpad=3,rotation=0,fontsize=13)
    axs[chID][0].set_xlim(-3,260 / sampling_rate)
    axs[chID][1].set_xlim(-3, 1000)

    fig.text(0.03, 0.5, 'voltage [V]', ha='center', va='center', rotation='vertical',fontsize=18)
    plt.xticks(size=13)
    plt.suptitle(title)

    if show:
        plt.show()
    else:
        print(f'saving to {saveLoc}')
        plt.savefig(saveLoc, format='png')
    plt.clf()
    plt.close()

    return



def get_index_max_weights(weights_file, num=50):
    weights = np.load(weights_file)

#    top_idx = np.argsort(weights)[-num:]
    top_idx = np.argsort(weights)

    ind = 1
    weights_used = np.zeros(num)
    non_repeat_idx = np.zeros(num, dtype=int)
    weights_used[0] = weights[top_idx[-1]]
    non_repeat_idx[0] = int(top_idx[-1])

    for id in reversed(top_idx):
        weight = weights[id]

        if not weight == weights_used[ind-1]:
#            print(f'new weight {weight} not in used {weights_used}, addind id {id}')
            weights_used[ind] = weight
            non_repeat_idx[ind] = int(id)
            ind += 1
            if ind == num:
#                print(f'done')
#                print(f'ids {non_repeat_idx}')
#                quit()
                return non_repeat_idx
                break        

    return non_repeat_idx

"""    for id in top_idx:
        print(f'id {id}')
        print(f'weight for id {weights[id]}')
    quit()
    return top_idx
"""
def get_max_weight_traces(traces_file, weights_file, type, amp, num=50):
    top_idx = get_index_max_weights(weights_file, num)

    traces = np.load(traces_file)
    
    for id in top_idx:
        id = int(id)
        trace = traces[id]

        print(f'shape trace {np.shape(trace)}')

        vtrace = []
        for ch in trace:
            print(f'shape ch {np.shape(ch)}')
            vtrace.append(ch * units.V)
            print(f'shape vtrace {np.shape(vtrace)}')
            
        print(f'shape vtrace {np.shape(vtrace)}')
        plot_trace(vtrace, f'High Weight {type} {amp}', saveLoc=f'DeepLearning/plots/{type}_{amp}_{id}.png')

    return

def get_subindex(list, indeces):
    return [list[index] for index in indeces]

def plot_hists(weights_backlobe_file, params_backlobe_file, weights_RCR_file, params_RCR_file):
    #Make a separate histogram of energy zeniths, azimuths of top events in RCR and Backlobe
    #As well as a weighted histogram of all events
    #Then make a combined weighted hist of both for energy/zenith/azimuth
    weights_BL = np.load(weights_backlobe_file)
    weights_RCR = np.load(weights_RCR_file)
    params_BL = np.load(params_backlobe_file)
    params_RCR = np.load(params_RCR_file)

    energy_RCR = [params[0] for params in params_RCR]
    zenith_RCR = [params[1] for params in params_RCR]
    azimuth_RCR = [params[2] for params in params_RCR]
    energy_BL = [params[0] for params in params_BL]
    zenith_BL = [params[1] for params in params_BL]
    azimuth_BL = [params[2] for params in params_BL]

    max_weights_ind_BL = get_index_max_weights(weights_backlobe_file)
    max_weights_ind_RCR = get_index_max_weights(weights_RCR_file)

    #Energy hists
    bins = np.linspace(17, 20, 30)
    plt.hist(get_subindex(energy_RCR,max_weights_ind_RCR), weights=get_subindex(weights_RCR,max_weights_ind_RCR), label='Top 50', density=True, bins=bins)
    plt.hist(energy_RCR, weights=weights_RCR, density=True, label='All Weighted', bins=bins)
    plt.legend()
    plt.xlabel('Energy (log10eV)')
    plt.savefig(f'DeepLearning/plots/RCR_Backlobe/Energy_RCR_hist.png')
    plt.clf()

    plt.hist(get_subindex(energy_BL,max_weights_ind_BL), weights=get_subindex(weights_BL,max_weights_ind_BL), label='Top 50', density=True, bins=bins)
    plt.hist(energy_BL, weights=weights_BL, density=True, label='All Weighted', bins=bins)
    plt.legend()
    plt.xlabel('Energy (log10eV)')
    plt.savefig(f'DeepLearning/plots/RCR_Backlobe/Energy_BL_hist.png')
    plt.clf()

    plt.hist(energy_RCR, weights=weights_RCR, density=True, label='RCR', bins=bins)
    plt.hist(energy_BL, weights=weights_BL, density=True, label='BL', bins=bins)
    plt.legend()
    plt.xlabel('Energy (log10eV)')
    plt.savefig(f'DeepLearning/plots/RCR_Backlobe/Energy_BL&RCR_hist.png')
    plt.clf()

    plt.hist(energy_BL, weights=weights_BL, density=False, label='BL', bins=bins)
    plt.hist(energy_RCR, weights=weights_RCR, density=False, label='RCR', bins=bins)
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Energy (log10eV)')
    plt.savefig(f'DeepLearning/plots/RCR_Backlobe/Energy_Nondensity_BL&RCR_hist.png')
    plt.clf()


    #Zenith hists
    bins = np.linspace(0, 90, 30)
    RCR_zen_mean = np.average(zenith_RCR, weights=weights_RCR)
    RCR_zen_std = np.sqrt(np.average( (zenith_RCR - RCR_zen_mean)**2, weights=weights_RCR))
    BL_zen_mean = np.average(zenith_BL, weights=weights_BL)
    BL_zen_std = np.sqrt(np.average( (zenith_BL - BL_zen_mean)**2, weights=weights_BL))

    plt.hist(get_subindex(zenith_RCR,max_weights_ind_RCR), weights=get_subindex(weights_RCR,max_weights_ind_RCR), label='Top 50', density=True, bins=bins, alpha=0.5)
    plt.hist(zenith_RCR, weights=weights_RCR, density=True, label=f'All Weighted, Mean {RCR_zen_mean:.0f}±{RCR_zen_std:.1f}deg', bins=bins, alpha=0.5)
    plt.legend()
    plt.xlabel('Zenith (deg)')
    plt.savefig(f'DeepLearning/plots/RCR_Backlobe/Zenith_RCR_hist.png')
    plt.clf()

    plt.hist(get_subindex(zenith_BL,max_weights_ind_BL), weights=get_subindex(weights_BL,max_weights_ind_BL), label='Top 50', density=True, bins=bins, alpha=0.5)
    plt.hist(zenith_BL, weights=weights_BL, density=True, label=f'All Weighted, Mean {BL_zen_mean:.0f}±{BL_zen_std:.1f}deg', bins=bins, alpha=0.5)
    plt.legend()
    plt.xlabel('Zenith (deg)')
    plt.savefig(f'DeepLearning/plots/RCR_Backlobe/Zenith_BL_hist.png')
    plt.clf()

    plt.hist(zenith_RCR, weights=weights_RCR, density=True, label=f'RCR, Mean {RCR_zen_mean:.0f}±{RCR_zen_std:.1f}deg', bins=bins, alpha=0.5)
    plt.hist(zenith_BL, weights=weights_BL, density=True, label=f'BL, Mean {BL_zen_mean:.0f}±{BL_zen_std:.1f}deg', bins=bins, alpha=0.5)
    plt.legend()
    plt.xlabel('Zenith (deg)')
    plt.savefig(f'DeepLearning/plots/RCR_Backlobe/Zenith_BL&RCR_hist.png')
    plt.clf()


    #Azimuth hists
    bins = np.linspace(0, 360, 60)
    plt.hist(get_subindex(azimuth_RCR,max_weights_ind_RCR), weights=get_subindex(weights_RCR,max_weights_ind_RCR), label='Top 50', density=True, bins=bins)
    plt.hist(azimuth_RCR, weights=weights_RCR, density=True, label='All Weighted', bins=bins)
    plt.legend()
    plt.xlabel('Azimuth (deg)')
    plt.savefig(f'DeepLearning/plots/RCR_Backlobe/Azimuth_RCR_hist.png')
    plt.clf()

    plt.hist(get_subindex(azimuth_BL,max_weights_ind_BL), weights=get_subindex(weights_BL,max_weights_ind_BL), label='Top 50', density=True, bins=bins)
    plt.hist(azimuth_BL, weights=weights_BL, density=True, label='All Weighted', bins=bins)
    plt.legend()
    plt.xlabel('Azimuth (deg)')
    plt.savefig(f'DeepLearning/plots/RCR_Backlobe/Azimuth_BL_hist.png')
    plt.clf()

    plt.hist(azimuth_RCR, weights=weights_RCR, density=True, label='RCR', bins=bins)
    plt.hist(azimuth_BL, weights=weights_BL, density=True, label='BL', bins=bins)
    plt.legend()
    plt.xlabel('Azimuth (deg)')
    plt.savefig(f'DeepLearning/plots/RCR_Backlobe/Azimuth_BL&RCR_hist.png')
    plt.clf()
    plt.close()

    return

amps = ['100s', '200s']
#amps = ['200s']
order = '4th'
path = f'DeepLearning/data/{order}pass/'

for amp in amps:
    #RCR files
    RCR_files = []
    for filename in os.listdir(path):
        if filename.startswith(f'SimRCR_{amp}'):
            RCR_files.append(os.path.join(path, filename))
            RCR_weights_file = f'DeepLearning/data/{order}pass/SimWeights_{filename}'
            RCR_params_file = f'DeepLearning/data/{order}pass/SimParams_{filename}'


    Backlobe_files = []
    for filename in os.listdir(path):
        if filename.startswith(f'Backlobe_{amp}'):
#        if filename.startswith(f'Backlobe_{amp}_Noiseless'):
            Backlobe_files.append(os.path.join(path, filename))
            Backlobe_weights_file = f'DeepLearning/data/{order}pass/SimWeights_{filename}'
            Backlobe_params_file = f'DeepLearning/data/{order}pass/SimParams_{filename}'


    get_max_weight_traces(RCR_files[0], RCR_weights_file, type=f'2.24_Sims/RCR/{amp}/RCR', amp=amp, num=100)
    get_max_weight_traces(Backlobe_files[0], Backlobe_weights_file, type=f'2.24_Sims/Backlobe/{amp}/Backlobe', amp=amp, num=100)
#    get_max_weight_traces(Backlobe_files[0], Backlobe_weights_file, type='Backlobe_Noiseless', amp=amp, num=50)


    plot_hists(Backlobe_weights_file, Backlobe_params_file, RCR_weights_file, RCR_params_file)


