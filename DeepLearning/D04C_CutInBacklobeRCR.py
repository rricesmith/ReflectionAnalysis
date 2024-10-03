import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from DeepLearning.D04B_reprocessNurPassingCut import plotSimSNRChi
import DeepLearning.D00_helperFunctions as D00_helperFunctions
from NuRadioReco.utilities import units

def RCRChiSNRCut(SNR):
    # Function that returns the Chi for given SNR of the cut line
    if SNR < 6:
        return 0.65
    elif SNR < 20:
        return 0.76 - 0.00004 * (20-SNR)**3
    else:
        return 0.76
    
def RCRChiSNRCutMask(SNR_array, Chi_array):
    # Function that returns a boolean array checking if each Chi value passes the cut at its corresponding SNR
    mask = np.zeros(len(SNR_array), dtype=bool)
    for i in range(len(SNR_array)):
        mask[i] = Chi_array[i] > RCRChiSNRCut(SNR_array[i])
    return mask 

def RCRChiSNRCutEfficiency(SNR_array, Chi_array, weights=None):
    # Calculated how many events of total pass the cut
    if weights == None:
        cut_mask = RCRChiSNRCutMask(SNR_array, Chi_array)
        return np.sum(cut_mask) / len(SNR_array)
    else:
        cut_mask = RCRChiSNRCutMask(SNR_array, Chi_array)
        return np.sum(weights[cut_mask]) / np.sum(weights)


def plotRCRChiSNRCut(ax=None, label=True):
    SNR = np.logspace(np.log10(3), np.log10(100), 100)
    Chi = []
    for snr in SNR:
        Chi.append(RCRChiSNRCut(snr))
    if ax is None:
        if label:
            plt.plot(SNR, Chi, label='RCR Chi-SNR cut', color='black', linestyle='--')
        else:
            plt.plot(SNR, Chi, color='black', linestyle='--')
    else:
        if label:
            ax.plot(SNR, Chi, label='RCR Chi-SNR cut', color='black', linestyle='--')
        else:
            ax.plot(SNR, Chi, color='black', linestyle='--')

    return 

if __name__ == '__main__':

    templates_RCR = D00_helperFunctions.loadSingleTemplate('200')
    template_series_RCR = D00_helperFunctions.loadMultipleTemplates('200', date='9.16.24')
    # template_series_RCR.append(templates_RCR)
    noiseRMS = 22.53 * units.mV

    fig, ax = plt.subplots()
    plotSimSNRChi(template_series_RCR, noiseRMS, ax=ax, cut=True)



    plotRCRChiSNRCut(ax)
    ax.set_xlim((3, 100))
    ax.set_ylim((0, 1))
    ax.set_xscale('log')
    ax.set_xlabel('SNR')
    ax.set_ylabel('Chi')
    ax.legend()
    savename = 'DeepLearning/plots/RCRChiSNRCut.png'
    fig.savefig(savename)
    print(f'Saved {savename}')

    plotSimSNRChi(template_series_RCR, noiseRMS, type='Backlobe', ax=ax, cut=True)
    plotRCRChiSNRCut(ax)
    ax.legend()
    savename = 'DeepLearning/plots/RCRChiSNRCut_wBacklobe.png'
    fig.savefig(savename)
    print(f'Saved {savename}')