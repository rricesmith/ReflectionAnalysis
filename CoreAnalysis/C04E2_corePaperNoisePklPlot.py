from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pickle
from icecream import ic


def getCoreBinWeightArray(f=0.3, dB=50):
    #Iterate through a simulation of CR distribution and get the weight of each core bin

    #Load the CR sims
    atm_overburden = 680
    with open(f"data/output_TA_19_{atm_overburden}.pkl", "rb") as fin:
        shower_energies_TA, weights_shower_energies_TA, shower_xmax_TA = pickle.load(fin)
        fin.close()
    with open(f"data/output_auger_19_{atm_overburden}.pkl", "rb") as fin:
        shower_energies_Auger, weights_shower_energies_Auger, shower_xmax_Auger = pickle.load(fin)
        fin.close()


    #Set the parameters that bin the CR simulation
    shower_E_bins = np.arange(17, 20.01, 0.1)
    logEs = 0.5 * (shower_E_bins[1:] + shower_E_bins[:-1])

    dCos = 0.05
    coszen_bin_edges = np.arange(np.cos(np.deg2rad(60)), 1.01, dCos)
    coszen_bin_edges = np.flip(coszen_bin_edges)
    coszens = 0.5 * (coszen_bin_edges[1:] + coszen_bin_edges[:-1])
    cos_edges = np.arccos(coszen_bin_edges)
    cos_edges[np.isnan(cos_edges)] = 0
    zen_bin_edges = np.rad2deg(np.nan_to_num(np.arccos(coszen_bin_edges)))

    core_E_bins = np.arange(15, 18.11, 0.1)
    core_E_bins_shift = core_E_bins - np.log10(f * 10**(-dB/20))

    core_weight_array_TA = np.zeros((len(core_E_bins-1), len(coszen_bin_edges)-1))
    core_weight_array_Auger = np.zeros_like(core_weight_array_TA)

    for iE, logE in enumerate(logEs):
        for iC, coszen in enumerate(coszens):

            ###First we will add CRs for TA flux
            niC = len(coszens) - 1 - iC
            mask = ~np.isnan(shower_energies_TA[iE][niC])
            engiEiC = np.log10(shower_energies_TA[iE][niC][mask])
            eRatesiEiC = weights_shower_energies_TA[iE][niC][mask]
            xMaxiEiC = shower_xmax_TA[iE][niC][mask]

            #Find digit the core energy corresponds to of deposited energy to get event rate from Edep
            coreDigs = np.digitize(engiEiC, core_E_bins_shift) - 1
            coreDigsMask = coreDigs >= 0
            for coreN, coreEng in enumerate(engiEiC):
                if coreDigs[coreN] < 0:
                    continue
                else:
                    core_weight_array_TA[coreDigs[coreN], iC] += eRatesiEiC[coreN]

            ###Then we add Auger CR parents
            niC = len(coszens) - 1 - iC
            mask = ~np.isnan(shower_energies_Auger[iE][niC])
            engiEiC = np.log10(shower_energies_Auger[iE][niC][mask])
            eRatesiEiC = weights_shower_energies_Auger[iE][niC][mask]
            xMaxiEiC = shower_xmax_Auger[iE][niC][mask]

            #Find digit the core energy corresponds to of deposited energy to get event rate from Edep
            coreDigs = np.digitize(engiEiC, core_E_bins_shift) - 1
            coreDigsMask = coreDigs >= 0
            for coreN, coreEng in enumerate(engiEiC):
                if coreDigs[coreN] < 0:
                    continue
                else:
                    core_weight_array_Auger[coreDigs[coreN], iC] += eRatesiEiC[coreN]

    return core_weight_array_TA, core_weight_array_Auger


class ColorSet(object):
    def __init__(self, colors=[]):
        self.colors = colors

    def __len__(self):
        return len(self.colors)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[ii] for ii in range(*idx.indices(len(self)))]
        return self.colors[idx % len(self.colors)]

def qualitative_colors(n):
    if n < 1:
        raise ValueError("Minimum number of qualitative colors is 1.")
    elif n > 12:
        raise ValueError("Maximum number of qualitative colors is 12.")
    cols = [
        "#4477AA",
        "#332288",
        "#6699CC",
        "#88CCEE",
        "#44AA99",
        "#117733",
        "#999933",
        "#DDCC77",
        "#661100",
        "#CC6677",
        "#AA4466",
        "#882255",
        "#AA4499",
    ]
    indices = [
        [0],
        [0, 9],
        [0, 7, 9],
        [0, 5, 7, 9],
        [1, 3, 5, 7, 9],
        [1, 3, 5, 7, 9, 12],
        [1, 3, 4, 5, 7, 9, 12],
        [1, 3, 4, 5, 6, 7, 9, 12],
        [1, 3, 4, 5, 6, 7, 9, 11, 12],
        [1, 3, 4, 5, 6, 7, 8, 9, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ]
    return ColorSet([cols[ix] for ix in indices[n - 1]])

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"



e_bins = np.arange(15, 18.11, 0.1)
e_center = (e_bins[1:] + e_bins[:-1])/2

dCos = 0.05
coszen_bin_edges = np.arange(np.cos(np.deg2rad(60)), 1.01, dCos)
coszen_bin_edges = np.flip(coszen_bin_edges)
coszens = 0.5 * (coszen_bin_edges[1:] + coszen_bin_edges[:-1])
cos_edges = np.arccos(coszen_bin_edges)
cos_edges[np.isnan(cos_edges)] = 0
zen_bin_edges = np.rad2deg(np.nan_to_num(np.arccos(coszen_bin_edges)))


station = 'station_1001'
trigger_antennas = [ ['LPDA_2of4_100Hz', [0, 1, 2, 3]], ['PA_8ch_100Hz', [8, 9, 10, 11]]]
layer_depths = [ 300, 500, 830 ]
dB = 50
f = 0.3

core_weight_array_TA, core_weight_array_Auger = getCoreBinWeightArray(f=f, dB=dB)


ncols = len(trigger_antennas[0])
nrows = len(layer_depths)
fig_SNR_hist_sum, ax_SNR_hist_sum = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
ax_SNR_hist_sum = np.atleast_2d(ax_SNR_hist_sum)
fig_SNR_hist_sum_unweighted, ax_SNR_hist_sum_unweighted = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
ax_SNR_hist_sum_unweighted = np.atleast_2d(ax_SNR_hist_sum_unweighted)


for iT, (trigger_name, antennas) in enumerate(trigger_antennas):
    for iD, depth in enumerate(layer_depths):


        # Load the data
        with open(f'plots/CoreAnalysis/CorePaper/diagnostics_SNR_{station}_{trigger_name}_{depth}_{dB:.1f}dB_{f}f.pkl', 'rb') as fin:
            SNR_hist_raw, SNR_hist_weights = pickle.load(fin)
            fin.close()

        step = 6
        colors = qualitative_colors(int(len(e_bins) / step)+1)
        for iE in range(0, len(e_bins) - 1, step):
            if e_bins[iE] > 16.5:
                continue

            eng_sum_SNR = []
            eng_sum_weight = []
            for j in range(step):
                if iE+j >= len(e_bins)-1:
                    j += -1
                    break
                for iC in range(len(coszen_bin_edges)-1):
                    if not np.any(SNR_hist_raw[iC, iE+j]):
                        continue
                    # ic(SNR_hist_raw[iC, iE], SNR_hist_weights[iC, iE])
                    eng_sum_SNR.extend(SNR_hist_raw[iC, iE+j].tolist())
                    # SNR_hist_weights[iC, iE+j] *= core_weight_array_TA[iE+j, iC]
                    eng_sum_weight.extend(SNR_hist_weights[iC, iE+j].tolist())
                    ic(min(SNR_hist_raw[iC, iE+j]), e_bins[iE+j])
            # ic(eng_sum_SNR, eng_sum_weight)
            # ic(iE, len(eng_sum_SNR), len(eng_sum_weight))
            bins = np.arange(0, 4, 0.1)
            # bins = np.logspace(np.log10(0.001), np.log10(100), 100)
            bins[0] = 0
            ax_SNR_hist_sum[iD, iT].hist(eng_sum_SNR, bins=bins, weights=eng_sum_weight, histtype='bar', label=f'{e_bins[iE]:.2f}-{e_bins[iE+j]:.2f}eV', density=False, stacked=True)
            ax_SNR_hist_sum_unweighted[iD, iT].hist(eng_sum_SNR, bins=bins, histtype='bar', label=f'{e_bins[iE]:.2f}-{e_bins[iE+j]:.2f}eV', density=False, stacked=True)

        ax_SNR_hist_sum[iD, iT].set_title(f'{trigger_name} {depth}m')
        ax_SNR_hist_sum[-1, iT].set_xlabel('SNR')
        ax_SNR_hist_sum[iD, iT].set_ylabel('Effective Area (km$^2$)')
        # ax_SNR_hist_sum[iD, iT].set_xscale('log')
        # ax_SNR_hist_sum[iD, iT].set_yscale('log')
        ax_SNR_hist_sum[iD, iT].legend(loc='lower right', prop={'size': 12})
        ax_SNR_hist_sum_unweighted[iD, iT].set_title(f'{trigger_name} {depth}m')
        ax_SNR_hist_sum_unweighted[-1, iT].set_xlabel('SNR')
        ax_SNR_hist_sum_unweighted[iD, iT].set_ylabel('Trigger Count')
        # ax_SNR_hist_sum_unweighted[iD, iT].set_xscale('log')
        # ax_SNR_hist_sum_unweighted[iD, iT].set_yscale('log')
        ax_SNR_hist_sum_unweighted[iD, iT].legend(loc='lower right', prop={'size': 12})
    
        


fig_SNR_hist_sum.tight_layout()
savename = f'plots/CoreAnalysis/CorePaper/diagnostics_SNR_sum_{station}_{dB}dB_{f}f.png'
fig_SNR_hist_sum.savefig(savename)
print(f'saved {savename}')

fig_SNR_hist_sum_unweighted.tight_layout()
savename = f'plots/CoreAnalysis/CorePaper/diagnostics_SNR_sum_unweighted_{station}_{dB}dB_{f}f.png'
fig_SNR_hist_sum_unweighted.savefig(savename)
print(f'saved {savename}')


