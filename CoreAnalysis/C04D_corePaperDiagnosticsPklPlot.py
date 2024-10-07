import pickle
import numpy as np
import pickle
import matplotlib.pyplot as plt
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

def GetFigAndAx(nrows, ncols):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8 * 0.7, nrows * 5 * 0.7), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
    return fig, ax

def SetPlotAxis(ax, minor_ticks=0, y_minor_ticks=0):
    from matplotlib.ticker import MultipleLocator

    ax.tick_params(axis="both", which="both", direction="in")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")
    if minor_ticks:
        ax.xaxis.set_minor_locator(MultipleLocator(minor_ticks))
    if y_minor_ticks:
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor_ticks))


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


trigger_antennas = [ ['LPDA_2of4_100Hz', [0, 1, 2, 3]], ['PA_8ch_100Hz', [8, 9, 10, 11]]]
layer_depths = [ 300, 500, 830 ]
# layer_depths = [830]
station = 'station_1001'
dB = 50.0
f = 0.3

core_weight_array_TA, core_weight_array_Auger = getCoreBinWeightArray(f=f, dB=dB)
# core_weight_array_TA_flat, core_weight_array_Auger_flat = getCoreBinWeightArray(f=0.3, dB=40)
# e_bins = np.arange(15, 18.11, 0.1)
# dCos = 0.05
# coszen_bin_edges = np.arange(np.cos(np.deg2rad(60)), 1.01, dCos)
# coszen_bin_edges = np.flip(coszen_bin_edges)
# for iT, (trigger_name, antennas) in enumerate(trigger_antennas):
#     for iD, depth in enumerate(layer_depths):

#         with open(f'plots/CoreAnalysis/CorePaper/diagnostics_viewing_angle_perbin_{station}_{trigger_name}_{depth}_{dB}dB_{f}f.pkl', 'rb') as fin:
#             viewing_hist_raw, viewing_hist_weights = pickle.load(fin)
#             fin.close()

#         for iE in range(len(e_bins)-1):
#             for iC in range(len(coszen_bin_edges)-1):
#                     ic(len(viewing_hist_raw[iC, iE]))
#                     ic(core_weight_array_TA[iE, iC],  viewing_hist_weights[iC, iE], viewing_hist_weights[iC, iE]/(core_weight_array_TA[iE, iC]**(len(viewing_hist_raw[iC, iE]))))
#                     # viewing_hist_weights[iC, iE] *= 1/(core_weight_array_TA[iE, iC]*(len(viewing_hist_raw[iC, iE])))
#                     # plot_weights.append(viewing_hist_weights[iC, iE+j])
#         quit()

#         with open(f'plots/CoreAnalysis/CorePaper/diagnostics_viewing_angle_sum_{station}_{trigger_name}_{depth}_{dB}dB_{f}f.pkl', 'rb') as fin:
#             all_viewing_angles, all_viewing_angles_weights = pickle.load(fin)
#             fin.close()

# quit()



colors = qualitative_colors(len(layer_depths))

fig_viewing_hist_sum, ax_viewing_hist_sum = plt.subplots(ncols=2, nrows=1, figsize=(16, 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
# fig_viewing_hist_sum, ax_viewing_hist_sum = GetFigAndAx(1, 2)

bins=np.arange(-20, 20.01, 1) 
for iT, (trigger_name, antennas) in enumerate(trigger_antennas):
    for iD, depth in enumerate(layer_depths):

        # Load the viewing angle data
        with open(f'plots/CoreAnalysis/CorePaper/diagnostics_viewing_angle_sum_{station}_{trigger_name}_{depth}_{dB}dB_{f}f.pkl', 'rb') as fin:
            all_viewing_angles, all_viewing_angles_weights = pickle.load(fin)
            fin.close()

        # ic(len(all_viewing_angles), len(all_viewing_angles_weights))
        # continue

        # Plot the viewing angle histogram
        ax_viewing_hist_sum[iT].hist(all_viewing_angles, bins=bins, weights=all_viewing_angles_weights, histtype='step', label=f'depth {depth}', color=colors[iD], density=True)

    # ax_viewing_hist_sum[iT].set_title(f'{trigger_name}')
    ax_viewing_hist_sum[iT].set_xlabel('Viewing - Cherenkov angle (deg)')
    ax_viewing_hist_sum[iT].legend(fontsize=12)
    ax_viewing_hist_sum[iT].set_xlim(-20, 20)
    ax_viewing_hist_sum[iT].grid(b=None)
    ax_viewing_hist_sum[iT].minorticks_on()
    ax_viewing_hist_sum[iT].tick_params(axis='x', which='minor', bottom=True, top=True)
    SetPlotAxis(ax_viewing_hist_sum[iT])
    ax_viewing_hist_sum[iT].set_ylabel('Normalized Events')

    xmin, xmax = ax_viewing_hist_sum[iT].get_xlim()
    ymin, ymax = ax_viewing_hist_sum[iT].get_ylim()
    ymax *= 1.0
    ax_viewing_hist_sum[iT].set_ylim(ymin, ymax)
    x_text = 0.03 * (xmax - xmin) + xmin
    y_text = 0.97 * (ymax - ymin) + ymin
    msg = f"reflected\nf: {f}\nReflectivity: -{dB:.0f}dB\n{trigger_name.replace('_', ' ')}"
    ax_viewing_hist_sum[iT].text(x_text, y_text, msg, verticalalignment="top", horizontalalignment="left", fontsize=12)


fig_viewing_hist_sum.savefig(f'plots/CoreAnalysis/CorePaper/viewing_angles/diagnostics_viewing_angle_sum_{station}_{dB}dB_{f}f.pdf')
ic(f'saved plots/CoreAnalysis/CorePaper/diagnostics_viewing_angle_sum_{station}_{dB}dB_{f}f.pdf')
plt.clf()


# Do the same but for the per-energy-bin histograms

bins=np.arange(-20, 20.01, 1)

# Bins for the energy and cos(zenith) of simulations
e_bins = np.arange(15, 18.11, 0.1)
e_center = (e_bins[1:] + e_bins[:-1])/2
dCos = 0.05
coszen_bin_edges = np.arange(np.cos(np.deg2rad(60)), 1.01, dCos)
coszen_bin_edges = np.flip(coszen_bin_edges)
coszens = 0.5 * (coszen_bin_edges[1:] + coszen_bin_edges[:-1])
cos_edges = np.arccos(coszen_bin_edges)
cos_edges[np.isnan(cos_edges)] = 0
zen_bin_edges = np.rad2deg(np.nan_to_num(np.arccos(coszen_bin_edges)))


for iD, depth in enumerate(layer_depths):
    fig_viewing_hist, ax_viewing_hist = plt.subplots(ncols=2, nrows=1, figsize=(16, 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
    for iT, (trigger_name, antennas) in enumerate(trigger_antennas):

        # Load the viewing angle data
        with open(f'plots/CoreAnalysis/CorePaper/diagnostics_viewing_angle_perbin_{station}_{trigger_name}_{depth}_{dB}dB_{f}f.pkl', 'rb') as fin:
            viewing_hist_raw, viewing_hist_weights = pickle.load(fin)
            fin.close()

        # colors = qualitative_colors(len(e_bins)-1)
        step = 6
        for iE in range(0, len(e_bins) - 1, step):
            plot_angles = []
            plot_weights = []
            for iC in range(len(coszen_bin_edges)-1):
                # ic(iC, iE, viewing_hist_raw.shape, viewing_hist_weights.shape)
                # plot_angles.extend(viewing_hist_raw[iC, iE:iE].tolist())
                for j in range(step):
                    if iE+j >= len(e_bins)-1:
                        j += -1
                        break
                    plot_angles.extend(viewing_hist_raw[iC, iE+j].tolist())

                    # Because data was saved at f=0.3, -50dB, rescaling it to f=1 and 0dB for cleaner plots
                    # append = viewing_hist_weights[iC, iE+j] / core_weight_array_TA[iE+j, iC]
                    # append *= core_weight_array_TA_flat[iE+j, iC]

                    for i in range(len(viewing_hist_raw[iC, iE+j])):
                    # viewing_hist_weights[iC, iE] *= core_weight_array_TA[iE, iC]
                        plot_weights.append(viewing_hist_weights[iC, iE+j])
                        # plot_weights.append(append)
            # Plot the viewing angle histogram
            ax_viewing_hist[iT].hist(plot_angles, bins=bins, weights=plot_weights, histtype='bar', label=f'{e_bins[iE]:.2f}-{e_bins[iE+j]:.2f}eV', density=False, stacked=True)
            # ax_viewing_hist[iT].hist(plot_angles, bins=bins, weights=plot_weights, histtype='bar', label=f'{e_bins[iE]:.2f}-{e_bins[iE+j]:.2f}eV', density=True, stacked=True)

        ax_viewing_hist[iT].semilogy(True)
        # ax_viewing_hist[iT].set_title(f'{trigger_name}')
        ax_viewing_hist[iT].set_xlabel('Viewing - Cherenkov angle (deg)')
        ax_viewing_hist[iT].legend(loc='upper right',fontsize=12)
        ax_viewing_hist[iT].set_xlim(-20, 20)
        ax_viewing_hist[iT].grid(b=None)
        ax_viewing_hist[iT].minorticks_on()
        ax_viewing_hist[iT].tick_params(axis='x', which='minor', bottom=True, top=True)
        SetPlotAxis(ax_viewing_hist[iT])
        # ax_viewing_hist[iT].set_ylabel('Normalized Events')
        ax_viewing_hist[iT].set_ylabel('Events/Stn/Yr')

        xmin, xmax = ax_viewing_hist[iT].get_xlim()
        ymin, ymax = ax_viewing_hist[iT].get_ylim()
        ymax *= 1.0
        ax_viewing_hist[iT].set_ylim(ymin, ymax)
        x_text = 0.03 * (xmax - xmin) + xmin
        y_text = 0.97 * (ymax - ymin) + ymin
        msg = f"reflected {depth} depth\nf: {f}\nReflectivity: -{dB:.0f}dB\n{trigger_name.replace('_', ' ')}"
        ax_viewing_hist[iT].text(x_text, y_text, msg, verticalalignment="top", horizontalalignment="left", fontsize=12)


    save_name = f'plots/CoreAnalysis/CorePaper/viewing_angles/diagnostics_viewing_angle_perbin_{station}_{depth}_{dB}dB_{f}f.pdf'
    fig_viewing_hist.savefig(save_name)
    ic(f'saved {save_name}')
    plt.clf()


fig_viewing_sum_hist, ax_viewing_sum_hist = plt.subplots(ncols=2, nrows=1, figsize=(16, 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
for iT, (trigger_name, antennas) in enumerate(trigger_antennas):
    colors = qualitative_colors(len(layer_depths))
    for iD, depth in enumerate(layer_depths):
        # Load the viewing angle data
        with open(f'plots/CoreAnalysis/CorePaper/diagnostics_viewing_angle_perbin_{station}_{trigger_name}_{depth}_{dB}dB_{f}f.pkl', 'rb') as fin:
            viewing_hist_raw, viewing_hist_weights = pickle.load(fin)
            fin.close()

        sum_plot_angles = []
        sum_plot_weights = []
        for iE in range(len(e_bins)-1):
            for iC in range(len(coszen_bin_edges)-1):
                sum_plot_angles.extend(viewing_hist_raw[iC, iE].tolist())
                for i in range(len(viewing_hist_raw[iC, iE+j])):
                    sum_plot_weights.append(viewing_hist_weights[iC, iE])

        # Plot the viewing angle histogram
        ax_viewing_sum_hist[iT].hist(sum_plot_angles, bins=bins, weights=sum_plot_weights, histtype='step', label=f'depth {depth}', color=colors[iD], density=True)

    # ax_viewing_sum_hist[iT].set_title(f'{trigger_name}')
    ax_viewing_sum_hist[iT].set_xlabel('Viewing - Cherenkov angle (deg)')
    ax_viewing_sum_hist[iT].legend(fontsize=12)
    ax_viewing_sum_hist[iT].set_xlim(-20, 20)
    ax_viewing_sum_hist[iT].grid(b=None)
    ax_viewing_sum_hist[iT].minorticks_on()
    ax_viewing_sum_hist[iT].tick_params(axis='x', which='minor', bottom=True, top=True)
    SetPlotAxis(ax_viewing_sum_hist[iT])
    ax_viewing_sum_hist[iT].set_ylabel('Normalized Events')

    xmin, xmax = ax_viewing_sum_hist[iT].get_xlim()
    ymin, ymax = ax_viewing_sum_hist[iT].get_ylim()
    ymax *= 1.0
    ax_viewing_sum_hist[iT].set_ylim(ymin, ymax)
    x_text = 0.03 * (xmax - xmin) + xmin
    y_text = 0.97 * (ymax - ymin) + ymin
    msg = f"reflected\nf: {f}\nReflectivity: -{dB:.0f}dB\n{trigger_name.replace('_', ' ')}"
    ax_viewing_sum_hist[iT].text(x_text, y_text, msg, verticalalignment="top", horizontalalignment="left", fontsize=12)

save_name = f'plots/CoreAnalysis/CorePaper/viewing_angles/diagnostics_viewing_angle_sum_{station}_{dB}dB_{f}f.pdf'
fig_viewing_sum_hist.savefig(save_name)
ic(f'saved {save_name}')
plt.clf()


# Do it again but with the already summed data
fig_viewing_sum_hist, ax_viewing_sum_hist = plt.subplots(ncols=2, nrows=1, figsize=(16, 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
for iT, (trigger_name, antennas) in enumerate(trigger_antennas):
    colors = qualitative_colors(len(layer_depths))
    for iD, depth in enumerate(layer_depths):
        # Load the viewing angle data
        with open(f'plots/CoreAnalysis/CorePaper/diagnostics_viewing_angle_sum_{station}_{trigger_name}_{depth}_{dB}dB_{f}f.pkl', 'rb') as fin:
            all_viewing_angles, all_viewing_angles_weights = pickle.load(fin)
            fin.close()

        # Plot the viewing angle histogram
        ax_viewing_sum_hist[iT].hist(all_viewing_angles, bins=bins, weights=all_viewing_angles_weights, histtype='step', label=f'depth {depth}', color=colors[iD], density=True)

    # ax_viewing_sum_hist[iT].set_title(f'{trigger_name}')
    ax_viewing_sum_hist[iT].set_xlabel('Viewing - Cherenkov angle (deg)')
    ax_viewing_sum_hist[iT].legend(fontsize=12)
    ax_viewing_sum_hist[iT].set_xlim(-20, 20)
    ax_viewing_sum_hist[iT].grid(b=None)
    ax_viewing_sum_hist[iT].minorticks_on()
    ax_viewing_sum_hist[iT].tick_params(axis='x', which='minor', bottom=True, top=True)
    SetPlotAxis(ax_viewing_sum_hist[iT])
    ax_viewing_sum_hist[iT].set_ylabel('Normalized Events')

    xmin, xmax = ax_viewing_sum_hist[iT].get_xlim()
    ymin, ymax = ax_viewing_sum_hist[iT].get_ylim()
    ymax *= 1.0
    ax_viewing_sum_hist[iT].set_ylim(ymin, ymax)
    x_text = 0.03 * (xmax - xmin) + xmin
    y_text = 0.97 * (ymax - ymin) + ymin
    msg = f"reflected\nf: {f}\nReflectivity: -{dB:.0f}dB\n{trigger_name.replace('_', ' ')}"
    ax_viewing_sum_hist[iT].text(x_text, y_text, msg, verticalalignment="top", horizontalalignment="left", fontsize=12)

save_name = f'plots/CoreAnalysis/CorePaper/viewing_angles/diagnostics_viewing_angle_sumsaved_{station}_{dB}dB_{f}f.pdf'
fig_viewing_sum_hist.savefig(save_name)
ic(f'saved {save_name}')
plt.clf()
