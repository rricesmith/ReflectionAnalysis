from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import argparse
import h5py
import os
#import glob
import hdf5AnalysisUtils as hdau
import pickle
from icecream import ic
from radiotools import helper as hp
import coreDataObjects as CDO
from coreDataObjects import coreStatistics
from NuRadioMC.utilities import medium
import C04_plotRayTracingSolutions as C04_p

def set_bad_imshow(array, value):
    ma = np.ma.masked_where(array == value, array)
    cmap = matplotlib.cm.viridis
    cmap.set_bad(color='white')
    return ma, cmap


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


DEBUG = False
station = 'station_1001'
type = 'SP'
folder = '../CorePaperhdf5/gen2'

identifier = 'CorePaper'
trigger_antennas = [ ['LPDA_2of4_100Hz', [0, 1, 2, 3]], ['PA_8ch_100Hz', [8, 9, 10, 11]]]
# trigger_antennas = [ ['LPDA_2of4_100Hz', [0, 1, 2, 3]]]
# trigger_antennas = [ ['PA_8ch_100Hz', [8, 9, 10, 11] ]]
layer_depths = [ 300, 500, 830 ]
# layer_depths = [ 300 ]
if type == 'MB':
    layer_depths = [ 576 ]

###########################################
#End Configs
###########################################

if type == 'SP':
    atm_overburden = 680
    f = 0.3
elif type == 'MB':
    atm_overburden = 1000
    f = 0.06
elif type == 'GL':
    print(f'not configured yet')
    quit()

nxmax = 2000
with open(f"data/output_TA_19_{atm_overburden}.pkl", "rb") as fin:
    shower_energies_TA, weights_shower_energies_TA, shower_xmax_TA = pickle.load(fin)
    fin.close()
with open(f"data/output_auger_19_{atm_overburden}.pkl", "rb") as fin:
    shower_energies_Auger, weights_shower_energies_Auger, shower_xmax_Auger = pickle.load(fin)
    fin.close()


def f_per_zenith(zenith_center):
    #This function returns the variable f value to use for a given zenith bin
    #Center of the zenith bin should be given in degrees
    
    if zenith_center < 18:
        return 0.52
    elif zenith_center < 26:
        return 0.4
    elif zenith_center < 32:
        return 0.3
    elif zenith_center < 37:
        return 0.22
    elif zenith_center < 41:
        return 0.16
    elif zenith_center < 46:
        return 0.11
    elif zenith_center < 49:
        return 0.08
    elif zenith_center < 53:
        return 0.06
    elif zenith_center < 57:
        return 0.05
    elif zenith_center < 60:
        return 0.05
    else:
        print(f'zenith {zenith_center} too large')
        quit()

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



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
    parser.add_argument('dB', type=float, help='dB of reflector')
    parser.add_argument('f', type=float, help='f factor of core energy going to radio')
    parser.add_argument('--path', type=str, help='path to look at of triggers, reflected or direct. Default refl', default='refl')
    parser.add_argument('--fVariable', type=bool, help='Use a different f for each zenith bin. Default False', default=False)

    args = parser.parse_args()
    dB = args.dB
    f = args.f
    path = args.path
    fVariable = args.fVariable

    refl_coef = 1
    dBfactor = 10**(-dB/20)
    feff = f * dBfactor 
    if fVariable:
        f = 'fZenVariable'

    print(f'dB {dB} f {f} from R {refl_coef} so feff {feff}')

#    e_bins = np.arange(17, 20.01, 0.1)     #for testing R=0.01, when simulating with a reflective layer
    e_bins = np.arange(15, 18.11, 0.1)
    e_center = (e_bins[1:] + e_bins[:-1])/2


    dCos = 0.05
    coszen_bin_edges = np.arange(np.cos(np.deg2rad(60)), 1.01, dCos)
    coszen_bin_edges = np.flip(coszen_bin_edges)
    coszens = 0.5 * (coszen_bin_edges[1:] + coszen_bin_edges[:-1])
    cos_edges = np.arccos(coszen_bin_edges)
    cos_edges[np.isnan(cos_edges)] = 0
    zen_bin_edges = np.rad2deg(np.nan_to_num(np.arccos(coszen_bin_edges)))

    ice = medium.southpole_simple()

    core_weight_array_TA, core_weight_array_Auger = getCoreBinWeightArray(f=f, dB=dB)
    # ic(core_weight_array_TA)
    # ic(core_weight_array_TA[0, 0], core_weight_array_TA[10, 7], core_weight_array_TA[-1, -1])
    # quit()

    ncols = len(trigger_antennas[0])
    nrows = len(layer_depths)
    fig_SNR_hist_sum, ax_SNR_hist_sum = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
    ax_SNR_hist_sum = np.atleast_2d(ax_SNR_hist_sum)
    fig_SNR_hist_sum_unweighted, ax_SNR_hist_sum_unweighted = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
    ax_SNR_hist_sum_unweighted = np.atleast_2d(ax_SNR_hist_sum_unweighted)

    for iT, (trigger_name, antennas) in enumerate(trigger_antennas):
        if trigger_name == 'LPDA_2of4_100Hz':
            antenna_depth = -3
        else:
            antenna_depth = -150
        for iD, depth in enumerate(layer_depths):
            ncols = len(e_bins) - 1
            nrows = len(coszen_bin_edges) - 1
            ic(ncols, nrows)
            fig_SNR_hist, ax_SNR_hist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
            ax_SNR_hist = np.atleast_2d(ax_SNR_hist)
            SNR_hist_raw = np.zeros_like(ax_SNR_hist)
            SNR_hist_weights = np.zeros_like(ax_SNR_hist)


            for iE in range(len(e_bins) - 1):
                # if not (iE == 17 or iE == 18):
                #     continue
                for iC in range(len(coszen_bin_edges)-1):
                    # if not (iC == 0 or iC == 2):
                    #     continue
                    filename = f'lgE_{e_bins[iE]:.2f}-cos_{coszen_bin_edges[iC+1]:.2f}-depth_{depth:.0f}.hdf5'

                    output_filename = os.path.join(folder, filename)
                    fin = h5py.File(output_filename, 'r')
                    print(f'input filename {output_filename}')
                    #Number of cores per bin from hdf5 file
                    cores = fin.attrs['n_events']
                    area = fin.attrs['area'] / units.km / units.km
                    r_max = fin.attrs['rmax'] / units.km

                    Vrms = fin[station].attrs['Vrms'][antennas[0]]  # All antennas of a trigger have same Vrms
                    ic(Vrms/units.microvolt)

                    keys = fin.attrs.keys()
                    triggered = 'trigger_names' in keys
                    multi_triggered = 'multiple_triggers' in keys
                    if not triggered:
                        continue

                    trig_mask, trig_index = hdau.trigger_mask(fin, trigger = trigger_name, station = station)
                    if trig_index == []:
                        print(f'No triggers {r_max}km {e_center[iE]}eV {coszens[iC]}coszen with trigger {trigger_name}')
                        continue
                    ra_mask, rb_mask, da_mask, db_mask, arrival_z, launch_z = hdau.multi_array_direction_masks(fin, antennas, station, trig_index)

                    if path == 'direct':
#                        refl_mask = da_mask | ra_mask      #This is incorrectly combining all direct rates, do not want to use
                        refl_mask = da_mask | db_mask
                    else:
                        if DEBUG:
                            print(f'refl mask make from len db {len(db_mask)} or rb {len(rb_mask)}')
#                        refl_mask = db_mask | rb_mask      #Also incorrect
                        refl_mask = ra_mask | rb_mask

                    xx = np.array(fin['xx'])
                    yy = np.array(fin['yy'])
                    zz = np.array(fin['zz'])
                    rr = (xx**2 + yy**2)**0.5
#                    print(f'len rr {len(rr)}')
#                    print(f'sum trig {trig_mask.sum()} and len {len(trig_mask)}')
#                    print(f'sum refl {refl_mask.sum()} and len {len(refl_mask)}')
                    rr = rr[trig_mask]
                    rr = rr[refl_mask]


                    # rr is now list of triggered radii
                    # zen_arrival is list of arrival angles triggered
                    # zen_launch is list of launch angles triggered
                    zenBins = np.linspace(0, np.pi, 30)
                    zenBins = np.rad2deg(zenBins)
                    rr_bins = np.linspace(0, r_max * 10**3 * 1.05, num=100)

                    rr_hist, rr_bins = np.histogram(rr, bins=rr_bins)
                    error_hist = np.sqrt(rr_hist)
                    throwsPerRadBin = CDO.throwsPerRadBin(r_max * 10**3, rr_bins, cores, shape='square')

                    reflTrigFrac = rr_hist / throwsPerRadBin                
                    reflTrigFrac[reflTrigFrac > 1] = 1              # hist of n_trig/n_throw per rad bin
                    errorTrigHigh = (rr_hist + error_hist) / throwsPerRadBin
                    errorTrigLow = (rr_hist - error_hist) / throwsPerRadBin
                    errorTrigLow[errorTrigLow < 0] = 0

                    areas = np.zeros(len(rr_hist))     #Will end up being units km^2
                    for iR in range(len(areas)):
                        areas[iR] = np.pi * (rr_bins[iR+1]**2 - rr_bins[iR]**2) * 10**-6

                    # max_amp_per_ray = hdau.ray_tracing_max_amp(fin, antennas[0], station, trig_index)
                    max_amps = hdau.multi_ray_tracing_max_amp(fin, antennas, station, trig_index)

                    SNR = max_amps / Vrms
                    SNR_hist_raw[iC, iE] = SNR
                    SNR_hist_weights[iC, iE] = np.ones_like(SNR) * area / cores

                    ax_SNR_hist[iC, iE].hist( SNR, bins=np.arange(0, 100, 1))
                    ax_SNR_hist[iC, iE].set_title(f'{e_bins[iE]:.1f}-{e_bins[iE+1]:.1f} log10eV, {zen_bin_edges[iC]:.1f}-{zen_bin_edges[iC+1]:.1f} deg')
                    # ax_SNR_hist[iC, iE].set_xscale('log')

            for iE in range(len(e_bins) - 1):
                ax_SNR_hist[-1, iE].set_xlabel('SNR')
            for iC in range(len(coszen_bin_edges)-1):
                ax_SNR_hist[iC, 0].set_ylabel('Count')


            savename = f'plots/CoreAnalysis/CorePaper/diagnostics{depth}m/diagnostics_SNR_{trigger_name}_{station}_{depth}_{dB}dB_{f}f.png'
            fig_SNR_hist.savefig(savename)
            print(f'saved {savename}')
            plt.clf()


            # for iE in range(len(e_bins) - 1):
            step = 6
            # colors = qualitative_colors(int(len(e_bins) / step)+1)
            for iE in range(0, len(e_bins) - 1, step):
                eng_sum_SNR = []
                eng_sum_weight = []
                for j in range(step):
                    if iE+j >= len(e_bins)-1:
                        j += -1
                        break
                    for iC in range(len(coszen_bin_edges)-1):
                        if not np.any(SNR_hist_raw[iC, iE]):
                            continue
                        # ic(SNR_hist_raw[iC, iE], SNR_hist_weights[iC, iE])
                        eng_sum_SNR.extend(SNR_hist_raw[iC, iE].tolist())
                        eng_sum_weight.extend(SNR_hist_weights[iC, iE].tolist())
                # ic(eng_sum_SNR, eng_sum_weight)

                # ic(iE, len(eng_sum_SNR), len(eng_sum_weight))
                ax_SNR_hist_sum[iD, iT].hist(eng_sum_SNR, bins=np.arange(0, 100, 1), weights=eng_sum_weight, histtype='bar', label=f'{e_bins[iE]:.2f}-{e_bins[iE+j]:.2f}eV', density=False, stacked=True)
                ax_SNR_hist_sum_unweighted[iD, iT].hist(eng_sum_SNR, bins=np.arange(0, 100, 1), histtype='bar', label=f'{e_bins[iE]:.2f}-{e_bins[iE+j]:.2f}eV', density=False, stacked=True)

            ax_SNR_hist_sum[iD, iT].set_title(f'{trigger_name} {depth}m')
            ax_SNR_hist_sum[-1, iT].set_xlabel('SNR')
            ax_SNR_hist_sum[iD, iT].set_ylabel('Events/Station/Year')
            # ax_SNR_hist_sum[iD, iT].set_xscale('log')
            ax_SNR_hist_sum[iD, iT].legend()
            ax_SNR_hist_sum_unweighted[iD, iT].set_title(f'{trigger_name} {depth}m')
            ax_SNR_hist_sum_unweighted[-1, iT].set_xlabel('SNR')
            ax_SNR_hist_sum_unweighted[iD, iT].set_ylabel('Trigger Count')
            # ax_SNR_hist_sum_unweighted[iD, iT].set_xscale('log')
            ax_SNR_hist_sum_unweighted[iD, iT].legend()
        
            

            # Save the histogram data with pickle for opening and replotting later
            with open(f'plots/CoreAnalysis/CorePaper/diagnostics_SNR_{station}_{trigger_name}_{depth}_{dB}dB_{f}f.pkl', 'wb') as fout:
                pickle.dump([SNR_hist_raw, SNR_hist_weights], fout)
                fout.close()



    fig_SNR_hist_sum.tight_layout()
    savename = f'plots/CoreAnalysis/CorePaper/diagnostics_SNR_sum_{station}_{dB}dB_{f}f.png'
    fig_SNR_hist_sum.savefig(savename)
    print(f'saved {savename}')

    fig_SNR_hist_sum_unweighted.tight_layout()
    savename = f'plots/CoreAnalysis/CorePaper/diagnostics_SNR_sum_unweighted_{station}_{dB}dB_{f}f.png'
    fig_SNR_hist_sum_unweighted.savefig(savename)
    print(f'saved {savename}')




