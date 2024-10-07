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



DEBUG = False
station = 'station_1001'
type = 'SP'
folder = '../CorePaperhdf5/gen2'

identifier = 'CorePaper'
# trigger_antennas = [ ['LPDA_2of4_100Hz', [0, 1, 2, 3]], ['PA_8ch_100Hz', [8, 9, 10, 11]]]
# trigger_antennas = [ ['LPDA_2of4_100Hz', [0, 1, 2, 3]]]
trigger_antennas = [ ['PA_8ch_100Hz', [8, 9, 10, 11] ]]
# layer_depths = [ 300, 500, 830 ]
layer_depths = [ 300 ]
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

    file_prefix = 'run/CoreRefl'

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

    fig_viewing_hist_sum, ax_viewing_hist_sum = plt.subplots(ncols=2, nrows=1, figsize=(16, 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
    # ax_viewing_hist_sum = np.atleast_2d(ax_viewing_hist_sum)

    for iT, (trigger_name, antennas) in enumerate(trigger_antennas):
        if trigger_name == 'LPDA_2of4_100Hz':
            antenna_depth = -3
        else:
            antenna_depth = -150
        for depth in layer_depths:
            ncols = len(e_bins) - 1
            nrows = len(coszen_bin_edges) - 1
            ic(ncols, nrows)
            fig_rr_arrival, ax_rr_arrival = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
            ax_rr_arrival = np.atleast_2d(ax_rr_arrival)
            fig_rr_launch, ax_rr_launch = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
            ax_rr_launch = np.atleast_2d(ax_rr_launch)
            fig_launch_arrival, ax_launch_arrival = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
            ax_launch_arrival = np.atleast_2d(ax_launch_arrival)
            fig_viewing_hist, ax_viewing_hist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
            ax_viewing_hist = np.atleast_2d(ax_viewing_hist)
            fig_bot_refl_angle_hist, ax_bot_refl_angle_hist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
            ax_bot_refl_angle_hist = np.atleast_2d(ax_bot_refl_angle_hist)

            viewing_hist_raw = np.zeros_like(ax_viewing_hist)
            viewing_hist_weights = np.zeros_like(ax_viewing_hist)


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

                    #Possible we never triggered, check to see if key of 'trigger_names' exits
                    # ic('--------------key in fin')
                    # for key in fin:
                    #     ic(key)
                    # ic('--------------key in fin.attrs')
                    # for key in fin.attrs:
                    #     ic(key)
                    # ic(f'--------------key in fin[{station}]')
                    # for key in fin[station]:
                    #     ic(key)
                    # quit()
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

                    # ic(len(ra_mask), len(rb_mask), len(da_mask), len(db_mask), len(arrival_z), len(xx), len(rr))
                    # ic(len(trig_mask))
                    # ic(len(refl_mask))
                    # ic(len(arrival_z))

                    # Make a hist of reflection angles
                    refl_angles = []
                    ray_tracing_max_amps = hdau.ray_tracing_solutions_max_amp(fin, antennas[0], station, trig_index)[refl_mask]
                    for i in range(len(xx[trig_mask][refl_mask])):
                        try:
                            tx, ty, tz = C04_p.get_path_for_solution(np.array([xx[trig_mask][refl_mask][i], yy[trig_mask][refl_mask][i], zz[trig_mask][refl_mask][i]]), station_location=np.array([0., 0., antenna_depth])*units.m, iS=ray_tracing_max_amps[i])  
                            refl_angle = C04_p.get_reflection_angle(tx, tz)
                            refl_angles.append(refl_angle)
                        except:
                            refl_angles.append(np.nan)

                    refl_angles = np.array(refl_angles)
                    nanmask = ~np.isnan(refl_angles)
                    # Turning off weight b/c n-throws per bin is already weighted, and so n-triggers per bin is already weighted
                    # ax_bot_refl_angle_hist[iC, iE].hist( refl_angles[nanmask], weights=np.pi*rr[nanmask]**2, bins=np.arange(0, 90, 1))
                    ax_bot_refl_angle_hist[iC, iE].hist( refl_angles[nanmask], bins=np.arange(0, 90, 1))
                    ax_bot_refl_angle_hist[iC, iE].set_title(f'{e_bins[iE]:.1f}-{e_bins[iE+1]:.1f} log10eV, {zen_bin_edges[iC]:.1f}-{zen_bin_edges[iC+1]:.1f} deg')
                    # continue

                    zen_arrival = np.arccos(arrival_z[refl_mask])
                    zen_arrival[np.isnan(zen_arrival)] = 0
                    zen_arrival = np.rad2deg(zen_arrival)
                    # zen_arrival = zen_arrival[refl_mask]
                    zen_launch = np.arccos(launch_z[refl_mask])
                    zen_launch[np.isnan(zen_launch)] = 0
                    zen_launch = np.rad2deg(zen_launch)


                    # Make a hist of viewing angles
                    shower_axis = -1 * hp.spherical_to_cartesian(np.array(fin['zeniths'])[trig_mask][refl_mask], np.array(fin['azimuths'])[trig_mask][refl_mask])
                    launch_vectors = hdau.launch_angles_max_amp(fin, antennas[0], station, trig_index)[refl_mask]
                    # Launch vectors is in format [n_showers, n_channels, n_ray_tracing_solutions, 3]
                    # We want all showers, specifically 1 channel for whether this is LPDA or PA, the ray tracing solution that triggered, and all 3 cartesian coords
                    # ic(shower_axis)
                    # ic(launch_vectors)
                    viewing_angles = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors)])
                    n_indexs = np.array([ice.get_index_of_refraction(x) for x in np.array([xx[trig_mask][refl_mask], yy[trig_mask][refl_mask], zz[trig_mask][refl_mask]]).T])
                    rho = np.arccos(1. / n_indexs)
                    nanmask = ~np.isnan(viewing_angles)

                    ax_viewing_hist[iC, iE].hist( (viewing_angles[nanmask]-rho[nanmask]) / units.deg, bins=np.arange(-20, 20, 1))
                    ax_viewing_hist[iC, iE].set_title(f'{e_bins[iE]:.1f}-{e_bins[iE+1]:.1f} log10eV, {zen_bin_edges[iC]:.1f}-{zen_bin_edges[iC+1]:.1f} deg')
                    # ic(area, np.sum(refl_mask), cores, area * np.sum(refl_mask) / cores)
                    viewing_hist_raw[iC, iE] = (viewing_angles[nanmask]-rho[nanmask]) / units.deg
                    viewing_hist_weights[iC, iE] = area * np.sum(refl_mask) / cores
                    continue

                    # rr is now list of triggered radii
                    # zen_arrival is list of arrival angles triggered
                    # zen_launch is list of launch angles triggered

                    zenBins = np.linspace(0, np.pi, 30)
                    zenBins = np.rad2deg(zenBins)
                    rr_bins = np.linspace(0, r_max * 10**3 * 1.05, num=100)

                    arrival_hist, arrival_bins = np.histogram(zen_arrival, bins=zenBins, density=True)
                    launch_hist, launch_bins = np.histogram(zen_launch, bins=zenBins, density=True)
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

                    ic(len(reflTrigFrac), len(zen_arrival))
                    rr_arrival_2dhist = np.zeros((len(rr_bins)-1, len(arrival_bins)-1))  # radius vs arrival angle
                    rr_launch_2dhist = np.zeros((len(rr_bins)-1, len(launch_bins)-1))   # radius vs launch angle
                    launch_arrival_2dhist = np.zeros((len(launch_bins)-1, len(arrival_bins)-1))   # launch vs arrival angle
                    for iR, r in enumerate(rr):
                        rDig = np.digitize(r, rr_bins) - 1
                        aDig = np.digitize(zen_arrival[iR], arrival_bins) - 1
                        lDig = np.digitize(zen_launch[iR], launch_bins) - 1

                        # ic(iR, rDig, aDig, lDig)
                        # ic(areas[rDig], reflTrigFrac[rDig], arrival_hist[aDig])
                        # rr_arrival_2dhist[rDig, aDig] += areas[rDig] * reflTrigFrac[rDig] * arrival_hist[aDig]     # area * n_trig/n_throw * n_arrival_bin/total_arrivals
                        # rr_launch_2dhist[rDig, lDig] += areas[rDig] * reflTrigFrac[rDig] * launch_hist[lDig]
                        rr_arrival_2dhist[rDig, aDig] += areas[rDig] * reflTrigFrac[rDig] / rr_hist[rDig]     # area * n_trig/n_throw * n_arrival_bin/total_arrivals
                        rr_launch_2dhist[rDig, lDig] += areas[rDig] * reflTrigFrac[rDig]  / rr_hist[rDig]
                        launch_arrival_2dhist[lDig, aDig] += areas[rDig] * reflTrigFrac[rDig] / rr_hist[rDig]
                    
                    # Below method is wrong I believe
                    # for iR, rFrac in enumerate(rr_hist):
                    #     for iA, aFrac in enumerate(arrival_hist):
                    #         rr_arrival_2dhist[iR, iA] = areas[iR] * rFrac * aFrac
                    #     for iL, lFrac in enumerate(launch_hist):
                    #         rr_launch_2dhist[iR, iL] = areas[iR] * rFrac * lFrac
                    # for iL, lFrac in enumerate(launch_hist):
                    #     for iA, aFrac in enumerate(arrival_hist):
                    #         launch_arrival_2dhist[iL, iA] = lFrac * aFrac * sum(areas * rr_hist)

                    # Make radius-arrival plot
                    rr_arrival_2dhist = np.fliplr(rr_arrival_2dhist)
                    rr_arrival_2dhist, cmap = set_bad_imshow(rr_arrival_2dhist, 0)
                    rr_arrival_2dhist = rr_arrival_2dhist.T
                    sum_2dhist = np.sum(rr_arrival_2dhist)
                    if sum_2dhist == 0:
                        norm = None
                    else:
                        norm = matplotlib.colors.LogNorm()
                    ax_rr_arrival[iC, iE].imshow(rr_arrival_2dhist, cmap=cmap, norm=norm, extent=[rr_bins[0], rr_bins[-1], arrival_bins[0], arrival_bins[-1]], aspect='auto')
                    ax_rr_arrival[iC, iE].set_title(f'{e_bins[iE]:.1f}-{e_bins[iE+1]:.1f} log10eV, {zen_bin_edges[iC]:.1f}-{zen_bin_edges[iC+1]:.1f} deg')

                    # Make radius-launch plot
                    rr_launch_2dhist = np.fliplr(rr_launch_2dhist)
                    rr_launch_2dhist, cmap = set_bad_imshow(rr_launch_2dhist, 0)
                    rr_launch_2dhist = rr_launch_2dhist.T
                    sum_2dhist = np.sum(rr_launch_2dhist)
                    if sum_2dhist == 0:
                        norm = None
                    else:
                        norm = matplotlib.colors.LogNorm()
                    ax_rr_launch[iC, iE].imshow(rr_launch_2dhist, cmap=cmap, norm=norm, extent=[rr_bins[0], rr_bins[-1], launch_bins[0], launch_bins[-1]], aspect='auto')
                    ax_rr_launch[iC, iE].set_title(f'{e_bins[iE]:.1f}-{e_bins[iE+1]:.1f} log10eV, {zen_bin_edges[iC]:.1f}-{zen_bin_edges[iC+1]:.1f} deg')

                    # Make launch-arrival plot
                    launch_arrival_2dhist = np.fliplr(launch_arrival_2dhist)
                    launch_arrival_2dhist, cmap = set_bad_imshow(launch_arrival_2dhist, 0)
                    launch_arrival_2dhist = launch_arrival_2dhist.T
                    sum_2dhist = np.sum(launch_arrival_2dhist)
                    if sum_2dhist == 0:
                        norm = None
                    else:
                        norm = matplotlib.colors.LogNorm()
                    ax_launch_arrival[iC, iE].imshow(launch_arrival_2dhist, cmap=cmap, norm=norm, extent=[launch_bins[0], launch_bins[-1], arrival_bins[0], arrival_bins[-1]], aspect='auto')
                    ax_launch_arrival[iC, iE].set_title(f'{e_bins[iE]:.1f}-{e_bins[iE+1]:.1f} log10eV, {zen_bin_edges[iC]:.1f}-{zen_bin_edges[iC+1]:.1f} deg')                    


                    # fig, ax = plt.subplots(ncols=1, nrows=1, gridspec_kw={"wspace": 0.2, "hspace": 0.2})
                    # ax = np.atleast_2d(ax)
                    # ax[0, 0].imshow(rr_arrival_2dhist, cmap=cmap, norm=norm, extent=[0, r_max * 10**3, 0, 180], aspect='auto')
                    # ax_labels = []
                    # for iZ, zen in enumerate(zenBins):
                    #     ax_labels.append('{:.0f}'.format(zenBins[iZ]))
                    # plt.yticks(range(len(zenBins)), ax_labels)
                    # plt.savefig(f'plots/CoreAnalysis/CorePaper/diagnostics_test.png')
                    # quit()

            # ax_labels = []
            # for iZ, zen in enumerate(zenBins):
            #     ax_labels.append('{:.0f}'.format(zenBins[iZ]))
            # plt.yticks(range(len(zenBins)), ax_labels)
            # plt.xlabel('Core distance (m)')
            # plt.ylabel('Arrival angle (deg)')
            # plt.colorbar(label=f'{np.sum(eRate):.5f} Evts/Stn/Yr')
            for iE in range(len(e_bins) - 1):
                ax_rr_arrival[-1, iE].set_xlabel('Core Distance (m)')
                ax_rr_launch[-1, iE].set_xlabel('Core Distance (m)')
                ax_launch_arrival[-1, iE].set_xlabel('Launch Angle (deg)')
                ax_viewing_hist[-1, iE].set_xlabel('viewing - cherenkov angle (deg)')
                ax_bot_refl_angle_hist[-1, iE].set_xlabel('Angle of Reflection off bottom (deg)')
            for iC in range(len(coszen_bin_edges)-1):
                ax_rr_arrival[iC, 0].set_ylabel('Arrival Angle (deg)')
                ax_rr_launch[iC, 0].set_ylabel('Launch Angle (deg)')
                ax_launch_arrival[iC, 0].set_ylabel('Arrival Angle (deg)')
                ax_viewing_hist[iC, 0].set_ylabel('Count')
                ax_bot_refl_angle_hist[iC, 0].set_ylabel('Weighted Count')



            plt.savefig(f'plots/CoreAnalysis/CorePaper/diagnostics_test.png')
            # fig_launch_arrival.savefig(f'plots/CoreAnalysis/CorePaper/diagnostics{depth}m/diagnostics_launch_arrival_{trigger_name}_{station}_{depth}_{dB}dB_{f}f.png')
            # fig_rr_arrival.savefig(f'plots/CoreAnalysis/CorePaper/diagnostics{depth}m/diagnostics_rr_arrival_{trigger_name}_{station}_{depth}_{dB}dB_{f}f.png')
            # fig_rr_launch.savefig(f'plots/CoreAnalysis/CorePaper/diagnostics{depth}m/diagnostics_rr_launch_{trigger_name}_{station}_{depth}_{dB}dB_{f}f.png')
            fig_viewing_hist.savefig(f'plots/CoreAnalysis/CorePaper/diagnostics{depth}m/diagnostics_viewing_angle_{trigger_name}_{station}_{depth}_{dB}dB_{f}f.png')
            # fig_bot_refl_angle_hist.savefig(f'plots/CoreAnalysis/CorePaper/diagnostics{depth}m/diagnostics_bot_reflection_angles_{trigger_name}_{station}_{depth}_{dB}dB_{f}f.png')
            plt.clf()

            # Add layer to sum hist
            bins=np.arange(-20, 20, 1) 
            all_viewing_angles = []
            all_viewing_angles_weights = []
            for iE in range(len(e_bins) - 1):
                for iC in range(len(coszen_bin_edges)-1):
                    ic(iE, iC)
                    # Skip bins that had nothing added
                    if not np.any(viewing_hist_raw[iC, iE]):
                        continue
                    # ic(viewing_hist_raw[iC, iE])
                    # ic(viewing_hist_raw[iC, iE].tolist())
                    all_viewing_angles.extend(viewing_hist_raw[iC, iE].tolist())
                    # ic(viewing_hist_raw[iC, iE])
                    viewing_hist_weights[iC, iE] *= core_weight_array_TA[iE, iC]
                    for i in range(len(viewing_hist_raw[iC, iE])):
                        all_viewing_angles_weights.append(viewing_hist_weights[iC, iE])
                    # ic(len(all_viewing_angles), len(all_viewing_angles_weights))

            ax_viewing_hist_sum[iT].hist(all_viewing_angles, bins=bins, weights=all_viewing_angles_weights, histtype='step', label=f'{depth}m', density=True)

            # Save the histogram data with pickle for opening and replotting later
            with open(f'plots/CoreAnalysis/CorePaper/diagnostics_viewing_angle_sum_{station}_{trigger_name}_{depth}_{dB}dB_{f}f.pkl', 'wb') as fout:
                pickle.dump([all_viewing_angles, all_viewing_angles_weights], fout)
                fout.close()
            with open(f'plots/CoreAnalysis/CorePaper/diagnostics_viewing_angle_perbin_{station}_{trigger_name}_{depth}_{dB}dB_{f}f.pkl', 'wb') as fout:
                pickle.dump([viewing_hist_raw, viewing_hist_weights], fout)
                fout.close()


        ax_viewing_hist_sum[iT].set_title(f'{trigger_name}')
        ax_viewing_hist_sum[iT].set_xlabel('viewing - cherenkov angle (deg)')
        ax_viewing_hist_sum[iT].legend()

    fig_viewing_hist_sum.savefig(f'plots/CoreAnalysis/CorePaper/diagnostics_viewing_angle_sum_{station}_{dB}dB_{f}f.png')
    plt.clf()