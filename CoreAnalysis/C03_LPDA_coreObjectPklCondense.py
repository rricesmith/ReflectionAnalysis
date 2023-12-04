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

import coreDataObjects as CDO
from coreDataObjects import coreStatistics

DEBUG = False
station = 'station_1'
type = 'SP'
cores = 2000

#identifier = 'Gen2_AllTriggers'
identifier = 'Gen2_AllTriggers_DipFix'
#identifier = 'Gen2_40dB_comparison'
trigger_antennas = [ ['LPDA_2of4_2sigma', [0, 1, 2, 3]], ['LPDA_2of4_3.8sigma', [0, 1, 2, 3]] ]
#trigger_antennas = [ ['single_dipole_trigger_2sig', [4]], ['single_dipole_trigger_3sig', [4]]]

#identifier = 'DipoleTesting'
#trigger_antennas = [ ['Gen2_Dipole_2sig', [0]], ['Gen2_Dipole_3sig', [0]],
#                     ['RNOG_3sig', [1]], ['ARA_2sig', [2]],
#                     ['Old_80to500_Simple_2sig', [3]], ['Old_80to500_HighLow_2sig', [3]],
#                     ['Old_80to250_Simple_2sig', [4]], ['Old_80to250_HighLow_2sig', [4]]
#                     ]

#layer_depths = [ [300, 1.7] ]
layer_depths = [ [300, 1.7], [500, 1.7], [800, 1.7] ]
if type == 'MB':
    layer_depths = [ [576, 1.7] ]

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
    parser.add_argument('dB', type=float, help='dB of reflector')
    parser.add_argument('f', type=float, help='f factor of core energy going to radio')
    parser.add_argument('--direction', type=str, help='direction to look at of triggers, above or below. Default below', default='below')

    args = parser.parse_args()
    dB = args.dB
    f = args.f
    direction = args.direction


    refl_coef = 1
    dBfactor = 10**(-dB/20)
    feff = f * dBfactor 
    f = 'fZenVariable'

    print(f'dB {dB} f {f} from R {refl_coef} so feff {feff}')

#    e_bins = np.arange(17, 20.01, 0.1)     #for testing R=0.01, when simulating with a reflective layer
    e_bins = np.arange(14, 17.5, 0.1)
    e_center = (e_bins[1:] + e_bins[:-1])/2

    file_prefix = 'run/CoreRefl'

    dCos = 0.05
    coszen_bin_edges = np.arange(np.cos(np.deg2rad(60)), 1.01, dCos)
    coszen_bin_edges = np.flip(coszen_bin_edges)
    coszens = 0.5 * (coszen_bin_edges[1:] + coszen_bin_edges[:-1])
    cos_edges = np.arccos(coszen_bin_edges)
    cos_edges[np.isnan(cos_edges)] = 0

    ############
    #Testing variable f from zenith
    ############
    f_centers = np.zeros_like(coszens)
    for iC, coszen in enumerate(coszens):
        f_centers[iC] = f_per_zenith(np.rad2deg(np.arccos(coszen)))
    feff_array = f_centers * dBfactor

    #Here we shift the energy to of the core to that of the simulated shower energy
    #Ecore = Eshower / (f * 10^-dB/20)
    #Take log of both sides since working in log-scale
    e_bins_shift = e_bins - np.log10(feff)
    e_center_shift = e_center - np.log10(feff)

    ############
    #Testing variable f from zenith
    ############
    e_bins_shift = {}
    e_center_shift = {}
    for iC, coszen in enumerate(coszens):
        e_bins_shift[iC] = e_bins - np.log10(feff_array[iC])
        e_center_shift[iC] = e_center - np.log10(feff_array[iC])

    n_zen_bins = len(coszen_bin_edges)-1
    shower_E_bins = np.arange(17, 20.01, 0.1)
    logEs = 0.5 * (shower_E_bins[1:] + shower_E_bins[:-1])


    for trigger_name, antennas in trigger_antennas:
        for depth, space in layer_depths:
            CoreObjectsList = []
            for iE in range(len(e_bins) - 1):
                for iC in range(len(coszen_bin_edges)-1):
                    folder = file_prefix + f'{space:.2f}km'
                    Estring = f'{e_bins[iE]:.1f}log10eV'
                    zString = f'coszen{coszen_bin_edges[iC]:.3f}'

                    cores = coreStatistics(e_bins[iE])       #New line testing changing core statistics
                    filename = f"{identifier}_Cores_Gen2Design_{depth}mRefl_{type}_{refl_coef}R_{space:.2f}km_{Estring}_{zString}_cores{cores}.hdf5"

                    output_filename = os.path.join(folder, Estring, filename)
                    fin = h5py.File(output_filename, 'r')
                    print(f'output filename {output_filename}')



                    #Possible we never triggered, check to see if key of 'trigger_names' exits
                    keys = fin.attrs.keys()
                    triggered = 'trigger_names' in keys
                    if not triggered:
                        print(f'No triggers {space}km {e_center[iE]}eV {coszens[iC]}coszen')
#                        newCore = CDO.coresBin(E_low=e_bins_shift[iE], E_high=e_bins_shift[iE+1], coszen_low=coszen_bin_edges[iC], coszen_high=coszen_bin_edges[iC+1],
                        newCore = CDO.coresBin(E_low=e_bins_shift[iC][iE], E_high=e_bins_shift[iC][iE+1], coszen_low=coszen_bin_edges[iC], coszen_high=coszen_bin_edges[iC+1],
                                                rr_trig_hist=[], rad_bins=[], max_radius=space, n_cores=cores, shape='square')
                        CoreObjectsList.append(newCore)
                        continue


                    trig_mask, trig_index = hdau.trigger_mask(fin, trigger = trigger_name, station = station)
                    ra_mask, rb_mask, da_mask, db_mask, arrival_z = hdau.multi_array_direction_masks(fin, antennas, station, trig_index)

                    if direction == 'above':
                        refl_mask = da_mask | ra_mask
                    else:
                        if DEBUG:
                            print(f'refl mask make from len db {len(db_mask)} or rb {len(rb_mask)}')
                        refl_mask = db_mask | rb_mask

                    xx = np.array(fin['xx'])
                    yy = np.array(fin['yy'])
                    rr = (xx**2 + yy**2)**0.5
                    rr = rr[trig_mask]
                    rr = rr[refl_mask]

                    zen_arrival = np.arccos(arrival_z[refl_mask])

                    rr_bins = np.linspace(0, space * 10**3 * 1.05, num=100)
                    rr_hist, rr_bins = np.histogram(rr, bins=rr_bins)
                    error_hist = np.sqrt(rr_hist)
                    throwsPerRadBin = CDO.throwsPerRadBin(space * 10**3, rr_bins, cores, shape='square')
                    ##########################
                    ######Note this is only an approximation of throws/bin. It is not exact, which is why we lower it
                    ##########################
                    #Need high statistics to make up for this issue
                    reflTrigFrac = rr_hist / throwsPerRadBin                
                    reflTrigFrac[reflTrigFrac > 1] = 1
                    errorTrigHigh = (rr_hist + error_hist) / throwsPerRadBin
                    errorTrigLow = (rr_hist - error_hist) / throwsPerRadBin
                    errorTrigLow[errorTrigLow < 0] = 0


#                    newCore = CDO.coresBin(E_low=e_bins_shift[iE], E_high=e_bins_shift[iE+1], coszen_low=coszen_bin_edges[iC], coszen_high=coszen_bin_edges[iC+1],
                    newCore = CDO.coresBin(E_low=e_bins_shift[iC][iE], E_high=e_bins_shift[iC][iE+1], coszen_low=coszen_bin_edges[iC], coszen_high=coszen_bin_edges[iC+1],
                                            rr_trig_hist=rr_hist, rad_bins = rr_bins, max_radius=space, n_cores=cores, shape='square', zeniths=zen_arrival)
                    newCore.setZeniths(zen_arrival)

                    CoreObjectsList.append(newCore)


            #Now we will iterate through the CR's to append CR event rates to cores
            for iE, logE in enumerate(logEs):
                for iC, coszen in enumerate(coszens):

                    ###First we will add CRs for TA flux
                    niC = len(coszens) - 1 - iC
                    mask = ~np.isnan(shower_energies_TA[iE][niC])
                    engiEiC = np.log10(shower_energies_TA[iE][niC][mask])
                    eRatesiEiC = weights_shower_energies_TA[iE][niC][mask]
                    xMaxiEiC = shower_xmax_TA[iE][niC][mask]

                    #Find digit the core energy corresponds to of deposited energy to get event rate from Edep
                    coreDigs = np.digitize(engiEiC, e_bins_shift[iC]) - 1
                    coreDigsMask = coreDigs >= 0
                    for coreN, coreEng in enumerate(engiEiC):
                        if coreDigs[coreN] < 0:
                            continue
                        for core in CoreObjectsList:
                            if not core.engInBins(coreEng) or not core.coszenInBins(coszen):
                                continue
                            else:
                                if DEBUG:
                                    print(f'CR {logE} coszen {coszen} has core eng {coreEng} inside corebin {core.e_bins} with eRate {eRatesiEiC[coreN]}')
                                core.addCrParent(logE, xMaxiEiC[coreN], eRatesiEiC[coreN], type='TA')
                                break       #We break here because search is done, found Core bin with energy/zenith we are searching for


                    ###Then we add Auger CR parents
                    niC = len(coszens) - 1 - iC
                    mask = ~np.isnan(shower_energies_Auger[iE][niC])
                    engiEiC = np.log10(shower_energies_Auger[iE][niC][mask])
                    eRatesiEiC = weights_shower_energies_Auger[iE][niC][mask]
                    xMaxiEiC = shower_xmax_Auger[iE][niC][mask]

                    #Find digit the core energy corresponds to of deposited energy to get event rate from Edep
                    coreDigs = np.digitize(engiEiC, e_bins_shift[iC]) - 1
                    coreDigsMask = coreDigs >= 0
                    for coreN, coreEng in enumerate(engiEiC):
                        if coreDigs[coreN] < 0:
                            continue
                        for core in CoreObjectsList:
                            if not core.engInBins(coreEng) or not core.coszenInBins(coszen):
                                continue
                            else:
                                print(f'CR {logE} coszen {coszen} has core eng {coreEng} inside corebin {core.e_bins} with eRate {eRatesiEiC[coreN]}')
                                core.addCrParent(logE, xMaxiEiC[coreN], eRatesiEiC[coreN], type='Auger')
                                break       #We break here because search is done, found Core bin with energy/zenith we are searching for




            save_filename = f'data/CoreDataObjects/{identifier}_CoreDataObjects_{trigger_name}_{direction}_{depth}mRefl_{type}_{refl_coef}R_{f}f_{dB}dB_{space}km_{cores}cores.pkl'
            with open(save_filename, 'wb') as fout:
                pickle.dump(CoreObjectsList, fout)
                fout.close()
            print(f'saved file as {save_filename}')


    print(f'Done!')
    quit()

