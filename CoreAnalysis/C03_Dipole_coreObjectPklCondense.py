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


antennas = [13]
station = 'station_1'

if type == 'SP':
    atm_overburden = 680
    f = 0.3
elif type == 'MB':
    atm_overburden = 1000
    f = 0.06
elif type == 'GL':
    print(f'not configured yet')
    quit()
flux = 'auger_19'


#f = 0.3
refl_coef = 1
#magn = np.sqrt(1.6)
dB = 40	


#Factor change based off difference from db of 40
dBfactor = 10**-(dB/20)	
#dBfactor = 10**-(dB/20) / 0.01	#is 1 for 40dB, scaled to work for lower dB



feff = f * dBfactor


nxmax = 2000
with open(f"data/output_{flux}_{atm_overburden}.pkl", "rb") as fin:
#with open(f"data/output_{flux}_{atm_overburden}_{nxmax}xmax.pkl", "rb") as fin:
    shower_energies, weights_shower_energies = pickle.load(fin)


#layer_depths = [ [300, 1], [500, 1.2], [800, 1.5], [1000, 2], [1170, 2.5] ]
#layer_depths = [ [300,1.5], [500,2] , [800,2.5], [1000,3], [1170,3.5]]
layer_depths = [ [300, 1.7] ]



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
    parser.add_argument('dB', type=float, help='dB of reflector')
    parser.add_argument('f', type=float, help='f factor of core energy going to radio')
    parser.add_argument('--direction', type=str, help='direction to look at of triggers, above or below', default='below')

    args = parser.parse_args()
    dB = args.dB
    f = args.f
    direction = args.direction

    dBfactor = 10**(-dB/20)
    feff = f * dBfactor 

    print(f'dB {dB} f {f} from R {refl_coef} so feff {feff}')

#    e_bins = np.arange(17, 20.01, 0.1)
    e_bins = np.arange(14, 20.01, 0.1)
#    e_bins = np.array([19, 20])
    e_center = (e_bins[1:] + e_bins[:-1])/2

    cores = 1000
#    spacing = [500, 750, 1000, 1500]
#    spacing = [2000]
#    spacing = [1]
#    file_prefix = 'run/mooresCoreRefl'
    file_prefix = 'run/CoreRefl'
#    file_prefix = 'data/'

#    depth = 300

    dCos = 0.05
    coszen_bin_edges = np.arange(np.cos(np.deg2rad(60)), 1.01, dCos)
    coszen_bin_edges = np.flip(coszen_bin_edges)
    coszens = 0.5 * (coszen_bin_edges[1:] + coszen_bin_edges[:-1])
    cos_edges = np.arccos(coszen_bin_edges)
    cos_edges[np.isnan(cos_edges)] = 0
    """  #Old code for refraction, not used for particle cores
    n_ice = 1.78
    angles_refracted = np.arcsin(np.sin(np.arccos(coszen_bin_edges)) / n_ice)
    angles_refracted[np.isnan(angles_refracted)] = 0
    coszen_bin_edges_refracted = np.cos(angles_refracted)

    r_coszen_bin_edges = coszen_bin_edges_refracted
    """
    r_coszen_bin_edges = coszen_bin_edges
    r_cos_edges = np.arccos(coszen_bin_edges)
    r_cos_edges[np.isnan(cos_edges)] = 0

    e_bins_shift = e_bins - np.log10(feff)
    e_center_shift = e_center - np.log10(feff)

    n_zen_bins = len(coszen_bin_edges)-1
    shower_E_bins = np.arange(17, 20.01, 0.1)
    logEs = 0.5 * (shower_E_bins[1:] + shower_E_bins[:-1])


#    for iS, space in enumerate(spacing):
    for depth, space in layer_depths:
        CoreObjectsList = []
        for iE in range(len(e_bins) - 1):
            for iC in range(len(coszen_bin_edges)-1):
                folder = file_prefix + f'{space:.2f}km'
#                folder = file_prefix
                Estring = f'{e_bins[iE]:.1f}log10eV'
                zString = f'coszen{r_coszen_bin_edges[iC]:.3f}'

                filename = f"Cores_Gen2Design_{depth}mRefl_{type}_{refl_coef}R_{space:.2f}km_{Estring}_{zString}_cores{throws}.hdf5"
#                filename = f'RefractedCoresFocused_{depth}mRefl_{type}_{refl_coef}R_{thresh}muV_{space:.2f}km_{Estring}_{zString}_cores{cores}.hdf5'
#                filename = f'RefractedCoresFocused_{depth}mRefl_{type}_{thresh}muV_{space:.2f}km_{Estring}_{zString}_cores{cores}.hdf5'    #Once focusing is done, only 800m has focusing done
#                filename = f'Grid_{type}_{thresh}muV_{space:.2f}km_{Estring}_{zString}_cores{cores}.hdf5'
#                filename = f'Grid_{type}_{space:.2f}km_{Estring}_{zString}_cores{cores}.hdf5'			#Use for MB 26muV, no thresh in name for 1000 cores

                output_filename = os.path.join(folder, Estring, filename)
                fin = h5py.File(output_filename, 'r')
                print(f'output filename {output_filename}')

                #Checking amp factor
                """
                try:
                    focus = np.array(fin['station_1']['focusing_factor'])
                    print(f'focusing factor {focus}')
                except KeyError:
                    print(f'key doesnt exist')
                
                continue
                """


#                for key in fin.keys():
#                    print(key)
                #Possible we never triggered, check to see if key of 'trigger_names' exits
                keys = fin.attrs.keys()
                triggered = 'trigger_names' in keys
                if not triggered:
                    print(f'No triggers {space}km {e_center[iE]}eV {coszens[iC]}coszen')
#                    reflection_fraction[iS][iE][iC] = 0
#                    tot_trig_fraction[iS][iE][iC] = 0
#                    Aeff[iS][iE][iC] = 0
#                    newCore = CDO.coresBin(E_low=e_bins_shift[iE], E_high=e_bins_shift[iE+1], coszen_low=r_coszen_bin_edges[iC], coszen_high=r_coszen_bin_edges[iC+1],
                    newCore = CDO.coresBin(E_low=e_bins_shift[iE], E_high=e_bins_shift[iE+1], coszen_low=coszen_bin_edges[iC], coszen_high=coszen_bin_edges[iC+1],
                                            trigRatesPerRadBin=[], rad_bins=[])
                    CoreObjectsList.append(newCore)
                    continue


                trig_mask, trig_index = hdau.trigger_mask(fin, trigger = 'single_dipole_trigger_2sig', station = station)
                ra_mask, rb_mask, da_mask, db_mask, arrival_z = hdau.multi_array_direction_masks(fin, antennas, station, trig_index)

                if direction == 'above':
                    refl_mask = da_mask | ra_mask
                else:
                    refl_mask = db_mask | rb_mask

                xx = np.array(fin['xx'])
                yy = np.array(fin['yy'])
                rr = (xx**2 + yy**2)**0.5
                rr = rr[trig_mask]
                rr = rr[refl_mask]

                zen_arrival = np.arccos(arrival_z[refl_mask])

                rr_bins = np.linspace(0, space * 10**3 * 1.05, num=100)
#                rr_bins = [0, 5000]
                rr_hist, rr_bins = np.histogram(rr, bins=rr_bins)
#                rr_hist, rr_bins = np.histogram(rr, bins=10)
                throwsPerRadBin = CDO.throwsPerRadBin(space * 10**3, rr_bins, cores, shape='square')
                print(f'rr hist {rr_hist} in bins {rr_bins} has throws per bin {throwsPerRadBin}, sum of {np.sum(throwsPerRadBin)}')
                reflTrigFrac = rr_hist / throwsPerRadBin
                reflTrigFrac[reflTrigFrac > 1] = 1
#                reflTrigFrac = [sum(refl_mask)/cores]

#                newCore = CDO.coresBin(E_low=e_bins_shift[iE], E_high=e_bins_shift[iE+1], coszen_low=r_coszen_bin_edges[iC], coszen_high=r_coszen_bin_edges[iC+1],
                newCore = CDO.coresBin(E_low=e_bins_shift[iE], E_high=e_bins_shift[iE+1], coszen_low=coszen_bin_edges[iC], coszen_high=coszen_bin_edges[iC+1],
                                        trigRatesPerRadBin=reflTrigFrac, rad_bins = rr_bins, zeniths=zen_arrival)
                newCore.setZeniths(zen_arrival)

                CoreObjectsList.append(newCore)


        #Now we will iterate through the CR's to append CR event rates to cores
        for iE, logE in enumerate(logEs):
            for iC, coszen in enumerate(coszens):

                niC = len(coszens) - 1 - iC
                mask = ~np.isnan(shower_energies[iE][niC])
                engiEiC = np.log10(shower_energies[iE][niC][mask])
                eRatesiEiC = weights_shower_energies[iE][niC][mask]


#                print(f'we are looking at engiEiC {engiEiC} and eRatesiEiC {eRatesiEiC} for logE {logE} coszen {coszen}')
#                quit()

                #Find digit the core energy corresponds to of deposited energy to get event rate from Edep
                coreDigs = np.digitize(engiEiC, e_bins_shift) - 1
                coreDigsMask = coreDigs >= 0
#                engiEiC = engiEiC[coreDigsMask]
                for coreN, coreEng in enumerate(engiEiC):
#                    print(f'Checking core {coreEng} of CR {logE} in coszen {coszen}')
                    if coreDigs[coreN] < 0:
                        continue
                    for core in CoreObjectsList:
                        if not core.engInBins(coreEng) or not core.coszenInBins(coszen):
                            continue
                        else:
                            print(f'CR {logE} coszen {coszen} has core eng {coreEng} inside corebin {core.e_bins} with eRate {eRatesiEiC[coreN]}')
                            core.addCrParent(logE, eRatesiEiC[coreN])
#                            newParent = CDO.crEvent(logE, eRatesiEiC[coreN])
#                            core.addCrParent(newParent)
#                            core.parentCRs.append(newParent)
                            break


                """
                orig_shower_digit = np.digitize(np.log10(engiEiC), shower_E_bins)-1		#Digit of the origonal core energy
                engiEiC = engiEiC * f
                weightsiEiC = weights_shower_energies[iE][niC][mask]
#                surf_shower_digit = np.digitize(np.log10(engiEiC), shower_E_bins)-1     #Digit of the deposited energy based on core energy
                for n in range(len(engiEiC)):
                    cr_events_weights[iE][iC] += weightsiEiC[n]
                    coreN = orig_shower_digit[n]
                    total_full_evtrte_cr[iS][iE][iC] += weightsiEiC[n] * space ** 2
                    if (coreN < 0) or (coreN == len(logEs)):
                        continue
                    core_events_weights[coreN][iC] += weightsiEiC[n] 
                    total_full_evtrte_core[iS][coreN][iC] += weightsiEiC[n] * space ** 2

                    depN = surf_shower_digit[n]
                    if (depN < 0) or (depN == len(logEs)):
                        continue
                    dep_core_events_weights[depN][iC] += weightsiEiC[n] 
                    full_evtrte_cr[iS][iE][iC] += weightsiEiC[n] * Aeff[iS][depN][iC]
                    full_evtrte_core[iS][coreN][iC] += weightsiEiC[n] * Aeff[iS][depN][iC]



			#Now we are going to do a basic tagging method based on imported LPDA trigger rates binned in energy and cos_zenith
                    nC = np.digitize(coszen, lpda_cos_zen_bins) - 1
                    nE = np.digitize(logE, lpda_logEs) - 1
                    if (nC < 0) or (nC > len(lpda_zen_center)-1) or (nE < 0) or (nE > len(lpda_E_center)-1):
                        lpda_undet_cr[iE][iC] += weightsiEiC[n] * Aeff[iS][depN][iC]
                        lpda_undet_core[coreN][iC] += weightsiEiC[n] * Aeff[iS][depN][iC]
                    else:
                        lpda_event_rate_cr[iE][iC] += weightsiEiC[n] * Aeff[iS][depN][iC] * lpda_trig_rates[nC][nE]
                        lpda_undet_cr[iE][iC] += weightsiEiC[n] * Aeff[iS][depN][iC] * (1-lpda_trig_rates[nC][nE])
                        lpda_event_rate_core[coreN][iC] += weightsiEiC[n] * Aeff[iS][depN][iC] * lpda_trig_rates[nC][nE]
                        lpda_undet_core[coreN][iC] += weightsiEiC[n] * Aeff[iS][depN][iC] * (1-lpda_trig_rates[nC][nE])
                """


        with open(f'data/CoreDataObjects_Dipole_{direction}_{depth}mRefl_{type}_{refl_coef}R_{f}f_{dB}dB_{space}km_{cores}cores.pkl', 'wb') as fout:
            pickle.dump(CoreObjectsList, fout)
        print(f'saved file!')


    quit()

    xx = np.linspace(-1200, 1200, 100)
    yy = np.linspace(-1200, 1200, 100)
    zz = np.zeros( (len(xx), len(yy)) )

    for core in CoreObjectsList:
#        print(f'core info ebins {core.e_bins} rad bins {core.rad_bins} parentCRs num {len(core.parentCRs)}')
#        print(f'trig rates {core.trigRatesPerRadBin} and Aeff {core.Aeff_per_rad_bin} over bins {core.rad_bins}')
        core.setTotalEventRatePerArea()
#        print(f'core info ebins {core.e_bins} coszenbins {core.coszen_bins} parentCRs num {len(core.parentCRs)} with event rate total per area {core.totalEventRatePerArea}')
        xx, yy, zz = core.addEventsToArea(xx, yy, zz)

#        plot = plt.figure(1)
#        if core.plotEventRatePerRad():
#            print(f'can we show?')
#            plt.show()

    """
    plt.contourf(xx, yy, zz, cmap='YlOrRd', norm=matplotlib.colors.LogNorm())
    plt.colorbar(label=f'Events/Stn/Yr, Net {np.sum(zz):.5f}')
    plt.scatter(0, 0, s=30, marker='x', color='black')
    plt.title(f'Core Event Rate Distribution over all CRs for {type}')
    plt.show()
    """

    with open(f'data/CoreDataObjects_{depth}mRefl_{type}_{refl_coef}R_{f}f_{spacing[0]}km_{cores}cores_{thresh}muV.pkl', 'wb') as fout:
#    with open(f'data/CoreDataObjectsFocused_{depth}mRefl_{type}_{f}f_{magn}magn_{dB}dB_{spacing[0]}km_{cores}cores_{thresh}muV.pkl', 'wb') as fout:
        pickle.dump(CoreObjectsList, fout)


