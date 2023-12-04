from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import argparse
import h5py
import os
import glob
import hdf5AnalysisUtils as hdau
import pickle


cmap = plt.cm.plasma
cmap.set_bad(color='black')

def func_powerlaw(x, m, c, c0):
    return c0 + x**m * c

#Function presumes a particular orientation of the 2d input, check result is as expected
def plot_colormap(array_input, extent, aspect, title, xlabel, ylabel, colorlabel=None, ylim=None):
    array_input = array_input.T
    input_mask = np.ma.masked_where(array_input == 0, array_input)
    plt.imshow(input_mask, extent=extent, aspect = aspect, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
    if colorlabel is not None:
        plt.colorbar(label=colorlabel)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    return None


antennas = [0, 1, 2, 3]
station = 'station_1'

type = 'SP'
thresh = 26

if type == 'MB':
    atm_overburden = 1000
    f = 0.06
else:
    atm_overburden = 680
    f = 0.3
flux = 'auger_19'
#f = 1

nxmax = 2000
with open(f"data/output_{flux}_{atm_overburden}.pkl", "rb") as fin:
#with open(f"data/output_{flux}_{atm_overburden}_{nxmax}xmax.pkl", "rb") as fin:
    shower_energies, weights_shower_energies = pickle.load(fin)


if __name__ == '__main__':
    e_bins = np.arange(17, 20.01, 0.1)
#    e_bins = np.array([19, 20])
    e_center = (e_bins[1:] + e_bins[:-1])/2

    cores = 5000
#    spacing = [500, 750, 1000, 1500]
#    spacing = [2000]
    spacing = [5]
#    file_prefix = 'run/mooresCoreRefl'
    file_prefix = 'run/CoreRefl'


    dCos = 0.05
    coszen_bin_edges = np.arange(np.cos(np.deg2rad(60)), 1.01, dCos)
    coszen_bin_edges = np.flip(coszen_bin_edges)
    coszens = 0.5 * (coszen_bin_edges[1:] + coszen_bin_edges[:-1])
    cos_edges = np.arccos(coszen_bin_edges)
    cos_edges[np.isnan(cos_edges)] = 0

    n_ice = 1.78
    angles_refracted = np.arcsin(np.sin(np.arccos(coszen_bin_edges)) / n_ice)
    angles_refracted[np.isnan(angles_refracted)] = 0
    coszen_bin_edges_refracted = np.cos(angles_refracted)

    r_coszen_bin_edges = coszen_bin_edges_refracted
    r_cos_edges = np.arccos(coszen_bin_edges)
    r_cos_edges[np.isnan(cos_edges)] = 0


    n_zen_bins = len(coszen_bin_edges)-1
    shower_E_bins = np.arange(17, 20.01, 0.1)
    logEs = 0.5 * (shower_E_bins[1:] + shower_E_bins[:-1])

    event_rate_cr_eng = np.zeros((len(spacing), len(logEs)))
    event_rate_surface_shower = np.zeros_like(event_rate_cr_eng)
    total_rate_cr = np.zeros_like(event_rate_cr_eng)
    total_rate_surface = np.zeros_like(event_rate_cr_eng)

    full_evtrte_cr = np.zeros( (len(spacing), len(e_center), len(coszens)) )
    full_evtrte_core = np.zeros_like(full_evtrte_cr)
    total_full_evtrte_cr = np.zeros_like(full_evtrte_cr)
    total_full_evtrte_core = np.zeros_like(full_evtrte_cr)

    Aeff = np.zeros((len(spacing), len(e_center), len(coszens)))
    reflection_fraction = np.zeros_like(Aeff)
    tot_trig_fraction = np.zeros_like(Aeff)

    e_bins_shift = e_bins - np.log10(f)
    e_center_shift = e_center - np.log10(f)

    shift_digit = np.digitize(e_center_shift, e_bins) - 1
    Aeff_shift = np.zeros_like(Aeff)

    up_lpda = 4
    edges = 6
    grid_spacing = 0.75
#    with open(f"data/Up_LPDA_Trig_fraction_2of{up_lpda}_edges{edges}m_{grid_spacing:.2f}km.pkl", "rb") as fin:
#    with open(f"data/Up_LPDA_Trig_fraction_{grid_spacing:.2f}km.pkl", "rb") as fin:
#    with open(f'data/Up_LPDA_Trig_fraction_2of3_0.75km.pkl', 'rb') as fin:
#        lpda_logEs, lpda_cos_zen_bins, SP_trig_rates, MB_trig_rates = pickle.load(fin)
#    lpda_cos_zen_bins = np.arccos(lpda_cos_zen_bins)

	#Doing it while also including the fit setup
    with open(f'data/Up_LPDA_Trig_fraction_2of4_edges6m_1.00km_wFit.pkl', 'rb') as fin:
        lpda_logEs, lpda_cos_zen_bins, SP_trig_rates, MB_trig_rates, fit_edges, SP_fit_rates, MB_fit_rates = pickle.load(fin)
    lpda_cos_zen_bins = np.arccos(lpda_cos_zen_bins)
    SP_trig_rates = SP_fit_rates
    MB_trig_rates = MB_fit_rates
    lpda_logEs = fit_edges

    lpda_zen_center = (lpda_cos_zen_bins[1:] + lpda_cos_zen_bins[:-1])/2
    lpda_E_center = (lpda_logEs[1:] + lpda_logEs[:-1])/2

    lpda_trig_rates = 0
    if type == 'SP':
        lpda_trig_rates = SP_trig_rates
    else:
        lpda_trig_rates = MB_trig_rates


    lpda_event_rate_cr = np.zeros_like(full_evtrte_cr[0])
    lpda_event_rate_core = np.zeros_like(lpda_event_rate_cr)
    lpda_undet_cr = np.zeros_like(lpda_event_rate_cr)
    lpda_undet_core = np.zeros_like(lpda_event_rate_cr)

    #Following is a calibration test of the core spectrum
    cr_events_weights = np.zeros_like(full_evtrte_cr[0])
    core_events_weights = np.zeros_like(full_evtrte_cr[0])
    dep_core_events_weights = np.zeros_like(full_evtrte_cr[0])

    rad_bin_const = 50
    rad_bins = np.zeros( (len(spacing), rad_bin_const+1) )
    rads = np.zeros( (len(spacing), rad_bin_const) )
    for iS, space in enumerate(spacing):
        rad_bins[iS] = np.linspace(0, 1500, rad_bin_const+1)
        rads[iS] = (rad_bins[iS][1:] + rad_bins[iS][:-1]) / 2

    rr_trigs_dep = np.zeros( (len(spacing), len(e_center), rad_bin_const) )
    rr_trigs_core = np.zeros_like(rr_trigs_dep)
    rr_trigs_dep_zen = np.zeros( (len(spacing), len(coszens), len(e_center), rad_bin_const) )
    rr_trigs_core_zen = np.zeros_like(rr_trigs_dep_zen)
#    rad_tagging_total_bins_dep = np.zeros_like(rr_trigs_dep_zen)
#    rad_tagging_total_bins_core = np.zeros_like(rr_trigs_dep_zen)

    self_rad_tag = func_powerlaw(logEs, 18.4797, 6.8*10**-22, 5.06)
    other_rad_tag_high = func_powerlaw(logEs, 18.4797, 6.8*10**-22, 5.06+ grid_spacing*1000)
    other_rad_tag_low = func_powerlaw(logEs, 18.4797, -6.8*10**-22, 5.06+ grid_spacing*1000)
#    self_rad_tag = func_powerlaw(logEs + np.log10(f), 18.4797, 6.8*10**-22, 5.06)
#    other_rad_tag_high = func_powerlaw(logEs + np.log10(f), 18.4797, 6.8*10**-22, 5.06+ grid_spacing*1000)
#    other_rad_tag_low = func_powerlaw(logEs + np.log10(f), 18.4797, -6.8*10**-22, 5.06+ grid_spacing*1000)

    rad_tagging_dep_zen = np.zeros_like(rr_trigs_dep_zen)
    rad_tagging_core_zen = np.zeros_like(rr_trigs_dep_zen)
    rad_tagging_cr_zen = np.zeros_like(rr_trigs_dep_zen)

    rr_throw_dep = np.zeros_like(rr_trigs_dep_zen)
    rr_throw_core = np.zeros_like(rr_trigs_dep_zen)

    rr_event_rates_core = np.zeros_like(rr_trigs_core_zen)
    rr_event_rates_cr = np.zeros_like(rr_trigs_core_zen)
    rr_tag_event_rates_core = np.zeros_like(rr_trigs_core_zen)
    rr_tag_event_rates_cr = np.zeros_like(rr_trigs_core_zen)



    for iS, space in enumerate(spacing):
        for iE in range(len(e_bins) - 1):                
            for iC in range(len(coszen_bin_edges)-1):
                folder = file_prefix + f'{space:.2f}km'
#                eLog = e_bins[iE]
                Estring = f'{e_bins[iE]:.1f}log10eV'
                zString = f'coszen{r_coszen_bin_edges[iC]:.2f}'


                filename = f'RefractedCores_{type}_{thresh}muV_{space:.2f}km_{Estring}_{zString}_cores{cores}.hdf5'
#                filename = f'Grid_{type}_{thresh}muV_{space:.2f}km_{Estring}_{zString}_cores{cores}.hdf5'
#                filename = f'Grid_{type}_{space:.2f}km_{Estring}_{zString}_cores{cores}.hdf5'			#Use for MB 26muV, no thresh in name for 1000 cores

                output_filename = os.path.join(folder, Estring, filename)
                fin = h5py.File(output_filename, 'r')
                print(f'output filename {output_filename}')

#                for key in fin.keys():
#                    print(key)
                #Possible we never triggered, check to see if key of 'trigger_names' exits
                keys = fin.attrs.keys()
                triggered = 'trigger_names' in keys
                if not triggered:
                    print(f'No triggers {space}km {e_center[iE]}eV {coszens[iC]}coszen')
                    reflection_fraction[iS][iE][iC] = 0
                    tot_trig_fraction[iS][iE][iC] = 0
                    Aeff[iS][iE][iC] = 0
                    continue

                trig_mask, trig_index = hdau.trigger_mask(fin, trigger = 'LPDA_2of4_3.5sigma', station = station)
                ra_mask, rb_mask, da_mask, db_mask, zz = hdau.multi_array_direction_masks(fin, antennas, station, trig_index)

                refl_mask = ra_mask | rb_mask

                xx = np.array(fin['xx'])
                yy = np.array(fin['yy'])
                rr = (xx**2 + yy**2)**0.5
                rr_dig = np.digitize(rr, rad_bins[iS]) - 1
                for iR, r in enumerate(rr):
                    nR = rr_dig[iR]
                    rr_throw_dep[iS][iC][iE][nR] += 1
                    engN = np.digitize(e_center_shift[iE], e_bins) - 1
                    if (engN < 0) or (engN >= len(e_center_shift) ):
                        continue
                    rr_throw_core[iS][iC][engN][nR] += 1
                rr = rr[trig_mask]
                rr = rr[refl_mask]
#                print(f'rr check {rr}')


                rr_dig = np.digitize(rr, rad_bins[iS]) - 1
                #As is would get the radi for energy deposited, want to relate the energy to the core energy causing it to get plot of cores
                for iR, r in enumerate(rr):
                    nR = rr_dig[iR]
                    engN = np.digitize(e_center_shift[iE], e_bins) - 1
                    if (engN < 0) or (engN >= len(e_center_shift) ):
                        continue
#                    print(f'r check of {r} between {rad_bins[iS][nR]}-{rad_bins[iS][nR+1]}')
#                    print(f'eng check of {logEs[iE]} shifted to {e_center_shift[iE]} between {e_bins[engN]}-{e_bins[engN+1]}')
                    rr_trigs_dep[iS][iE][nR] += 1
                    rr_trigs_core[iS][engN][nR] += 1
                    rr_trigs_dep_zen[iS][iC][iE][nR] += 1
                    rr_trigs_core_zen[iS][iC][engN][nR] += 1
                    if (r <= self_rad_tag[iE]) or (other_rad_tag_low[iE] <= r <= other_rad_tag_high[iE]):
                        rad_tagging_dep_zen[iS][iC][iE][nR] += 1
                        rad_tagging_core_zen[iS][iC][engN][nR] += 1



#            reflection_fraction[iS][iE] = sum(refl_mask)/len(refl_mask)
                reflection_fraction[iS][iE][iC] = sum(refl_mask)/cores
                tot_trig_fraction[iS][iE][iC] = len(refl_mask)/cores
                Aeff[iS][iE][iC] = reflection_fraction[iS][iE][iC] * space**2

                


            #Coszen input parameters were flipped compared to calculated weights. So need to reflip to have proper coszen order matching generation
#            reflection_fraction[iS][iE] = np.flip(reflection_fraction[iS][iE])
#            tot_trig_fraction[iS][iE] = np.flip(tot_trig_fraction[iS][iE])
#            Aeff[iS][iE] = np.flip(Aeff[iS][iE])
		####IS THIS BAD ABOVE? WHY FLIPPING HERE?????? - Its bad, not needed here





            if not (shift_digit[iE] >= len(shift_digit)):
                shift = shift_digit[iE]
                Aeff_shift[iS][shift] = Aeff[iS][iE]
            

        """
        Aeff_temp = np.zeros_like(logEs)
        for iE, logE in enumerate(logEs):
            spot = np.digitize(logE, e_bins)-1
            if spot < 0:
                Aeff_temp[iE] = 0
            else:
                Aeff_temp[iE] = reflection_fraction[iS][spot] * space ** 2
            print(f'logE of {logE} has spot {spot} in {e_bins}, for Aeff of {Aeff_temp[iE]}')
        """

        #This turns the arrays into trigger rates in radi bins
        rr_trigs_dep_zen[iS] /= rr_throw_dep[iS]
        rr_trigs_core_zen[iS] /= rr_throw_core[iS]
        rr_trigs_dep_zen[np.isnan(rr_trigs_dep_zen)] = 0
        rr_trigs_core_zen[np.isnan(rr_trigs_core_zen)] = 0
        rad_tagging_dep_zen[iS] /= rr_throw_dep[iS]
        rad_tagging_core_zen[iS] /= rr_throw_core[iS]
        rad_tagging_dep_zen[np.isnan(rad_tagging_dep_zen)] = 0
        rad_tagging_core_zen[np.isnan(rad_tagging_core_zen)] = 0

        #Now we convert bins into effective areas, each bin represents a disk w/min and max radii
        for iR, r in enumerate(rads[iS]):
            area_bin = np.pi * (rad_bins[iS][iR+1] ** 2 - rad_bins[iS][iR] ** 2) * 10**-6 	#rad in m originally, convert to km for Aeff
            print(f'area bin is {area_bin} for rad {r}, iR {iR}')
            rr_trigs_dep_zen[iS][...,iR] *= area_bin
            rr_trigs_core_zen[iS][...,iR] *= area_bin
            rad_tagging_dep_zen[iS][...,iR] *= area_bin
            rad_tagging_core_zen[iS][...,iR] *= area_bin



        for iE, logE in enumerate(logEs):
            for iC, coszen in enumerate(coszens):
                

                niC = len(coszens) - 1 - iC
                mask = ~np.isnan(shower_energies[iE][niC])
                engiEiC = shower_energies[iE][niC][mask]
                orig_shower_digit = np.digitize(np.log10(engiEiC), shower_E_bins)-1		#Digit of the origonal core energy
                engiEiC = engiEiC * f
                weightsiEiC = weights_shower_energies[iE][niC][mask]
                surf_shower_digit = np.digitize(np.log10(engiEiC), shower_E_bins)-1          	#Digit of the deposited energy based on core energy
                for n in range(len(engiEiC)):
                    if not (surf_shower_digit[n] == 7):
                        continue
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

                    rr_event_rates_cr[iS][iC][iE] += rr_trigs_core_zen[iS][iC][depN] * weightsiEiC[n]
                    rr_tag_event_rates_cr[iS][iC][iE] += rad_tagging_core_zen[iS][iC][depN] * weightsiEiC[n]
                    rr_event_rates_core[iS][iC][coreN] += rr_trigs_core_zen[iS][iC][depN] * weightsiEiC[n]
                    rr_tag_event_rates_core[iS][iC][coreN] += rad_tagging_core_zen[iS][iC][depN] * weightsiEiC[n]


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
                        
                    
                    #good method, just done as sum below now
#                    event_rate_cr_eng[iS][iE] += weightsiEiC[n] * Aeff[iS][engN][iC]
#                    event_rate_surface_shower[iS][engN] += weightsiEiC[n] * Aeff[iS][engN][iC]
#                    total_rate_cr[iS][iE] += weightsiEiC[n] * space ** 2
#                    total_rate_surface[iS][engN] += weightsiEiC[n] * space ** 2
                     #obsilete method
#                    event_rate_cr_eng[iS][iE] += weightsiEiC[n] * Aeff_shift[iS][engN][iC]
#                    event_rate_surface_shower[iS][engN] += weightsiEiC[n] * Aeff_shift[iS][engN][iC]
#                    event_rate_cr_eng[iS][iE] += weightsiEiC[n] * Aeff_temp[engN]
#                    event_rate_surface_shower[iS][engN] += weightsiEiC[n] * Aeff_temp[engN]


        for iE, logE in enumerate(logEs):
             event_rate_cr_eng[iS][iE] = sum(full_evtrte_cr[iS][iE])
             event_rate_surface_shower[iS][iE] = sum(full_evtrte_core[iS][iE])
             total_rate_cr[iS][iE] = sum(total_full_evtrte_cr[iS][iE])
             total_rate_surface[iS][iE] = sum(total_full_evtrte_core[iS][iE])

        


#        plt.scatter(e_center, reflection_fraction[iS], label=f'{space}m')
#        Aeff[iS] = reflection_fraction[iS] * space ** 2
#        plt.scatter(e_center, Aeff[iS], label=f'Area {space**2}km^2')


    """
    for iS, space in enumerate(spacing):
        plt.scatter(e_center, reflection_fraction[iS], label=f'{space**2}km^2')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Energy (log10eV)')
    plt.ylabel('Trigger Fraction Refl')
    plt.title('Trigger reflected fraction at {type}')
    plt.show()
    """
    for iS, space in enumerate(spacing):
        sum = 0
        for iE, eng in enumerate(logEs):
            if eng >= 18:
                sum += event_rate_surface_shower[iS][iE]
        print(f'Event rate for {space}km above 18eV is {sum}')
#    quit()
    """
	#Old in case I mess up
        for iE, logE in enumerate(logEs):
            for iC, coszen in enumerate(coszens):
                nC = np.digitize(coszen, lpda_cos_zen_bins) - 1
                nE = np.digitize(eng, lpda_logEs) - 1
                if (nC < 0) or (nC > len(lpda_zen_center)-1) or (nE < 0) or (nE > len(lpda_E_center)-1):
                    lpda_undet_cr[iE][iC] += full_evt_rte_cr[0][iE][iC]
                else:
                    lpda_event_rate_cr[iE][iC] += full_evtrte_cr[0][iE][iC] * lpda_trig_rates[nC][nE]
                    lpda_undet_cr[iE][iC] += full_evtrte_cr[0][iE][iC] * (1-lpda_trig_rates[nC][nE])
                 
                lpda_undet_core[iE][iC] += full_evt_rte_core[0][iE][iC]

                    
                    lpda_event_rate_core[iE][iC] += full_evtrte_core[0][iE][iC] * lpda_trig_rates[nC][nE]
                    lpda_undet_core[iE][iC] += full_evtrte_core[0][iE][iC] * (1-lpda_trig_rates[nC][nE])
    """


    #Currently have event rates in shape [iE][iC], ie
    #18.0 eV [ 0deg, 10deg, 20deg, ...]
    #19.0 eV [0deg, 10, 20, ...]
    #imshow plots as pixels, so we need to both flip the cosbins and then transpose in order to have result of
    # 60deg [18.0ev, 18.1ev, ...]
    # 50deg [18.0ev, 18.1ev, ...]

    for iE, eng in enumerate(logEs):
        lpda_event_rate_cr[iE] = np.flip(lpda_event_rate_cr[iE])
        lpda_event_rate_core[iE] = np.flip(lpda_event_rate_core[iE])
        lpda_undet_cr[iE] = np.flip(lpda_undet_cr[iE])
        lpda_undet_core[iE] = np.flip(lpda_undet_core[iE])
        cr_events_weights[iE] = np.flip(cr_events_weights[iE])
        core_events_weights[iE] = np.flip(core_events_weights[iE])
        dep_core_events_weights[iE] = np.flip(dep_core_events_weights[iE])

    
    #Core spectrum plots, can remove for speed as doesn't change much
    cr_events_weights = cr_events_weights.T
    masked4  = np.ma.masked_where(cr_events_weights == 0, cr_events_weights)
    plt.imshow(masked4, extent=[17, 20, coszen_bin_edges[0], coszen_bin_edges[len(coszen_bin_edges)-1]], aspect=5, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
    plt.colorbar(label=f'Events/stn/yr/km^2')
    plt.ylabel('Cos(zenith)')
    plt.xlabel('$E_{CR}$ eV')
    plt.title(f'CR Flux {type} {flux} atm overburden {atm_overburden}')
#    plt.show()
    plt.savefig(f'CoreAnalysis/plots/CR_Flux_{type}_{atm_overburden}atm.png')
    core_events_weights = core_events_weights.T
    masked4  = np.ma.masked_where(core_events_weights == 0, core_events_weights)
    plt.imshow(masked4, extent=[17, 20, coszen_bin_edges[0], coszen_bin_edges[len(coszen_bin_edges)-1]], aspect=5, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
    plt.colorbar(label=f'Events/stn/yr/km^2')
    plt.ylabel('Cos(zenith)')
    plt.xlabel('$E_{Core}$ eV')
    plt.title(f'Core Flux {type} {flux} atm overburden {atm_overburden}')
#    plt.show()
    plt.savefig(f'CoreAnalysis/plots/Core_Flux_{type}_{atm_overburden}atm.png')
    dep_core_events_weights = dep_core_events_weights.T
    masked4  = np.ma.masked_where(dep_core_events_weights == 0, dep_core_events_weights)
    plt.imshow(masked4, extent=[17, 20, coszen_bin_edges[0], coszen_bin_edges[len(coszen_bin_edges)-1]], aspect=5, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
    plt.colorbar(label=f'Events/stn/yr/km^2')
    plt.ylabel('Cos(zenith)')
    plt.xlabel('$E_{Dep}$ eV')
    plt.title(f'Deposited Energy Flux {type} {flux} atm overburden {atm_overburden}')
#    plt.show()
    plt.savefig(f'CoreAnalysis/plots/Eng_Dep_Flux_{type}_{atm_overburden}atm.png')
    



    """
    ###Tagged event rates
    #
    lpda_event_rate_cr = lpda_event_rate_cr.T
    masked4  = np.ma.masked_where(lpda_event_rate_cr == 0, lpda_event_rate_cr)
    plt.imshow(masked4, extent=[17, 20, coszen_bin_edges[0], coszen_bin_edges[len(coszen_bin_edges)-1]], aspect=5, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
    plt.colorbar(label=f'Events/stn/yr, total {np.sum(lpda_event_rate_cr):.4f}')
    plt.ylabel('Cos(zenith)')
    plt.xlabel('$E_{CR}$ eV')
    plt.title(f'Event Rate of CRs tagged by Up LPDA {type} Grid {grid_spacing:.2f}km, {thresh}muV, f={f}, {cores}cores/bin')
    plt.show()
    lpda_event_rate_core = lpda_event_rate_core.T
    masked4  = np.ma.masked_where(lpda_event_rate_core == 0, lpda_event_rate_core)
    plt.imshow(masked4, extent=[17, 20, coszen_bin_edges[0], coszen_bin_edges[len(coszen_bin_edges)-1]], aspect=5, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
    plt.colorbar(label=f'Events/stn/yr, total {np.sum(lpda_event_rate_core):.4f}')
    plt.ylabel('Cos(zenith)')
    plt.xlabel('$E_{Core}$ eV')
    plt.title(f'Event Rate of Cores tagged by Up LPDA {type} Grid {grid_spacing:.2f}km, {thresh}muV, f={f}, {cores}cores/bin')
    plt.show()
    lpda_undet_cr = lpda_undet_cr.T
    masked4  = np.ma.masked_where(lpda_undet_cr == 0, lpda_undet_cr)
    plt.imshow(masked4, extent=[17, 20, coszen_bin_edges[0], coszen_bin_edges[len(coszen_bin_edges)-1]], aspect=5, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
    plt.colorbar(label=f'Events/stn/yr, total {np.sum(lpda_undet_cr):.4f}')
    plt.ylabel('Cos(zenith)')
    plt.xlabel('$E_{CR}$ eV')
    plt.title(f'Event Rate of CRs NOT tagged by Up LPDA {type} Grid {grid_spacing:.2f}km, {thresh}muV, f={f}, {cores}cores/bin')
    plt.show()
    lpda_undet_core = lpda_undet_core.T
    masked4  = np.ma.masked_where(lpda_undet_core == 0, lpda_undet_core)
    plt.imshow(masked4, extent=[17, 20, coszen_bin_edges[0], coszen_bin_edges[len(coszen_bin_edges)-1]], aspect=5, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
    plt.colorbar(label=f'Events/stn/yr, total {np.sum(lpda_undet_core):.4f}')
    plt.ylabel('Cos(zenith)')
    plt.xlabel('$E_{Core}$ eV')
    plt.title(f'Event Rate of Cores NOT tagged by Up LPDA {type} Grid {grid_spacing:.2f}km, {thresh}muV, f={f}, {cores}cores/bin')
    plt.show()
    """





    print(f'rr trigs core {rr_trigs_core}')
    print(f'rr trigs dep {rr_trigs_dep}')

    rad_tagging_dep = np.zeros( (len(spacing), len(coszens)) )
    rad_tagging_core = np.zeros( (len(spacing), len(coszens)) )

    for iS, space in enumerate(spacing):
        for iE, eng in enumerate(logEs):
            rr_trigs_core[iS][iE] = np.flip(rr_trigs_core[iS][iE])
            rr_trigs_dep[iS][iE] = np.flip(rr_trigs_dep[iS][iE])
            reflection_fraction[iS][iE] = np.flip(reflection_fraction[iS][iE])
            tot_trig_fraction[iS][iE] = np.flip(tot_trig_fraction[iS][iE])
            Aeff[iS][iE] = np.flip(Aeff[iS][iE])
            Aeff_shift[iS][iE] = np.flip(Aeff_shift[iS][iE])
            full_evtrte_cr[iS][iE] = np.flip(full_evtrte_cr[iS][iE])
            full_evtrte_core[iS][iE] = np.flip(full_evtrte_core[iS][iE])

        for iC, coszen in enumerate(coszens):
            for iE, eng in enumerate(logEs):
                rr_trigs_core_zen[iS][iC][iE] = np.flip(rr_trigs_core_zen[iS][iC][iE])
                rr_trigs_dep_zen[iS][iC][iE] = np.flip(rr_trigs_dep_zen[iS][iC][iE])
                rad_tagging_dep_zen[iS][iC][iE] = np.flip(rad_tagging_dep_zen[iS][iC][iE])
                rad_tagging_core_zen[iS][iC][iE] = np.flip(rad_tagging_core_zen[iS][iC][iE])

                rr_event_rates_cr[iS][iC][iE] = np.flip(rr_event_rates_cr[iS][iC][iE])
                rr_event_rates_core[iS][iC][iE] = np.flip(rr_event_rates_core[iS][iC][iE])
                rr_tag_event_rates_cr[iS][iC][iE] = np.flip(rr_tag_event_rates_cr[iS][iC][iE])
                rr_tag_event_rates_core[iS][iC][iE] = np.flip(rr_event_rates_core[iS][iC][iE])


            rad_tagging_dep[iS][iC] = np.sum(rad_tagging_dep_zen[iS][iC]) / np.sum(rr_trigs_dep_zen[iS][iC])
            rad_tagging_core[iS][iC] = np.sum(rad_tagging_core_zen[iS][iC]) / np.sum(rr_trigs_core_zen[iS][iC])



    #As m, c, c0
    hor_prop_eqn = [18.45, 6.8*10**-22, 5.06]

    cos_edges = np.rad2deg(cos_edges)
    #Plots showing radial distribution per coszen bin
    #commented out absolute rates, now does event rates
    for iS, space in enumerate(spacing):
        for iC, coszen in enumerate(coszens):        
            ylabel = 'Radius (m)'
            xlabel = '$E_{Core}$ eV'
            title = f'Rad EvtRate {type} {cos_edges[iC]:.1f}-{cos_edges[iC+1]:.1f}deg, {100*rad_tagging_core[iS][iC]:.2f}% tagged, {thresh}muV, f={f}, {cores}cores/bin'
            colorlabel = f'Evts/Stn/Yr, total {np.sum(rr_event_rates_core[iS][iC]):.4f}'
            plot_colormap(rr_event_rates_core[iS][iC], extent=[17, 20, rad_bins[iS][0], rad_bins[iS][rad_bin_const-1]], aspect=1/500,
                          title=title, xlabel=xlabel, ylabel=ylabel, colorlabel=colorlabel, ylim=[0, 1500])
            plt.plot(logEs, func_powerlaw(logEs + np.log10(f), 18.4797, 6.8*10**-22, 5.06), color='blue')
            plt.plot(logEs, func_powerlaw(logEs + np.log10(f), 18.4797, 6.8*10**-22, 5.06+grid_spacing*1000), color='green')
            plt.plot(logEs, func_powerlaw(logEs + np.log10(f), 18.4797, -6.8*10**-22, 5.06+grid_spacing*1000), color='green')
#            plt.plot(logEs, self_rad_tag, color='blue')
#            plt.plot(logEs, other_rad_tag_high, color='green')
#            plt.plot(logEs, other_rad_tag_low, color='green')
#            plt.show()
            plt.savefig(f'CoreAnalysis/plots/Rad_EvntRate_radOverlay_{type}_{thresh}muV_{f}f_{cores}coresPerBin.png')

            ylabel = 'Radius (m)'
            xlabel = '$E_{CR}$ eV'
            title = f'Rad EvtRate {type} {cos_edges[iC]:.1f}-{cos_edges[iC+1]:.1f}deg, {100*rad_tagging_core[iS][iC]:.2f}% tagged, {thresh}muV, f={f}, {cores}cores/bin'
            colorlabel = f'Evts/Stn/Yr, total {np.sum(rr_event_rates_cr[iS][iC]):.4f}'
            plot_colormap(rr_event_rates_cr[iS][iC], extent=[17, 20, rad_bins[iS][0], rad_bins[iS][rad_bin_const-1]], aspect=1/500,
                          title=title, xlabel=xlabel, ylabel=ylabel, colorlabel=colorlabel, ylim=[0, 1500])
#            plt.plot(logEs, self_rad_tag, color='blue')
#            plt.plot(logEs, other_rad_tag_high, color='green')
#            plt.plot(logEs, other_rad_tag_low, color='green')
#            plt.show()
            plt.savefig(f'CoreAnalysis/plots/Rad_EvntRate_{type}_{thresh}muV_{f}f_{cores}coresPerBin.png')


            """
#            rad_T = rr_trigs_core_zen[iS][iC].T
            rad_T = rr_event_rates_core[iS][iC].T
            rad_mask = np.ma.masked_where(rad_T == 0, rad_T)
            plt.imshow(rad_mask, extent=[17, 20, rad_bins[iS][0], rad_bins[iS][rad_bin_const-1] ], aspect=1/500, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
            plt.colorbar(label=f'Evts/Stn/Yr, total {np.sum(rr_event_rates_core[iS][iC]):.4f}')
            plt.plot(logEs, self_rad_tag, color='blue')
            plt.plot(logEs, other_rad_tag_high, color='green')
            plt.plot(logEs, other_rad_tag_low, color='green')
            plt.ylim(0, 1500)
            plt.ylabel('Radius (m)')
            plt.xlabel('$E_{Core}$ eV')
            plt.title(f'Rad EvtRate {type} {cos_edges[iC]:.1f}-{cos_edges[iC+1]:.1f}deg, {100*rad_tagging_core[iS][iC]:.2f}% tagged, {thresh}muV, f={f}, {cores}cores/bin')
            plt.show()
            """


    #Plots showing the radial distribution of all events
    for iS, space in enumerate(spacing):
        
        rad_T = rr_trigs_dep[iS].T
        rad_mask = np.ma.masked_where(rad_T == 0, rad_T)
        plt.imshow(rad_mask, extent=[17, 20, rad_bins[iS][0], rad_bins[iS][rad_bin_const-1] ], aspect=1/500, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
        plt.colorbar(label=f'Num of Trigs in Bin')
        plt.plot(logEs, func_powerlaw(logEs, 18.4797, 6.8*10**-22, 5.06), color='blue')
        plt.plot(logEs, func_powerlaw(logEs, 18.4797, 6.8*10**-22, 5.06+1000), color='green')
        plt.plot(logEs, func_powerlaw(logEs, 18.4797, -6.8*10**-22, 5.06+1000), color='green')
        plt.ylim(0, 1500)
        plt.ylabel('Radius (m)')
        plt.xlabel('$E_{Deposited}$ eV')
        plt.title(f'Radius Trigger Nums at {type}, {thresh}muV, f={f}, {cores}cores/bin')
#        plt.show()
        plt.savefig(f'CoreAnalysis/plots/Rad_TriggerNums_{type}_{thresh}muV_{f}f_{cores}coresPerBin.png')
        rad_T = rr_trigs_core[iS].T
        rad_mask = np.ma.masked_where(rad_T == 0, rad_T)
        plt.imshow(rad_mask, extent=[17, 20, rad_bins[iS][0], rad_bins[iS][rad_bin_const-1] ], aspect=1/500, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
        plt.colorbar(label=f'Num of Trigs in Bin')
        plt.plot(logEs, func_powerlaw(logEs + np.log10(f), 18.4797, 6.8*10**-22, 5.06), color='blue')
        plt.plot(logEs, func_powerlaw(logEs + np.log10(f), 18.4797, 6.8*10**-22, 5.06+1000), color='green')
        plt.plot(logEs, func_powerlaw(logEs + np.log10(f), 18.4797, -6.8*10**-22, 5.06+1000), color='green')
        plt.ylim(0, 1500)
        plt.ylabel('Radius (m)')
        plt.xlabel('$E_{Core}$ eV')
        plt.title(f'Radius Trigger Rate at {type}, {thresh}muV, f={f}, {cores}cores/bin')
#        plt.show()
        plt.savefig(f'CoreAnalysis/plots/Rad_TriggerRates_{type}_{thresh}muV_{f}f_{cores}coresPerBin.png')


        refl_frac = reflection_fraction[iS].T
        masked  = np.ma.masked_where(refl_frac == 0, refl_frac)
        plt.imshow(masked, extent=[17, 20, coszen_bin_edges[0], coszen_bin_edges[len(coszen_bin_edges)-1]], aspect=5, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
#        plt.imshow(reflection_fraction[iS].T, interpolation='nearest', aspect=5)
        plt.colorbar(label='Fraction of Reflected Triggers')
        plt.ylabel('Cos(zenith)')
        plt.xlabel('$E_{Deposited}$ eV')
        plt.title(f'Reflection Rate at {type}, {thresh}muV, f={f}, {cores}cores/bin')
#        plt.show()
        plt.savefig(f'CoreAnalysis/plots/ReflRate_{type}_{thresh}muV_{f}f_{cores}coresPerBin.png')

        refl_frac_shift = Aeff_shift[iS] / (space ** 2)
        print(f' refl shift = {refl_frac_shift}')
        refl_frac_shift = refl_frac_shift.T
        masked2 = np.ma.masked_where(refl_frac_shift == 0, refl_frac_shift)
        plt.imshow(masked2, extent=[17, 20, coszen_bin_edges[0], coszen_bin_edges[len(coszen_bin_edges)-1]], aspect=5, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
        plt.colorbar(label='Fraction of Reflected Triggers')
        plt.ylabel('Cos(zenith)')
        plt.xlabel('$E_{Core}$ w/' + f'f={f} eV')
        plt.title(f'Coupled Reflection Rate at {type}, {thresh}muV, f={f}, {cores}cores/bin')
#        plt.show()
        plt.savefig(f'CoreAnalysis/plots/CoupledReflRate_{type}_{thresh}muV_{f}f_{cores}coresPerBin.png')

        evt_rt_cr = full_evtrte_cr[iS].T
        masked3  = np.ma.masked_where(evt_rt_cr == 0, evt_rt_cr)
        plt.imshow(masked3, extent=[17, 20, coszen_bin_edges[0], coszen_bin_edges[len(coszen_bin_edges)-1]], aspect=5, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
        plt.colorbar(label=f'Events/stn/yr, total {np.sum(event_rate_cr_eng[iS]):.4f}')
        plt.ylabel('Cos(zenith)')
        plt.xlabel('$E_{CR}$ eV')
        plt.title(f'Event Rate of CR at {type}, {thresh}muV, f={f}, {cores}cores/bin')
#        plt.show()
        plt.savefig(f'CoreAnalysis/plots/EvtRate_CR_{type}_{thresh}muV_{f}f_{cores}coresPerBin.png')

        evt_rt_core = full_evtrte_core[iS].T
        masked4  = np.ma.masked_where(evt_rt_core == 0, evt_rt_core)
        plt.imshow(masked4, extent=[17, 20, coszen_bin_edges[0], coszen_bin_edges[len(coszen_bin_edges)-1]], aspect=5, interpolation='none', cmap=cmap, norm=matplotlib.colors.LogNorm())
        plt.colorbar(label=f'Events/stn/yr, total {np.sum(event_rate_surface_shower[iS]):.4f}')
        plt.ylabel('Cos(zenith)')
        plt.xlabel('$E_{Core}$ eV')
        plt.title(f'Event Rate of Cores at {type}, {thresh}muV, f={f}, {cores}cores/bin')
#        plt.show()
        plt.savefig(f'CoreAnalysis/plots/EvtRate_Core_{type}_{thresh}muV_{f}f_{cores}coresPerBin.png')



    


    print(f'Aeff')
    print(Aeff)
    print(f'Aeff shifted')
    print(Aeff_shift)


    Aeff_cos = np.zeros((len(spacing), len(coszens), len(logEs)))
    for iS, space in enumerate(spacing):
        for iE, logE in enumerate(logEs):
            for iC, coszen in enumerate(coszens):
                Aeff_cos[iS][iC][iE] += Aeff[iS][iE][iC]
            print(f'Aeff for {logE} is {Aeff[iS][iE]}')



    for iS, space in enumerate(spacing):
        for iC, coszen in enumerate(coszens):
            plt.scatter(logEs, Aeff_cos[iS][iC], label=f'{space}km, {np.rad2deg(cos_edges[iC])}-{np.rad2deg(cos_edges[iC+1])}deg')
#        plt.scatter(logEs, Aeff_no_cos[iS], label=f'{space}km spacing')
#        for iE, logE in enumerate(logEs):
#            plt.scatter(coszens, Aeff[iS][iE], label=f'{space}km, {logE}eV')
    plt.legend()
    plt.xlabel('Energy log10eV')
#    plt.xlabel('Cos zenith')
    plt.ylabel('Aeff km^2')
    plt.yscale('log')
    plt.title(f'Aeff per energy of core {type} {thresh}muV f={f}')
#    plt.show()
    plt.savefig(f'CoreAnalysis/plots/Aeff_Core_{type}_{thresh}muV_{f}f_{cores}coresPerBin.png')

    for iS, space in enumerate(spacing):
        plt.scatter(logEs, event_rate_cr_eng[iS], label=f'A {space**2}km^2, {np.sum(event_rate_cr_eng[iS]):.4f}evts/yr/stn')
    plt.legend()
    plt.xlabel('Energy (log10eV)')
#    plt.ylabel('Fraction of Reflected Triggers')
    plt.ylabel('Event Rate/yr/stn')
    plt.yscale('log')
    plt.title(f'Event Rate at {type}, {thresh}muV, f={f}, {cores}cores/bin')
#    plt.show()
    plt.savefig(f'CoreAnalysis/plots/EventRate_CR_{type}_{thresh}muV_{f}f_{cores}coresPerBin.png')

    for iS, space in enumerate(spacing):
        plt.scatter(logEs, event_rate_surface_shower[iS], label=f'A {space**2}km^2, {np.sum(event_rate_surface_shower[iS]):.4f}evts/yr/stn')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Energy of Core(log10eV)')
    plt.ylabel('Events/yr/stn')
    plt.title(f'Event rate as core energy, {type}, {thresh}muV, f={f}, {cores}cores/bin')
#    plt.show()
    plt.savefig(f'CoreAnalysis/plots/EventRate_Core_{type}_{thresh}muV_{f}f_{cores}coresPerBin.png')
    




    #######Method for coarser energy binning. Not sure useful/needed
    """
    e_bins_coarse = np.arange(17, 20.01, 0.5)
    e_center_coarse = (e_bins_coarse[1:] + e_bins_coarse[:-1])/2
    coarse_digit = np.digitize(logEs, e_bins_coarse)-1

    c_Aeff_cos = np.zeros((len(spacing), len(coszens), len(e_center_coarse)))
    c_event_rate_cr_eng = np.zeros((len(spacing), len(e_center_coarse)))
    c_event_rate_surface_shower = np.zeros_like(c_event_rate_cr_eng)
    for iS, space in enumerate(spacing):
        for iE, eng in enumerate(logEs):
            cE = coarse_digit[iE]
            for iC, coszen in enumerate(coszens):
                c_Aeff_cos[iS][iC][cE] += Aeff_cos[iS][iC][iE]
            c_event_rate_cr_eng[iS][cE] += event_rate_cr_eng[iS][iE]
            c_event_rate_surface_shower[iS][cE] += event_rate_surface_shower[iS][iE]

    num_for_mean = np.zeros(len(e_center_coarse))
    for iE, eng in enumerate(coarse_digit):
        cE = coarse_digit[iE]
        num_for_mean[cE] += 1
    for iS, space in enumerate(spacing):
        for iC, coszen in enumerate(coszens):
            for nE, num in enumerate(num_for_mean):
                c_Aeff_cos[iS][iC][nE] = c_Aeff_cos[iS][iC][nE]/num

    Aeff_cos = c_Aeff_cos
    event_rate_cr_eng = c_event_rate_cr_eng
    event_rate_surface_shower = c_event_rate_surface_shower





    for iS, space in enumerate(spacing):
        for iC, coszen in enumerate(coszens):
            plt.scatter(e_center_coarse, Aeff_cos[iS][iC], label=f'{space}km, {np.rad2deg(cos_edges[iC])}-{np.rad2deg(cos_edges[iC+1])}deg')
#            plt.scatter(logEs, Aeff_cos[iS][iC], label=f'{space}km, {np.rad2deg(cos_edges[iC])}-{np.rad2deg(cos_edges[iC+1])}deg')
#        plt.scatter(logEs, Aeff_no_cos[iS], label=f'{space}km spacing')
#        for iE, logE in enumerate(logEs):
#            plt.scatter(coszens, Aeff[iS][iE], label=f'{space}km, {logE}eV')
    plt.legend()
    plt.xlabel('Energy log10eV')
#    plt.xlabel('Cos zenith')
    plt.ylabel('Aeff km^2')
    plt.yscale('log')
    plt.title(f'Aeff per energy of core {type} {thresh}muV f={f}')
    plt.show()

    for iS, space in enumerate(spacing):
        plt.scatter(e_center_coarse, event_rate_cr_eng[iS], label=f'A {space**2}km^2, {np.sum(event_rate_cr_eng[iS]):.4f}evts/yr/stn')
#        plt.scatter(logEs, event_rate_cr_eng[iS], label=f'A {space**2}km^2, {sum(event_rate_cr_eng[iS]):.4f}evts/yr/stn')
    plt.legend()
    plt.xlabel('Energy (log10eV)')
#    plt.ylabel('Fraction of Reflected Triggers')
    plt.ylabel('Event Rate/yr/stn')
    plt.yscale('log')
    plt.title(f'Event Rate at {type}, {thresh}muV, f={f}, {cores}cores/bin')
    plt.show()

    for iS, space in enumerate(spacing):
        plt.scatter(e_center_coarse, event_rate_surface_shower[iS], label=f'A {space**2}km^2, {np.sum(event_rate_surface_shower[iS]):.4f}evts/yr/stn')
#        plt.scatter(logEs, event_rate_surface_shower[iS], label=f'A {space**2}km^2, {sum(event_rate_surface_shower[iS]):.4f}evts/yr/stn')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Energy of Core(log10eV)')
    plt.ylabel('Events/yr/stn')
    plt.title(f'Event rate as core energy, {type}, {thresh}muV, f={f}, {cores}cores/bin')
    plt.show()
    """
