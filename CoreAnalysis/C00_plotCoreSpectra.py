from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
import os
import glob
import hdf5AnalysisUtils as hdau
import pickle
import astrotools.auger as auger
from NuRadioReco.utilities import cr_flux
import utilities.plottingUtils as pltUtil
from matplotlib.legend_handler import HandlerTuple


dCos = 0.05
coszen_bin_edges = np.arange(np.cos(np.deg2rad(60)), 1.01, dCos)
coszens = 0.5 * (coszen_bin_edges[1:] + coszen_bin_edges[:-1])
n_zen_bins = len(coszen_bin_edges)-1
shower_E_bins = np.arange(17, 20.01, 0.1)
logEs = 0.5 * (shower_E_bins[1:] + shower_E_bins[:-1])


def get_Energies_Weights(type, year, loc):
    if loc == 'SP':
        with open(f'data/output_{type}_{year}_680.pkl', 'rb') as fin:
            shower_energies, weights_shower_energies, shower_xmax = pickle.load(fin)
            fin.close()
    elif loc == 'MB':
        with open(f'data/output_{type}_{year}_1000.pkl', 'rb') as fin:
            shower_energies, weights_shower_energies, shower_xmax = pickle.load(fin)
            fin.close()
    else:
        print(f'Location {loc} not made')
        quit()
    return shower_energies, weights_shower_energies


def coreDualHists(type, year):
    SP_shower_energies, SP_weights_shower_energies = get_Energies_Weights(type, year, 'SP')
    MB_shower_energies, MB_weights_shower_energies = get_Energies_Weights(type, year, 'MB')


    eLowDig = 17
    eHighDig = 20
    zLowDig = 9
    zHighDig = 1

    eLowRange = f'{shower_E_bins[eLowDig]:.1f}-{shower_E_bins[eLowDig+1]:.1f} bin'
    eHighRange = f'{shower_E_bins[eHighDig]:.1f}-{shower_E_bins[eHighDig+1]:.1f} bin'
    zLowRange = f'{np.rad2deg(np.arccos(coszen_bin_edges[zLowDig])):.0f}-{np.rad2deg(np.arccos(coszen_bin_edges[zLowDig-1])):.0f} deg'
    zHighRange = f'{np.rad2deg(np.arccos(coszen_bin_edges[zHighDig])):.0f}-{np.rad2deg(np.arccos(coszen_bin_edges[zHighDig-1])):.0f} deg'

    #First make a plot showing the change due to energy

    eLow = SP_shower_energies[eLowDig][zLowDig].flatten()
    mask = ~np.isnan(eLow)
    eLow = np.log10(SP_shower_energies[eLowDig][zLowDig][mask])
    eHigh = SP_shower_energies[eHighDig][zLowDig].flatten()
    mask = ~np.isnan(eHigh)
    eHigh = np.log10(SP_shower_energies[eHighDig][zLowDig][mask])

    plt.hist(eHigh, bins=shower_E_bins, weights=np.ones_like(eHigh)/2000, label=eHighRange + ', ' + zLowRange, edgecolor='blue', linestyle='-', fill=False, histtype='step')
    plt.hist(eLow, bins=shower_E_bins, weights=np.ones_like(eLow)/2000, label=eLowRange + ', ' + zLowRange, edgecolor='red', linestyle='--', fill=False, histtype='step')
    plt.xlabel(r'Core Energy ($log_{10}$eV)', fontsize=16)
    plt.ylabel('Normalized Cores', fontsize=16)    
    plt.legend()
    plt.savefig("plots/CoreAnalysis/CoreHistEnergyChange.png")
    plt.clf()


    #Now a plot showing change due to zenith
    zLow = SP_shower_energies[eHighDig][zLowDig].flatten()
    mask = ~np.isnan(zLow)
    zLow = np.log10(SP_shower_energies[eHighDig][zLowDig][mask])
    zHigh = SP_shower_energies[eHighDig][zHighDig].flatten()
    mask = ~np.isnan(zHigh)
    zHigh = np.log10(SP_shower_energies[eHighDig][zHighDig][mask])

    plt.hist(zHigh, bins=shower_E_bins, weights=np.ones_like(zHigh)/2000, label=eHighRange + ', ' + zHighRange, edgecolor='blue', linestyle='-', fill=False)
    plt.hist(zLow, bins=shower_E_bins, weights=np.ones_like(zLow)/2000, label=eHighRange + ', ' + zLowRange, edgecolor='red', linestyle='--', fill=False)
    plt.xlabel(r'Core Energy ($log_{10}$eV)', fontsize=16)
    plt.ylabel('Normalized Cores', fontsize=16)
    plt.legend()
    plt.savefig("plots/CoreAnalysis/CoreHistZenithChange.png")
    plt.clf()

    #Now a plot showing change from depth
    sp = SP_shower_energies[eHighDig][zLowDig].flatten()
    mask = ~np.isnan(sp)
    sp = np.log10(SP_shower_energies[eHighDig][zLowDig][mask])
    mb = MB_shower_energies[eHighDig][zLowDig].flatten()
    mask = ~np.isnan(mb)
    mb = np.log10(MB_shower_energies[eHighDig][zLowDig][mask])

    plt.hist(sp, bins=shower_E_bins, weights=np.ones_like(sp)/2000, label=eHighRange + ', ' + zLowRange + ' SP', edgecolor='blue', fill=False, histtype='step')
    plt.hist(mb, bins=shower_E_bins, weights=np.ones_like(mb)/2000, label=eHighRange + ', ' + zLowRange + ' MB', edgecolor='green', linestyle='-', fill=False, histtype='step')
    plt.xlabel(r'Core Energy ($log_{10}$eV)', fontsize=16)
    plt.ylabel('Normalized Cores', fontsize=16)
    plt.legend()
    plt.savefig("plots/CoreAnalysis/CoreHistLocChange.png")
    plt.clf()

    #TODO - plots above are just single bin. Check last plot of A02 to see how to make sum over all energy bins. Do I want to use one of those? Yes/no
    eLow = SP_shower_energies[eLowDig].flatten()
    mask = ~np.isnan(eLow)
    eLow = np.log10(eLow[mask])
    eLowWeights = SP_weights_shower_energies[eLowDig].flatten()[mask]
    eHigh = SP_shower_energies[eHighDig].flatten()
    mask = ~np.isnan(eHigh)
    eHigh = np.log10(eHigh[mask])
    eHighWeights = SP_weights_shower_energies[eHighDig].flatten()[mask]

    plt.plot([shower_E_bins[eHighDig], shower_E_bins[eHighDig]], [-1, 10**3], color='blue', linestyle='--')
    plt.text(shower_E_bins[eHighDig]+0.01, 0.08, 'CR ' + eHighRange, rotation=270, va='center', fontsize=12, color='blue')
    plt.plot([shower_E_bins[eLowDig], shower_E_bins[eHighDig]], [-1, 10**3], color='red', linestyle='--')
    plt.text(shower_E_bins[eLowDig]+0.01, 0.08, 'CR ' + eLowRange, rotation=270, va='center', fontsize=12, color='red')
    plt.hist(eLow, bins=shower_E_bins, weights=eLowWeights, label='CR ' + eLowRange, edgecolor='red', linestyle='-', fill=False, histtype='step')
    plt.hist(eHigh, bins=shower_E_bins, weights=eHighWeights, label='CR ' + eHighRange, edgecolor='blue', linestyle='-', fill=False, histtype='step')
    plt.xlabel(r'$E_{core}$ ($log_{10}$eV)', fontsize=16)
    plt.ylabel(r'$\phi$ ($km^{-2}$ $yr^{-1}$) per energy bin', fontsize=16)    
    plt.legend(loc='upper right')
    plt.xlim((17, 20))
    plt.ylim((0, 0.149))
    plt.savefig("plots/CoreAnalysis/CoreHistEnergyZenSumChange.png")
    pltUtil.savePlot(plt.gca(), 'CoreHistEnergyChange')
    plt.clf()

    weighted_mean = np.average(eLow, weights=eLowWeights)
    print(f'Average for Elow is {weighted_mean}')
    weighted_mean = np.average(eHigh, weights=eHighWeights)
    print(f'Average for Ehigh is {weighted_mean}')

    sp = SP_shower_energies[eHighDig].flatten()
    mask = ~np.isnan(sp)
    sp = np.log10(sp[mask])
    spWeights = SP_weights_shower_energies[eHighDig].flatten()[mask]
    mb = MB_shower_energies[eHighDig].flatten()
    mask = ~np.isnan(mb)
    mb = np.log10(mb[mask])
    mbWeights = MB_weights_shower_energies[eHighDig].flatten()[mask]

    plt.plot([shower_E_bins[eHighDig], shower_E_bins[eHighDig]], [-1, 10**3], color='black', linestyle='--')
    plt.text(shower_E_bins[eHighDig]+0.01, 0.03, 'CR ' + eHighRange, rotation=270, va='center', fontsize=12, )
    plt.hist(sp, bins=shower_E_bins, weights=spWeights, label='South Pole', edgecolor='blue', fill=False, histtype='step')
    plt.hist(mb, bins=shower_E_bins, weights=mbWeights, label='Moores Bay', edgecolor='green', linestyle='-', fill=False, histtype='step')
    plt.xlabel(r'$E_{core}$ ($log_{10}$eV)', fontsize=16)
    plt.ylabel(r'$\phi$ ($km^{-2}$ $yr^{-1}$) per energy bin', fontsize=16)    
    plt.legend()
    plt.xlim((17, 20))
    plt.ylim((0, 0.0545))
    plt.savefig("plots/CoreAnalysis/CoreHistLocZenSumChange.png")
    pltUtil.savePlot(plt.gca(), 'CoreHistLocChange')
    plt.clf()

    weighted_mean = np.average(sp, weights=spWeights)
    print(f'Average for sp is {weighted_mean}')
    weighted_mean = np.average(mb, weights=mbWeights)
    print(f'Average for mb is {weighted_mean}')



    return

def getSP_MB_coreFlux(type, year):

    SP_shower_energies, SP_weights_shower_energies = get_Energies_Weights(type, year, 'SP')
    MB_shower_energies, MB_weights_shower_energies = get_Energies_Weights(type, year, 'MB')

    SP_event_rate_core = np.zeros_like(logEs)
    MB_event_rate_core = np.zeros_like(logEs)

    #Didn't work
    #SP_event_rate_binned = np.zeros( (len(coszens), len(logEs)) )
    #MB_event_rate_binned = np.zeros_like(SP_event_rate_binned)

    #Using list method as it is clearer
    SP_event_rate_binned = []
    MB_event_rate_binned = []
    for iC, coszen in enumerate(coszens):
        engArray = []
        engArray2 = []
        for iE, logE in enumerate(logEs):
            engArray.append(0)
            engArray2.append(0)
        SP_event_rate_binned.append(engArray)
        MB_event_rate_binned.append(engArray2)

    #Iterate through cores, add weight (event rate) to bin based off of the core energy
    for iE, logE in enumerate(logEs):
        for iC, coszen in enumerate(coszens):
            SPmask = ~np.isnan(SP_shower_energies[iE][iC])
            SPengiEiC = SP_shower_energies[iE][iC][SPmask]
            SPweightsiEiC = SP_weights_shower_energies[iE][iC][SPmask]
            SP_surf_shower_digit = np.digitize(np.log10(SPengiEiC), shower_E_bins)-1


            MBmask = ~np.isnan(MB_shower_energies[iE][iC])
            MBengiEiC = MB_shower_energies[iE][iC][MBmask]
            MBweightsiEiC = MB_weights_shower_energies[iE][iC][MBmask]
            MB_surf_shower_digit = np.digitize(np.log10(MBengiEiC), shower_E_bins)-1
            for n in range(len(SPengiEiC)):
                engN = SP_surf_shower_digit[n]
                if (engN < 0) or (engN == len(logEs)):
                    continue
                SP_event_rate_core[engN] += SPweightsiEiC[n]
                SP_event_rate_binned[iC][engN] += SPweightsiEiC[n]
    #            SP_event_rate_binned[iC][engN] += SPweightsiEiC[n]
            for n in range(len(MBengiEiC)):
                engN = MB_surf_shower_digit[n]
                if (engN < 0) or (engN == len(logEs)):
                    continue
                MB_event_rate_core[engN] += MBweightsiEiC[n]
    #            MB_event_rate_binned[iC][engN] += MBweightsiEiC[n]
                MB_event_rate_binned[iC][engN] += MBweightsiEiC[n]

    return SP_event_rate_binned, SP_event_rate_core, MB_event_rate_binned, MB_event_rate_core


def plotCoreCRFluxBackground(year=19, text=False):

    SP_event_rate_binned_auger, SP_event_rate_core_auger, MB_event_rate_binned_auger, MB_event_rate_core_auger = getSP_MB_coreFlux('auger', year)

    SP_event_rate_binned_TA, SP_event_rate_core_TA, MB_event_rate_binned_TA, MB_event_rate_core_TA = getSP_MB_coreFlux('TA', 19)


    plt.scatter(logEs, SP_event_rate_core_auger, marker='v', color='blue', alpha=0.5)
    plt.scatter(logEs, SP_event_rate_core_TA, label='South Pole', marker='^', color='blue')
    plt.scatter(logEs, MB_event_rate_core_auger, marker='v', color='orange', alpha=0.5)
    plt.scatter(logEs, MB_event_rate_core_TA, label='Moores Bay', marker='^', color='orange')
    #plt.scatter(logEs, auger19_event_rate, label='Auger', marker='+')
    plt.scatter(logEs, auger19_event_rate, marker='v', color='gray')
    plt.scatter(logEs, TA19_event_rate, marker='^', color='black')
    if text:
        plt.text(18, 2.5*10**2, 'TA', color='black')
        plt.text(17.4, 2.5*10**2, 'Auger', color='gray')

    return




if __name__ == '__main__':
    type = 'auger'
    #type = 'TA'
    year = 19

    SP_event_rate_binned_auger, SP_event_rate_core_auger, MB_event_rate_binned_auger, MB_event_rate_core_auger = getSP_MB_coreFlux('auger', year)

    SP_event_rate_binned_TA, SP_event_rate_core_TA, MB_event_rate_binned_TA, MB_event_rate_core_TA = getSP_MB_coreFlux('TA', 19)

    coreDualHists(type, year)

    #Get the Auger & TA spectrum to plot
    auger19_event_rate = []
    TA19_event_rate = []
    for iE, logE in enumerate(logEs):
    #    auger_event_rate = auger.event_rate(shower_E_bins[iE], shower_E_bins[iE+1], zmax=90, area=1, year=19)  #Method for getting event rate from astrotools.auger. Result agrees with below method
        solid_angle = 2 * np.pi   # solid angle in sr
        ns_to_year_conversion = 1 / (3.17098 * 10**-17)
        m_to_km_conversion = 1000**2
        auger19_event_rate.append(cr_flux.get_flux_per_energy_bin(shower_E_bins[iE], shower_E_bins[iE+1], type='auger_19') * solid_angle * ns_to_year_conversion * m_to_km_conversion)
        TA19_event_rate.append(cr_flux.get_flux_per_energy_bin(shower_E_bins[iE], shower_E_bins[iE+1], type='TA_19') * solid_angle * ns_to_year_conversion * m_to_km_conversion)




    zen_edges = np.arccos(coszen_bin_edges)
    zen_edges[np.isnan(zen_edges)] = 0
    zen_edges = np.rad2deg(zen_edges)
    for iC, coszen in enumerate(coszens):
        #Not sure what this did, don't think its necessary?
    #    SP_event_rate_binned[iC][SP_event_rate_binned[iC] > SP_event_rate_binned[0][0]] = 0
    #    SP_event_rate_binned[iC][::-1].sort()        
        plt.plot(logEs, SP_event_rate_binned_TA[iC], label=f'{zen_edges[iC+1]:.0f}-{zen_edges[iC]:.0f}°', marker='^')
    #    plt.plot(logEs, SP_event_rate_binned_auger[iC], label=f'{zen_edges[iC+1]:.0f}-{zen_edges[iC]:.0f}°', marker='8')
    if False:
        plt.scatter(logEs, TA19_event_rate, marker='*', color='black')
        #plt.scatter(logEs, auger19_event_rate, marker='*', color='black')
        plt.text(18, 2.5*10**2, 'TA')
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12 )
    plt.tick_params(axis='both', which='minor', direction='in', top=True, right=True)
    plt.legend(loc='upper right', prop={'size':14})
    plt.yscale('log')
    plt.ylim((7*10**-8, 2*10**3))
    #plt.ylabel('Event Rate/yr/km^2')
    plt.ylabel(r'$\phi_{core}$ ($km^{-2}$ $yr^{-1}$) per steradian bin', fontsize=16)
    plt.xlabel(r'$E_{core}$ ($log_{10}$eV)', fontsize=16)
    #plt.title('Core Energy Spectrum South Pole')
    plt.savefig("plots/CoreAnalysis/CoreEngSpectrumSP")
    #plt.show()
    pltUtil.savePlot(plt.gca(), 'CoreEngSpectrumSP')
    plt.clf()


    for iC, coszen in enumerate(coszens):
        plt.plot(logEs, MB_event_rate_binned_TA[iC], label=f'{zen_edges[iC+1]:.0f}-{zen_edges[iC]:.0f}°', marker='o', linestyle='--')
    #    plt.plot(logEs, MB_event_rate_binned_auger[iC], label=f'{zen_edges[iC+1]:.0f}-{zen_edges[iC]:.0f}°', marker='8', linestyle='--')
    if False:
        plt.scatter(logEs, TA19_event_rate, marker='*', color='black')
        plt.text(18, 2.5*10**2, 'TA')
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12 )
    plt.tick_params(axis='both', which='minor', direction='in', top=True, right=True)
    plt.legend(loc='upper right', prop={'size':14})
    plt.yscale('log')
    #plt.ylabel('Event Rate/yr/km^2')
    plt.ylabel(r'$\phi_{core}$ ($km^{-2}$ $yr^{-1}$) per steradian bin', fontsize=16)
    plt.xlabel(r'$E_{core}$ ($log_{10}$eV)', fontsize=16)
    plt.ylim((7*10**-8, 2*10**3))

    #plt.title('Core Energy Spectrum')
    #plt.title('Core Energy Spectrum Moores Bay')
    plt.savefig("plots/CoreAnalysis/CoreEngSpectrumMB")
    #plt.show()
    pltUtil.savePlot(plt.gca(), 'CoreEngSpectrumMB')
    plt.clf()



    #plt.scatter(logEs, SP_event_rate_core, label='TA MB')
    #plt.scatter(logEs, MB_event_rate_core, label='Auger MB', marker='^')

    plt.scatter(logEs, auger19_event_rate, marker='v', color='gray')
    plt.scatter(logEs, TA19_event_rate, label=r'E_${CR}$, Cosmic Ray Flux',marker='^', color='black')
    plt.scatter(logEs, SP_event_rate_core_auger, marker='v', color='blue', alpha=0.5)
    plt.scatter(logEs, SP_event_rate_core_TA, label=r'E_${core}$, South Pole', marker='^', color='blue')
    plt.scatter(logEs, MB_event_rate_core_auger, marker='v', color='orange', alpha=0.5)
    plt.scatter(logEs, MB_event_rate_core_TA, label=r'E_${core}$, Moores Bay', marker='^', color='orange')
    #plt.scatter(logEs, auger19_event_rate, label='Auger', marker='+')
#    plt.text(18, 2.5*10**2, 'TA', color='black')
#    plt.text(17.4, 2.5*10**2, 'Auger', color='gray')

    #plt.text(17.3, 10**-7, f'{type} {year}', ha='center', va='center', fontsize=14)
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12 )
    plt.tick_params(axis='both', which='minor', direction='in', top=True, right=True)

    #legend1 = plt.legend(prop={'size':16})
    #plt.gca().add_artist(legend1)
    legend1 = plt.legend(handles=[(plt.scatter([], [], marker='^', color='black'), plt.scatter([], [], marker='v', color='gray')),
                                (plt.scatter([], [], marker='^', color='blue'), plt.scatter([], [], marker='v', color='blue', alpha=0.5)),
                                (plt.scatter([], [], marker='^', color='orange'), plt.scatter([], [], marker='v', color='orange', alpha=0.5))],
                                labels=[r'$E_{CR}$, Cosmic Ray Flux', r'$E_{core}$, South Pole', r'$E_{core}$, Moores Bay'],
                                loc='upper right', prop={'size':16}, handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.gca().add_artist(legend1)
    legend2 = plt.legend(handles=[plt.scatter([], [], marker='^', facecolor='none', edgecolor='black', label='TA'),
                                plt.scatter([], [], marker='v', facecolor='none', edgecolor='black', label='Auger')],
                                loc='upper right', prop={'size':12}, bbox_to_anchor=(0.996, 0.7856))
    plt.gca().add_artist(legend2)

    plt.yscale('log')
    #plt.ylabel(r'$\phi_{core}$ ($km^{-2}$ $yr^{-1}$)', fontsize=16)
    #plt.xlabel('$E_{core}$ (eV)', fontsize=16)
    plt.ylabel(r'$\phi$ ($km^{-2}$ $yr^{-1}$) per energy bin', fontsize=16)
    plt.xlabel(r'$E$ ($log_{10}$eV)', fontsize=16)
    #plt.title('Core Energy Spectrum')
    plt.ylim((10**-3, 10**3))
    plt.savefig("plots/CoreAnalysis/CoreEngSpectrumTotal")
    #plt.show()
    pltUtil.savePlot(plt.gca(), 'CoreEngSpectrumTotal')
    plt.clf()

    print(f'Done')
