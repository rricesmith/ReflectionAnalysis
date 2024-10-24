from NuRadioReco.utilities import units
import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.colors
import numpy as np
#import argparse
import h5py
import os
#import glob
#import hdf5AnalysisUtils as hdau
import argparse

#import CoreAnalysis.coreDataObjects as coreDataObjects
import coreDataObjects as CDO
import pickle
import CoreAnalysis.C00_coreAnalysisUtils as CDO_util
plt.style.use('plotsStyle.mplstyle')
from icecream import ic



"""
nxmax = 2000
with open(f"data/output_{flux}_{atm_overburden}.pkl", "rb") as fin:
#with open(f"data/output_{flux}_{atm_overburden}_{nxmax}xmax.pkl", "rb") as fin:
    shower_energies, weights_shower_energies = pickle.load(fin)
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data sets to plot')
    parser.add_argument('inputfilenamecores', type=str, help='path to Cores analysis pickle file')
    parser.add_argument('location', type=str, help='MB or SP location')
    parser.add_argument('--title_comment', type=str, default='', help='Extra string to put into title')
    parser.add_argument('--savePrefix', type=str, default='', help='Prefix to save file with to separate from others')
    parser.add_argument('--f', type=float, default=0.3, help='f value used in calculation, DEPRECIATED')
    parser.add_argument('--dB', type=float, default=40, help='dB value of reflector used, DEPRECIATED')
    args = parser.parse_args()
    core_input_file = args.inputfilenamecores
    location = args.location
    title_comment = args.title_comment
    savePrefix = args.savePrefix
    dB = args.dB
    f = args.f


    type = location

    """
    if type == 'MB':
        atm_overburden = 1000
        f = 0.06
#        core_input_file = 'data/CoreDataObjects_SP_5km_5000cores_26muV.pkl'
    else:
        atm_overburden = 680
        f = 0.3
#        core_input_file = 'data/CoreDataObjects_MB_5km_5000cores_26muV.pkl'
    flux = 'auger_19'
    """

    e_bins = np.arange(17, 20.01, 0.1)
#    e_bins = np.array([19, 20])
    e_center = (e_bins[1:] + e_bins[:-1])/2

    cores = 5000
#    spacing = [500, 750, 1000, 1500]
#    spacing = [2000]
    spacing = [1.7]
#    file_prefix = 'run/mooresCoreRefl'
#    file_prefix = 'run/CoreRefl'
    file_prefix = 'data/'

    xx = np.linspace(-1000, 1000, 100)
    yy = np.linspace(-1000, 1000, 100)
    zz = np.zeros( (len(xx), len(yy)) )


#    core_input_file = f'data/CoreDataObjects_Dipole2sigma_above_300mRefl_SP_1R_0.1f_0.0dB_1.5km_1000cores.pkl'

    with open(core_input_file, 'rb') as fin:
        CoreObjectsList = pickle.load(fin)


    zenBins = np.linspace(0, np.pi, 30)
    zenCent = (zenBins[1:] + zenBins[:-1])/2
    zenCent = np.rad2deg(zenCent)
    plotEng = 0
    zenCount = np.zeros_like(zenCent)
    scatterEng = []
    scatterZen = []
    scatterCount = []
    for core in CoreObjectsList:
        eng = core.e_center
        eventRate = core.totalEventRateAreaCore()
        if not eng == plotEng:
            for iZ, z in enumerate(zenCount):
                if z > 0:
#                    plt.scatter(plotEng, zenCent[iZ], c=z)
                    scatterEng.append(plotEng)
                    scatterZen.append(zenCent[iZ])
                    scatterCount.append(z)
            plotEng = eng
            zenCount = np.zeros_like(zenCent)
        hist = core.getZeniths()
#        print(f'shape event rate {np.shape(eventRate)} and hist {np.shape(hist)}')
#        print(f'event rate {eventRate}')
#        print(f'hist {hist}')
        eventRateWeight = eventRate / np.sum(hist)
        zenCount += hist * eventRateWeight

#    plt.scatter(plotEng, zenCent[iZ], c=z)


#        plt.bar(zenBins[:-1], core.getZeniths(), width=np.diff(zenBins), align='edge')
#    plt.scatter(scatterEng, scatterZen, c=scatterCount, norm=matplotlib.colors.LogNorm())
#    plt.clim(10**-5, 100)
#    plt.colorbar(label='# of triggers')
#    plt.xlabel('E core')
#    plt.ylabel('Signal angle at LPDA (deg)')
#    plt.title(title_comment + f' {dB}dB f={f}' )
#    plt.show()

    #plot again with second below data

#    core_input_file = f'data/CoreDataObjects_Dipole2sigma_below_300mRefl_SP_1R_0.3f_40.0dB_1.5km_1000cores.pkl'

    """
    with open(core_input_file, 'rb') as fin:
        CoreObjectsList = pickle.load(fin)


    zenBins = np.linspace(0, np.pi, 30)
    zenCent = (zenBins[1:] + zenBins[:-1])/2
    zenCent = np.rad2deg(zenCent)
    plotEng = 0
    zenCount = np.zeros_like(zenCent)
    scatterEng = []
    scatterZen = []
    scatterCount = []
    for core in CoreObjectsList:
        eng = core.e_center
        eventRate = core.totalEventRateCore()
        if not eng == plotEng:
            for iZ, z in enumerate(zenCount):
                if z > 0:
#                    plt.scatter(plotEng, zenCent[iZ], c=z)
                    scatterEng.append(plotEng)
                    scatterZen.append(zenCent[iZ])
                    scatterCount.append(z)
            plotEng = eng
            zenCount = np.zeros_like(zenCent)
        hist = core.getZeniths()
        eventRateWeight = eventRate / np.sum(hist)
        zenCount += hist * eventRateWeight

#    plt.scatter(plotEng, zenCent[iZ], c=z)
    """

    CDO_util.plotCoreDiagnostics(CoreObjectsList, title_comment=savePrefix)
    quit()


#        plt.bar(zenBins[:-1], core.getZeniths(), width=np.diff(zenBins), align='edge')
    plt.scatter(scatterEng, scatterZen, c=scatterCount, norm=matplotlib.colors.LogNorm())
    plt.clim(10**-5, 100)
    plt.colorbar(label='Event Rate')
    plt.xlabel('E core')
    plt.ylabel('Signal angle at Dipole (deg)')
    plt.title(f'{title_comment} Antenna Arrival Zen vs Energy')
#    plt.show()
    plt.savefig(f'plots/CoreAnalysis/{savePrefix}_ArrivalAngles_dual_above_0.1f_below_300m_40dB_0.3f.png')
    plt.clf()





    print(f'here1')


    print(f'plotting CR vs Shower energy flux')
    CDO_util.plotCrShowerHeatFlux(CoreObjectsList, loc=location)
    plt.savefig(f'plots/CoreAnalysis/{savePrefix}_CRvsShowerFlux.png')
    plt.clf()

    print(f'doing single aeff vs multi aeff test')
    CDO_util.plotCoreEnergyAeff(CoreObjectsList, label='Multi Aeff')
    CDO_util.plotCoreEnergyAeff(CoreObjectsList, singleAeff=True, label='Single Aeff')
    plt.legend()
    plt.ylim(10**-6, 10**2)
    plt.savefig(f'plots/CoreAnalysis/{savePrefix}_SingleVsMultiAeff.png')
    plt.clf()


    #Make XY heatmap of event rate
    for core in CoreObjectsList:
        x1, y1, z1 = core.addEventsToArea(xx, yy, zz)
        zz += z1
    ic(xx)
    ic(yy)
    ic(zz)
    plt.contourf(xx, yy, zz, cmap='YlOrRd', norm=matplotlib.colors.LogNorm()) 
    # plt.contourf(xx, yy, zz, cmap='YlOrRd') 
    plt.colorbar(label=f'Events/Stn/Yr, Net {np.sum(zz):.5f}')
    plt.scatter(0, 0, s=30, marker='x', color='black')
    plt.xlim(min(xx), max(xx))
    plt.ylim(min(yy), max(yy))
    plt.title(f'Core Event Rate Distribution over all CRs for {type}')
    plt.savefig(f'plots/CoreAnalysis/{savePrefix}_CoreEvntRate_XY_Distribution.png')
    plt.clf()
#    plt.show()

    print(f'here2')


    CDO_util.plotArrivalDirectionHist(CoreObjectsList, cut=[40, 140])
#    CDO_util.plotArrivalDirectionHist(CoreObjectsList)
    plt.savefig(f'plots/CoreAnalysis/{savePrefix}_ArrivalHist.png')
    plt.clf()

    #Make a plot of event rate as function of core energy
    CDO_util.plotCoreEnergyEventRate(CoreObjectsList)
    plt.savefig(f'plots/CoreAnalysis/{savePrefix}_CoreEnergy_Erate.png')
    plt.clf()
    print(f'here3')

    #Plot single vs multi aeff event rate result
    CDO_util.plotCoreEnergyEventRate(CoreObjectsList, label='multi Aeff', lowerLim=18)
    CDO_util.plotCoreEnergyEventRate(CoreObjectsList, singleAeff=True, label='single Aeff', lowerLim=18)
#    plt.yscale('log')
    plt.savefig(f'plots/CoreAnalysis/{savePrefix}_SingleVsMulti_CoreEnergy_Erate.png')
    plt.clf()
#    quit()

    #Make plot of event rate per CR energy
    CDO_util.plotCREnergyEventRate(CoreObjectsList)
    plt.savefig(f'plots/CoreAnalysis/{savePrefix}_CrEnergy_Erate.png')
    plt.clf()

    print(f'here4')

    #Make plot of Aeff in CR/Core energy
    CDO_util.plotCoreEnergyAeff(CoreObjectsList)
#    plt.yscale('log')
    plt.savefig(f'plots/CoreAnalysis/{savePrefix}_CoreEnergy_Aeff.png')
    plt.clf()


    #Not working, needs to be coded
#    CDO_util.plotCREnergyAeff(CoreObjectsList)
#    plt.savefig(f'plots/CoreAnalysis/{savePrefix}_CrEnergy_Aeff.png')
#    plt.clf()
    print(f'here5')

    #Make heatmap of energy/zenith event rate distribution
    CDO_util.plotCoreEnergyZenithHeat(CoreObjectsList)
    plt.savefig(f'plots/CoreAnalysis/{savePrefix}_CoreEngZen_Heatmap.png')
    plt.clf()
    print(f'here6')

    CDO_util.plotCrEnergyZenithHeat(CoreObjectsList)
    plt.savefig(f'plots/CoreAnalysis/{savePrefix}_CREngZen_Heatmap.png')
    plt.clf()
