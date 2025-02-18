import os
import pickle
import argparse
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import astrotools.auger as auger
import NuRadioReco.utilities.cr_flux as cr_flux
from NuRadioReco.utilities import units

import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*', 'x', '^', 'v', 'd'))
color = itertools.cycle(('black', 'blue', 'green', 'orange', 'brown'))
#plt.style.use('plotsStyle.mplstyle')

def dBfactor(dB):
    return 10**(-dB/20)

def linefit(x, min=17):
    vals = .8/3 * (x-min) + 0.2
    vals[x<min] = 0
    vals[vals > 1] = 1
    return vals

def piecefit(x):
    vals = linefit(x)
    vals[x < 17.2] = 0
    return vals

#def logfit(x, min=17, a=0.14, b=0.16, c=1):    #Hand Fit
def logfit(x, min=17, a=0.49, b=0.22, c=0.78):    #Scipy Fit
    vals = np.log(a*(x-min-.3))*b + c
    vals[vals < 0] = 0
    vals[np.isnan(vals)] = 0
    return vals

def set_bad_imshow(array, value):
    ma = np.ma.masked_where(array == value, array)
    cmap = matplotlib.cm.viridis
    cmap.set_bad(color='white')
    return ma, cmap



def singleBinERate(eLow, eHigh, zLow, zHigh, Aeff):
#    print(f'check elow {eLow} ehigh {eHigh} zlow {zLow} zhigh {zHigh} aeff {Aeff}')
#    high = auger.event_rate(eLow, eHigh, zHigh, Aeff)
#    low = auger.event_rate(eLow, eHigh, zLow, Aeff)
#Need geometric exposure?
    #Full geometric exposure is pi * (1+cos)(1-cos)
    #auger.event_rate has 2*pi*(1-cos), so just need to remove 2 and add other factor
    high = auger.event_rate(eLow, eHigh, zHigh, Aeff * 0.5*(1 + np.cos(np.deg2rad(zHigh))))
    low = auger.event_rate(eLow, eHigh, zLow, Aeff * 0.5*(1 + np.cos(np.deg2rad(zLow))))

#What about using NuRadioReco cr_flux?
#    high = cr_flux.get_cr_event_rate(log10energy=eLow, zenith=zHigh*units.deg, a_eff=Aeff*units.km) * 10**eLow*units.eV
#    low = cr_flux.get_cr_event_rate(log10energy=eHigh, zenith=zLow*units.deg, a_eff=Aeff*units.km) * 10**eHigh*units.eV
#    print(f'for eLow-eHigh {eLow}-{eHigh} and zLow-zHigh {zLow}-{zHigh}')
#    print(f'high of {high/units.year} and low {low/units.year}')

    return high - low

def binnedAeff(energies, zeniths, nTrig, nThrown, trigMask, xLocs, yLocs, maxR, nRadBins=20, binWidth=.1, logScale=False):
    energy = energies
    if not logScale:
        energy = np.log10(energies)

    rads = np.sqrt(xLocs**2 + yLocs**2)

    engBins = np.arange(np.min(energy), np.max(energy), binWidth)
    cosZenBins = np.arange(1, 0.5, -0.05)
    zenBins = np.arccos(cosZenBins)
    zenBins[np.isnan(zenBins)] = 0

    radBins = np.linspace(0, maxR, nRadBins)
    nTrigBins = np.zeros((len(engBins)-1, len(zenBins)-1))
    nThrowBins = np.zeros_like(nTrigBins)

    rDigitize = np.digitize(rads, radBins)-1
    zDigitize = np.digitize(zeniths, zenBins)-1


    for iE in range(len(engBins)-1):
        lowMask = engBins[iE] <= energy
        highMask = energy < engBins[iE+1]
        engMask = lowMask & highMask
        for iZ in range(len(zenBins)-1):
            lowMask = zenBins[iZ] <= zeniths
            highMask = zeniths < zenBins[iZ+1]
            zenMask = lowMask & highMask

#            fMask = trigMask & engMask & zenMask
            fMask = engMask & zenMask
            tMask = trigMask[fMask]
            print(f'shape tmask {np.shape(tMask)}')
            print(f'shape before trigged {np.shape(nTrig)}')
            trigged = nTrig[fMask]
#            trigged = trigged[tMask]
            print(f'shape of trigged {np.shape(trigged)}')
            thrown = nThrown[fMask]
#            thrown = thrown[tMask]
            print(f'shape of rDigitize {np.shape(rDigitize)}')
            rdigits = rDigitize[fMask]
            print(f'shape of rdigits {np.shape(rdigits)}')
#            rdigits = rdigits[tMask]
            zdigits = zDigitize[fMask]
#            zdigits = zdigits[tMask]

            print(f'doing thrown {thrown} and trigged {trigged} and rdig {rdigits} and zdig {zdigits}')
            for iT in range(len(thrown)):
                rs = rdigits.flatten()
                for r in rs:
                    nThrowBins[r][zdigits[0]] += 1

            rdigits = rdigits[tMask]
#            if trigged.size == 0:
            for r in rdigits:
#                for iT in range(len(trigged)):
                nTrigBins[r][zdigits[0]] += 1

    print(f'final ntrig {nTrigBins}')
    print(f'final throw {nThrowBins}')

    nThrowBins[nThrowBins == 0] = 1
    nTrigBins = nTrigBins / nThrowBins
    aeffBins = np.zeros_like(nTrigBins)

    aeffBins = nTrigBins
#    radBins *= 10**-3   #convert from m to km
#    for iE in range(len(radBins)-1):
#        areaHigh = np.pi * radBins[iR+1]**2
#        areaLow = np.pi * radBins[iR]**2
#        aeffBins[iR] = nTrigBins[iR] * (areaHigh - areaLow)

    return aeffBins, nTrigBins, engBins, zenBins
"""
dd_rate = auger.event_rate(np.log10(np.min(energy[dd_mask])), np.log10(np.max(energy[dd_mask])), np.rad2deg(np.max(zenith[dd_mask])), Aeff_dd)
def returnAugerRate(aeffBins, radBins, zenBins):
    eRate = 0
    rCenter = (radBins[1:] + radBins[:-1])/2
    zCenter = (zenBins[1:] + zenBins[:-1])/2
    zCenter = np.rad2deg(zCenter)
    for iR in rCenter:
        for iZ in zCenter:
            
"""
def getERate(energy, zenith, xs, ys, trigMask, ntrig, nthrow, length, title='', plot=True):
    AREA = (length/2)**2 * np.pi
#    maxR = np.sqrt( 2 * (length/2)**2 ) * 1000  #do in m, old for square throw
    maxR = (length/2) * 1000  #do in m, new for circular throw
#    radBins = np.linspace(0, maxR*1.1, 20)
    radBins = np.linspace(0, maxR, 2)
#    engBins = np.arange(np.log10(np.min(energy)), 20.1, .2)
#    cosZenBins = np.arange(1, -0.001, -0.05)
    engBins = np.arange(15.9, 20.5, 0.2)       #Standard Fine Binning
#    engBins = np.arange(15.9, 20.5, 0.5)        #Standard Coarse Binning
#    engBins = np.arange(15.6, 20.5, 0.5)        #Test Binning of SP
#    engBins = np.arange(16, 20.1, 0.2)

    cosZenBins = np.arange(1, -0.001, -0.2)
#    cosZenBins = np.flip(np.linspace(0, 1, 7))     #Anna's bins
    zenBins = np.arccos(cosZenBins)
    zenBins[np.isnan(zenBins)] = 0



    rads = np.sqrt(xs**2 + ys**2)
    e_dig = np.digitize(np.log10(energy), engBins) - 1
    z_dig = np.digitize(zenith, zenBins) - 1
    r_dig = np.digitize(rads, radBins)-1

    trig = np.zeros( (len(engBins)-1, len(zenBins)-1, len(radBins)-1) )
    throw = np.zeros( (len(engBins)-1, len(zenBins)-1, len(radBins)-1) )

    trigBins = []
    throwBins = []
    engs = []
    for iE, eng in enumerate(energy):
        nE = e_dig[iE]
        zE = z_dig[iE]
        rE = r_dig[iE]
        for iR, r in enumerate(rE):
            if r == len(radBins)-1:
#                print(f'r of {r} in bins {radBins}, maxR {maxR}')
                continue
            if zE == len(zenBins)-1:
                print(f'ze of {zE} in bins {zenBins} from zen {zenith[iE]}')
#            throw[nE,zE,r] += 1
#            trig[nE,zE,r] += int(trigMask[iE,iR])
#            print(f'r dig {r} corresponding to {rads[iE][iR]}')
            throw[nE][zE][r] += 1
            trig[nE][zE][r] += int(trigMask[iE][iR])

    #Poisson variance
#    error_trig = trig + trig
#    error_trig_low = trig - trig
    #Gaussian variance
    error_trig = trig + np.sqrt(trig)
    error_trig_low = trig - np.sqrt(trig)
    error_trig_low[error_trig_low < 0] = 0



    plotRadBins = radBins / 1000
    areaBins = np.zeros(len(radBins)-1)
    for iR in range(len(areaBins)):
        areaBins[iR] = np.pi * (plotRadBins[iR+1]**2 - plotRadBins[iR]**2)

    """
    for iE in range(len(engBins)-1):
        if not 17.1 < engBins[iE] < 17.2:
            print(f'eng {engBins[iE]}, skip')
            continue
        print(f'iZ 1 for {np.rad2deg(zenBins[1])}')
        plotArea = areaBins / (np.pi * (length/2)**2)
        plt.scatter( (plotRadBins[1:]+plotRadBins[:-1])/2, throw[iE][1], label='Num Throws')
        plt.scatter( (plotRadBins[1:]+plotRadBins[:-1])/2, plotArea * 1600*4, label='Area_bin/Area_total')
        plt.legend()
        plt.show()
    """



    plotEngRadRate = False
    if plotEngRadRate:
        throwrate = np.sum(throw,axis=1)
        for iE in range(len(engBins)-1):
            throwrate[iE] /= areaBins
#        throwrate = np.sum(throw,axis=0)
#        for iZ in range(len(cosZenBins)-1):
#            throwrate[iZ] /= areaBins

        throwrate, cmap = set_bad_imshow(throwrate, 0)
        plt.imshow(throwrate.T, extent=[min(engBins), max(engBins), max(plotRadBins), min(plotRadBins)], aspect='auto', cmap=cmap)
#        plt.imshow(throwrate.T, extent=[max(zenBins), min(zenBins), max(plotRadBins), min(plotRadBins)], aspect='auto', cmap=cmap)
        plt.xlabel('Energy (log10eV)')
#        plt.xlabel('Cos Zen')
        plt.ylabel('Radius from Station (km)')
        plt.colorbar(label=f'Density n_trig_bin/Area_bin')
        plt.title(title)
        plt.show()
        quit()

        trate = np.sum(trig, axis=1)
    #        plt.imshow(rrate.T, extent=[min(engBins), max(engBins), max(radBins), min(radBins)], aspect='auto', interpolation='none', norm=matplotlib.colors.LogNorm())
        trate, cmap = set_bad_imshow(trate, 0)
        plt.imshow(trate.T, extent=[min(engBins), max(engBins), max(plotRadBins), min(plotRadBins)], aspect='auto', cmap=cmap)
    #        plt.imshow(rrate.T, extent=[min(engBins), max(engBins), max(radBins), min(radBins)], aspect='auto', norm=matplotlib.colors.LogNorm())
        plt.xlabel('Energy (log10eV)')
        plt.ylabel('Radius from Station (km)')
        plt.colorbar(label=f'Density n_trig_bin/Area_bin')
        plt.title(title)
        plt.show()


        trrate = np.sum(throw, axis=1)
    #        plt.imshow(rrate.T, extent=[min(engBins), max(engBins), max(radBins), min(radBins)], aspect='auto', interpolation='none', norm=matplotlib.colors.LogNorm())
        trrate, cmap = set_bad_imshow(trrate, 0)
        plt.imshow(trrate.T, extent=[min(engBins), max(engBins), max(plotRadBins), min(plotRadBins)], aspect='auto', cmap=cmap)
    #        plt.imshow(rrate.T, extent=[min(engBins), max(engBins), max(radBins), min(radBins)], aspect='auto', norm=matplotlib.colors.LogNorm())
        plt.xlabel('Energy (log10eV)')
        plt.ylabel('Radius from Station (km)')
        plt.colorbar(label=f'Num Throws')

        plt.title(title)
        plt.show()

#        if np.sum(trig[nE]) > 0:
#            print(f'for energy {eng} got a trig number of {np.sum(trig[nE])}')
#            trigBins.append(np.sum(trig[nE]))
#            engs.append(eng)
    for iE in range(len(engBins)-1):
        for iZ in range(len(zenBins)-1):
            trigBins.append(np.sum(trig[iE][iZ]))
            throwBins.append(np.sum(throw[iE][iZ]))
            engs.append(engBins[iE])

    plotTrigScatter = False
    if plotTrigScatter:
        plt.scatter(engs, throwBins, label='Throw')
        plt.scatter(engs, trigBins, label='Trig')
        plt.legend()
        plt.xlabel('Energy')
        plt.ylabel('Number Triggers')
        plt.yscale('log')
        plt.title(title)
        plt.show()



    throw[throw == 0] = 1
    Aeff = trig/throw
    error_Aeff = error_trig/throw
    error_Aeff_low = error_trig_low/throw


    plotEffRad = False
    if plotEffRad:
        Arate = np.mean(Aeff, axis=1)
    #        plt.imshow(rrate.T, extent=[min(engBins), max(engBins), max(radBins), min(radBins)], aspect='auto', interpolation='none', norm=matplotlib.colors.LogNorm())
        Arate, cmap = set_bad_imshow(Arate, 0)
    #    plt.imshow(Arate.T, extent=[min(engBins), max(engBins), max(radBins), min(radBins)], aspect='auto', cmap=cmap)
        plt.imshow(Arate.T, extent=[min(engBins), max(engBins), max(plotRadBins), min(plotRadBins)], aspect='auto', norm=matplotlib.colors.LogNorm(), cmap=cmap)
        plt.xlabel('Energy (log10eV)')
        plt.ylabel('Radius from Station (km)')
        plt.colorbar(label=f'Efficiency (nTrig/nThrow)')
        plt.title(title)
        plt.show()


    engCenterBins = (engBins[1:] + engBins[:-1])/2
    zenCenterBins = (zenBins[1:] + zenBins[:-1])/2
    skip = True
    for iE, eng in enumerate(engCenterBins):
        if skip == True:
            continue
        for iZ, zen in enumerate(zenCenterBins):
    #        Arate = np.mean(Arate, axis=0)
#            plotRate = Arate[iE]
            plotRate = Aeff[iE][iZ]
            if not np.any(plotRate):
                print(f'No rate for energy {eng}')
                continue
            scatterRadBins = (plotRadBins[1:] + plotRadBins[:-1])/2
            areaBins = np.zeros_like(scatterRadBins)
            for iR in range(len(scatterRadBins)):
                areaBins[iR] = np.pi * (plotRadBins[iR+1]**2 - plotRadBins[iR]**2)
        #    plt.plot( scatterRadBins, Arate, label='Trig Rate')
        #    plt.plot( scatterRadBins, areaBins, label='Area')
        #    plt.plot( scatterRadBins, Arate * areaBins, label='Aeff')
            fig, host = plt.subplots()
            par1 = host.twinx()
            par2 = host.twinx()
            host.set_xlabel('Radius (km)')
            host.set_ylabel('Trigger Efficiency')
            host.set_ylim(-0.01,1.1)
            par1.set_ylim(-0.01, max(areaBins) * 1.1)
            par2.set_ylim(0, max(plotRate * areaBins) * 1.1)
            par1.set_ylabel('Area of bin at radius (km^2)')
            par2.set_ylabel(f'Aeff (km^2), Sum {np.sum(plotRate * areaBins):.4f}')

            rad_between = plotRadBins[1]-plotRadBins[0]
    #        p1, = host.plot( scatterRadBins, plotRate, color='red', label='Trig Rate')
    #        p2, = par1.plot( scatterRadBins, areaBins, color='blue', label='Area')
    #        p3, = par2.plot( scatterRadBins, plotRate * areaBins, color='green', label='Aeff')
    #        p1, = host.step( scatterRadBins, Arate, where='mid', color='red', label='Trig Rate')
    #        p2, = par1.step( scatterRadBins, areaBins, where='mid', color='blue', label='Area')
    #        p3, = par2.step( scatterRadBins, Arate * areaBins, where='mid', color='green', label='Aeff')
            p1, = host.step( scatterRadBins, plotRate, where='mid', color='red', label='Trig Rate')
            p2, = par1.step( scatterRadBins, areaBins, where='mid', color='blue', label='Area')
            p3, = par2.step( scatterRadBins, plotRate * areaBins, where='mid', color='green', label='Aeff')
            lns = [p1, p2, p3]
            host.legend(handles=lns, loc='best')
            par2.spines['right'].set_position(('outward', 60))
            host.yaxis.label.set_color('red')
            par1.yaxis.label.set_color('blue')
            par2.yaxis.label.set_color('green')
        #    host.yaxis.label.set_color(p1.get_color())
        #    par1.yaxis.label.set_color(p2.get_color())
        #    par2.yaxis.label.set_color(p3.get_color())
            fig.tight_layout()
            plt.title(f'Energy {eng:.2f} Zenith {zen:.2f} Num Bins {len(scatterRadBins)}, delta_r = {plotRadBins[1]-plotRadBins[0]:.2f}km')
            plt.savefig(f'plots/RadiusStudy/RadiusAeff_{eng:.2f}eV_{np.rad2deg(zen):.2f}deg_{len(scatterRadBins)}bins.jpg')
            plt.clf()
#            plt.show()
            print(f'For eng {eng}, zen {zen}, Aeffs of {plotRate * areaBins}, sum Aeff is {np.sum(plotRate * areaBins)}')
            print(f'Trig Rates are {plotRate}')
            print(f'Area is {areaBins}')
#    quit()




    plotEffEngZen = False
    if plotEffEngZen:
        Arate = np.mean(Aeff, axis=2)
    #        plt.imshow(rrate.T, extent=[min(engBins), max(engBins), max(radBins), min(radBins)], aspect='auto', interpolation='none', norm=matplotlib.colors.LogNorm())
        Arate, cmap = set_bad_imshow(Arate, 0)
    #    plt.imshow(Arate.T, extent=[min(engBins), max(engBins), max(radBins), min(radBins)], aspect='auto', cmap=cmap)
        plt.imshow(Arate.T, extent=[min(engBins), max(engBins), max(cosZenBins), min(cosZenBins)], aspect=4, norm=matplotlib.colors.LogNorm(), cmap=cmap)
        ax_labels = []
        for zen in zenBins:
            ax_labels.append('{:.0f}'.format(np.rad2deg(zen)))
        ax = plt.gca()
        ax.set_yticks(cosZenBins)
        ax.set_yticklabels(ax_labels)
        plt.xlabel('Energy (log10eV)')
        plt.ylabel('CR Zenith (deg)')
        plt.colorbar(label=f'Efficiency (nTrig/nThrow)')
        plt.title(title)
        plt.show()


    areas = np.zeros(len(radBins)-1)
    for iR in range(len(radBins)-1):
        aHigh = np.pi * (radBins[iR+1]*10**-3)**2
        aLow = np.pi * (radBins[iR]*10**-3)**2
        a = aHigh - aLow
        areas[iR] = a
#        Aeff[:][:][iR] *= a
#        Aeff[:,:,iR] *= a
#        error_Aeff[:,:,iR] *= a
#        error_Aeff_low[:,:,iR] *= a
    for iE in range(len(engBins)-1):
        for iZ in range(len(zenBins)-1):
            Aeff[iE][iZ] *= areas
            error_Aeff[iE][iZ] *= areas
            error_Aeff_low[iE][iZ] *= areas
    print(f'Sum of Aeff with nbins {len(radBins)} is {np.sum(areas)}')





    eRate = np.zeros_like(Aeff)
    error_eRate = np.zeros_like(Aeff)
    error_eRate_low = np.zeros_like(Aeff)
    print(f'len eng {len(engBins)} zen {len(zenBins)}')
    print(f'shape {np.shape(np.sum(eRate, axis=0))}')
    print(f'shape {np.shape(np.sum(eRate, axis=1))}')
    print(f'shape {np.shape(np.sum(eRate, axis=2))}')



        #Test starts here
    minE = 0
    for iE in range(len(engBins)-1):
        if np.any(Aeff[iE]):
            minE = engBins[iE]
            break
    eCenter = (engBins[1:] + engBins[:-1])/2
#    chiCut = linefit(eCenter, min=minE)
#    chiCut = piecefit(eCenter)
#    minE = 16.7
    chiCut = logfit(eCenter, min=minE)
    print(f'minE is {minE}')


    for iE in range(len(engBins)-1):
#        if engBins[iE+1] < 17.6:
#            continue
        for iZ in range(len(zenBins)-1):
            for iR in range(len(radBins)-1):
#                eRate[iE][iZ][iR] = singleBinERate(engBins[iE], engBins[iE+1], np.rad2deg(zenBins[iZ]), np.rad2deg(zenBins[iZ+1]), 1) * (10**((engBins[iE+1] + engBins[iE])/2))**2
                eRate[iE][iZ][iR] = singleBinERate(engBins[iE], engBins[iE+1], np.rad2deg(zenBins[iZ]), np.rad2deg(zenBins[iZ+1]), Aeff[iE][iZ][iR])
#                eRate[iE][iZ][iR] = singleBinERate(engBins[iE], engBins[iE+1], np.rad2deg(zenBins[iZ]), np.rad2deg(zenBins[iZ+1]), Aeff[iE][iZ][iR]) * chiCut[iE]
                error_eRate[iE][iZ][iR] = singleBinERate(engBins[iE], engBins[iE+1], np.rad2deg(zenBins[iZ]), np.rad2deg(zenBins[iZ+1]), error_Aeff[iE][iZ][iR])
                error_eRate_low[iE][iZ][iR] = singleBinERate(engBins[iE], engBins[iE+1], np.rad2deg(zenBins[iZ]), np.rad2deg(zenBins[iZ+1]), error_Aeff_low[iE][iZ][iR])

    skip = True
    for iE, eng in enumerate(engCenterBins):
        if skip == True:
            continue
        for iZ, zen in enumerate(zenCenterBins):
            zen = np.rad2deg(zen)
    #        Arate = np.mean(Arate, axis=0)
#            plotRate = Arate[iE]
            plotRate = eRate[iE][iZ]
            AeffRate = Aeff[iE][iZ]
            if not np.any(plotRate):
                print(f'No rate for energy {eng} zen {zen}')
                continue
            scatterRadBins = (plotRadBins[1:] + plotRadBins[:-1])/2
            areaBins = np.zeros_like(scatterRadBins)
            for iR in range(len(scatterRadBins)):
                areaBins[iR] = np.pi * (plotRadBins[iR+1]**2 - plotRadBins[iR]**2)
#            AeffRate = AeffRate / areaBins
        #    plt.plot( scatterRadBins, Arate, label='Trig Rate')
        #    plt.plot( scatterRadBins, areaBins, label='Area')
        #    plt.plot( scatterRadBins, Arate * areaBins, label='Aeff')
            fig, host = plt.subplots()
            par1 = host.twinx()
            par2 = host.twinx()
            host.set_xlabel('Radius (km)')
            host.set_ylabel(f'Event Rate in Bin (Evts/Stn/Yr), Sum {np.sum(plotRate):.0f}')
            host.set_ylim(max(plotRate)*10**-3,max(plotRate)*1.1)
#            host.set_yscale('log')
            par1.set_ylim(-0.01, max(areaBins) * 1.1)
            par2.set_ylim(0.0001, 1.1)
            par2.set_yscale('log')
            par1.set_ylabel('Area of bin at radius (km^2)')
            par2.set_ylabel(f'Aeff (km^2) Sum {np.sum(AeffRate):.2f}')

            rad_between = plotRadBins[1]-plotRadBins[0]
    #        p1, = host.plot( scatterRadBins, plotRate, color='red', label='Trig Rate')
    #        p2, = par1.plot( scatterRadBins, areaBins, color='blue', label='Area')
    #        p3, = par2.plot( scatterRadBins, plotRate * areaBins, color='green', label='Aeff')
    #        p1, = host.step( scatterRadBins, Arate, where='mid', color='red', label='Trig Rate')
    #        p2, = par1.step( scatterRadBins, areaBins, where='mid', color='blue', label='Area')
    #        p3, = par2.step( scatterRadBins, Arate * areaBins, where='mid', color='green', label='Aeff')
            p1, = host.step( scatterRadBins, plotRate, where='mid', color='red', label='Event Rate')
            p2, = par1.step( scatterRadBins, areaBins, where='mid', color='blue', label='Area in Bin')
            p3, = par2.step( scatterRadBins, AeffRate, where='mid', color='green', label='Aeff')
            lns = [p1, p2, p3]
            host.legend(handles=lns, loc='lower right')
            par2.spines['right'].set_position(('outward', 60))
            host.yaxis.label.set_color('red')
            par1.yaxis.label.set_color('blue')
            par2.yaxis.label.set_color('green')
        #    host.yaxis.label.set_color(p1.get_color())
        #    par1.yaxis.label.set_color(p2.get_color())
        #    par2.yaxis.label.set_color(p3.get_color())
            fig.tight_layout()
            plt.title(f'Energy {eng:.2f} Zenith {zen:.2f} Num Bins {len(scatterRadBins)}, delta_r = {plotRadBins[1]-plotRadBins[0]:.2f}km')
            plt.savefig(f'plots/RadiusStudy/RadiusErateArea_{eng:.2f}eV_{zen:.2f}deg_{len(scatterRadBins)}bins.jpg')
            plt.clf()
#            plt.show()
#            print(f'For eng {eng}, zen {zen}, Aeffs of {plotRate * areaBins}, sum Aeff is {np.sum(plotRate * areaBins)}')
#            print(f'Trig Rates are {plotRate}')
#            print(f'Area is {areaBins}')




#    error_eRate = error_eRate - eRate
#    error_eRate[eRate == 0] = 0
#    error_eRate_low = eRate - error_eRate


    asp = 4
    if plot:
#    while not asp == -1:
#        print(f'give aspect')
#        asp = float(input())

        rrate = np.sum(eRate, axis=1)
        print(f'shape rrate {np.shape(rrate)} and len eng {len(engBins)} and rad {len(radBins)}')
        radBins = radBins / 1000
#        plt.imshow(rrate.T, extent=[min(engBins), max(engBins), max(radBins), min(radBins)], aspect='auto', interpolation='none', norm=matplotlib.colors.LogNorm())
        rrate, cmap = set_bad_imshow(rrate, 0)
        plt.imshow(rrate.T, extent=[min(engBins), max(engBins), max(radBins), min(radBins)], aspect='auto', cmap=cmap)
#        plt.imshow(rrate.T, extent=[min(engBins), max(engBins), max(radBins), min(radBins)], aspect='auto', norm=matplotlib.colors.LogNorm())
        plt.xlabel('Energy (log10eV)')
        plt.ylabel('Radius from Station (km)')
        plt.colorbar(label=f'{np.sum(rrate):.5f} Evts/Stn/Yr')
        plt.title(title)
        plt.show()



        rate = np.sum(eRate, axis=2)
        zenBins = np.rad2deg(zenBins)
        rate, cmap = set_bad_imshow(rate, 0)
        plt.imshow(rate.T, extent=[min(engBins), max(engBins), max(cosZenBins), min(cosZenBins)], aspect=asp, interpolation='none', cmap=cmap)
#        plt.imshow(rate.T, extent=[min(engBins), max(engBins), max(cosZenBins), min(cosZenBins)], aspect=asp, interpolation='none', norm=matplotlib.colors.LogNorm())
        """
        eCenter = (engBins[1:] + engBins[:-1])/2
        zCenter = (zenBins[1:] + zenBins[:-1])/2
        zCenter = np.rad2deg(zCenter)
        for iE in range(len(eCenter)):
            for iZ in range(len(zCenter)):
#                rate = np.sum(eRate[iE][iZ])
                plt.scatter(eCenter[iE], zCenter[iZ], c=rate[iE][iZ])
        plt.clim(rate.min(), rate.max())
        """
        print(f'max coszen {max(cosZenBins)} and min {min(cosZenBins)}')
#        plt.scatter(np.log10(energy), 1-np.cos(np.rad2deg(zenith))**2, color='black', marker='x')
        ax_labels = []
        for zen in zenBins:
            ax_labels.append('{:.0f}'.format(zen))
        ax = plt.gca()
        ax.set_yticks(cosZenBins)
        ax.set_yticklabels(ax_labels)
        plt.xlabel('Energy (log10eV)')
        plt.ylabel('CR Zenith (deg)')
        print(f'rate is {np.sum(rate)}')
        plt.colorbar(label=f'{np.sum(rate):.5f} Evts/Stn/Yr')
        plt.title(title)
#        plt.legend()
        plt.show()

        rate = np.sum(eRate, axis=2)
        error_rate=np.sum(error_eRate, axis=2)
        error_rate_low=np.sum(error_eRate_low, axis=2)
        for iZ in range(len(zenBins)-1):
            prate = rate[:,iZ]
            pErr_rate = error_rate[:,iZ]
            pErrLow_rate = error_rate_low[:,iZ]
            c = next(color)
#            plt.plot( (engBins[1:] + engBins[:-1])/2, prate, label=f'{zenBins[iZ]:.1f}-{zenBins[iZ+1]:.1f} deg', linestyle='--', color=c) 
            plt.fill_between((engBins[1:] + engBins[:-1])/2, pErrLow_rate, pErr_rate, label=f'{zenBins[iZ]:.1f}-{zenBins[iZ+1]:.1f} deg',color=c)

        rate = np.sum(rate, axis=1)
        error_rate=np.sum(error_rate, axis=1)
        error_rate_low=np.sum(error_rate_low, axis=1)


#        plt.plot( (engBins[1:] + engBins[:-1])/2, rate, label=f'{np.sum(rate):.0f} Evts/Stn/Yr', linestyle='--', color='red') 
        plt.fill_between((engBins[1:] + engBins[:-1])/2, error_rate_low, error_rate, label=f'{np.sum(rate):.0f} Evts/Stn/Yr', color='red')
        plt.legend()
        plt.title(title)
        plt.xlabel('Energy (log10eV)')
        plt.ylabel('Evts/Stn/Yr')
#        plt.yscale('log')
        plt.show()

        arate = np.sum(Aeff, axis=2)
        arate = np.sum(arate, axis=1)
        plt.plot( (engBins[1:] + engBins[:-1])/2, arate, label=f'{np.sum(arate):.5f} Evts/Stn/Yr')
#        plt.title(title + ' for 35 stations')
        plt.title(title)
        plt.xlabel('Energy (log10eV)')
        plt.ylabel('Aeff (km^2)')
#        plt.yscale('log')
        plt.show()

    return eRate




parser = argparse.ArgumentParser(description='Run file on footprint refraction file')
#parser.add_argument('file', type=str, default='None', help='file path to analyze')
parser.add_argument('area', type=float, default=5, help='Length of area thrown over in km, default 5km square')
parser.add_argument('--comment', type=str, default='', help='Title comment to add')
parser.add_argument('-files', '--list', nargs='+', help='file path to analyze', required=True)

args = parser.parse_args()
#file = args.file
files = args.list
area = args.area
comment = args.comment

#with open(file, 'rb') as fin:
#    output = pickle.load(fin)


n = []
n_dip_dir = []
n_dip_refl = []
n_lpda_refl = []
n_lpda_dir = []
energy = []
zenith = []
azimuth = []
x = []
y = []
dip_dir_mask = []
dip_refl_mask = []
lpda_dir_mask = []
lpda_refl_mask = []
ant_zen = []
dip_dir_SNR = []
dip_refl_SNR = []
lpda_dir_SNR = []
lpda_refl_SNR = []

for file in files:
    with open(file, 'rb') as fin:
        output = pickle.load(fin)
    for runid in output:
        print(runid)

    #    if np.log10(output[runid]['energy']) < 17.7:
    #        continue

        n.append(output[runid]['n'])
#        n_dip_dir.append(output[runid]['n_dip_dir'])
#        n_dip_refl.append(output[runid]['n_dip_refl'])
        n_lpda_refl.append(output[runid]['n_lpda_refl'])
        n_lpda_dir.append(output[runid]['n_lpda_dir'])
        energy.append(output[runid]['energy'])
        zenith.append(output[runid]['zenith'])
#        azimuth.append(output[runid]['azimuth'])               #ADD BACK IN
        x.append(output[runid]['x_dir_lpda'])
        y.append(output[runid]['y_dir_lpda'])
#        dip_dir_mask.append(output[runid]['dip_dir_mask'])
#        dip_refl_mask.append(output[runid]['dip_refl_mask'])
        lpda_dir_mask.append(output[runid]['lpda_dir_mask'])
        lpda_refl_mask.append(output[runid]['lpda_refl_mask'])
#        ant_zen.append(output[runid]['ant_zen'])
#        dip_dir_SNR.append(output[runid]['dip_dir_SNR'])
#        dip_refl_SNR.append(output[runid]['dip_refl_SNR'])
#        lpda_dir_SNR.append(output[runid]['lpda_dir_SNR'])
#        lpda_refl_SNR.append(output[runid]['lpda_refl_SNR'])
    fin.close()

n = np.array(n)
#n_dip_dir = np.array(n_dip_dir)
#n_dip_refl = np.array(n_dip_refl)
n_lpda_refl = np.array(n_lpda_refl)
n_lpda_dir = np.array(n_lpda_dir)
energy = np.array(energy)
zenith = np.array(zenith)
#azimuth = np.array(azimuth)                                    #ADD BACK IN
#print(f'x is {x}')
#x = np.array(x, dtype=object)
#y = np.array(y, dtype=object)
#x = np.asarray(x)
#y = np.asarray(y)
print(f'1')
x = np.array(list(itertools.zip_longest(*x, fillvalue=np.NaN))).T
print(f'2')
y = np.array(list(itertools.zip_longest(*y, fillvalue=np.NaN))).T
print(f'3')
dip_dir_mask = np.array(list(itertools.zip_longest(*dip_dir_mask, fillvalue=False))).T
print(f'4')
dip_refl_mask = np.array(list(itertools.zip_longest(*dip_refl_mask, fillvalue=False))).T
print(f'5')
lpda_dir_mask = np.array(list(itertools.zip_longest(*lpda_dir_mask, fillvalue=False))).T
print(f'6')
lpda_refl_mask = np.array(list(itertools.zip_longest(*lpda_refl_mask, fillvalue=False))).T

"""
dip_dir_mask = np.array(dip_dir_mask, dtype=object)
dip_refl_mask = np.array(dip_refl_mask, dtype=object)
lpda_refl_mask = np.array(lpda_refl_mask, dtype=object)
lpda_dir_mask = np.array(lpda_dir_mask, dtype=object)
"""
ant_zen = np.array(ant_zen)
#dip_dir_SNR = np.array(dip_dir_SNR)
#dip_refl_SNR = np.array(dip_refl_SNR)
#print(f'dir dir snr {lpda_dir_SNR}')
#lpda_dir_SNR = np.array(lpda_dir_SNR)
#print(f'dir dir snr {lpda_dir_SNR}')
#lpda_refl_SNR = np.array(lpda_refl_SNR)


"""
length = 1000  #m
maxR = np.sqrt( 2 * (length/2)**2 )
dlAeff, dlTrig, dlEngBins, dlZenBins = binnedAeff(energy, zenith, n_lpda_dir, n, lpda_dir_mask, x_dir_lpda, y_dir_lpda, maxR)
dlAeff = dlTrig

eCenter = (dlEngBins[1:] + dlEngBins[:-1])/2
zCenter = (dlZenBins[1:] + dlZenBins[:-1])/2
print(f'Aeff orig {dlAeff}')
print(f'Transposed {dlAeff}')


for iZ, zen in enumerate(zCenter):
    print(f'shape ecent {np.shape(eCenter)} and dl segment {np.shape(dlAeff[:][iZ])}')
    plt.scatter(eCenter, dlAeff[:][iZ], label=f'Zens {dlZenBins[iZ]}-{dlZenBins[iZ+1]}')
    plt.xlabel('Energy (log10 eV)')
    plt.legend()
    plt.title('Aeff of direct LPDA triggers')
    plt.show()
"""


"""
#Code for plotting trigger eff vs SNR
#lpda_dir_SNR = lpda_dir_SNR / 0.001
maxes = max(lpda_dir_SNR)
SNR_bins = np.linspace(0, 12, num=50)
#SNR_bins = np.linspace(0, max(maxes)/40, num=100)
#snrDigs = np.digitize(lpda_dir_SNR, SNR_bins) - 1

throws = np.zeros(len(SNR_bins)-1)
trigs = np.zeros(len(SNR_bins)-1)

for iS, SNRs in enumerate(lpda_dir_SNR):
    if len(SNRs) == 0:
        continue
#    print(f'snrs {SNRs} and bins {SNR_bins}')
    SNRs = np.array(SNRs)
    SNRs = SNRs * (10*10**-6 * 10**2)
#    SNRs = SNRs * (15*10**-6 * 10**2)
    snrDigs = np.digitize(SNRs, SNR_bins)-1
    print(f'len snrs {len(SNRs)} and mask {len(lpda_dir_mask[iS])}')
    for iD, dig in enumerate(snrDigs):
        if dig >= len(SNR_bins)-1:
            print(f'SNR {SNRs[iD]} dig too big')
            continue
#        print(f'dig {dig} and iS {iS}')
#        trigs[dig] += np.sum(n_lpda_dir[iS])
#        throws[dig] += np.sum(n[iS])
        throws[dig] += 1
        trigs[dig] += int(lpda_dir_mask[iS][iD])


trigs[throws == 0] = 1
throws[throws == 0] = 0
plotSNR = (SNR_bins[1:] + SNR_bins[:-1])/(2)
#plotSNR = (SNR_bins[1:] + SNR_bins[:-1])/(2) * (0.000015 * 10**2)
plt.plot(plotSNR, trigs/throws)
plt.xlabel('SNR')
plt.ylabel('Trigger efficiency')
plt.title(comment)
plt.show()
xquit()
"""
#Plot weighted zenith/azimuth plot
plotAzi = False
if plotAzi:
    #anyMask = np.any(lpda_refl_mask, axis=1)
    anyMask = np.any(lpda_dir_mask, axis=1)
    pltZens = zenith[anyMask]
    pltAzi = azimuth[anyMask]
    engs = energy[anyMask]
    engs = engs / min(engs)
    weights = n_lpda_dir[anyMask] / n[anyMask]
    weights = weights * np.cos(pltZens) * np.sin(pltZens) / (engs**2)
    print(f'weights {weights}')

    rbins = np.linspace(0, 90, 6)
    abins = np.linspace(0, 2*np.pi, 21)
    hist, _, _ = np.histogram2d(pltAzi, np.rad2deg(pltZens), bins=(abins, rbins), weights=weights)
    A, R = np.meshgrid(abins, rbins)
    plt.axes(projection='polar')
    plt.pcolormesh(A, R, hist.T)
    #plt.pcolormesh(A, R, hist.T, norm=matplotlib.colors.LogNorm())

    #plt.scatter(np.rad2deg(pltAzi), np.rad2deg(pltZens), c=weights, norm=matplotlib.colors.LogNorm())
    #plt.axes(projection='polar')
    plt.colorbar(label='Trig,Eng,Zenith Weighting')
    plt.title('Detected CR distribution')
    plt.show()
    #quit()

    #plt.scatter(np.log10(energy), np.rad2deg(zenith))
    #plt.title('New Distribution with IceTop')
    #plt.show()

#GOOD CODE, for event rate
print(f'Starting to plot')
if True:
#    RLeRate = getERate(energy, zenith, x, y, lpda_refl_mask, n_lpda_refl, n, area, title='Reflected LPDA '+comment)
#    print(f'total eRate refl LPDA {np.sum(RLeRate)}')
#    DDeRate = getERate(energy, zenith, x, y, dip_dir_mask, n_dip_dir, n, area, title='Direct Dipole '+comment)
#    print(f'total eRate direct dipole {np.sum(DDeRate)}')
    DLeRate = getERate(energy=energy, zenith=zenith, xs=x, ys=y, trigMask=lpda_dir_mask, ntrig=n_lpda_dir, nthrow=n, length=area, title='Direct LPDA '+comment)
    print(f'total eRate direct LPDA {np.sum(DLeRate)}')
#    RDeRate = getERate(energy, zenith, x, y, dip_refl_mask, n_dip_refl, n, area, title='Reflected Dipole '+comment)
#    print(f'total eRate refl dipole {np.sum(RDeRate)}')
    quit()

    engBins = np.arange(np.log10(np.min(energy)), 20.1, .1)
    DDengRate = []
    DLengRate = []
    RDengRate = []
    RLengRate = []
    for iE in range(len(engBins)-1):
        DDengRate.append(np.sum(DDeRate[iE]))
        DLengRate.append(np.sum(DLeRate[iE]))
        RDengRate.append(np.sum(RDeRate[iE]))
        RLengRate.append(np.sum(RLeRate[iE]))
    plt.plot( (engBins[1:]+engBins[:-1])/2, DLengRate, label=f'{np.sum(DLengRate):2f} evts/stn/yr')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Energy (log10eV)')
    plt.ylabel('Event Rate (evts/stn/yr)')
    plt.title('Event Rate per Energy for Upwardfacing LPDA')
    plt.show()

    plt.plot( (engBins[1:]+engBins[:-1])/2, DDengRate, label=f'Direct Dipole {np.sum(DDengRate):.4f} evts/stn/yr', color=next(color))
    plt.plot( (engBins[1:]+engBins[:-1])/2, RDengRate, label=f'Reflected Dipole {np.sum(RDengRate):.4f} evts/stn/yr', color=next(color))
    plt.plot( (engBins[1:]+engBins[:-1])/2, RLengRate, label=f'Reflected LPDA {np.sum(RLengRate):.4f} evts/stn/yr', color=next(color))
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Energy (log10eV)')
    plt.ylabel('Event Rate (evts/stn/yr)')
    plt.title('Event Rate per Energy, SP')
    plt.show()



mask = np.any(lpda_refl_mask, axis=1)
print(f'mask shape {np.shape(mask)} and energy {np.shape(energy)}')
et = energy[mask]
print(f'et {et}')
print(f'true energies are dip dir {energy[np.any(dip_dir_mask, axis=1)]}, dip refl {energy[np.any(dip_refl_mask, axis=1)]}, lpda refl {energy[np.any(lpda_refl_mask, axis=1)]}')
#quit()

print(f' dip dir {np.sum(dip_dir_mask)} dip relf {np.sum(dip_refl_mask)} lpda refl {np.sum(lpda_refl_mask)} lpda dir {np.sum(lpda_dir_mask)}')

plt.scatter(x[dip_dir_mask], y[dip_dir_mask])
plt.title('Direct Dipole xy positions')
plt.show()

plt.scatter(x[dip_refl_mask], y[dip_refl_mask])
plt.title('Reflected Dipole xy Positions')
plt.show()

plt.scatter(x[lpda_refl_mask], y[lpda_refl_mask])
plt.title('Reflected LPDA xy Positions')
plt.show()

plt.scatter(x[lpda_dir_mask], y[lpda_dir_mask])
plt.title('LPDA Triggered Direct core locations')
plt.show()

#area = 3
area = area**2


dd_mask = np.any(dip_dir_mask, axis=1)
rd_mask = np.any(dip_refl_mask, axis=1)
rl_mask = np.any(lpda_refl_mask, axis=1)
dl_mask = np.any(lpda_dir_mask, axis=1)

print(f'n {np.sum(n)} dd {np.sum(n_dip_dir)} rd {np.sum(n_dip_refl)} rl {np.sum(n_lpda_refl)}')
Aeff_dd = area * np.sum(n_dip_dir) / np.sum(n)
Aeff_rd = area * np.sum(n_dip_refl) / np.sum(n)
Aeff_rl = area * np.sum(n_lpda_refl) / np.sum(n)
#Aeff_rl = area * np.sum(n_lpda_refl[rl_mask]) / np.sum(n[rl_mask])
Aeff_dl = area * np.sum(n_lpda_dir) / np.sum(n)

dd_rate = auger.event_rate(np.log10(np.min(energy[dd_mask])), np.log10(np.max(energy[dd_mask])), np.rad2deg(np.max(zenith[dd_mask])), Aeff_dd)
rd_rate = auger.event_rate(np.log10(np.min(energy[rd_mask])), np.log10(np.max(energy[rd_mask])), np.rad2deg(np.max(zenith[rd_mask])), Aeff_rd)
rl_rate = auger.event_rate(np.log10(np.min(energy[rl_mask])), np.log10(np.max(energy[rl_mask])), np.rad2deg(np.max(zenith[rl_mask])), Aeff_rl)
dl_rate = auger.event_rate(np.log10(np.min(energy[dl_mask])), np.log10(np.max(energy[dl_mask])), np.rad2deg(np.max(zenith[dl_mask])), Aeff_dl)

print(f'Event rate dir dip {dd_rate}')
print(f'Event rate refl dip {rd_rate}')
print(f'Event rate refl lpda {rl_rate}')
print(f'Event rate dir lpda {dl_rate}')

#plt.scatter(np.log10(energy[~dd_mask]), np.rad2deg(zenith[~dd_mask]), color='black', marker='x')
#plt.scatter(np.log10(energy[dd_mask]), np.rad2deg(zenith[dd_mask]), c=n_dip_dir[dd_mask] / n[dd_mask])
plt.scatter(np.log10(energy[~dd_mask]), 1-np.cos(zenith[~dd_mask]), color='black', marker='x')
plt.scatter(np.log10(energy[dd_mask]), 1-np.cos(zenith[dd_mask]), c=n_dip_dir[dd_mask] / n[dd_mask])
plt.ylabel('1-cos(zenith)')
plt.title('Footprint Triggers of Direct Dipole')
plt.show()

plt.scatter(np.log10(energy[~rd_mask]), np.rad2deg(zenith[~rd_mask]), color='black', marker='x')
plt.scatter(np.log10(energy[rd_mask]), np.rad2deg(zenith[rd_mask]), c=n_dip_refl[rd_mask] / n[rd_mask])
plt.title('Footprint Triggers of Reflected Dipole')
plt.show()


plt.scatter(np.log10(energy[~rl_mask]), np.rad2deg(zenith[~rl_mask]), color='black', marker='x')
plt.scatter(np.log10(energy[rl_mask]), np.rad2deg(zenith[rl_mask]), c=n_lpda_refl[rl_mask] / n[rl_mask])
plt.title('Footprint Triggers of Reflected LPDA')
plt.show()

plt.scatter(np.log10(energy[~dl_mask]), np.rad2deg(zenith[~dl_mask]), color='black', marker='x')
plt.scatter(np.log10(energy[dl_mask]), np.rad2deg(zenith[dl_mask]), c=n_lpda_dir[dl_mask] / n[dl_mask])
plt.title('Footprint Triggers of Direct LPDA')
plt.show()



plt.hist(np.rad2deg(ant_zen[dd_mask].flatten()))
plt.xlabel('Dipole Direct Recieve Zenith')
plt.show()

plt.hist(np.rad2deg(ant_zen[rd_mask].flatten()))
plt.xlabel('Dipole Reflected Recieve Zenith')
plt.show()

plt.hist(np.rad2deg(ant_zen[rl_mask].flatten()))
plt.xlabel('LPDA Reflected Recieve Zenith')
plt.show()



