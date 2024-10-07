import matplotlib.pyplot as plt
import coreDataObjects as CDO
import numpy as np
import matplotlib.colors
import pickle
from icecream import ic


def set_bad_imshow(array, value):
    ma = np.ma.masked_where(array == value, array)
    cmap = matplotlib.cm.viridis
    cmap.set_bad(color='white')
    return ma, cmap

def plotCoreDiagnostics(CoreObjectsList, title_comment='', legend_comment='', type='TA', density=False):
    #Make a page of histograms of the event rate as a function of radius per energy and coszen bin

    zenBins = np.linspace(0, np.pi, 30)
    zenBins = np.rad2deg(zenBins)
    all_energies = []
    all_coszen = []

    all_data = {}

    for core in CoreObjectsList:
        energy = f'{core.e_bins[0]:.1f}-{core.e_bins[1]:.1f}'
        coszen = f'{np.rad2deg(np.arccos(core.coszen_bins[0])):.1f}-{np.rad2deg(np.arccos(core.coszen_bins[1])):.1f}'
        if not energy in all_energies:
            all_energies.append(energy)
            all_data[energy] = {}
        if not coszen in all_coszen:
            all_coszen.append(coszen)
            all_data[energy][coszen] = {}
        if not coszen in all_data[energy]:
            all_data[energy][coszen] = {}
        all_data[energy][coszen]['Aeff_per_rad'] = [core.rad_bins, core.Aeff_per_rad_bin]
        all_data[energy][coszen]['ArrivalAngles'] = core.getZeniths()

    all_energies.sort()
    all_coszen.sort()

    ncols = len(all_energies)
    nrows = len(all_coszen)
    ic(all_energies)
    ic(all_coszen)
    ic(ncols, nrows)
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
    ax = np.atleast_2d(ax)

    for iE, energy in enumerate(all_energies):
        for iC, coszen in enumerate(all_coszen):
            ic(energy, coszen)
            prad = all_data[energy][coszen]['Aeff_per_rad'][0]
            pAeff = all_data[energy][coszen]['Aeff_per_rad'][1]

            ax[iC, iE].bar(prad[:-1], pAeff, width=np.diff(prad), align='edge')
            ax[iC, iE].set_title(f'{energy} log10eV, {coszen} deg')
            ax[iC, iE].set_ylabel('Effective Area (m^2)')
            ax[iC, iE].set_yscale('log')
            ax[iC, iE].set_ylim(10**-9, 10**-1)
        ax[-1, iE].set_xlabel('Radius (m)')

    ic('saving')
    plt.savefig(f'plots/CoreAnalysis/{title_comment}_Diagnostics_AeffRads.png', bbox_inches='tight')
    plt.clf()
    ic('saved')

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
    ax = np.atleast_2d(ax)

    for iE, energy in enumerate(all_energies):
        for iC, coszen in enumerate(all_coszen):
            ic(energy, coszen)
            pArrAng = all_data[energy][coszen]['ArrivalAngles']

            ax[iC, iE].bar(zenBins[:-1], pArrAng, width=np.diff(zenBins), align='edge')
            ax[iC, iE].set_title(f'{energy} log10eV, {coszen} deg')
            ax[iC, iE].set_ylabel('Count of Events')
            ax[iC, iE].set_yscale('log')
            ax[iC, iE].set_ylim(bottom=1)
        ax[-1, iE].set_xlabel('Arrival Angle (deg)')

    plt.savefig(f'plots/CoreAnalysis/{title_comment}_Diagnostics_ArrivalAngles.png', bbox_inches='tight')
    plt.clf()

    #2D histogram of Arrival angle vs Radius
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
    ax = np.atleast_2d(ax)

    for iE, energy in enumerate(all_energies):
        for iC, coszen in enumerate(all_coszen):
            prad = all_data[energy][coszen]['Aeff_per_rad'][0]      #x-axis
            pAeff = all_data[energy][coszen]['Aeff_per_rad'][1]     #Aeff per x-axis bin, rad
            pArrAng = all_data[energy][coszen]['ArrivalAngles']     #num of triggers per zenith arrival angle




    #Also do one with launch vector rather than arrival angle
    #And plot some of the event displays of events from lowest energy
    # https://github.com/nu-radio/NuRadioMC/blob/develop/NuRadioMC/SignalProp/examples/example_3d.py
    #   Line 39, rather than plot all of them, find the one with max amplitude
    #also do 2d launch vs radius, launch vs arrival

    return

def plotArrivalDirectionHist(CoreObjectsList, cut=None, title_comment='', legend_comment='', type='TA', density=False):
    #Make a histogram of arrival angles
    #If passed, make a cut at some arrival angle range [cut_low, cut_high]
    #Show the rate with/without cut, and plot the cut lines


    
    zenBins = np.linspace(0, np.pi, 30)
    if not cut == None:
        binCut = np.digitize(np.deg2rad(cut), zenBins)
    zenCent = (zenBins[1:] + zenBins[:-1])/2
    zenCent = np.rad2deg(zenCent)
    arrivalCounts = np.zeros_like(zenCent)
    arrivalCutCounts = np.zeros_like(zenCent)
    for core in CoreObjectsList:
        eventRate = core.totalEventRateCore()
        if eventRate == 0:
            continue
        arrivalHist = core.getZeniths()
#        print(f'arrival hist {arrivalHist}')
        eventRateWeight = eventRate / np.sum(arrivalHist)
        arrivalCounts += arrivalHist * eventRateWeight
#        print(f'arrivalCounts {arrivalCounts}')
        if not cut == None:
            arrivalHist[:binCut[0]] = 0
            arrivalHist[binCut[1]:] = 0
            arrivalCutCounts += arrivalHist * eventRateWeight

    zenBins = np.rad2deg(zenBins)
    if density:
        arrivalCounts = arrivalCounts / max(arrivalCounts)
    plot = plt.bar(zenBins[:-1], arrivalCounts, width=np.diff(zenBins), edgecolor='black', facecolor=None, align='edge')
    if not cut == None:
        plt.axvline(x=cut[0], color='red', linestyle='--')
        plt.axvline(x=cut[1], color='red', linestyle='--')
        sumEvts = np.sum(arrivalCounts)
        cutEvts = np.sum(arrivalCutCounts)
        legend1 = plt.legend(handles=[plt.plot([], [], color='black')[0], plt.plot([], [], color='red')[0]],
                            labels=[f'{sumEvts:.3f} Evts/Stn/Year', f'{cutEvts:.3f} Evts/Stn/Year'], loc='upper right', prop={'size':16})
        plt.gca().add_artist(legend1)
    plt.xlabel('Arrival Angle (deg)')
    plt.title(f'{title_comment}')


    return plot


def plotCREnergyEventRate(CoreObjectsList, error=True, title_comment='', legend_comment='', type='TA'):
    #Make plot of event rate per CR energy
    plotCrEng = []
    plotCrErate = []
    plotErrorErate = []
#    errorErateHigh = []
#    errorErateLow = []
    for core in CoreObjectsList:
#        CRs, rates, errorhigh, errorlow = core.eventRatePerEcr()
        CRs, rates, errorrate = core.eventRatePerEcr(type=type)
        for iC, cr in enumerate(CRs):
            if not cr in plotCrEng:
                plotCrEng.append(cr)
                plotCrErate.append(rates[iC])
#                errorErateHigh.append(errorhigh[iC])
#                errorErateLow.append(errorlow[iC])
                plotErrorErate.append([errorrate[iC]])
            else:
                index = plotCrEng.index(cr)
                plotCrErate[index] += rates[iC]
#                errorErateHigh[index] += errorhigh[iC]
#                errorErateLow[index] += errorlow[iC]
                plotErrorErate[index].append(errorrate[iC])


    if error:
#        errorErateHigh = np.array(errorErateHigh)
#        errorErateLow = np.array(errorErateLow)
        plotErr = np.zeros(len(plotErrorErate))
        for iE, err in enumerate(plotErrorErate):
            err = np.array(err)
            plotErr[iE] = np.sqrt(np.sum(err**2))
        errorErateHigh = np.zeros(len(plotCrErate))
        errorErateLow = np.zeros(len(plotCrErate))
        for iR, rate in enumerate(plotCrErate):
            errorErateHigh[iR] = rate + plotErr[iR]
            errorErateLow[iR] = rate - plotErr[iR]
        errorErateLow[errorErateLow < 0] = 0
        sum_error = np.sqrt(np.sum((plotErr)**2 ))
        plot = plt.fill_between(plotCrEng, errorErateHigh, errorErateLow, label=f'{np.sum(plotCrErate):.4f}±{sum_error:.4f} Evts/Stn/Yr')
        #Don't believe we want a weighted sum across energy bins, so ignore below
#        weightedSumRate = CDO.weightedAverage(plotCrErate, plotErrorErate, DEBUG=True)
#        weightedSumError = CDO.weightedError(plotErrorErate, DEBUG=True)
#        print(f'weighted sum rate {weightedSumRate} and rated error {weightedSumError}')
#        print(f'normal rate {np.sum(plotCrErate)} and error {sum_error}')
#        quit()
#        plt.fill_between(plotCrEng, errorErateHigh, errorErateLow, label=f'{weightedSumRate:.4f}±{weightedSumError:.4f} Evts/Stn/Yr')
#        plt.plot(plotCrEng, plotCrErate, color='black')
    else:
        plot = plt.plot(plotCrEng, plotCrErate, label=f'{legend_comment} {np.sum(plotCrErate):.4f} Evts/Stn/Yr')
    plt.xlabel('CR Energy (log10eV)')
    plt.ylabel('Evts/Stn/Yr')
    plt.legend(loc='upper left')
    plt.title(title_comment)
#    plt.savefig(f'plots/CoreAnalysis/{savePrefix}_CrEnergy_Erate.png')
#    plt.clf()
    return plot
    
def getCoreEnergyEventRate(CoreObjectsList, singleAeff=False, type='TA', DEBUG=False):    
    plotCoreEng = []
    plotCoreErate = []
#    plotErrorErate = []
    errorErateHigh = []
    errorErateLow = []

    finalError = []
    testZen = []
    for core in CoreObjectsList:
        energy = core.e_center
        erate = core.totalEventRateCore(singleAeff=singleAeff, type=type)
        zen = core.zen_center
        sigma_error = core.totalEventErrorCore(singleAeff=singleAeff, highLow=False, type=type)

        if DEBUG:
            print(f'eng {energy} zen {zen} has rate {erate} and error {sigma_error}')
            if not erate == 0:
                print(f'based off of total trigs {sum(core.statistics_per_bin)}')
        errorhigh, errorlow = core.totalEventErrorCore(singleAeff=singleAeff, type=type)
        if not zen in testZen:
            testZen.append(zen)
        if not energy in plotCoreEng:
            plotCoreEng.append(energy)
            plotCoreErate.append(erate)
            errorErateHigh.append(errorhigh)
            errorErateLow.append(errorlow)

            finalError.append([core.totalEventErrorCore(singleAeff=singleAeff, highLow=False, type=type)])
        else:
            index = plotCoreEng.index(energy)
            plotCoreErate[index] += erate
            errorErateHigh[index] += errorhigh
            errorErateLow[index] += errorlow


            finalError[index].append(core.totalEventErrorCore(singleAeff=singleAeff, highLow=False,type=type))

    #Sort in case its unsorted
    finalError = [x for _,x in sorted(zip(plotCoreEng, finalError))]
    plotCoreErate = [x for _,x in sorted(zip(plotCoreEng, plotCoreErate))]
    plotCoreEng = sorted(plotCoreEng)

    return plotCoreEng, plotCoreErate, finalError

def plotCoreEnergyEventRate(CoreObjectsList, error=True, singleAeff=False, title_comment='', label='', type='TA', lowerLim=0, DEBUG=False):
    #Make a plot of event rate as function of core energy
    plotCoreEng, plotCoreErate, finalError = getCoreEnergyEventRate(CoreObjectsList, singleAeff)

    if error:
        plotErr = np.zeros(len(finalError))
        for iE, err in enumerate(finalError):
            err = np.array(err)
            plotErr[iE] = np.sqrt(np.sum(err**2))
            if DEBUG:
                print(f'eng {plotCoreEng[iE]}')
                print(f'rate {plotCoreErate[iE]}')
                print(f'err \n{err}')
                print(f'tot err {plotErr[iE]}')
        plotErrHigh = np.zeros_like(plotErr)
        plotErrLow = np.zeros_like(plotErr)
        for iR, rate in enumerate(plotCoreErate):
            plotErrHigh[iR] = rate + plotErr[iR]
            plotErrLow[iR] = rate - plotErr[iR]
        plotErrLow[plotErrLow < 0] = 0
        sumErr = np.sqrt(np.sum(plotErr**2))
        if DEBUG:
            print(f'tot rate {np.sum(plotCoreErate)} and tot error {sumErr}')
        plot = plt.fill_between(plotCoreEng, plotErrHigh, plotErrLow, label=f'{label} {np.sum(plotCoreErate):.4f}±{sumErr:.4f} Evts/Stn/Yr')
    #    plt.plot(plotCoreEng, plotCoreErate, color='blue', label='err prop')

        if DEBUG:
            plt.legend(loc='upper left')
            plt.xlabel('Core Energy (log10eV)')
            plt.ylabel('Evts/Stn/Yr')
            plt.savefig(f'plots/CoreAnalysis/finaltest.png')
            plt.clf()

             #Code that makes an error only plot for debugging
            plt.plot(plotCoreEng, plotErr, label=f'Tot err {sumErr:.6f}')
            plt.yscale('log')
            plt.xlabel('Core Energy')
            plt.ylabel('Error Evts/Stn/Yr')
            plt.ylim((10**-6, 10**-3))
            plt.legend(loc='upper left')
            plt.savefig(f'plots/CoreAnalysis/finaltesterror.png')
            plt.clf()
    else:
        plot = plt.plot(plotCoreEng, plotCoreErate, label=f'{label} {np.sum(plotCoreErate):.4f} Evts/Stn/Yr')
    plt.xlabel('Core Energy (log10eV)')
    plt.ylabel('Evts/Stn/Yr')
    plt.legend(loc='upper left')
    plt.title(title_comment)
#    plt.savefig(f'plots/CoreAnalysis/{savePrefix}_CoreEnergy_Erate.png')
#    plt.clf()
    return plot

def plotCoreEnergyAeff(CoreObjectsList, singleAeff=False, label='', title_comment=''):
    #Make a plot of effective area as function of core energy
    plotCoreEng = []
    plotCoreAeff = []
    for core in CoreObjectsList:
        energy = core.e_center
        aeff = core.totalAeffCore(singleAeff=singleAeff)
        if not energy in plotCoreEng:
            plotCoreEng.append(energy)
            plotCoreAeff.append(aeff)
        else:
            index = plotCoreEng.index(energy)
            plotCoreAeff[index] += aeff
    plt.scatter(plotCoreEng, plotCoreAeff, label=label)
    plt.xlabel('Core Energy (log10eV)')
    plt.ylabel(r'Effective Area ($km^{2}$)')
    plt.yscale('log')
#    plt.legend()
    plt.title(title_comment)
    return

def plotCREnergyAeff(CoreObjectsList, title_comment=''):
    #make plot of effective area as function of CR energy
    plotCREng = []
    plotCRAeff = []
    for core in CoreObjectsList:
        energy = core.e_center
        aeff = core.totalAeffCore()
        if not energy in plotCREng:
            plotCREng.append(energy)
            plotCRAeff.append(aeff)
        else:
            index = plotCREng.index(energy)
            plotCRAeff[index] += aeff
    plt.scatter(plotCREng, plotCRAeff)
    plt.xlabel('CR Energy (log10eV)')
    plt.ylabel(r'Effective Area ($km^{2}$)')
    plt.yscale('log')
#    plt.legend()
    plt.title(title_comment)
    return

def plotCoreEnergyZenithHeat(CoreObjectsList, title_comment='', type='TA'):
    #Make a heatmap of the event rate distributed in energy and zenith phase space
    plotCoreEng = []
    plotCoreZen = []
    labelCosZen = []
    labelEng = []
    for core in CoreObjectsList:
        energy = core.e_center
        zen = core.zen_center
        if not energy in plotCoreEng:
            plotCoreEng.append(energy)
        if not zen in plotCoreZen:
            plotCoreZen.append(zen)
        for coszen in core.coszen_bins:
            if not coszen in labelCosZen:
               labelCosZen.append(coszen)
        for eng in core.e_bins:
            if not eng in labelEng:
                labelEng.append(eng)
    plotCoreEng.sort()
    plotCoreZen.sort()
    labelEng.sort()

    labelZen = []
    for coszen in labelCosZen:
        labelZen.append(np.rad2deg(np.nan_to_num(np.arccos(coszen))))
    labelZen.sort()

    eRate = np.zeros((len(plotCoreEng), len(plotCoreZen)))
    for core in CoreObjectsList:
        eIndex = plotCoreEng.index(core.e_center)
        zIndex = plotCoreZen.index(core.zen_center)
        eRate[eIndex, zIndex] += core.totalEventRateCore(type=type)


    eRate = np.fliplr(eRate)
    eRate, cmap = set_bad_imshow(eRate, 0)
    eRate = eRate.T
    sumErate = np.sum(eRate)
    if sumErate == 0:
        norm = None
    else:
        norm = matplotlib.colors.LogNorm()
    plt.imshow(eRate, extent=[min(labelEng), max(labelEng), 0, len(plotCoreZen)], aspect='auto',
                norm=norm, cmap=cmap, vmin=np.max(eRate)*0.01)

    ax_labels = []
    for iZ, zen in enumerate(labelZen):
        ax_labels.append('{:.0f}'.format(labelZen[iZ]))
    plt.yticks(range(len(labelZen)), ax_labels)
    plt.xlabel('Core Energy (log10eV)')
    plt.ylabel('Zenith (deg)')
    plt.colorbar(label=f'{np.sum(eRate):.5f} Evts/Stn/Yr')
    plt.title(title_comment)

    return

def plotCrEnergyZenithHeat(CoreObjectsList, title_comment='', type='TA'):
    #Make a heatmap of the event rate distributed in energy and zenith phase space
    plotCrEng = []
    plotCrZen = []
    labelCosZen = []
    labelEng = []
    for core in CoreObjectsList:
        CRs, rates, errorrate = core.eventRatePerEcr(type=type)
        zen = core.zen_center
        for iC, cr in enumerate(CRs):
            if not cr in plotCrEng:
                plotCrEng.append(cr)
        if not zen in plotCrZen:
            plotCrZen.append(zen)
        for coszen in core.coszen_bins:
            if not coszen in labelCosZen:
               labelCosZen.append(coszen)
        for eng in core.e_bins:
            if not eng in labelEng:
                labelEng.append(eng)

    plotCrEng.sort()
    plotCrZen.sort()
    labelEng.sort()
    ic(labelEng)

    labelZen = []
    for coszen in labelCosZen:

        labelZen.append(np.rad2deg(np.nan_to_num(np.arccos(coszen))))
    labelZen.sort()


    eRate = np.zeros((len(plotCrEng), len(plotCrZen)))
    for core in CoreObjectsList:
        CRs, rates, errorrate = core.eventRatePerEcr(type=type)
        for iC, cr in enumerate(CRs):
            eIndex = plotCrEng.index(cr)
            zIndex = plotCrZen.index(core.zen_center)
            eRate[eIndex, zIndex] += rates[iC]


    eRate = np.fliplr(eRate).T
    eRate, cmap = set_bad_imshow(eRate, 0)
    sumErate = np.sum(eRate)
    if sumErate == 0:
        norm = None
    else:
        norm = matplotlib.colors.LogNorm()
    plt.imshow(eRate, extent=[min(labelEng), max(labelEng), 0, len(plotCrZen)], aspect='auto', 
                norm=norm, cmap=cmap, vmin=np.max(eRate)*0.01)
    ax_labels = []
    for zen in labelZen:
        ax_labels.append('{:.0f}'.format(zen))
    plt.yticks(range(len(labelZen)), ax_labels)
    plt.xlabel('CR Energy (log10eV)')
    plt.ylabel('Zenith (deg)')
    plt.colorbar(label=f'{np.sum(eRate):.5f} Evts/Stn/Yr')
    plt.title(title_comment)

    return
    
def getMeanEcoreEcr(CoreObjectsList, type='TA'):

    EcoreEcrArray = []
    EcoreEcrWeights = []
    for core in CoreObjectsList:
        coreEnergy = core.e_center
        CRs, rates, errorrate = core.eventRatePerEcr(type=type)
        for iC, Ecr in enumerate(CRs):
            #Both values are in log10, need to do exponential of them
            EcoreEcrArray.append( 10 ** (coreEnergy - Ecr))
            EcoreEcrWeights.append(rates[iC])
        
    weighted_mean = np.average(EcoreEcrArray, weights=EcoreEcrWeights)
#    plt.hist(EcoreEcrArray, bins=20, weights=EcoreEcrWeights)
#    plt.xlabel(r'$E_{core}$/$E_{CR}$')

    return weighted_mean, EcoreEcrArray, EcoreEcrWeights

def plotMeanEcoreEcr(CoreObjectsList, title_comment='', type='TA'):

    weighted_mean, EcoreEcrArray, EcoreEcrWeights = getMeanEcoreEcr(CoreObjectsList, type=type)

    bins = np.arange(0, 1.05, 0.1)
    plt.hist(EcoreEcrArray, bins=bins, weights=EcoreEcrWeights, label=f'Mean {weighted_mean:.2f}')
    plt.xlabel(f'Ecore/Ecr')

    return


def getXSlantHistogram(CoreObjectsList, loc='SP', type='TA'):
    #Set atmosphere depth in g/cm^2
    if loc == 'SP':
        Xice = 680
    elif loc == 'MB':
        Xice = 1000

    XmaxArray = []
    EventRateWeighting = []     #This event rate weighting should be in events/stn/year, combining weighting of Aeff and event rate

    for core in CoreObjectsList:
        coszen = core.coszen_center
        CRs, Xmaxs, rates, errorrate = core.eventRatePerEcr(type=type, return_Xmax=True)
        for iC, cr in enumerate(CRs):
            if rates[iC] == 0:
                continue
            #Slant depth is plotted to match Simon's work
#            print(f'math Xice {Xice} - Xmax {Xmaxs[iC]} / coszen {coszen} = {(Xice - Xmaxs[iC]) / coszen}, weight of {rates[iC]}')
#            continue
            XmaxArray.append( Xice / coszen - Xmaxs[iC])
            EventRateWeighting.append(rates[iC])

    return XmaxArray, EventRateWeighting 

def getXSlantPerVarHistogram(CoreObjectsList, loc='SP', type='TA', var='zenith'):
    #Returns an array of an array per energy, as well as an array of the energies

    #Set atmosphere depth in g/cm^2
    if loc == 'SP':
        Xice = 680
    elif loc == 'MB':
        Xice = 1000

    BinXmaxDict = {}

    for core in CoreObjectsList:
        coszen = core.coszen_center
        if var == 'E_core':
            bin = f'{core.e_bins[0]:.1f}-{core.e_bins[1]:.1f} log10eV Core'
        elif var == 'zenith':
            bin = f'{np.nan_to_num(np.rad2deg(np.arccos(core.coszen_bins[1]))):.0f}-{np.rad2deg(np.arccos(core.coszen_bins[0])):.0f} deg'
        CRs, Xmaxs, rates, errorrate = core.eventRatePerEcr(type=type, return_Xmax=True)
        for iC, cr in enumerate(CRs):
            energy = cr
            if rates[iC] == 0:
                continue
            if var == 'E_cr':
                bin = f'{cr:.1f} lov10eV CR'
            if not bin in BinXmaxDict:
                BinXmaxDict[bin] = {}
                BinXmaxDict[bin]['Xmax'] = []
                BinXmaxDict[bin]['weight'] = []
            BinXmaxDict[bin]['Xmax'].append( Xice / coszen - Xmaxs[iC])
            BinXmaxDict[bin]['weight'].append(rates[iC])

    XmaxArray = []
    VarArray = []
    EventRateWeighting = []

    for bin in BinXmaxDict:
        VarArray.append(bin)
        XmaxArray.append(BinXmaxDict[bin]['Xmax'])
        EventRateWeighting.append(BinXmaxDict[bin]['weight'])

    return XmaxArray, EventRateWeighting, VarArray

def plotSimonsXSlant(color='black'):
    #Plot Simon's best fit overlay
    A = 0.242
    B = -0.00808
    C = 0.0429
    x = np.linspace(-50, 350, 400)
    y = A * np.exp(B * x) + C
    plt.plot(x, y, label=f'{A}*exp^({B}*x) + {C}', color=color)
    return

def plotXSlantHistogram(CoreObjectsList, loc='SP', type='TA'):
    XmaxArray, EventRateWeighting = getXSlantHistogram(CoreObjectsList, loc, type)

    weightedMean = np.average(XmaxArray, weights=EventRateWeighting)

    """
    plotSimonsXSlant()
    plt.ylabel('En/Eres = f')
    plt.xlabel(f'Xice - Xmax (slant) (g/cm^2)')
    plt.legend(loc='upper left')
    """
    ax2 = plt.twinx()
    ax2.hist(XmaxArray, bins=20, weights=EventRateWeighting, color='orange', alpha=0.5)
    ax2.set_yticks([])
    ax2.yaxis.set_tick_params(labelright=False)
    ax2.set_xlim((-300, 400))

    return weightedMean

def plotXSlantPerVarHistogram(CoreObjectsList, loc='SP', type='TA', var='zenith'):
    XmaxArray, EventRateWeighting, VarArray = getXSlantPerVarHistogram(CoreObjectsList, loc, type, var)

#    weightedMean = np.average(XmaxArray, weights=EventRateWeighting)

    plotSimonsXSlant()
    plt.ylabel('En/Eres = f')
    plt.xlabel(f'Xice - Xmax (slant) (g/cm^2)')
    ax2 = plt.twinx()
    labels = []
    for iB, bin in enumerate(VarArray):
        mean = np.average(XmaxArray[iB], weights=EventRateWeighting[iB])
        variance = np.sqrt(np.average([(value-mean)**2 for value in XmaxArray[iB]], weights=EventRateWeighting[iB]))
        labels.append(f'{bin}, {mean:.0f}±{variance:.0f}')
#    ax2.hist(XmaxArray, bins=20, weights=EventRateWeighting, alpha=0.5, stacked=True, label=[f'{bin}' for bin in VarArray])
    ax2.hist(XmaxArray, bins=20, weights=EventRateWeighting, alpha=0.5, stacked=True, label=labels)
    ax2.set_yticks([])
    ax2.yaxis.set_tick_params(labelright=False)
    ax2.set_xlim((-300, 400))
    ax2.legend(loc='upper right')

    return 0


def plotCrShowerHeatFlux(CoreObjectsList, f=0.3, type='TA', loc='SP'):
    if loc == 'SP':
        atm_overburden = 680
    elif loc == 'MB':
        atm_overburden = 1000

    if type == 'TA':
        with open(f"data/output_TA_19_{atm_overburden}.pkl", "rb") as fin:
            shower_energies, weights_shower_energies, shower_xmax = pickle.load(fin)
            fin.close()
    elif type == 'Auger':
        with open(f"data/output_auger_19_{atm_overburden}.pkl", "rb") as fin:
            shower_energies, weights_shower_energies, shower_xmax = pickle.load(fin)
            fin.close()
    else:
        print(f'Type {type} is wrong')
        quit()


    #For iterating through the generated showers
    shower_E_bins = np.arange(17, 20.01, 0.1)
    logEs = 0.5 * (shower_E_bins[1:] + shower_E_bins[:-1])
    dCos = 0.05
    coszen_bin_edges = np.arange(np.cos(np.deg2rad(60)), 1.01, dCos)
    coszen_bin_edges = np.flip(coszen_bin_edges)
    coszens = 0.5 * (coszen_bin_edges[1:] + coszen_bin_edges[:-1])
    cos_edges = np.arccos(coszen_bin_edges)
    cos_edges[np.isnan(cos_edges)] = 0


    plotCrEng = np.arange(17.0, 20.01, 0.1)
    plotShEng = np.arange(15.0, 20.01, 0.1)
    eRate = np.zeros((len(plotCrEng), len(plotShEng)))

    #Add showers to right bins
    for iE, logE in enumerate(logEs):
        eIndex = np.digitize(logE, plotCrEng)
#        print(f'logE {logE} index {eIndex} in plot {plotCrEng}')
        for iC, coszen in enumerate(coszens):
            mask = ~np.isnan(shower_energies[iE][iC])
            shEngs = shower_energies[iE][iC][mask]
            shRates = weights_shower_energies[iE][iC][mask]
            for shN, shEng in enumerate(shEngs):
                shIndex = np.digitize(np.log10(shEng * f), plotShEng)
#                print(f'sh eng {np.log10(shEng * f)} index {shIndex} in plot {plotShEng}')
                eRate[eIndex, shIndex] += shRates[shN]


    eRate = np.fliplr(eRate)
    eRate, cmap = set_bad_imshow(eRate, 0)
    eRate = eRate.T
    norm = matplotlib.colors.LogNorm()
    plt.imshow(eRate, extent=[17.0, 20.0, 15.0, 20.0], aspect='auto',
                norm=norm, cmap=cmap, vmin=3*10**-7, vmax=5*10**2)
    plt.xlabel('lg(Ecr/eV)')
    plt.ylabel('lg(Esh/eV) = lg(f*Ecore/eV)')
    plt.colorbar()

    return


if __name__ == "__main__":

    #Plot multiple f values
    if 0:
        plotSimonsXSlant(color='black')
        plt.ylabel('En/Eres = f')
        plt.xlabel(f'Xice - Xmax (slant) (g/cm^2)')

        labels = []

        core_input_file = 'data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_3.8sigma_below_300mRefl_SP_1R_0.3f_40.0dB_1.7km_1000cores.pkl'
        with open(core_input_file, 'rb') as fin:
            CoreObjectsList = pickle.load(fin)

        XmaxArray, EventRateWeighting = getXSlantHistogram(CoreObjectsList)
        mean = np.average(XmaxArray, weights=EventRateWeighting)
        variance = np.sqrt(np.average([(value-mean)**2 for value in XmaxArray], weights=EventRateWeighting))
        ax2 = plt.twinx()
        ax2.hist(XmaxArray, bins=20, weights=EventRateWeighting, edgecolor='red', fill=False, histtype='step', label=f'Refl LPDA, mean {mean:.0f}±{variance:.1f}')
        ax2.set_yticks([])
        ax2.yaxis.set_tick_params(labelright=False)
        labels.append(f'Refl LPDA, mean {mean:.0f}±{variance:.0f}')

        core_input_file = 'data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_single_dipole_trigger_3sig_below_300mRefl_SP_1R_0.3f_40.0dB_1.7km_1000cores.pkl'
        with open(core_input_file, 'rb') as fin:
            CoreObjectsList = pickle.load(fin)

        XmaxArray, EventRateWeighting = getXSlantHistogram(CoreObjectsList)
        mean = np.average(XmaxArray, weights=EventRateWeighting)
        variance = np.sqrt(np.average([(value-mean)**2 for value in XmaxArray], weights=EventRateWeighting))
    #    ax2.hist(XmaxArray, bins=20, weights=EventRateWeighting, edgecolor='orange', fill=False, histtype='step', label=f'Refl Dip, mean {mean:.0f}')
        ax3 = plt.twinx()
        ax3.hist(XmaxArray, bins=20, weights=EventRateWeighting, edgecolor='orange', fill=False, histtype='step', label=f'Refl Dip, mean {mean:.0f}±{variance:.1f}')
        ax3.set_yticks([])
        ax3.yaxis.set_tick_params(labelright=False)
        labels.append(f'Refl Dip, mean {mean:.0f}±{variance:.0f}')

        core_input_file = 'data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_single_dipole_trigger_3sig_above_300mRefl_SP_1R_0.3f_0.0dB_1.7km_1000cores.pkl'
        with open(core_input_file, 'rb') as fin:
            CoreObjectsList = pickle.load(fin)

        XmaxArray, EventRateWeighting = getXSlantHistogram(CoreObjectsList)
        mean = np.average(XmaxArray, weights=EventRateWeighting)
        variance = np.sqrt(np.average([(value-mean)**2 for value in XmaxArray], weights=EventRateWeighting))
    #    ax2.hist(XmaxArray, bins=20, weights=EventRateWeighting, edgecolor='pink', fill=False, histtype='step', label=f'Dir Dip, mean {mean:.0f}')
        ax5 = plt.twinx()
        ax5.hist(XmaxArray, bins=20, weights=EventRateWeighting, edgecolor='blue', fill=False, histtype='step', label=f'Dir Dip, mean {mean:.0f}±{variance:.1f}')
        ax5.set_yticks([])
        ax5.yaxis.set_tick_params(labelright=False)
        labels.append(f'Dir Dip, mean {mean:.0f}±{variance:.0f}')

        core_input_file = 'data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_2sigma_below_576mRefl_MB_1R_0.05f_0.0dB_1.7km_1000cores.pkl'
        with open(core_input_file, 'rb') as fin:
            CoreObjectsList = pickle.load(fin)

        XmaxArray, EventRateWeighting = getXSlantHistogram(CoreObjectsList, loc='MB')
        mean = np.average(XmaxArray, weights=EventRateWeighting)
        variance = np.sqrt(np.average([(value-mean)**2 for value in XmaxArray], weights=EventRateWeighting))
    #    ax2.hist(XmaxArray, bins=20, weights=EventRateWeighting, edgecolor='green', fill=False, histtype='step', label=f'Refl LPDA MB, mean {mean:.0f}')
        ax4 = plt.twinx()
        ax4.hist(XmaxArray, bins=20, weights=EventRateWeighting, edgecolor='green', fill=False, histtype='step', label=f'Refl LPDA MB, mean {mean:.0f}±{variance:.1f}')
        ax4.set_yticks([])
        ax4.yaxis.set_tick_params(labelright=False)
        labels.append(f'Refl LPDA MB, mean {mean:.0f}±{variance:.0f}')


        legend1 = plt.legend(handles=[plt.plot([], [], color='red')[0], plt.plot([], [], color='orange')[0], plt.plot([], [], color='blue')[0], plt.plot([], [], color='green')[0]],
                            labels=labels, loc='upper right', prop={'size':16})
        plt.gca().add_artist(legend1)

        plt.xlim((-300, 700))
    #    plt.title(f'3.8sigma lpda, 300m 40dB 0.4f SP, 0.05 MB')
        plt.savefig(f'plots/CoreAnalysis/xSlantHistMultiLayer.png')
        plt.clf()
        quit()

    core_input_file = 'data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_3.8sigma_below_300mRefl_SP_1R_fZenVariablef_40.0dB_1.7km_1000cores.pkl'
    with open(core_input_file, 'rb') as fin:
        CoreObjectsList = pickle.load(fin)

    plotXSlantPerVarHistogram(CoreObjectsList)
    plt.savefig(f'plots/CoreAnalysis/SP/July2023/fVarZenith_XSlantHistZeniths300m.png')
    plt.clf()
    quit()

    mean = plotXSlantHistogram(CoreObjectsList)
    plt.title(f'3.8sigma lpda, 300m 40dB 0.4f, Mean {mean:.0f}')
    plt.savefig(f'plots/CoreAnalysis/xSlantHistTest300m.png')
    plt.clf()


    plotMeanEcoreEcr(CoreObjectsList)
    plt.title('3.8sigma lpda, 300m 40dB 0.4f')
    plt.savefig(f'plots/CoreAnalysis/MeanEcoreEcrTest300m.png')
    plt.clf()

    core_input_file = 'data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_3.8sigma_below_500mRefl_SP_1R_0.4f_40.0dB_1.7km_1000cores.pkl'
    with open(core_input_file, 'rb') as fin:
        CoreObjectsList = pickle.load(fin)
    mean = plotXSlantHistogram(CoreObjectsList)
    plt.title(f'3.8sigma lpda, 500m 40dB 0.4f, Mean {mean:.0f}')
    plt.savefig(f'plots/CoreAnalysis/xSlantHistTest500m.png')
    plt.clf()


    plotMeanEcoreEcr(CoreObjectsList)
    plt.title('3.8sigma lpda, 500m 40dB 0.4f')
    plt.savefig(f'plots/CoreAnalysis/MeanEcoreEcrTest500m.png')
    plt.clf()

    core_input_file = 'data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_3.8sigma_below_800mRefl_SP_1R_0.4f_40.0dB_1.7km_1000cores.pkl'
    with open(core_input_file, 'rb') as fin:
        CoreObjectsList = pickle.load(fin)
    mean = plotXSlantHistogram(CoreObjectsList)
    plt.title(f'3.8sigma lpda, 800m 40dB 0.4f, Mean {mean:.0f}')
    plt.savefig(f'plots/CoreAnalysis/xSlantHistTest800m.png')
    plt.clf()


    plotMeanEcoreEcr(CoreObjectsList)
    plt.title('3.8sigma lpda, 800m 40dB 0.4f')
    plt.savefig(f'plots/CoreAnalysis/MeanEcoreEcrTest800m.png')
    plt.clf()
