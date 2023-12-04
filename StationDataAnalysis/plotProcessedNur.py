import os
import argparse
import numpy as np
import NuRadioReco.modules.io.eventReader
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.detector import detector
from astropy.time import Time
import datetime
import pickle
from NuRadioReco.utilities import units
import itertools

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors

import StationDataAnalysis.Nu_RCR_ChiCut as ChiCut

color = itertools.cycle(('black', 'blue', 'green', 'orange'))



endAnalysisTime = Time('2019-02-04').unix

parser = argparse.ArgumentParser(description='Run template matching analysis on on data')
parser.add_argument('--station', type=int, default=14, help='Station data is being ran on, default 14')
parser.add_argument('--num_traces', type=int, default=0, help='Number of traces to save, default 0')
parser.add_argument('-files', '--list', type=str, nargs='+',default='', help='File to run on', required=True)


args = parser.parse_args()
filesToRead = args.list
station_id = args.station
num_traces = args.num_traces

det = detector.Detector()
det.update(datetime.datetime(2015, 12, 12))

parallelChannels = det.get_parallel_channels(station_id)

eventReader = NuRadioReco.modules.io.eventReader.eventReader()

nu_maxCorr = []
cr_maxCorr = []
nu_SNRs = []
cr_SNRs = []
nu_forcedMask = []
cr_forcedMask = []


s100 = [13, 15, 18]
s200 = [14, 17, 19, 30]
#if station_id in s100:
#    noise_rms = 20 * units.mV
#elif station_id in s200:
#    noise_rms = 22.5 * units.mV
noise_rms = 20 * units.mV	#RMS used by Chris


tracesPlotted = 0
for file in filesToRead:
    eventReader.begin(file)
#if True:
#    eventReader.begin(filesToRead)
    print(f'running file {file}')


    for evt in eventReader.run():
        station = evt.get_station(station_id)

        stationTime = station.get_station_time().unix
        if stationTime > endAnalysisTime:
            continue

        nu_xCorr = 0
        cr_xCorr = 0
        nu_maxSNR = 0
        cr_maxSNR = 0
        for parChans in parallelChannels:
            nu_avgCorr = []
            cr_avgCorr = []
            nu_SNR = []
            cr_SNR = []
            for channel in station.iter_channels(use_channels=parChans):
                nu_avgCorr.append(np.abs(channel.get_parameter(chp.nu_xcorrelations)))
                cr_avgCorr.append(np.abs(channel.get_parameter(chp.cr_xcorrelations)))


                tempSNR = channel.get_parameter(chp.SNR)['peak_2_peak_amplitude']
                if isinstance(tempSNR, np.float64):
                    tempSNR = tempSNR / (2*noise_rms)
                    nu_SNR.append(tempSNR)
                    cr_SNR.append(tempSNR)
                else:
                    nu_SNR.append(max(channel.get_parameter(chp.SNR)['peak_2_peak_amplitude']) / (2*noise_rms))	#SNR
                    cr_SNR.append(max(channel.get_parameter(chp.SNR)['peak_2_peak_amplitude']) / (2*noise_rms))	#SNR

                if (tracesPlotted < num_traces and cr_avgCorr[-1] > 0.7) or ChiCut.passesCut(nu_avgCorr[-1], cr_avgCorr[-1]):		#Save channel traces if high xCorr to refl Cr template
                    print(f'Printing trace {tracesPlotted} CR xCorr {cr_avgCorr[-1]}')
                    plt.plot(channel.get_times(), channel.get_trace())
                    plt.xlabel('ns')
                    plt.title(f'Stn {station_id} CR Chi {cr_avgCorr[-1]:.2f} SNR {cr_SNR[-1]:.2f} ' + station.get_station_time().fits)
                    plt.savefig(f'StationDataAnalysis/plots/traces/station_{station_id}/station{station_id}_trace_Chi{cr_avgCorr[-1]:.2f}_SNR{cr_SNR[-1]:.2f}_{station.get_station_time().fits}.png')
                    plt.clf()

                    plt.plot(channel.get_frequencies()/units.MHz, np.abs(channel.get_frequency_spectrum()))
                    plt.xlabel('Freq (MHz)')
                    plt.xlim([0, 500])
                    plt.title(f'Stn {station_id} CR Chi {cr_avgCorr[-1]:.2f} SNR {cr_SNR[-1]:.2f} ' + station.get_station_time().fits)
                    plt.savefig(f'StationDataAnalysis/plots/traces/station_{station_id}/station{station_id}_freqs_Chi{cr_avgCorr[-1]:.2f}_SNR{cr_SNR[-1]:.2f}_{station.get_station_time().fits}.png')
                    plt.clf()

                    tracesPlotted = tracesPlotted + 1


            nu_avgCorr = np.mean(np.abs(nu_avgCorr))	#xcorr can be negative, take mean of the absolute values
            cr_avgCorr = np.mean(np.abs(cr_avgCorr))
            nu_SNR = max(nu_SNR)
            cr_SNR = max(cr_SNR)
            if nu_avgCorr > nu_xCorr:
                nu_xCorr = nu_avgCorr
                nu_maxSNR = max(nu_maxSNR, nu_SNR)		#SNR
            if cr_avgCorr > cr_xCorr:
                cr_xCorr = cr_avgCorr
                cr_maxSNR = max(cr_maxSNR, cr_SNR)		#SNR

        if ChiCut.passesCut(nu_xCorr, cr_xCorr):
            figint = 411
            smalint = 0
            fig, ax = plt.subplots(4, 1, sharey='col')
            for parChans in parallelChannels:
                for channel in station.iter_channels(use_channels=parChans):
                    cTimes = channel.get_times()
                    cTrace = channel.get_trace()
                    ax[smalint].plot(cTimes, cTrace, label=f'Ch{channel.get_id()}', color=next(color))
                    smalint += 1
                    figint += 1                

            plt.xlabel('ns')
            plt.legend()
            fig.suptitle(f'Stn {station_id} R-CR Chi {cr_xCorr:.2f} Nu Chi {nu_xCorr:.2f} ' + station.get_station_time().fits)
            plt.savefig(f'StationDataAnalysis/plots/traces/station_{station_id}/station{station_id}_PassedCut_{station.get_station_time().fits}.png', format='png')
#            plt.show()
            plt.clf()


        if (nu_xCorr > 0) or (cr_xCorr > 0):
            nu_maxCorr.append(nu_xCorr)
            nu_SNRs.append(nu_maxSNR)
            nu_forcedMask.append(station.has_triggered())
#        if cr_xCorr > 0:
            cr_maxCorr.append(cr_xCorr)
            cr_SNRs.append(cr_maxSNR)
            cr_forcedMask.append(station.has_triggered())



nu_maxCorr = np.array(nu_maxCorr)
cr_maxCorr = np.array(cr_maxCorr)
nu_SNRs = np.array(nu_SNRs)
cr_SNRs = np.array(cr_SNRs)
nu_forcedMask = np.array(nu_forcedMask)
cr_forcedMask = np.array(cr_forcedMask)

print(f'nu srns {nu_SNRs} shape {np.shape(nu_SNRs)}')
print(f'shape nu forced {np.shape(nu_forcedMask)} and cr {np.shape(cr_forcedMask)}')

#SNRbins = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#maxCorrBins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

SNRbins = np.logspace(0.477, 2, num=80)			#SNR bins
#SNRbins = np.linspace(1, 4, num=80)			#minlogp2p

maxCorrBins = np.arange(0, 1.0001, 0.01)


nuHighChiMask = nu_maxCorr > 0.7
crHighChiMask = cr_maxCorr > 0.7
numNuHigh = nuHighChiMask.sum()
numCrHigh = crHighChiMask.sum()


cutMask = ChiCut.cutMask(nu_maxCorr, cr_maxCorr)

plt.scatter(nu_maxCorr[np.logical_not(nu_forcedMask)], cr_maxCorr[np.logical_not(nu_forcedMask)], label=f'Forced Trigs', facecolors='none', edgecolor='black')
plt.scatter(nu_maxCorr[nu_forcedMask], cr_maxCorr[nu_forcedMask], label=f'{len(nu_maxCorr[nu_forcedMask])} Evts')
if np.any(cutMask):
    plt.scatter(nu_maxCorr[cutMask], cr_maxCorr[cutMask], label=f'{np.sum(cutMask)} Pass')
ChiCut.plotCut()
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xlabel('Nu Chi')
plt.ylabel('R-CR Chi')
plt.legend()
plt.title(f'Station {station_id} Nu R-CR Chis')
#plt.show()
plt.savefig(f'StationDataAnalysis/plots/station{station_id}_NuRCR_Chi_dist.png')
plt.clf()




#quit()
##################################################################



plt.hist(nu_SNRs[nu_forcedMask], bins=SNRbins)
plt.xlabel('SNR')
plt.xscale('log')
plt.title(f'Station {station_id} Triggered Amplitudes')
#plt.show()
plt.savefig(f'StationDataAnalysis/plots/SNRplots/station{station_id}_SNR_distribution.png')
plt.clf()



plt.hist2d(nu_SNRs[nu_forcedMask], nu_maxCorr[nu_forcedMask], bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
plt.colorbar()
if numNuHigh > 0:
    plt.scatter(nu_SNRs[nuHighChiMask], nu_maxCorr[nuHighChiMask], label=f'High Nu Events {numNuHigh}', facecolors='none', edgecolor='black')
if numCrHigh > 0:
    plt.scatter(nu_SNRs[crHighChiMask], nu_maxCorr[crHighChiMask], label=f'High CR Events {numCrHigh}', facecolors='none', edgecolor='orange')
if np.any(cutMask):
    plt.scatter(nu_SNRs[cutMask], nu_maxCorr[cutMask], label=f'Passed Chi-Chi Cut', facecolor='none', edgecolor='red')
if numNuHigh > 0 or numCrHigh > 0 or np.any(cutMask):
    plt.legend()
plt.ylim((0, 1))
plt.xlabel('SNR')
plt.ylabel('Nu Chi_avg')

plt.xlim((3, 100))
plt.xscale('log')

plt.tick_params(axis='x', which='minor', bottom=True)
plt.grid(which='both')
plt.title(f'Station {station_id} Neutrino Correlations')
#plt.show()
plt.savefig(f'StationDataAnalysis/plots/SNRplots/station{station_id}_NuXcorr.png')
plt.clf()





plt.hist2d(cr_SNRs[cr_forcedMask], cr_maxCorr[cr_forcedMask], bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
plt.colorbar()
if numNuHigh > 0:
    plt.scatter(cr_SNRs[nuHighChiMask], cr_maxCorr[nuHighChiMask], label=f'High Nu Events {numNuHigh}', facecolors='none', edgecolor='black')
if numCrHigh > 0:
    plt.scatter(cr_SNRs[crHighChiMask], cr_maxCorr[crHighChiMask], label=f'High CR Events {numCrHigh}', facecolors='none', edgecolor='orange')
if np.any(cutMask):
    plt.scatter(cr_SNRs[cutMask], cr_maxCorr[cutMask], label=f'Passed Chi-Chi Cut', facecolor='none', edgecolor='red')
if numNuHigh > 0 or numCrHigh > 0 or np.any(cutMask):
    plt.legend()
plt.ylim((0, 1))
plt.xlabel('SNR')
plt.ylabel('CR Chi_avg')

plt.xlim((3, 100))
plt.xscale('log')

plt.tick_params(axis='x', which='minor', bottom=True)
plt.grid(which='both')
plt.title(f'Station {station_id} Refl CR Correlations')
#plt.show()
plt.savefig(f'StationDataAnalysis/plots/SNRplots/station{station_id}_ReflCrXcorr.png')
plt.clf()

quit()

if True:
    data_dump = {}
    station = 'station_'+str(station_id)
    data_dump[station] = {}
    data_dump[station]['station_id'] = station_id
    data_dump[station]['nu_xCorr'] = nu_maxCorr
    data_dump[station]['cr_xCorr'] = cr_maxCorr
    data_dump[station]['nu_SNRs'] = nu_SNRs
    data_dump[station]['cr_SNRs'] = cr_SNRs
    data_dump[station]['nu_trigMask'] = nu_forcedMask
    data_dump[station]['cr_trigMask'] = cr_forcedMask


    with open(f'StationDataAnalysis/processedPkl/data_station_{station_id}.pkl', 'wb') as output:
        pickle.dump(data_dump, output)
    output.close()


