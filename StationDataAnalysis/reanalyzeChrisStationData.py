from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units
import argparse
import numpy as np
import time
import NuRadioReco.modules.channelBandPassFilter

from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from astropy.time import Time
import datetime

import matplotlib.pyplot as plt

import json
from NuRadioReco.utilities.io_utilities import read_pickle
import templateCrossCorr as txc

import NuRadioReco.modules.io.eventWriter
import os

def inBlackoutTime(time, blackoutTimes):
    for blackouts in blackoutTimes:
        if blackouts[0] < time and time < blackouts[1]:
            return True
    return False

blackoutFile = open('StationDataAnalysis/cuts/BlackoutCuts.json')
blackoutData = json.load(blackoutFile)
blackoutFile.close()

blackoutTimes = []

for iB, tStart in enumerate(blackoutData['BlackoutCutStarts']):
    tEnd = blackoutData['BlackoutCutEnds'][iB]
    blackoutTimes.append([tStart, tEnd])


parser = argparse.ArgumentParser(description='Run template matching analysis on on data')
parser.add_argument('files', type=str, default='', help='File to run on')
parser.add_argument('--station', type=int, default=14, help='Station data is being ran on, default 14')
parser.add_argument('--templates_nu', type=str, default='200', help='Pkl file of template for Xcorr, default 200series Nu')
parser.add_argument('--templates_cr', type=str, default='200', help='200 for 200series template, 100 for 100, otherwise pass pickle file of custom template')
parser.add_argument('--output', type=str, default='', help='File to save event writer to')

args = parser.parse_args()
filesToRead = args.files
station_id = args.station
templates_nu = args.templates_nu
templates_cr = args.templates_cr
output_filename = args.output



template_sampling_rate = 2*units.GHz
if templates_nu == '100':
    templates_nu = 'StationDataAnalysis/templates/NUdowntemplate_100hp1000lpFilter_100series_SST.pkl'
    t_zen = np.deg2rad(120)
    t_azi = np.deg2rad(30.0)
    t_view = np.deg2rad(0.0)
    templates_nu = read_pickle(templates_nu)[t_zen][t_azi][t_view]
elif templates_nu == '200':
    templates_nu = 'StationDataAnalysis/templates/NUdowntemplate_NoFilter_200series_SST.pkl'
    t_zen = np.deg2rad(120)
    t_azi = np.deg2rad(30.0)
    t_view = np.deg2rad(0.0)
    templates_nu = read_pickle(templates_nu)[t_zen][t_azi][t_view]


if templates_cr == '100':
    templates_cr = 'StationDataAnalysis/templates/reflectedCR_template_100series.pkl'
elif templates_cr == '200':
    templates_cr = 'StationDataAnalysis/templates/reflectedCR_template_200series.pkl'
if True:
    templates_cr = read_pickle(templates_cr)
    for key in templates_cr:
        temp = templates_cr[key]
    templates_cr = temp


if output_filename == '':
    filename = os.path.basename(filesToRead)
    output_filename = f'StationDataAnalysis/processedNur/station_{station_id}/'+filename
print(f'saving file to {output_filename}')



fin = NuRadioRecoio.NuRadioRecoio(filesToRead)

eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin(output_filename)
mode = {'Channels':True, 'ElectricFields':False, 'SimChannels':False, 'SimElectricFields':False}

det = detector.Detector()
det.update(datetime.datetime(2015, 12, 12))
parallelChannels = det.get_parallel_channels(station_id)
noiseRMS = det.get_noise_RMS(station_id, channel_id=0, stage='raw')


channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()


maxCorr = []
SNRs = []
forcedMask = []

#forcedNoiseRMS = []
noiseRMS = 22.53 * units.mV

for iE, evt in enumerate(fin.get_events()):

    station = evt.get_station(station_id)
    stationtime = station.get_station_time().unix
    if inBlackoutTime(stationtime, blackoutTimes):
        continue


#    channel = station.get_channel(0)
#    plt.plot(channel.get_times() / units.ns, channel.get_trace() / units.mV, label='pre-DC removal')
    channelBandPassFilter.run(evt, station, det, passband=[1*units.Hz, np.inf*units.MHz])
#    plt.plot(channel.get_times() / units.ns, channel.get_trace() / units.mV, label='post-DC removal')
#    plt.legend()
#    plt.show()


#    triggers = station.get_triggers()
#    print(f'triggers {triggers}')

#    maxCorrAndSNR = []

#    maxCorrAndSNR.append([0, 0])
    maxCorr.append(0)
    SNRs.append(0)
    forcedMask.append(station.has_triggered())
    for parChans in parallelChannels:
        nu_parCorr = 0
        cr_parCorr = 0
        Vp2p = []
        Vmax = []
        for channel in station.iter_channels(use_channels=parChans):
            cTrace = channel.get_trace()
#            plt.plot(cTrace, label='ctrace')
#            plt.plot(templates, label='temp')
#            plt.legend()
#            plt.show()
            nu_xCorr= txc.get_xcorr_for_channel(cTrace, templates_nu, channel.get_sampling_rate(), template_sampling_rate, times=channel.get_times(), debug=False)		#Chris template
            cr_xCorr= txc.get_xcorr_for_channel(cTrace, templates_cr, channel.get_sampling_rate(), template_sampling_rate, times=channel.get_times(), debug=False)		#CR refl template
            nu_parCorr += np.abs(nu_xCorr)
            cr_parCorr += np.abs(cr_xCorr)
            Vp2p.append(np.max(cTrace) + np.abs(np.min(cTrace)))
            Vmax.append(np.max(cTrace))

            channel.set_parameter(chp.nu_xcorrelations, nu_xCorr)
            channel.set_parameter(chp.cr_xcorrelations, cr_xCorr)
            SNR = {}
            SNR['peak_2_peak_amplitude'] = np.max(cTrace) + np.abs(np.min(cTrace))
            SNR['noise_rms'] = noiseRMS
            SNR['peak_amplitude'] = np.max(np.abs(cTrace))

            channel.set_parameter(chp.SNR, SNR)

        """
        avgCorr = parCorr / len(parChans)
#        print(f'avg corr {avgCorr}') 
#        if avgCorr > maxCorrAndSNR[-1][0]:
        if avgCorr > maxCorr[-1]:
            SNR = min(Vp2p) / (2*noiseRMS)
            maxCorr[-1] = avgCorr
            SNRs[-1] = SNR           
#            maxCorrAndSNR[-1][0] = avgCorr
#            maxCorrAndSNR[-1][1] = SNR
        """
#    print(f'max cor and snr {maxCorrAndSNR}')

    eventWriter.run(evt, mode=mode)


#    quit()

#print(f'forced noise of {np.mean(forcedNoiseRMS)}')

"""
SNRbins = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
maxCorrBins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

forcedMask = np.array(forcedMask)
SNRs = np.array(SNRs)
maxCorr = np.array(maxCorr)


#plt.hist2d(SNRs[forcedMask], maxCorr[forcedMask], bins=[SNRbins, maxCorrBins])
#plt.colorbar()
plt.scatter(SNRs[~forcedMask], maxCorr[~forcedMask], label='Forced', facecolors='none', edgecolor='black')
plt.scatter(SNRs[forcedMask], maxCorr[forcedMask], label='Trigger')
plt.xlim((3, 100))
plt.ylim((0, 1))
plt.xlabel('SNR')
plt.ylabel('Chi_avg')
plt.xscale('log')
plt.legend()
plt.show()

quit()
"""
