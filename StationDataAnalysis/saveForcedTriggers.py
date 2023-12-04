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
    output_filename = f'StationDataAnalysis/processedNur/station_{station_id}/forced_triggers_'+filename
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


    if not station.has_triggered():
        eventWriter.run(evt, mode=mode)

