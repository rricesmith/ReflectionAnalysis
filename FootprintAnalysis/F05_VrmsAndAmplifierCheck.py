import os
import re
import sys
import datetime
from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioReco.detector import generic_detector
import NuRadioReco.modules.io.coreas.readCoREAS
import NuRadioReco.modules.io.coreas.readCoREASStationGrid
import NuRadioReco.modules.io.coreas.simulationSelector
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelStopFilter
import NuRadioReco.modules.voltageToEfieldConverter
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.voltageToAnalyticEfieldConverter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import scipy
import glob
import pickle
from NuRadioReco.modules.base import module
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stnp
import time
import sys
import numpy as np
import astropy
from scipy import constants
import argparse

import matplotlib.pyplot as plt
import itertools

from NuRadioReco.utilities.io_utilities import read_pickle
from radiotools.helper import get_normalized_xcorr
import templateCrossCorr as txc
import NuRadioReco.modules.io.eventWriter


color = itertools.cycle(('black', 'blue', 'green', 'orange'))



parser = argparse.ArgumentParser(description='Run CR analysis for specific runids')
parser.add_argument('--amp_type', type=str, default='300', help='Amplifier type: 100, 200, or 300. Default 300')

args = parser.parse_args()

amp_type = args.amp_type

spacing  = 1 * units.km
cores = 1
CoREAS_mode = 'direct'
min_file = 0
max_file = -1
type = 'MB'
config = 'MB_old'


if not type == 'IceTop':
    min_file = int(min_file)
    max_file = int(max_file)


print(f'noise is {noise}')

input_files = []
i = min_file
if type == 'SP':
    attenuation_model = None
    if max_file == -1:
        max_file = 2100
    while i < max_file:
        file = 'none'
        if i < 1000:
            file = f'../SPFootprints/000{i:03d}.hdf5'
        else:
            file = f'../SPFootprints/SIM00{i}.hdf5'
        if os.path.exists(file):
            input_files.append(file)
        i += 1
elif type == 'IceTop':
    attenuation_model = None
    if min_file < 16.0:
        print(f'Setting IceTop min energy to 16.0 log10eV')
        min_file = 16.0
    if max_file == -1 or max_file > 18.5:
        print(f'Setting IceTop max energy to 18.5 log10eV')
        max_file = 18.5
    i = min_file
    while i < max_file: 
        if icetop_sin == -1:
            sin2Val = np.arange(0, 1.01, 0.1)
        else:
            sin2Val = [icetop_sin]
        for sin2 in sin2Val:
            num_in_bin = 0
            folder = f'../../../../pub/arianna/SIM/southpole/IceTopLibrary/lgE_{i:.1f}/sin2_{sin2:.1f}/'
            for (dirpath, dirnames, filenames) in os.walk(folder):
                for file in filenames:
                    if not 'highlevel' in file:
                        file = os.path.join(folder, file)
                        input_files.append(file)
                        num_in_bin += 1
                    if num_in_bin == num_icetop:
                        continue
        i += 0.1
elif type == 'MB':
#    attenuation_model = 'MB_flat'
    attenuation_model = 'MB_freq'
    if max_file == -1:
        max_file = 3999
    depthLayer = 576.0
    dB = 0.0
    while i < max_file:
        file = f'../MBFootprints/00{i:04d}.hdf5'
        if os.path.exists(file):
            input_files.append(file)
        i += 1


# Logging level
import logging
logger=logging.getLogger("module")
logger.setLevel(logging.WARNING)


#type = type + '_hpol'

lpda_coinc = 2
dip_sigma = 2
if config == 'SP' or config == 'MB_future':
    det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_{config}_{amp_type}s_footprint{depthLayer:.0f}m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
    lpda_sigma = 3.5
elif config == 'MB_old':
    det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_{config}_{amp_type}s_footprint{depthLayer:.0f}m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
    lpda_sigma = 4.4
    passband_high = 700
elif config == 'SP_old':
    det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_{config}_{amp_type}s_footprint{depthLayer:.0f}m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
    lpda_sigma = 5
    lpda_coinc = 3	#3/3 for Tingwei station 51 analysis
    passband_high = 700
elif config == 'Stn51':
    det = generic_detector.GenericDetector(json_filename=f'configurations/station51.json', assume_inf=False, antenna_by_depth=True, default_station=51)
    lpda_sigma = 5
    lpda_coinc = 2	#2/3 for Testing station 51 analysis
    passband_high = 700
else:
    print(f'no config of {config} used, use SP, MB_old, or MB_future')



det.update(datetime.datetime(2018, 10, 1))

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processung


#Using station grad custom code
readCoREAS = NuRadioReco.modules.io.coreas.readCoREASStationGrid.readCoREAS()
readCoREAS.begin(input_files, -spacing/2, spacing/2, -spacing/2, spacing/2, n_cores=cores, seed=None, log_level=logging.WARNING)	#use for direct triggers


simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
simulationSelector.begin()

electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()

efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter(log_level=logging.WARNING)
efieldToVoltageConverter.begin(debug=False)

hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()

channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()

channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelSignalReconstructor.begin()

channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()


simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()

####
#For Nur file generation
####
saveEvent = False
numToSave = 500
savedEvents = 0
output_filename = f'{config}_200s_Refl_CRs_{numToSave}Evts_Noise_{noise}_Amp_{add_amp}.nur'
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin(output_filename)
mode = {'Channels':True, 'ElectricFields':True, 'SimChannels':True, 'SimElectricFields':True}
####

station_id = 1

if config == 'SP' or config == 'MB_future':
    dir_LPDA_channels = [0, 1, 2]
    refl_LPDA_channels = [3, 4, 5, 6]
    dir_dipole_channels = [7]
    refl_dipole_channels = [8]
elif config == 'MB_old':
    dir_LPDA_channels = [0, 1, 2, 3]
    refl_LPDA_channels = [4, 5, 6, 7]
    dir_dipole_channels = [8]
    refl_dipole_channels = [9]
elif config == 'SP_old':
#    dir_LPDA_channels = [0, 1, 2, 3]	#4 upward facing for testing
    dir_LPDA_channels = [0, 1, 2]	#3 upward facing is station 51 configuration
    refl_LPDA_channels = [4, 5, 6, 7]
    dir_dipole_channels = [8]
    refl_dipole_channels = [9]
elif config == 'Stn51':
#    dir_LPDA_channels = [4, 5, 6]	#Station 51 upward channels on July 2018
    dir_LPDA_channels = [4, 5, 6, 8]	#Station 51 upward channels on July 2018 with extra test channel
    refl_LPDA_channels = [0]
    dir_dipole_channels = [0]
    refl_dipole_channels = [0]



def runCoREAS(CoREAS_mode=None):
    if CoREAS_mode=='direct':
        for evt, iE, x, y in readCoREAS.run(detector=det, output_mode=2):
            yield evt, iE, x, y
    elif CoREAS_mode=='refracted':
        for evt, iE, x, y in readCoREAS.run(detector=det, ray_type='refracted', layer_depth=depthLayer, layer_dB=dB, force_dB=True, attenuation_model=attenuation_model, output_mode=2):
            yield evt, iE, x, y
    else:
        print(f'CoREAS mode {CoREAS_mode} not implemented')

Vrms_per_channel = {}
Vrms_per_channel_amp = {}
output = {}
# Loop over all events in file as initialized in readCoRREAS and perform analysis
#for evt, iE, x, y in readCoREAS.run(detector=det, output_mode=2):
#for evt, iE, x, y in readCoREAS.run(detector=det, ray_type='refracted', layer_depth=depthLayer, layer_dB=dB, force_dB=True, attenuation_model=attenuation_model, output_mode=2):
for evt, iE, x, y in runCoREAS(CoREAS_mode):
    print(f'check : returned event. x {x} y {y}')
    runid = evt.get_run_number()
    sim_shower = evt.get_sim_shower(0)


    print(f'Run of {runid}')
    for station in evt.get_stations():

        ss = station.get_sim_station()

        if not simulationSelector.run(evt, ss, det):
            continue
        efieldToVoltageConverter.run(evt, station, det)



        #Calculate Bandwidth
        if Vrms_per_channel == {}:
            for channel_id in station.get_channel_ids():

                print(f'chan ids {station.get_channel_ids()}')
                print(f'ss chan ids {ss.get_channel_ids()}')
                dt = 1 / station.get_channel(channel_id).get_sampling_rate()/units.GHz
                print(f'dt {dt}')
                dt = 1 / 1*units.GHz
                ff = np.linspace(0, 0.5/dt /units.GHz, 10000)
                filt = np.ones_like(ff, dtype=np.complex)
                if not add_amp:
                    if amp_type == '100' or amp_type == '200':
                        passband_low = 80
                        passband_high = 700
                    else:
                        passband_low = 50
                        passband_high = 500	#300s upper filter
                    noise_figure = 1.4
                    filt *= channelBandPassFilter.get_filter(ff, station.get_id(), channel_id, det, passband=[passband_low * units.MHz, passband_high * units.MHz], filter_type="butter", order=10, rp=0.1)
                    bandwidth = np.trapz(np.abs(filt)**2, ff)

                    noise_temp = 400 * units.kelvin
                    Vrms = (noise_temp * 50 * constants.k * bandwidth/units.Hz * noise_figure) **0.5
                    Vrms = 15 * units.micro*units.V
                    print(f'Vrms {Vrms/units.micro/units.V}microV for channel {channel_id}')
                else:
                    filt *= hardwareResponseIncorporator.get_filter(ff, station.get_id(), channel_id, det, sim_to_data=True)
                    bandwidth = np.trapz(np.abs(filt)**2, ff)
                    noise_temp = 350 * units.kelvin
                    noise_figure = 1.4
                    Vrms = (noise_temp * 50 * constants.k * bandwidth/units.Hz * noise_figure) **0.5
                    print(f'Vrms {Vrms/units.milli/units.V}mV for channel {channel_id} with amp')

                Vrms_per_channel[channel_id] = Vrms

            lpda_thresh_high = {key: value * lpda_sigma for key, value in Vrms_per_channel.items()}
            lpda_thresh_low = {key: value * -lpda_sigma for key, value in Vrms_per_channel.items()}
            dip_thresh_high = {key: value * dip_sigma for key, value in Vrms_per_channel.items()}
            dip_thresh_low = {key: value * -dip_sigma for key, value in Vrms_per_channel.items()}


        station.set_station_time(astropy.time.Time('2019-01-01T00:00:00'))
        eventTypeIdentifier.run(evt, station, 'forced', 'cosmic_ray')


#        channelLengthAdjuster.run(evt, station, det)			#Do I want to run this in the future?

        print(f'Vrms per channels are {Vrms_per_channel}')


        for channel in station.iter_channels():
            cTrace = channel.get_trace()
            cTrace *= 0
            channel.set_trace(cTrace)


        print(f'Set trace to zero')
        print(f'First test : Seeing if Pre-amp calculated noise Vrms creates post-amp calculated noise Vrms')


        channelGenericNoiseAdder.run(evt, station, det, min_freq=50*units.MHz, max_freq=passband_high*units.MHz, type='rayleigh', amplitude=Vrms_per_channel)



        if add_amp:
            hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)

        if noise:
            channelGenericNoiseAdder.run(evt, station, det, min_freq=50*units.MHz, max_freq=passband_high*units.MHz, type="rayleigh",
                                         amplitude=Vrms_per_channel)
        if not add_amp:
            channelBandPassFilter.run(evt, station, det,
                                  passband=[80 * units.MHz, passband_high * units.MHz], filter_type="rectangular", order=2)



        channelSignalReconstructor.run(evt, station, det)


        if CoREAS_mode == 'direct':
            # Trigger for direct LPDAs        
            highLowThreshold.run(evt, station, det,
                                        threshold_high = lpda_thresh_high,
                                        threshold_low = lpda_thresh_low,
                                        coinc_window = 32 * units.ns,
                                        triggered_channels=dir_LPDA_channels,
                                        number_concidences=lpda_coinc,
                                        trigger_name='dir_LPDA_2of4_3.5sigma')
            if station.has_triggered(trigger_name='dir_LPDA_2of4_3.5sigma'):
                lpda_dir_trig = True
                SNR = 0
                max_amps = []
                for channel in dir_LPDA_channels:
                    max_amp = station.get_channel(channel).get_parameter(chp.maximum_amplitude)
                    max_amps.append(max_amp)
                SNR = max(max_amps)
                SNR = SNR / Vrms_per_channel[dir_LPDA_channels[0]]
                output[runid]['lpda_dir_SNR'].append(SNR)

        # Trigger for direct dipoles
            highLowThreshold.run(evt, station, det,
                                        threshold_high = dip_thresh_high,
                                        threshold_low = dip_thresh_low,
                                        triggered_channels=dir_dipole_channels,
                                        number_concidences=1,
                                        trigger_name='dipole_2.0sigma_dir')
            if station.has_triggered(trigger_name='dipole_2.0sigma_dir'):
                max_amps = []
                for channel in dir_dipole_channels:
                    max_amp = station.get_channel(channel).get_parameter(chp.maximum_amplitude)
                    max_amps.append(max_amp)
                SNR = max(max_amps)
                SNR = SNR / Vrms_per_channel[dir_dipole_channels[0]]
                output[runid]['dip_dir_SNR'].append(SNR)
                dip_dir_trig = True
        else:
            # Trigger for reflected dipoles
            highLowThreshold.run(evt, station, det,
                                        threshold_high = dip_thresh_high,
                                        threshold_low = dip_thresh_low,
                                        triggered_channels=refl_dipole_channels,
                                        number_concidences=1,
                                        trigger_name='dipole_2.0sigma_refl')
            if station.has_triggered(trigger_name='dipole_2.0sigma_refl'):
                max_amps = []
                for channel in refl_dipole_channels:
                    max_amp = station.get_channel(channel).get_parameter(chp.maximum_amplitude)
                    max_amps.append(max_amp)
                SNR = max(max_amps)
                SNR = SNR / Vrms_per_channel[refl_dipole_channels[0]]
                output[runid]['dip_refl_SNR'].append(SNR)
                dip_refl_trig = True

            # Trigger for reflected LPDA
            highLowThreshold.run(evt, station, det,
                                        threshold_high = lpda_thresh_high,
                                        threshold_low = lpda_thresh_low,
                                        coinc_window = 32 * units.ns,
                                        triggered_channels=refl_LPDA_channels,
                                        number_concidences=2,
                                        trigger_name='LPDA_2of4_3.5sigma')
            if station.has_triggered(trigger_name='LPDA_2of4_3.5sigma'):
                max_amps = []
                for channel in refl_LPDA_channels:
                    max_amp = station.get_channel(channel).get_parameter(chp.maximum_amplitude)
                    max_amps.append(max_amp)
                SNR = max(max_amps)
                SNR = SNR / Vrms_per_channel[refl_LPDA_channels[0]]
                output[runid]['lpda_refl_SNR'].append(SNR)
                lpda_refl_trig = True



        for field in station.get_electric_fields_for_channels(channel_ids=[dir_LPDA_channels[0]]):
            output[runid]['ant_zen'].append(field.get_parameter(efp.zenith))
            print(f'LD zen {field.get_parameter(efp.zenith)}')

        station.remove_triggers()


    if saveEvent and station.has_triggered(trigger_name='LPDA_2of4_3.5sigma'):
        savedEvents += 1
        print(f'saving event {savedEvents}')
        eventWriter.run(evt, mode=mode)
        if savedEvents >= numToSave:
            quit()


    print(f'triggered DD {dip_dir_trig} DR {dip_refl_trig} LR {lpda_refl_trig} LD {lpda_dir_trig}')

    output[runid]['dip_dir_mask'].append(dip_dir_trig)
    output[runid]['dip_refl_mask'].append(dip_refl_trig)
    output[runid]['lpda_refl_mask'].append(lpda_refl_trig)
    output[runid]['lpda_dir_mask'].append(lpda_dir_trig)

    output[runid]['n_dip_dir'] += int(dip_dir_trig)
    output[runid]['n_dip_refl'] += int(dip_refl_trig)
    output[runid]['n_lpda_refl'] += int(lpda_refl_trig)
    output[runid]['n_lpda_dir'] += int(lpda_dir_trig)




if type == 'SP_old':
    config += f'_{lpda_coinc}of{len(dir_LPDA_channels)}'
if noise:
    config += '_wNoise'
if add_amp:
    config += f'_wAmp'
config += f'{amp_type}s'
if type == 'IceTop':
    config += f'_{icetop_sin}Sin_{num_icetop}PerBin'
config += '_StatPrep'


with open(f'output/CRFootprintRates/CoREAS_{CoREAS_mode}_{config}_Layer{depthLayer}m_{dB}dB_Area{spacing/units.km:.2f}_{cores}cores_id{min_file}_{max_file}.pkl', 'wb') as fout:
    pickle.dump(output, fout)

