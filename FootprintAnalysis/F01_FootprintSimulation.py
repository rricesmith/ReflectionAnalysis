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
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelLengthAdjuster
import NuRadioReco.modules.triggerTimeAdjuster
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

def pullFilesForSimulation(sim_type, min_file, max_file, num_icetop):
    """
    Creates a list of CoREAS footprints to use for this simulation based off of simulation configuration

    Parameters
    ----------
    sim_type : string
        MB: Moore's Bay CoREAS footprints
        SP: South Pole CoREAS footprints
        IceTop: IceCube CoREAS footprints, located in SP
        GL: Greenland CoREAS footprints
    min_file : float
        Minimum file number of footprints to use.
        SP goes 0-2100
        MB goes 0-4000
        IceTop is binned in energy, goes 16-18.4 log10eV
    max_file : float
        Maximum file number of footprints to use
    num_icetop : int
        IceTop has many footprint files per Energy-Zenith folder, this limits number used to value num_icetop

    Returns
    -------
    input_files : List
        List of file paths to individual CoREAS footprint files
    attenuation_model : string
        String of attenuation model to use for CoREAS simulation
        None for SP
        MB_freq for MB (frequency dependent attenuation in ice)
    """
    input_files = []
    i = min_file
    attenuation_model = None
    if type == 'SP':
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
        while i < max_file:
            file = f'../MBFootprints/00{i:04d}.hdf5'
            if os.path.exists(file):
                input_files.append(file)
            i += 1
    elif type == 'GL':
        if max_file == -1:
            max_file = 600
        while i < max_file:
            file = f'../../../../pub/arianna/SIM/greenland/output/hdf5/SIM{i:06d}.hdf5'
            if os.path.exists(file):
                input_files.append(file)
            i += 1

    else:
        print(f'Error, type {type} does not exist')
        quit()

    return input_files, attenuation_model

def getDetectorAndTriggersForSim(config, amp_type, depthLayer):
    """
    Pulls detector file and trigger types for simulation

    Parameters
    ----------
    config : string
        SP: Future South Pole configuration
        MB_future: Future Moore's Bay configuration
        MB_old: Old Moore's Bay configuration, 4 downward and 4 upward LPDAs
        SP_old: Old South Pole configuration, modeled after Station51
        Stn51:  Stn51 station configuration
        GL: Future Greenland station configuration
    amp_type: string
        100 or 200: Older amplifier types, used in MB_old
        300: Only used for SP_old
        future: Use Bandpass Filters to simulate amplifier, used for SP and MB_future
    depthLayer: float
        Depth of layer to simulate. Currently json detector files depend upon depth of reflective
        layer being simulater

    Returns
    -------
    det : GenericDetector object
    lpda_sigma : List
        List of pairs with sigma triggers and a name, ie [3.5, '3.5sigma']
    dip_sigma : Int
        Sigma trigger of dipole, only 1 tested currently but could be updated to be list as LPDA
    lpda_coinc : int
        Number of coincidences to use in any LPDA triggers. Only used as 3 for Stn51 3/3 tests, 2 otherwise
    """
    if config == 'MB_old' or config == 'MB_future':
        depthLayer = 576.0
    lpda_coinc = 2
    dip_sigma = 2
    PA_sigma = []
    if config == 'SP' or config == 'MB_future' or config == 'GL':
        if not amp_type == 'future':
            print(f'Future stations only use future, not {amp_type}')
            quit()
        det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_{config}_footprint{depthLayer:.0f}m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
    #    lpda_sigma = 3.5
    #    lpda_sigma = [[3.5, '3.5sigma'], [3.9498194908011524, '100Hz'], [4.919151494949084, '10mHz']]
        lpda_sigma = [[2, '2sigma'], [4, '4sigma'], [6, '6sigma'], [8, '8sigma'], [10, '10sigma']]
        PA_sigma = [[30.68, '100Hz'], [38.62, '1Hz'], [50.53, '1mHz']]
    elif config == 'MB_old':
        det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_{config}_{amp_type}s_footprint{depthLayer:.0f}m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
        lpda_sigma = [[4.4, '4.4sigma']]
    elif config == 'SP_old':
        if not amp_type == '300':
            print(f'SP_old only is 300s')
            quit()
        det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_{config}_footprint{depthLayer:.0f}m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
        lpda_sigma = [[5, '5sigma']]
        lpda_coinc = 3	#3/3 for Tingwei station 51 analysis
    elif config == 'Stn51':
        det = detector.Detector(json_filename=f'configurations/station51_InfAir.json', assume_inf=False, antenna_by_depth=False)
#        det = generic_detector.GenericDetector(json_filename=f'configurations/station51_InfAir.json', assume_inf=False, antenna_by_depth=False, default_station=51)
        lpda_sigma = [[5, '5sigma']]
        lpda_coinc = 2	#2/3 for Testing station 51 analysis
    elif config == 'TriggerTest':
        depthLayer = 576.0
        det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_MB_old_300s_footprint{depthLayer:.0f}m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)

    #    det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_MB_2of3_6m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
    #    det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_MB_2of3_13m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)

        det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_GL_2of3_6m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)

    #    det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_SP_old_footprint300m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
    #    det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_SP_footprint300m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
        lpda_sigma = [[3.5, '3.5sigma'], [3.9498194908011524, '100Hz'], [4.4, '4.4sigma'], [4.919151494949084, '10mHz']]
    elif config == 'PyrasTest':
        det = generic_detector.GenericDetector(json_filename=f'configurations/rnog_GL_2of3_6m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
        depthLayer = 576.0
        lpda_sigma = [[6, '6sigma']]
    elif config == 'BacklobeTest':
        depthLayer = 576.0
        det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_MB_BacklobeTest_{amp_type}s_footprint{depthLayer:.0f}m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
        lpda_sigma = [[4.4, '4.4sigma']]

    else:
        print(f'no config of {config} used, use SP, MB_old, or MB_future')
        quit()
    return det, lpda_sigma, dip_sigma, PA_sigma, lpda_coinc

def getAntennaChannels(config):
    """
    Returns lists of channels triggering together

    Parameters
    ----------
    config : string
        SP: Future South Pole configuration
        MB_future: Future Moore's Bay configuration
        MB_old: Old Moore's Bay configuration, 4 downward and 4 upward LPDAs
        SP_old: Old South Pole configuration, modeled after Station51
        Stn51:  Stn51 station configuration

    Returns
    -------
    dir_LPDA_channels : list of ints
        List of channels corresponding to upward facing LPDAs
    refl_LPDA_channels : list of ints
        List of channels corresponding to downward facing LPDAs
    dir_dip_channels : list of ints
        List of channels corresponding to direct dipole simulation
    refl_dip_channels : list of ints
        List of channels corresponding to reflected dipole simulation
    """

    dir_PA_channels = []
    refl_PA_channels = []
    if config == 'SP' or config == 'MB_future' or config == 'GL':
        dir_LPDA_channels = [0, 1, 2]
        refl_LPDA_channels = [3, 4, 5, 6]
        dir_dipole_channels = [7]
        refl_dipole_channels = [8]
        dir_PA_channels = [9, 10, 11, 12]
        refl_PA_channels = [13, 14, 15, 16]
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
        dir_LPDA_channels = [4, 5, 6]	#Station 51 upward channels on July 2018
    #    dir_LPDA_channels = [4, 5, 6, 8]	#Station 51 upward channels on July 2018 with extra test channel
        refl_LPDA_channels = [0]
        dir_dipole_channels = [0]
        refl_dipole_channels = [0]
    elif config == 'TriggerTest' or 'PyrasTest':
        dir_LPDA_channels = [0, 1, 2]
    #    dir_LPDA_channels = [0, 1, 2, 3]
        refl_LPDA_channels = [4, 5, 6, 7]
        dir_dipole_channels = [0]
        refl_dipole_channels = [0]
    elif config == 'BacklobeTest':
        dir_LPDA_channels = [0, 1, 2, 3]    #Direct are facing down, so as to test backlobe response
        refl_LPDA_channels = [4, 5, 6, 7]   #Reflected are also facing down, bot located 1152m down mirrored to be upwards
    else:
        print(f'No config {config} exists, quitting')
        quit()

    return dir_LPDA_channels, refl_LPDA_channels, dir_dipole_channels, refl_dipole_channels, dir_PA_channels, refl_PA_channels



def runCoREAS(CoREAS_mode, det, depthLayer, dB, attenuation_model):
    if CoREAS_mode=='direct':
        for evt, iE, x, y in readCoREAS.run(detector=det, output_mode=2):		#This mode is standard coreas
            yield evt, iE, x, y
    elif CoREAS_mode=='refracted':
        for evt, iE, x, y in readCoREAS.run(detector=det, ray_type='refracted', layer_depth=depthLayer, layer_dB=dB, force_dB=True, attenuation_model=attenuation_model, output_mode=2):
            yield evt, iE, x, y
    else:
        print(f'CoREAS mode {CoREAS_mode} not implemented')


def bandwidthAndVrmsConditions(amp_type):
    """
    Returns bandwidths used and temerature conditions

    Parameters
    ----------
    amp_type : string
        100, 200, or 300: Configure existing amplifiers
        future: Configure future detector, which used a bandpass filter
            From here: https://github.com/nu-radio/analysis-scripts/blob/gen2-tdr-2021/gen2-tdr-2021/detsim/D01detector_sim.py

    Returns
    -------
    noise_figure : float
        Assumed noise figure of the amplifier being used
    noise_temp : float
        Temperature of the amplifier being used in Kelvin
    passband : List of bandwidths
        Each bandwidth has structure [bandwidth low, bandwidth high, filter type, order]
    passband_dipole : List of bandwidths for dipole
        Same as passband, but for dipole channels if different
        Only used for future amp_type
    """


    noise_temp = 400 * units.kelvin
    noise_figure = 1.4
    passband = []   #Passband elements have structure [freq low, freq high, filter type, order]
    passband_dipole = []
    passband_PA = []

    if amp_type == '100':
        noise_figure = 1.4
        noise_temp = 400 * units.kelvin
        passband.append([100*units.MHz, 1000*units.MHz, 'butter', 2])
        passband_dipole = passband
    elif amp_type == '200':
        noise_figure = 1.4
        noise_temp = 400 * units.kelvin
        passband.append([50*units.MHz, 1000*units.MHz, 'butter', 2])
        passband_dipole = passband
    elif amp_type == 'future' or amp_type == 'TriggerTest':
        noise_figure = 1
        noise_temp = 350 * units.kelvin

        #2020 design
    #     passband.append([1*units.MHz, 250*units.MHz, 'butter', 10])
    #     passband.append([80*units.MHz, 500*units.GHz, 'butter', 5])

    #     passband_dipole.append([1*units.MHz, 250*units.MHz, 'cheby1', 9])
    #     passband_dipole.append([80*units.MHz, 500*units.GHz, 'cheby1', 4])

        #2021 design
        passband.append([1*units.MHz, 1000*units.MHz, 'butter', 10])    #passband low
    #    passband.append([1*units.MHz, 500*units.MHz, 'butter', 10])    #passband testing
        passband.append([1*units.MHz, 150*units.MHz, 'butter', 10])    #passband low_trigger
        passband.append([80*units.MHz, 800*units.GHz, 'butter', 5])     #passband high

        passband_dipole.append([1*units.MHz, 1000*units.MHz, 'cheby1', 7])
        passband_dipole.append([1*units.MHz, 220*units.MHz, 'cheby1', 7])
        passband_dipole.append([96*units.MHz, 100*units.GHz, 'cheby1', 4])

        passband_PA.append([1*units.MHz, 1000*units.MHz, 'cheby1', 7])
        passband_PA.append([1*units.MHz, 220*units.MHz, 'cheby1', 7])
        passband_PA.append([96*units.MHz, 100*units.GHz, 'cheby1', 4])

        if amp_type == 'TriggerTest':
    #        amp_type = 'future'
    #        passband[1] = [1*units.MHz, 150*units.MHz, 'butter', 10]
            passband[1] = [1*units.MHz, 250*units.MHz, 'butter', 10]
    #        passband[1] = [1*units.MHz, 300*units.MHz, 'butter', 10]
    #        passband[1] = [1*units.MHz, 500*units.MHz, 'butter', 10]
    #        passband = []
    #        passband.append([50*units.MHz, 500*units.GHz, 'butter', 4])
    #        passband.append([50*units.MHz, 500*units.GHz, 'butter', 2])

            noise_figure = 1.4
    #         noise_figure = 1.0
            noise_temp = 400 * units.kelvin
    #        noise_temp = 350 * units.kelvin
    elif amp_type == 'PyrasTest':
        noise_temp = 300 * units.kelvin
        noise_figure = 1

        coinc_window = 50 * units.ns

        passband.append([80*units.MHz, 180*units.MHz, 'butter', 10])
        sampling_rate = 3.2 * units.GHz
        

    else:
        #300s case
        noise_figure = 1.4
        noise_temp = 400 * units.kelvin
    #    passband.append([50*units.MHz, 500*units.GHz, 'butter', 2])
        passband_dipole = passband

    return noise_figure, noise_temp, passband, passband_dipole, passband_PA

#Parse arguments for simulation
parser = argparse.ArgumentParser(description='Run CR analysis for specific runids')
parser.add_argument('spacing', type=float, default=1., help='Station spacing')
parser.add_argument('cores', type=int, default=100, help='Number of cores to sim')
parser.add_argument('mode', type=str, default='direct', help='Mode to run CoREAS, direct or refracted')
parser.add_argument('--dB', type=float, default=40, help='dB of reflector for reflected air signals')
parser.add_argument('--depthLayer', type=float, default=300, help='Depth of reflector simulated')
parser.add_argument('--min_file', type=float, default=0, help='Min run to start on')
parser.add_argument('--max_file', type=float, default=-1, help='Max run to end on, default of -1 means goes through all') 
parser.add_argument('--type', type=str, default='SP', help='SP or MB footprints, default SP')
parser.add_argument('--config', type=str, default='SP', help='Config for detector/trigger. SP, SP_old, MB_old, MB_future are used')
parser.add_argument('--no_noise', default=True, action='store_false', help='Include noise or not, default True')
parser.add_argument('--add_amp', default=False, action='store_true', help='Pass flag to add amplifier')
parser.add_argument('--icetop_sin', type=float, default=-1, help='Lower edge of sin bin to use, -1 default is all')
parser.add_argument('--num_icetop', type=int, default=10, help='Number of IceTop files to use per energy/sin bin')
parser.add_argument('--amp_type', type=str, default='300', help='Amplifier type: 100, 200, or 300. Default 300')
parser.add_argument('--antenna', type=str, default='lpda', help='Whether to sim lpda or dipole')

args = parser.parse_args()

cores = args.cores
CoREAS_mode = args.mode
min_file = args.min_file
max_file = args.max_file
dB = args.dB
depthLayer = args.depthLayer
type = args.type
config = args.config
noise = args.no_noise
add_amp = args.add_amp
icetop_sin = args.icetop_sin
num_icetop = args.num_icetop
amp_type = args.amp_type
antenna = args.antenna

if not type == 'IceTop':
    min_file = int(min_file)
    max_file = int(max_file)


print(f'noise is {noise}')


#Pull in which files to use for simulation
input_files, attenuation_model = pullFilesForSimulation(type, min_file, max_file, num_icetop)


# Logging level
import logging
logger=logging.getLogger("module")
logger.setLevel(logging.WARNING)

spacing = args.spacing * units.km

#Get Detector and trigger sigmas
det, lpda_sigma, dip_sigma, PA_sigma, lpda_coinc = getDetectorAndTriggersForSim(config, amp_type, depthLayer)


#det.update(datetime.datetime(2018, 10, 1))
#det.update(datetime.datetime(2018, 1, 1))
det.update(astropy.time.Time('2018-1-1'))

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processung


#Using station grid custom code
readCoREAS = NuRadioReco.modules.io.coreas.readCoREASStationGrid.readCoREAS()
readCoREAS.begin(input_files, -spacing/2, spacing/2, -spacing/2, spacing/2, n_cores=cores, shape='radial', seed=None, log_level=logging.WARNING)


simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
simulationSelector.begin()

electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()

efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter(log_level=logging.WARNING)
efieldToVoltageConverter.begin(debug=False)

hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
if amp_type == 'PyrasTest':
    hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()

channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()

channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelSignalReconstructor.begin()

channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()

channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()

channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()

channelLengthAdjuster = NuRadioReco.modules.channelLengthAdjuster.channelLengthAdjuster()
channelLengthAdjuster.begin(number_of_samples = 5000, offset=1000)

triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
if CoREAS_mode == 'direct':
    pre = 'dir'
else:
    pre = 'refl'
triggerTimeAdjuster.begin(trigger_name=f'{pre}_LPDA_4.4sigma')
#triggerTimeAdjuster.begin(trigger_name='dir_LPDA_5sigma')

simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
phasedArrayTrigger = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()

####
#For Nur file generation, keep False if not testing it
####
saveEvent = True
numToSave = 10000
savedEvents = 0
output_filename = f'FootprintAnalysis/output/Backlobes_{config}_{amp_type}s_Refl_CRs_{numToSave}Evts_Noise_{noise}_Amp_{add_amp}_min{min_file}_max{max_file}.nur'
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin(output_filename)
#mode = {'Channels':True, 'ElectricFields':True, 'SimChannels':True, 'SimElectricFields':True}
mode = {'Channels':True, 'ElectricFields':False, 'SimChannels':False, 'SimElectricFields':False}
####

station_id = 1

#Get channels list for each trigger
dir_LPDA_channels, refl_LPDA_channels, dir_dipole_channels, refl_dipole_channels, dir_PA_channels, refl_PA_channels = getAntennaChannels(config)
#Set parameters for Phased array if needed
main_low_angle = np.deg2rad(-59.54968597864437)
main_high_angle = np.deg2rad(59.54968597864437)
phasing_angles_4ant = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))

#Configure temperatures for triggering
noise_figure, noise_temp, passband, passband_dipole, passband_PA = bandwidthAndVrmsConditions(amp_type)


Vrms_per_channel = {}
preAmpVrms_per_channel = {}
output = {}
# Loop over all events in file as initialized in readCoRREAS and perform analysis
for evt, iE, x, y in runCoREAS(CoREAS_mode, det, depthLayer, dB, attenuation_model):
    print(f'check : returned event. x {x} y {y}')
    runid = evt.get_run_number()
    sim_shower = evt.get_sim_shower(0)
    if(runid not in output):
        num_ran = 0

        xys = []
        SNRs = []

        output[runid] = {}
        output[runid]['n'] = 0
        output[runid]['energy'] = sim_shower[shp.energy]
        output[runid]['zenith'] = sim_shower[shp.zenith]
        output[runid]['azimuth'] = sim_shower[shp.azimuth]
        output[runid]['x'] = []
        output[runid]['y'] = []
        #Dipole flags
        output[runid]['n_dip_dir'] = 0
        output[runid]['n_dip_refl'] = 0
        output[runid]['dip_dir_mask'] = []
        output[runid]['dip_refl_mask'] = []
        output[runid]['dip_dir_SNR'] = []
        output[runid]['dip_refl_SNR'] = []
        #LPDA flags
        for sigma, name in lpda_sigma:
            output[runid][name] = {}
            output[runid][name]['n_lpda_refl'] = 0
            output[runid][name]['n_lpda_dir'] = 0
            output[runid][name]['lpda_refl_mask'] = []
            output[runid][name]['lpda_dir_mask'] = []
            output[runid][name]['lpda_dir_SNR'] = []
            output[runid][name]['lpda_refl_SNR'] = []
        for sigma, name in PA_sigma:
            if not name in output[runid]:
                output[runid][name] = {}
            output[runid][name]['n_PA_refl'] = 0
            output[runid][name]['n_PA_dir'] = 0
            output[runid][name]['PA_refl_mask'] = []
            output[runid][name]['PA_dir_mask'] = []
            output[runid][name]['PA_dir_SNR'] = []
            output[runid][name]['PA_refl_SNR'] = []

        output[runid]['ant_zen'] = []
        logger.warning(f"{runid} starting, cosmic ray with energy {output[runid]['energy']:.2g}eV zenith = {output[runid]['zenith']/units.deg:.0f}deg, azimuth = {output[runid]['azimuth']/units.deg:.0f}deg at location {x}m by {y}m")
        prevrunid = runid

    num_ran += 1

    output[runid]['n'] +=1
    output[runid]['x'].append(x)
    output[runid]['y'].append(y)

    dip_dir_trig = False
    dip_refl_trig = False
    lpda_refl_trig = {}
    lpda_dir_trig = {}
    PA_refl_trig = {}
    PA_dir_trig = {}
    for sigma, name in lpda_sigma:
        lpda_refl_trig[name] = False
        lpda_dir_trig[name] = False
    for sigma, name in PA_sigma:
        PA_refl_trig[name] = False
        PA_dir_trig[name] = False



    print(f'Run of {runid}')
    for station in evt.get_stations():
        station.set_station_time(datetime.datetime(2018, 10, 1))

        ss = station.get_sim_station()
#        if not simulationSelector.run(evt, ss, det):
#            continue
        eventTypeIdentifier.run(evt, station, mode='forced', forced_event_type='cosmic_ray')
        efieldToVoltageConverter.run(evt, station, det)
        #Phased array trigger can't handle different channel start times, don't run in refracted case
        if mode == 'direct':
            channelAddCableDelay.run(evt, station, det, mode='add')

        if saveEvent:
            channelResampler.run(evt, station, det, 2*units.GHz)


        #Calculate Vrms, only done once
        check_noise = False
        if Vrms_per_channel == {}:
            for channel_id in station.get_channel_ids():
                channel = station.get_channel(channel_id)
                if check_noise:
                    cTrace = channel.get_trace()
                    print(f'cTrace length {len(cTrace)}, sampling rate {channel.get_sampling_rate()}')
                    cTrace = np.zeros_like(cTrace)
                    channel.set_trace(cTrace, channel.get_sampling_rate())
                print(f'chan ids {station.get_channel_ids()}')
                print(f'ss chan ids {ss.get_channel_ids()}')
                sim_sampling_rate = channel.get_sampling_rate()

                if amp_type == 'PyrasTest':
                    sim_sampling_rate = 3.2 * units.GHz

                dt = 1 / sim_sampling_rate
                print(f'dt {dt} using sim sampling rate {sim_sampling_rate/units.GHz}GHz')
                ff = np.linspace(0, 0.5/dt /units.GHz, 10000)
                filt = np.ones_like(ff, dtype=complex)
                if not add_amp:
                    if not amp_type == 'future':
                        for pb in passband:
                            filt *= channelBandPassFilter.get_filter(ff, station.get_id(), channel_id, det, passband=[pb[0], pb[1]], filter_type=pb[2], order=pb[3], rp=0.1)
                    else:
                        if channel_id in dir_LPDA_channels or channel_id in refl_LPDA_channels:
                            for pb in passband:
                                filt *= channelBandPassFilter.get_filter(ff, station.get_id(), channel_id, det, passband=[pb[0], pb[1]], filter_type=pb[2], order=pb[3], rp=0.1)
#                            filt *= channelBandPassFilter.get_filter(ff, station.get_id(), channel_id, det, passband=[passband_low * units.MHz, 5 * units.GHz], filter_type=filter_type, order=order_high, rp=0.1)
                        elif channel_id in dir_dipole_channels or channel_id in refl_dipole_channels:
                            for pb in passband_dipole:
                                filt *= channelBandPassFilter.get_filter(ff, station.get_id(), channel_id, det, passband=[pb[0], pb[1]], filter_type=pb[2], order=pb[3], rp=0.1)
#                            filt *= channelBandPassFilter.get_filter(ff, station.get_id(), channel_id, det, passband=[passband_low * units.MHz, 800 * units.GHz], filter_type=dip_filter_type, order=dip_order_high, rp=0.1)
                        elif channel_id in dir_PA_channels or channel_id in refl_PA_channels:
                            for pb in passband_PA:
                                filt *= channelBandPassFilter.get_filter(ff, station.get_id(), channel_id, det, passband=[pb[0], pb[1]], filter_type=pb[2], order=pb[3], rp=0.1)

                    bandwidth = np.trapz(np.abs(filt)**2, ff)

                    Vrms = (noise_temp * 50 * constants.k * bandwidth/units.Hz * noise_figure) **0.5
                    preAmpVrms = Vrms
                    print(f'Vrms {Vrms/units.micro/units.V}microV for channel {channel_id}')
                else:
#                    if amp_type == '100' or amp_type == '200':
#                    if amp_type == '100' or amp_type == '200' or amp_type == 'PyrasTest':
                    if amp_type == 'PyrasTest':
                        for pb in passband:
                            filt *= channelBandPassFilter.get_filter(ff, station.get_id(), channel_id, det, passband=[pb[0], pb[1]], filter_type=pb[2], order=pb[3])
                    filt = hardwareResponseIncorporator.get_filter(ff, station.get_id(), channel_id, det, sim_to_data=True)
                    bandwidth = np.trapz(np.abs(filt)**2, ff)
                    Vrms = (noise_temp * 50 * constants.k * bandwidth/units.Hz * noise_figure) **0.5
                    print(f'Vrms {Vrms/units.milli/units.V}mV for channel {channel_id} with amp')
                    print(f'checking normalization method')
                    max_freq = 0.5 / dt
                    preAmpVrms = Vrms / (bandwidth / (max_freq))**0.5
                    print(f'PreAmp Vrms {preAmpVrms/units.micro/units.V}microV for channel {channel_id}')

                Vrms_per_channel[channel_id] = Vrms
                preAmpVrms_per_channel[channel_id] = preAmpVrms

            lpda_thresh_high = {key: value for key, value in Vrms_per_channel.items()}
            lpda_thresh_low = {key: value * -1 for key, value in Vrms_per_channel.items()}
            dip_thresh_high = {key: value for key, value in Vrms_per_channel.items()}
            dip_thresh_low = {key: value * -1 for key, value in Vrms_per_channel.items()}
            PA_thresh_high = {key: value for key, value in Vrms_per_channel.items()}
            PA_thresh_low = {key: value * -1 for key, value in Vrms_per_channel.items()}

        eventTypeIdentifier.run(evt, station, 'forced', 'cosmic_ray')
        #Cut each channel to be shorter to reduce amount of noise added and likelyhood of noise triggers. Do for each set of channels separately
        #Had to edit channelLengthAdjustor for this to work, as is it cuts all channels even if you only indicate some
#        channelLengthAdjuster.run(evt, station, channel_ids=refl_LPDA_channels)
#        channelLengthAdjuster.run(evt, station, channel_ids=dir_LPDA_channels)
#        channelLengthAdjuster.run(evt, station, channel_ids=dir_dipole_channels)
#        channelLengthAdjuster.run(evt, station, channel_ids=refl_dipole_channels)
#        if dir_PA_channels:
#            channelLengthAdjuster.run(evt, station, channel_ids=dir_PA_channels)
#            channelLengthAdjuster.run(evt, station, channel_ids=refl_PA_channels)

        #Because out Dipole is a stand in for a Phased Array, want to multiply signal by 2 relative to noise
        #To do this we multiply the channel trace by a factor of two, then add noise as normal for 1sigma Vrms
        for ch_id in refl_dipole_channels:
            channel = station.get_channel(ch_id)
            trace = channel.get_trace()
            channel.set_trace(trace * 2, channel.get_sampling_rate())
        for ch_id in dir_dipole_channels:
            channel = station.get_channel(ch_id)
            trace = channel.get_trace()
            channel.set_trace(trace * 2, channel.get_sampling_rate())

        if noise and not saveEvent:
            if not amp_type == 'PyrasTest':
                channelGenericNoiseAdder.run(evt, station, det, min_freq=0*units.MHz, max_freq=0.5*sim_sampling_rate, type="rayleigh",
                                         amplitude=preAmpVrms_per_channel)
            else:
                channelGenericNoiseAdder.run(evt, station, det, min_freq=0*units.MHz, max_freq=800*units.MHz, type="rayleigh",
                                         amplitude=preAmpVrms_per_channel)

        if check_noise:
            for channel_id in station.get_channel_ids():
                channel = station.get_channel(channel_id)
                cTrace = channel.get_trace()
                noise_rms = np.sqrt(np.mean(np.square(cTrace)))
                print(f'ch {channel_id} has noise rms {noise_rms} pre-amp')

        if add_amp:
            #There was filtering before amp response in the 100/200s amps to prevent blowup of amp. Not in 300 or future amps
            """
            if amp_type == '100' or amp_type == '200':
                for pb in passband:
                    channelBandPassFilter.run(evt, station, det, passband=[pb[0], pb[1]], filter_type=pb[2], order=pb[3])
            """
            hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)

        if check_noise:
            for channel_id in station.get_channel_ids():
                channel = station.get_channel(channel_id)
                cTrace = channel.get_trace()
                noise_rms = np.sqrt(np.mean(np.square(cTrace)))
                print(f'ch {channel_id} has noise rms {noise_rms} post-amp')
            quit()


        if not add_amp:
            if not amp_type == 'future':
                for pb in passband:
                    channelBandPassFilter.run(evt, station, det,
                                  passband=[pb[0], pb[1]], filter_type=pb[2], order=pb[3], rp=0.1)
            else:
                if antenna == 'lpda':
                    for pb in passband:
                        channelBandPassFilter.run(evt, station, det,
                                      passband=[pb[0], pb[1]], filter_type=pb[2], order=pb[3], rp=0.1)
                elif antenna == 'dipole':
                    for pb in passband_dipole:
                        channelBandPassFilter.run(evt, station, det,
                                      passband=[pb[0], pb[1]], filter_type=pb[2], order=pb[3], rp=0.1)
                elif antenna == 'PA':
                    for pb in passband_PA:
                        channelBandPassFilter.run(evt, station, det,
                                      passband=[pb[0], pb[1]], filter_type=pb[2], order=pb[3], rp=0.1)



        channelSignalReconstructor.run(evt, station, det)


        print(f'coreas mode {CoREAS_mode}')
        if CoREAS_mode == 'direct':
            # Trigger for direct LPDAs   
            for sigma, name in lpda_sigma:
                highLowThreshold.run(evt, station, det,
                                            threshold_high = {key: value * sigma for key, value in lpda_thresh_high.items()},
                                            threshold_low = {key: value * sigma for key, value in lpda_thresh_low.items()},
                                            coinc_window = 40 * units.ns,
                                            triggered_channels=dir_LPDA_channels,
                                            number_concidences=lpda_coinc,
                                            trigger_name='dir_LPDA_'+name)
                if station.has_triggered(trigger_name='dir_LPDA_'+name):
                    SNR = 0
                    max_amps = []
                    min_amps = []
                    for channel in dir_LPDA_channels:
                        ch_snr = station.get_channel(channel).get_parameter(chp.SNR)
                        max_amp = ch_snr['peak_2_peak_amplitude']
                        max_amps.append(max_amp)
                    SNR = max(max_amps)
#                    SNR = SNR / Vrms_per_channel[dir_LPDA_channels[0]]
                    output[runid][name]['lpda_dir_SNR'].append(SNR)
                    lpda_dir_trig[name] = True
#                output[runid][name]['lpda_dir_mask'].append(station.has_triggered(trigger_name='dir_LPDA_'+name))
                output[runid][name]['n_lpda_dir'] += int(station.has_triggered(trigger_name='dir_LPDA_'+name))


        # Trigger for direct dipoles
            highLowThreshold.run(evt, station, det,
                                        threshold_high = {key: value * dip_sigma for key, value in dip_thresh_high.items()},
                                        threshold_low = {key: value * dip_sigma for key, value in dip_thresh_low.items()},
                                        triggered_channels=dir_dipole_channels,
                                        number_concidences=1,
                                        trigger_name='dipole_2.0sigma_dir')
            if station.has_triggered(trigger_name='dipole_2.0sigma_dir'):
                max_amps = []
                for channel in dir_dipole_channels:
                    ch_snr = station.get_channel(channel).get_parameter(chp.SNR)
                    max_amp = ch_snr['peak_2_peak_amplitude']
                    max_amps.append(max_amp)
                SNR = max(max_amps)
#                SNR = SNR / Vrms_per_channel[dir_dipole_channels[0]]
                output[runid]['dip_dir_SNR'].append(SNR)
                dip_dir_trig = True
#            output[runid]['dip_dir_mask'].append(station.has_triggered(trigger_name='dipole_2.0sigma_dir'))
            output[runid]['n_dip_dir'] += int(station.has_triggered(trigger_name='dipole_2.0sigma_dir'))


            if config == 'Stn51':
                #For Stn51 sim, use reflected when doing direct for a 3/3 5sigma trigger
                simpleThreshold.run(evt, station, det,
                                            threshold = {key: value * 5 for key, value in lpda_thresh_high.items()},
                                            coinc_window = 40 * units.ns,
                                            triggered_channels=dir_LPDA_channels,
                                            number_concidences=3,
                                            trigger_name='LPDA_3of3_5sigma')
                if station.has_triggered(trigger_name='LPDA_3of3_5sigma'):
                    SNR = 0
                    max_amps = []
                    for channel in dir_LPDA_channels:
                        ch_snr = station.get_channel(channel).get_parameter(chp.SNR)
                        max_amp = ch_snr['peak_2_peak_amplitude']
                        max_amps.append(max_amp)
                    SNR = max(max_amps)
#                    SNR = SNR / Vrms_per_channel[dir_LPDA_channels[0]]
                    output[runid][name]['lpda_refl_SNR'].append(SNR)
                    lpda_refl_trig[name] = True
#                output[runid][name]['lpda_refl_mask'].append(station.has_triggered(trigger_name='LPDA_2of4_3.5sigma'))
                output[runid][name]['n_lpda_refl'] += int(station.has_triggered(trigger_name='LPDA_3of3_5sigma'))



        else:
            #Trigger for direct PA
#            sampling_rate_PA = 2.4 * units.GHz  #PA sampling rate
            sampling_rate_PA = 10 * units.GHz  #Use sim sampling rate
            window_4ant = int(16 * units.ns * sampling_rate_PA * 2.0)
            step_4ant = int(8 * units.ns * sampling_rate_PA * 2.0)
            for sigma, name in PA_sigma:
                Vrms_PA = Vrms_per_channel[dir_PA_channels[0]]
                phasedArrayTrigger.run(evt, station, det,
                                   Vrms=Vrms_PA,
                                   threshold= sigma * np.power(Vrms_PA, 2.0),  # see phased trigger module for explanation
                                   triggered_channels=dir_PA_channels,
                                   phasing_angles=phasing_angles_4ant,
                                   ref_index=1.75,
                                   trigger_name=f'dir_PA_4channel_{name}',  # the name of the trigger
                                   trigger_adc=True,  # Don't have a seperate ADC for the trigger
                                   adc_output=f'voltage',  # output in volts
                                   trigger_filter=None,
                                   upsampling_factor=4,
                                   window=window_4ant,
                                   step=step_4ant)
                if station.has_triggered(trigger_name='dir_PA_4channel_'+name):
                    SNR = 0
                    max_amps = []
                    min_amps = []
                    for channel in dir_PA_channels:
                        ch_snr = station.get_channel(channel).get_parameter(chp.SNR)
                        max_amp = ch_snr['peak_2_peak_amplitude']
                        max_amps.append(max_amp)
                    SNR = max(max_amps)
#                    SNR = SNR / Vrms_per_channel[dir_LPDA_channels[0]]
                    output[runid][name]['PA_dir_SNR'].append(SNR)
                    PA_dir_trig[name] = True
#                output[runid][name]['lpda_dir_mask'].append(station.has_triggered(trigger_name='dir_LPDA_'+name))
                output[runid][name]['n_PA_dir'] += int(station.has_triggered(trigger_name='dir_PA_4channel_'+name))

        #Trigger for refl PA
#            sampling_rate_PA = 2.4 * units.GHz  #PA sampling rate
            sampling_rate_PA = 10 * units.GHz  #Use sim sampling rate
            window_4ant = int(16 * units.ns * sampling_rate_PA * 2.0)
            step_4ant = int(8 * units.ns * sampling_rate_PA * 2.0)
            for sigma, name in PA_sigma:
                Vrms_PA = Vrms_per_channel[refl_PA_channels[0]]
                phasedArrayTrigger.run(evt, station, det,
                                   Vrms=Vrms_PA,
                                   threshold= sigma * np.power(Vrms_PA, 2.0),  # see phased trigger module for explanation
                                   triggered_channels=refl_PA_channels,
                                   phasing_angles=phasing_angles_4ant,
                                   ref_index=1.75,
                                   trigger_name=f'refl_PA_4channel_{name}',  # the name of the trigger
                                   trigger_adc=True,  # Don't have a seperate ADC for the trigger
                                   adc_output=f'voltage',  # output in volts
                                   trigger_filter=None,
                                   upsampling_factor=4,
                                   window=window_4ant,
                                   step=step_4ant)
                if station.has_triggered(trigger_name='refl_PA_4channel_'+name):
                    SNR = 0
                    max_amps = []
                    min_amps = []
                    for channel in dir_PA_channels:
                        ch_snr = station.get_channel(channel).get_parameter(chp.SNR)
                        max_amp = ch_snr['peak_2_peak_amplitude']
                        max_amps.append(max_amp)
                    SNR = max(max_amps)
#                    SNR = SNR / Vrms_per_channel[dir_LPDA_channels[0]]
                    output[runid][name]['PA_refl_SNR'].append(SNR)
                    PA_refl_trig[name] = True
#                output[runid][name]['lpda_dir_mask'].append(station.has_triggered(trigger_name='dir_LPDA_'+name))
                output[runid][name]['n_PA_refl'] += int(station.has_triggered(trigger_name='refl_PA_4channel_'+name))


            # Trigger for direct dipoles
            highLowThreshold.run(evt, station, det,
                                        threshold_high = {key: value * dip_sigma for key, value in dip_thresh_high.items()},
                                        threshold_low = {key: value * dip_sigma for key, value in dip_thresh_low.items()},
                                        triggered_channels=dir_dipole_channels,
                                        number_concidences=1,
                                        trigger_name='dipole_2.0sigma_dir')
            if station.has_triggered(trigger_name='dipole_2.0sigma_dir'):
                max_amps = []
                for channel in dir_dipole_channels:
                    ch_snr = station.get_channel(channel).get_parameter(chp.SNR)
                    max_amp = ch_snr['peak_2_peak_amplitude']
                    max_amps.append(max_amp)
                SNR = max(max_amps)
#                SNR = SNR / Vrms_per_channel[dir_dipole_channels[0]]
                output[runid]['dip_dir_SNR'].append(SNR)
                dip_dir_trig = True
#            output[runid]['dip_dir_mask'].append(station.has_triggered(trigger_name='dipole_2.0sigma_dir'))
            output[runid]['n_dip_dir'] += int(station.has_triggered(trigger_name='dipole_2.0sigma_dir'))

            # Trigger for reflected dipoles
            highLowThreshold.run(evt, station, det,
                                        threshold_high = {key: value * dip_sigma for key, value in dip_thresh_high.items()},
                                        threshold_low = {key: value * dip_sigma for key, value in dip_thresh_low.items()},
                                        triggered_channels=refl_dipole_channels,
                                        number_concidences=1,
                                        trigger_name='dipole_2.0sigma_refl')
            if station.has_triggered(trigger_name='dipole_2.0sigma_refl'):
                max_amps = []
                for channel in refl_dipole_channels:
                    ch_snr = station.get_channel(channel).get_parameter(chp.SNR)
                    max_amp = ch_snr['peak_2_peak_amplitude']
                    max_amps.append(max_amp)
                SNR = max(max_amps)
#                SNR = SNR / Vrms_per_channel[refl_dipole_channels[0]]
                output[runid]['dip_refl_SNR'].append(SNR)
                dip_refl_trig = True
#            output[runid]['dip_refl_mask'].append(station.has_triggered(trigger_name='dipole_2.0sigma_refl'))
            output[runid]['n_dip_refl'] += int(station.has_triggered(trigger_name='dipole_2.0sigma_refl'))

            # Trigger for reflected LPDA
            for sigma, name in lpda_sigma:
                highLowThreshold.run(evt, station, det,
                                            threshold_high = {key: value * sigma for key, value in lpda_thresh_high.items()},
                                            threshold_low = {key: value * sigma for key, value in lpda_thresh_low.items()},
                                            coinc_window = 40 * units.ns,
                                            triggered_channels=refl_LPDA_channels,
                                            number_concidences=2,
                                            trigger_name='refl_LPDA_'+name)
                if station.has_triggered(trigger_name='refl_LPDA_'+name):
                    max_amps = []
                    for channel in refl_LPDA_channels:
                        ch_snr = station.get_channel(channel).get_parameter(chp.SNR)
                        max_amp = ch_snr['peak_2_peak_amplitude']
                        max_amps.append(max_amp)
                    SNR = max(max_amps)
#                    SNR = SNR / Vrms_per_channel[refl_LPDA_channels[0]]
                    output[runid][name]['lpda_refl_SNR'].append(SNR)
                    lpda_refl_trig[name] = True
#                output[runid][name]['lpda_refl_mask'].append(station.has_triggered(trigger_name='refl_LPDA_'+name))
                output[runid][name]['n_lpda_refl'] += int(station.has_triggered(trigger_name='refl_LPDA_'+name))


        for field in station.get_electric_fields_for_channels(channel_ids=[dir_LPDA_channels[0]]):
            output[runid]['ant_zen'].append(field.get_parameter(efp.zenith))
            print(f'LD zen {field.get_parameter(efp.zenith)}')


        for sigma, name in lpda_sigma:
            if CoREAS_mode == 'direct':
                pre = 'dir'
            else:
                pre = 'refl'
            trigger_name = f'{pre}_LPDA_{name}'
            print(f'checking is trigger of {trigger_name}')
#            if saveEvent and station.has_triggered(trigger_name='refl_LPDA_'+name):
            if saveEvent and station.has_triggered(trigger_name=trigger_name):
                print(f'triggered')

                if 0:
                    trigger = station.get_trigger(trigger_name)
                    chs = refl_LPDA_channels
                    traces = {}
                    times = {}
                    for ch in chs:
                        traces[ch] = []
                        times[ch] = []
                        ch0trace = station.get_channel(ch).get_trace()
                        ch0times = station.get_channel(ch).get_times()

                        traces[ch].append(ch0trace)
                        times[ch].append(ch0times)
                    if noise:
                        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=False)
                        channelGenericNoiseAdder.run(evt, station, det, min_freq=0*units.MHz, max_freq=0.5*sim_sampling_rate, type="rayleigh",
                                                amplitude=preAmpVrms_per_channel)
                        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)
#                        channelGenericNoiseAdder.run(evt, station, det, min_freq=0*units.MHz, max_freq=0.5*sim_sampling_rate, type="rayleigh",
#                                                amplitude=Vrms_per_channel)
                        #Removing the cable delay that is added one way, but not removed in the other?
                        for channel in station.iter_channels():
                            cable_delay = det.get_cable_delay(station.get_id(), channel.get_id())
                            channel.add_trace_start_time(cable_delay)

                    for ch in chs:
                        ch0trace = station.get_channel(ch).get_trace()
                        ch0times = station.get_channel(ch).get_times()

                        traces[ch].append(ch0trace)
                        times[ch].append(ch0times)

                    triggerTimeAdjuster.run(evt, station, det)
                    for ch in chs:
                        ch0trace = station.get_channel(ch).get_trace()
                        ch0times = station.get_channel(ch).get_times()

                        traces[ch].append(ch0trace)
                        times[ch].append(ch0times)
                    channelResampler.run(evt, station, det, 2*units.GHz)
                    for ch in chs:
                        ch0trace = station.get_channel(ch).get_trace()
                        ch0times = station.get_channel(ch).get_times()

                        traces[ch].append(ch0trace)
                        times[ch].append(ch0times)

                    for ch in chs:
                        fig, axs = plt.subplots(nrows=4, ncols=1, sharex=False)
                        axs[0].plot(times[ch][0], traces[ch][0], label='pre-noise')
                        axs[0].plot(times[ch][0], np.ones(len(times[ch][0]))*4.4*lpda_thresh_high[0]/units.V, color='red', linestyle='--')
                        axs[0].plot(times[ch][0], np.ones(len(times[ch][0]))*4.4*lpda_thresh_low[0]/units.V, color='red', linestyle='--')
                        axs[0].axvline(x=times[ch][3][0], color='black', linestyle='--')
                        axs[0].axvline(x=times[ch][3][-1], color='black', linestyle='--')
                        axs[0].axvline(x=trigger.get_trigger_time(), color='green')

                        axs[1].plot(times[ch][0], traces[ch][0], label='pre-noise')
                        axs[1].plot(times[ch][0], np.ones(len(times[ch][0]))*4.4*lpda_thresh_high[0]/units.V, color='red', linestyle='--')
                        axs[1].plot(times[ch][0], np.ones(len(times[ch][0]))*4.4*lpda_thresh_low[0]/units.V, color='red', linestyle='--')
                        axs[1].axvline(x=times[ch][3][0], color='black', linestyle='--')
                        axs[1].axvline(x=times[ch][3][-1], color='black', linestyle='--')
                        axs[1].axvline(x=trigger.get_trigger_time(), color='green')
                        axs[1].set_xlim((times[ch][3][0]-100, times[ch][3][-1]+100))


                        axs[2].plot(times[ch][1], traces[ch][1], label='with noise')
                        axs[2].plot(times[ch][1], np.ones(len(times[ch][1]))*4.4*lpda_thresh_high[0]/units.V, color='red', linestyle='--')
                        axs[2].plot(times[ch][1], np.ones(len(times[ch][1]))*4.4*lpda_thresh_low[0]/units.V, color='red', linestyle='--')
                        axs[2].axvline(x=times[ch][3][0], color='black', linestyle='--')
                        axs[2].axvline(x=times[ch][3][-1], color='black', linestyle='--')
                        axs[2].axvline(x=trigger.get_trigger_time(), color='green')
                        axs[2].set_xlim((times[ch][3][0]-100, times[ch][3][-1]+100))

                        axs[3].plot(times[ch][2], traces[ch][2], label='resampled')
                        axs[3].plot(times[ch][2], np.ones(len(times[ch][2]))*4.4*lpda_thresh_high[0]/units.V, color='red', linestyle='--')
                        axs[3].plot(times[ch][2], np.ones(len(times[ch][2]))*4.4*lpda_thresh_low[0]/units.V, color='red', linestyle='--')
                        axs[3].axvline(x=times[ch][3][0], color='black', linestyle='--')
                        axs[3].axvline(x=times[ch][3][-1], color='black', linestyle='--')
                        axs[3].axvline(x=trigger.get_trigger_time(), color='green')

                        plt.savefig(f'testTrace{ch}.png')
                    quit()

                if noise:
                    hardwareResponseIncorporator.run(evt, station, det, sim_to_data=False)
                    channelGenericNoiseAdder.run(evt, station, det, min_freq=0*units.MHz, max_freq=0.5*sim_sampling_rate, type="rayleigh",
                                            amplitude=preAmpVrms_per_channel)
                    hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)
                    #Removing the cable delay that is added one way, but not removed the other. Needs correction below
                    for channel in station.iter_channels():
                        cable_delay = det.get_cable_delay(station.get_id(), channel.get_id())
                        channel.add_trace_start_time(cable_delay)
                triggerTimeAdjuster.run(evt, station, det)
                channelResampler.run(evt, station, det, 2*units.GHz)

                savedEvents += 1
                print(f'saving event {savedEvents}')
                eventWriter.run(evt, mode=mode)
                if savedEvents >= numToSave:
                    quit()

        station.remove_triggers()


    for sigma, name in lpda_sigma:
        output[runid][name]['lpda_dir_mask'].append(lpda_dir_trig[name])
        output[runid][name]['lpda_refl_mask'].append(lpda_refl_trig[name])
    output[runid]['dip_dir_mask'].append(dip_dir_trig)
    output[runid]['dip_refl_mask'].append(dip_refl_trig)
    for sigma, name in PA_sigma:
        output[runid][name]['PA_dir_mask'].append(PA_dir_trig[name])
        output[runid][name]['PA_refl_mask'].append(PA_refl_trig[name])
#    for sigma, name in lpda_sigma:
#        output[runid][name]['n_lpda_refl'] += int(lpda_refl_trig)
#        output[runid][name]['n_lpda_dir'] += int(lpda_dir_trig)
#    output[runid]['n_dip_dir'] += int(dip_dir_trig)
#    output[runid]['n_dip_refl'] += int(dip_refl_trig)




if type == 'SP_old':
    config += f'_{lpda_coinc}of{len(dir_LPDA_channels)}'
if noise:
    config += '_wNoise'
if add_amp:
    config += f'_wAmp'
config += f'{amp_type}s'
if type == 'IceTop':
#    config += f'_InfAir'		#for 5k sim, no InfAir has no add. InfAir has _REALInfAir added
#    config += f'_REALInfAir'		#for 10k sim, accidentally used InfAir for non-inf air, then this add for InfAir

    config += f'_{icetop_sin}Sin_{num_icetop}PerBin'

#config += '_StatPrep'
#config += f'_{noise_figure:.1f}NF_{noise_temp:.0f}KTemp'
#config += '_10GHztrig_noise_calc'
#config += '_Refract'
#config += f'_{antenna}'
#config += f'_500MHzLowpassTest'
#config += f'_RateCheck300'
#config += f'_TotalChange'
#config += f'_ThreshChange'
#config += f'_No300sBWTest'
#config += '_Temp350K'
#config += '_1.0NF'
#config += '_150MHzLowpass'
#config += '_250MHzLowpass'
#config += '_300MHzLowpass'
#config += '_500MHzLowpass'
#config += '_2of3Trigger6m'
#config += '_2of3Trigger13m'
#config += '_SPTest'
config += '_ProposalFlux'

with open(f'output/CRFootprintRates/CoREAS_{CoREAS_mode}_{config}_Layer{depthLayer}m_{dB}dB_Area{spacing/units.km:.2f}_{cores}cores_id{min_file}_{max_file}.pkl', 'wb') as fout:
    pickle.dump(output, fout)

