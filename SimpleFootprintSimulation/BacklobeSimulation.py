from NuRadioReco.utilities import units
import NuRadioReco.modules.io.coreas.readCoREAS
import NuRadioReco.modules.io.coreas.simulationSelector
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.electricFieldBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelStopFilter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelLengthAdjuster
import NuRadioReco.modules.triggerTimeAdjuster
import astropy
import argparse
import NuRadioReco.modules.io.eventWriter
import numpy as np
import os
import datetime
from icecream import ic
from scipy import constants

from NuRadioReco.detector import detector
from NuRadioReco.detector import generic_detector

import logging
logger=logging.getLogger("module")
logger.setLevel(logging.WARNING)


def calculateNoisePerChannel(det, station, amp=True):
    """
    Calculate the noise per channel
        det: detector object
        station: station object
        passband: passband settings for trigger
        amp: boolean for whether to include amplifier
    """
    # Setup conditions for amplifier. Assuming a 200s amplifier
    noise_figure = 1.4
    noise_temp = 400 * units.kelvin
    passband = [50*units.MHz, 1000*units.MHz, 'butter', 2]

    preAmpVrms_per_channel = {}
    postAmpVrms_per_channel = {}
    for channel_id in station.get_channel_ids():
        channel = station.get_channel(channel_id)
        sim_sampling_rate = channel.get_sampling_rate()

        dt = 1 / sim_sampling_rate
        ff = np.linspace(0, 0.5/dt /units.GHz, 10000)
        filt = np.ones_like(ff, dtype=complex)

        if amp:
            filt = hardwareResponseIncorporator.get_filter(ff, station.get_id(), channel_id, det, sim_to_data=True)
        else:
            filt *= channelBandPassFilter.get_filter(ff, station.get_id(), channel_id, det, passband=[passband[0], passband[1]], filter_type=passband[2], order=passband[3], rp=0.1)

        bandwidth = np.trapz(np.abs(filt)**2, ff)
        Vrms = (noise_temp * 50 * constants.k * bandwidth/units.Hz * noise_figure) **0.5

        if amp:
            max_freq = 0.5 / dt
            preAmpVrms_per_channel[channel_id] = Vrms / (bandwidth / max_freq)**0.5
            postAmpVrms_per_channel[channel_id] = Vrms
        else:
            preAmpVrms_per_channel[channel_id] = Vrms
            postAmpVrms_per_channel[channel_id] = Vrms

    return preAmpVrms_per_channel, postAmpVrms_per_channel


def pullFilesForSimulation(sim_type, min_file=0, max_file=-1):
    """
    Pull in IceTop files for simulation
    IceTop simulations range from 16.0-18.5 log10eV
    Sin^2(zenith) bins range from 0.0-1.0
    There are ~33 separate footprints per Energy/Sin^2 bin
    """

    i = min_file
    input_files = []
    if sim_type == 'SP':
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
    elif sim_type == 'MB':
        if max_file == -1:
            max_file = 3999
        while i < max_file:
            file = f'../MBFootprints/00{i:04d}.hdf5'
            if os.path.exists(file):
                input_files.append(file)
            i += 1
    elif sim_type == 'GL':
        if max_file == -1:
            max_file = 600
        while i < max_file:
            file = f'../../../../pub/arianna/SIM/greenland/output/hdf5/SIM{i:06d}.hdf5'
            if os.path.exists(file):
                input_files.append(file)
            i += 1

    return input_files

# Read in settings for simulation
parser = argparse.ArgumentParser(description='Run Cosmic Ray simulation for Station 51')
parser.add_argument('output_filename', type=str, help='Output filename for simulation')
parser.add_argument('n_cores', type=int, help='Number of cores to use for simulation')
parser.add_argument('--loc', type=str, default='MB', help='Location, MB for Moores Bay, SP for SouthPole, GL for Greenland')
parser.add_argument('--min_file', type=int, default=0, help='Minimum file number to use')
parser.add_argument('--max_file', type=int, default=-1, help='Maximum file number to use, -1 means use all files')
parser.add_argument('--sim_amp', type=bool, default=True, help='Include amplifier in simulation')
parser.add_argument('--amp_type', type=str, default='200', help='Amplifier type')
parser.add_argument('--add_noise', default=False, help='Include noise in simulation')

args = parser.parse_args()
output_filename = args.output_filename
n_cores = args.n_cores
loc = args.loc
min_file = args.min_file
max_file = args.max_file
sim_amp = args.sim_amp
amp_type = args.amp_type
add_noise = args.add_noise

# Get files for simulation
input_files = pullFilesForSimulation(loc, min_file, max_file)

# Setup detector
det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_{loc}_BacklobeTest_{amp_type}s_footprint576m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
det.update(astropy.time.Time('2018-1-1'))
station_id = 1

# Indicate channels for simulation
direct_LPDA_channels = [0, 1, 2, 3]    #Direct are facing down, so as to test backlobe response

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processung
readCoREAS = NuRadioReco.modules.io.coreas.readCoREAS.readCoREAS()
readCoREAS.begin(input_files, station_id, n_cores=n_cores, max_distance=1*units.km)

simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
simulationSelector.begin()

efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)

hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()

channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()

eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()

channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()

triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
triggerTimeAdjuster.begin(trigger_name=f'direct_LPDA_2of4_3.5sigma')


eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin(output_filename)

preAmpVrms_per_channel = {}

# Start simulation
for iE, evt in enumerate(readCoREAS.run(detector=det)):
    logger.info("processing event {:d} with id {:d}".format(iE, evt.get_id()))

    # for station in evt.get_stations():

    station = evt.get_station(station_id)
    station.set_station_time(datetime.datetime(2018, 10, 1))
    det.update(station.get_station_time())

    eventTypeIdentifier.run(evt, station, mode='forced', forced_event_type='cosmic_ray')
    # channelAddCableDelay.run(evt, station, det, mode='add')   # Not sure if necessary
    efieldToVoltageConverter.run(evt, station, det)
    channelResampler.run(evt, station, det, 2*units.GHz)


    if preAmpVrms_per_channel == {}:
        # Get noise levels for simulation
        preAmpVrms_per_channel, postAmpVrms_per_channel = calculateNoisePerChannel(det, station=station, amp=sim_amp)
        ic(preAmpVrms_per_channel, postAmpVrms_per_channel)
        if sim_amp:
            threshold_high_3_5 = {key: value * 3.5 for key, value in postAmpVrms_per_channel.items()}
            threshold_low_3_5 = {key: value * -3.5 for key, value in postAmpVrms_per_channel.items()}
            threshold_high_4_4 = {key: value * 4.4 for key, value in postAmpVrms_per_channel.items()}
            threshold_low_4_4 = {key: value * -4.4 for key, value in postAmpVrms_per_channel.items()}
        else:
            threshold_high_3_5 = {key: value * 3.5 for key, value in preAmpVrms_per_channel.items()}
            threshold_low_3_5 = {key: value * -3.5 for key, value in preAmpVrms_per_channel.items()}
            threshold_high_4_4 = {key: value * 4.4 for key, value in preAmpVrms_per_channel.items()}
            threshold_low_4_4 = {key: value * -4.4 for key, value in preAmpVrms_per_channel.items()}

        ic(preAmpVrms_per_channel, postAmpVrms_per_channel, threshold_high_3_5, threshold_high_4_4)
        # quit()

    if simulationSelector.run(evt, station.get_sim_station(), det):

        # efieldToVoltageConverter.run(evt, station, det)

        if add_noise:
            channelGenericNoiseAdder.run(evt, station, det, type='rayleigh', amplitude=preAmpVrms_per_channel)

        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)

        highLowThreshold.run(evt, station, det, threshold_high=threshold_high_3_5, 
                            threshold_low=threshold_low_3_5,
                            coinc_window = 40*units.ns,
                            triggered_channels=direct_LPDA_channels,
                            number_concidences=2,
                            trigger_name=f'direct_LPDA_2of4_3.5sigma')


        if station.get_trigger(f'direct_LPDA_2of4_3.5sigma').has_triggered():

            highLowThreshold.run(evt, station, det, threshold_high=threshold_high_4_4, 
                                threshold_low=threshold_low_4_4,
                                coinc_window = 40*units.ns,
                                triggered_channels=direct_LPDA_channels,
                                number_concidences=2,
                                trigger_name=f'direct_LPDA_2of4_4.44sigma')


            triggerTimeAdjuster.run(evt, station, det)
            # channelResampler.run(evt, station, det, 1*units.GHz)
            channelStopFilter.run(evt, station, det)
            eventWriter.run(evt, det)

    # Save every event for proper rate calculation
    # Now every event is saved regardless of if it triggers or not
    # When checking events in nur, now check if station.has_triggered()
    # eventWriter.run(evt, det)


nevents = eventWriter.end()
print("Finished processing, {} events".format(nevents))