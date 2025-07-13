from NuRadioReco.utilities import units
# import NuRadioReco.modules.io.coreas.readCoREAS
import readCoREASStationGrid
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

from SimpleFootprintSimulation.modifyEfieldForSurfaceReflection import modifyEfieldForSurfaceReflection
from NuRadioReco.framework.parameters import showerParameters as shp

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


def pullFilesForSimulation(sim_type, min_energy, max_energy, num_icetop=10, icetop_sin=-1):
    """
    Pull in IceTop files for simulation
    IceTop simulations range from 16.0-18.5 log10eV
    Sin^2(zenith) bins range from 0.0-1.0
    There are ~33 separate footprints per Energy/Sin^2 bin
    """

    input_files = []
    if min_energy < 16.0:
        print(f'Setting IceTop min energy to 16.0 log10eV')
        min_energy = 16.0
    if max_energy == -1 or max_energy > 18.5:
        print(f'Setting IceTop max energy to 18.5 log10eV')
        max_energy = 18.5

    i = min_energy
    while i < max_energy:
        #Currently just iterating through all sin's equally. Can separate sin bins if needed
        if icetop_sin == -1:
            sin2Val = np.arange(0, 1.01, 0.1)
        else:
            sin2Val = [icetop_sin]
        for sin2 in sin2Val:
            num_in_bin = 0
            folder = f'../../../../../dfs8/sbarwick_lab/arianna/SIM/southpole/IceTop/lgE_{i:.1f}/sin2_{sin2:.1f}/'
            for (dirpath, dirnames, filenames) in os.walk(folder):
                for file in filenames:
                    if num_in_bin == num_icetop:
                        continue
                    if not 'highlevel' in file:
                        file = os.path.join(folder, file)
                        input_files.append(file)
                        num_in_bin += 1
        i += 0.1

    return input_files

# Read in settings for simulation
parser = argparse.ArgumentParser(description='Run Cosmic Ray simulation for Station 51')
parser.add_argument('output_filename', type=str, help='Output filename for simulation')
parser.add_argument('n_cores', type=int, help='Number of cores to use for simulation')
parser.add_argument('--min_energy', type=float, default=16.0, help='Minimum energy for simulation')
parser.add_argument('--max_energy', type=float, default=18.5, help='Maximum energy for simulation')
parser.add_argument('--sin2', type=float, default=-1, help='Sin^2(zenith) value for simulation, range from 0.0-1.0')
parser.add_argument('--num_icetop', type=int, default=10, help='Number of IceTop footprints to simulate per bin')
parser.add_argument('--sim_amp', default=True, help='Include amplifier in simulation')
parser.add_argument('--add_noise', default=False, help='Include noise in simulation')

args = parser.parse_args()
output_filename = args.output_filename
n_cores = args.n_cores
min_energy = args.min_energy
max_energy = args.max_energy
sin2 = args.sin2
num_icetop = args.num_icetop
sim_amp = args.sim_amp
add_noise = args.add_noise

# Get files for simulation
input_files = pullFilesForSimulation('IceTop', min_energy, max_energy, num_icetop=num_icetop, icetop_sin=sin2)
ic(f'files for running {input_files}')

# Setup detector
# det = detector.Detector(json_filename=f'configurations/station51_InfAir.json', assume_inf=False, antenna_by_depth=False)
det = detector.Detector(json_filename=f'configurations/station51_InfAir.json', assume_inf=False, antenna_by_depth=False)
det.update(astropy.time.Time('2018-1-1'))
station_id = 51

# Indicate channels for simulation
direct_LPDA_channels = [4, 5, 6]

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processung
readCoREAS = readCoREASStationGrid.readCoREAS()
distance = 2 * units.km
readCoREAS.begin(input_files, -(distance)/2, (distance)/2, -(distance)/2, (distance)/2, n_cores=n_cores, shape='radial', seed=None, log_level=logging.WARNING)

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
triggerTimeAdjuster.begin(trigger_name=f'direct_LPDA_2of3_3.5sigma')


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

    sim_shower = evt.get_sim_shower(0)
    zenith = sim_shower[shp.zenith]/units.rad

    new_efields = []
    efields = station.get_electric_fields()
    for efield in efields:
        # modify the Efield for surface reflection
        # Doing this for backlobe antennas to. Needs to be removed in the future if backlobe signals wish to be looked at
        new_efields.append(modifyEfieldForSurfaceReflection(efield, incoming_zenith=zenith, antenna_height=1*units.m, n_index=1.35))

    station.set_electric_fields(new_efields)

    efieldToVoltageConverter.run(evt, station, det)
    channelResampler.run(evt, station, det, 1*units.GHz)


    if preAmpVrms_per_channel == {}:
        # Get noise levels for simulation
        preAmpVrms_per_channel, postAmpVrms_per_channel = calculateNoisePerChannel(det, station=station, amp=sim_amp)
        ic(preAmpVrms_per_channel, postAmpVrms_per_channel)
        if sim_amp:
            threshold_high_3_5 = {key: value * 3.5 for key, value in postAmpVrms_per_channel.items()}
            threshold_low_3_5 = {key: value * -3.5 for key, value in postAmpVrms_per_channel.items()}
            threshold_high_5 = {key: value * 5 for key, value in postAmpVrms_per_channel.items()}
            threshold_low_5 = {key: value * -5 for key, value in postAmpVrms_per_channel.items()}
        else:
            threshold_high_3_5 = {key: value * 3.5 for key, value in preAmpVrms_per_channel.items()}
            threshold_low_3_5 = {key: value * -3.5 for key, value in preAmpVrms_per_channel.items()}
            threshold_high_5 = {key: value * 5 for key, value in preAmpVrms_per_channel.items()}
            threshold_low_5 = {key: value * -5 for key, value in preAmpVrms_per_channel.items()}

        ic(preAmpVrms_per_channel, postAmpVrms_per_channel, threshold_high_3_5, threshold_high_5)
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
                            trigger_name=f'direct_LPDA_2of3_3.5sigma')


        if station.get_trigger(f'direct_LPDA_2of3_3.5sigma').has_triggered():

            highLowThreshold.run(evt, station, det, threshold_high=threshold_high_3_5, 
                                threshold_low=threshold_low_3_5,
                                coinc_window = 40*units.ns,
                                triggered_channels=direct_LPDA_channels,
                                number_concidences=3,
                                trigger_name=f'direct_LPDA_3of3_3.5sigma')

            highLowThreshold.run(evt, station, det, threshold_high=threshold_high_5, 
                                threshold_low=threshold_low_5,
                                coinc_window = 40*units.ns,
                                triggered_channels=direct_LPDA_channels,
                                number_concidences=3,
                                trigger_name=f'direct_LPDA_3of3_5sigma')


            triggerTimeAdjuster.run(evt, station, det)
            # channelResampler.run(evt, station, det, 1*units.GHz)
            channelStopFilter.run(evt, station, det, prepend=0*units.ns, append=0*units.ns)
            
    # Save every event for proper rate calculation
    # Now every event is saved regardless of if it triggers or not
    # When checking events in nur, now check if station.has_triggered()
    eventWriter.run(evt, det)


nevents = eventWriter.end()
print("Finished processing, {} events".format(nevents))
