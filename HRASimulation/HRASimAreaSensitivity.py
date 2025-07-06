from NuRadioReco.utilities import units
import NuRadioReco.modules.io.coreas.readCoREAS
# import NuRadioReco.modules.io.coreas.readCoREASStationGrid
import readCoREASStationGrid    # Use local version saved in this repository
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
import NuRadioReco.modules.correlationDirectionFitter
import astropy
import argparse
import NuRadioReco.modules.io.eventWriter
import numpy as np
import os
import datetime
from icecream import ic
from scipy import constants
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import eventParameters as evtp
from NuRadioReco.framework.parameters import showerParameters as shp
from HRASimulation.HRAAnalysis import HRAevent

from NuRadioReco.detector import detector
from NuRadioReco.detector import generic_detector

from SimpleFootprintSimulation.SimHelpers import pullFilesForSimulation, calculateNoisePerChannel

import logging
logger=logging.getLogger("module")
logger.setLevel(logging.WARNING)

# Read in settings for simulation
parser = argparse.ArgumentParser(description='Run Cosmic Ray simulation for Station 51')
parser.add_argument('output_filename', type=str, help='Output filename for simulation, without nur prepend')
parser.add_argument('n_cores', type=int, help='Number of cores to use for simulation')
parser.add_argument('--min_file', type=int, default=0, help='Minimum file number to use')
parser.add_argument('--max_file', type=int, default=-1, help='Maximum file number to use, -1 means use all files')
parser.add_argument('--add_noise', default=False, help='Include noise in simulation')
parser.add_argument('--distance', type=int, default=10, help='Distance in km to simulate')
parser.add_argument('--seed', type=int, default=0, help='Seed for simulation, needed to ensure all stations receive same CRs')

args = parser.parse_args()
output_filename = args.output_filename
n_cores = args.n_cores
min_file = args.min_file
max_file = args.max_file
add_noise = args.add_noise
distance = args.distance
seed = args.seed

ic(f'Simulation parameters are: {output_filename}, {n_cores}, {min_file}, {max_file}, {add_noise}, {distance}, {seed}')

# Get files for simulation
input_files = pullFilesForSimulation('MB', min_file, max_file)
# Select a single file for simple testing
input_files = [input_files[0]]  # For testing, use only the first file

# Setup detector
det = detector.Detector('HRASimulation/HRAStationLayoutForCoREAS.json', 'json')   #Relative path from running folder
det.update(datetime.datetime(2018, 10, 1))

# Indicate channels for simulation
primary_LPDA_channels = [0, 1, 2, 3]

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processung
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
triggerTimeAdjuster.begin() # Default trigger name will be overridden in the run call

correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
correlationDirectionFitter.begin(debug=False)

# UPDATED: Station list for testing
all_stations = [13, 17, 113]

# UPDATED: Sigma values for testing, in descending order
trigger_sigmas = [50, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

preAmpVrms_per_channel = {}
thresholds_high = {}
thresholds_low = {}
for station_id in all_stations:
    preAmpVrms_per_channel[station_id] = {}
    thresholds_high[station_id] = {}
    thresholds_low[station_id] = {}

# Start simulation
def run_stations(stations_list, mode='by_depth'):

    readCoREAS = readCoREASStationGrid.readCoREAS()
    # UPDATED: readCoREAS shape is now 'uniform'
    readCoREAS.begin(input_files, (-distance/2)*1000, (distance/2)*1000, (-distance/2)*1000, (distance/2)*1000, n_cores=n_cores, shape='uniform', seed=seed, log_level=logging.DEBUG)

    eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
    eventWriter.begin(output_filename + f'.nur')

    HRAEventList = []

    for evt, iE, x, y in readCoREAS.run(detector=det, ray_type=mode, layer_depth=-576*units.m, layer_dB=0, attenuation_model='MB_freq', output_mode=2):
        ic(f"processing event {iE} with id {evt.get_id()} at position {x}, {y}")
        sim_shower = evt.get_sim_shower(0)
        ic(f'Event parameters are: Eng {sim_shower[shp.energy]/units.eV}eV, Zen {sim_shower[shp.zenith]/units.deg}deg, Azi {sim_shower[shp.azimuth]/units.deg}deg')

        evt.set_parameter(evtp.coreas_x, x)
        evt.set_parameter(evtp.coreas_y, y)

        for station_id in stations_list:
            station = evt.get_station(station_id)
            station.set_station_time(datetime.datetime(2018, 10, 1))

            eventTypeIdentifier.run(evt, station, mode='forced', forced_event_type='cosmic_ray')
            efieldToVoltageConverter.run(evt, station, det)
            channelResampler.run(evt, station, det, 2*units.GHz)

            if preAmpVrms_per_channel[station_id] == {}:
                # Get noise levels for simulation
                preAmpVrms, postAmpVrms = calculateNoisePerChannel(det, station=station, amp=True, hardwareResponseIncorporator=hardwareResponseIncorporator, channelBandPassFilter=channelBandPassFilter)
                preAmpVrms_per_channel[station_id] = preAmpVrms

                ic(f'Station {station_id} has preAmpVrms {preAmpVrms}')
                ic(f'Station {station_id} has postAmpVrms {postAmpVrms}')

                # Calculate all possible thresholds beforehand
                for sigma in trigger_sigmas:
                    threshold_high = {key: value * sigma for key, value in postAmpVrms.items()}
                    threshold_low = {key: value * -sigma for key, value in postAmpVrms.items()}
                    thresholds_high[station_id][sigma] = threshold_high
                    thresholds_low[station_id][sigma] = threshold_low

            if True:
                if add_noise:
                    channelGenericNoiseAdder.run(evt, station, det, type='rayleigh', amplitude=preAmpVrms_per_channel[station_id])

                hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)

                # --- NEW TRIGGER LOGIC ---
                # Iterate through sigmas from highest to lowest to find the maximum sigma that triggers
                for sigma in trigger_sigmas:
                    trigger_name = f'primary_LPDA_2of4_{sigma}sigma'
                    highLowThreshold.run(evt, station, det,
                                         threshold_high=thresholds_high[station_id][sigma],
                                         threshold_low=thresholds_low[station_id][sigma],
                                         coinc_window=40 * units.ns,
                                         triggered_channels=primary_LPDA_channels,
                                         number_concidences=2,
                                         trigger_name=trigger_name)

                    # If it triggered, this is the highest sigma. Record it and stop trying lower values.
                    if station.get_trigger(trigger_name).has_triggered():

                        # Run post-trigger analysis modules
                        triggerTimeAdjuster.run(evt, station, det, trigger_name=trigger_name)
                        channelStopFilter.run(evt, station, det, prepend=0 * units.ns, append=0 * units.ns)
                        # correlationDirectionFitter.run(evt, station, det, n_index=1.35, ZenLim=[0 * units.deg, 180 * units.deg], channel_pairs=((primary_LPDA_channels[0], primary_LPDA_channels[2]), (primary_LPDA_channels[1], primary_LPDA_channels[3])))

                        # Break the loop since we found the highest sigma that triggered
                        break
                # --- END OF NEW TRIGGER LOGIC ---

            ic(f'{station_id} Triggered {station.has_triggered()}')
            if station.has_triggered():
                ic(f'{station_id} highest sigma was {station.get_parameter("highest_trigger_sigma")}')

        # Save every event for rate calculation
        eventWriter.run(evt)

        # Save event in condensed format as well
        HRAEventList.append(HRAevent(evt))

    nevents = eventWriter.end()
    dt = readCoREAS.end()
    ic(f"Finished processing All stations, {nevents} events processed, {dt} seconds elapsed")

    HRAEventList = np.array(HRAEventList)
    np.save(output_filename + f'_HRAeventList.npy', HRAEventList)

if __name__ == "__main__":
    # UPDATED: Run simulation only for the specified stations and mode
    run_stations(all_stations)