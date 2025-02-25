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

# Get files for simulation
input_files = pullFilesForSimulation('MB', min_file, max_file)

# Setup detector
# det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_{loc}_BacklobeTest_{amp_type}s_footprint576m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
# det = detector.Detector('../NuRadioMC/NuRadioReco/detector/ARIANNA/arianna_detector_db.json', 'json')   #Relative path from running folder
# Have to use a custom detector file rather than one in NuRadioReco as CoREAS throws events around the position 0,0
# Therefore the json is configured to be centered on station G/18, the approximate center of the detector, and then thrown in a large enough area to cover all stations
det = detector.Detector('HRASimulation/HRAStationLayoutForCoREAS.json', 'json')   #Relative path from running folder
det.update(datetime.datetime(2018, 10, 1))

# Indicate channels for simulation
primary_LPDA_channels = [0, 1, 2, 3]   #Downward for all stations except 32, which is upward
primary_LPDA_channels_52upward = [4, 5]
secondary_LPDA_channels = [4, 5, 6, 7]     #Only on station 52

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
triggerTimeAdjuster.begin(trigger_name=f'primary_LPDA_2of4_3.5sigma')

correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
correlationDirectionFitter.begin(debug=False)


all_stations = [13, 14, 15, 17, 18, 19, 30, 32, 52, 113, 114, 115, 117, 118, 119, 130, 132, 152]
# all_stations = [13, 14, 15, 17, 18, 19, 30, 32, 52]
# all_stations_reflected = [113, 114, 115, 117, 118, 119, 130, 132, 152]

trigger_sigmas = [3.5, 4, 4.5, 5, 5.5, 6]

preAmpVrms_per_channel = {}
thresholds_high = {}
thresholds_low = {}
for station_id in all_stations:
    preAmpVrms_per_channel[station_id] = {}
    thresholds_high[station_id] = {}
    thresholds_low[station_id] = {}
# for station_id in all_stations_reflected:
#     preAmpVrms_per_channel[station_id] = {}
#     thresholds_high[station_id] = {}
#     thresholds_low[station_id] = {}

# Start simulation
# Because CoREAS only simulates one station at a time, need to simulate each station in order using the same seed
# All events will be saved, therefore all events will be in order compared to each other across files
def run_stations(stations_list, mode='by_depth'):

    # readCoREAS = NuRadioReco.modules.io.coreas.readCoREASStationGrid.readCoREAS()
    readCoREAS = readCoREASStationGrid.readCoREAS()
    readCoREAS.begin(input_files, (-distance/2)*units.km, (distance/2)*units.km, (-distance/2)*units.km, (distance/2)*units.km, n_cores=n_cores, shape='radial', seed=seed, log_level=logging.WARNING)
    # readCoREAS.begin(input_files, (-distance/2)*units.km, (distance/2)*units.km, (-distance/2)*units.km, (distance/2)*units.km, n_cores=n_cores, shape='uniform', seed=seed, log_level=logging.WARNING)
    eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
    eventWriter.begin(output_filename + f'.nur')



    for evt, iE, x, y in readCoREAS.run(detector=det, ray_type=mode, layer_depth=-576*units.m, layer_dB=0, attenuation_model='MB_freq', output_mode=2):
        ic(f"processing event {iE} with id {evt.get_id()} at position {x}, {y}")

        evt.set_parameter(evtp.coreas_x, x)
        evt.set_parameter(evtp.coreas_y, y)

        for station_id in stations_list:
            ic(f'Starting station {station_id} simulation')
    
            # det.update(datetime.datetime(2018, 10, 1))
            # ic(det.__current_time)
            # ic(det.__valid_t0)
            # ic(det.__valid_t1)

            # for station in evt.get_stations():

            station = evt.get_station(station_id)
            station.set_station_time(datetime.datetime(2018, 10, 1))

            eventTypeIdentifier.run(evt, station, mode='forced', forced_event_type='cosmic_ray')
            # channelAddCableDelay.run(evt, station, det, mode='add')   # Not sure if necessary
            efieldToVoltageConverter.run(evt, station, det)
            channelResampler.run(evt, station, det, 2*units.GHz)


            if preAmpVrms_per_channel[station_id] == {}:
                # Get noise levels for simulation
                preAmpVrms, postAmpVrms = calculateNoisePerChannel(det, station=station, amp=True, hardwareResponseIncorporator=hardwareResponseIncorporator, channelBandPassFilter=channelBandPassFilter)
                preAmpVrms_per_channel[station_id] = preAmpVrms

                for sigma in trigger_sigmas:
                    threshold_high = {key: value * sigma for key, value in postAmpVrms.items()}
                    threshold_low = {key: value * -sigma for key, value in postAmpVrms.items()}
                    thresholds_high[station_id][sigma] = threshold_high
                    thresholds_low[station_id][sigma] = threshold_low

                # threshold_high_3_5 = {key: value * 3.5 for key, value in postAmpVrms.items()}
                # threshold_low_3_5 = {key: value * -3.5 for key, value in postAmpVrms.items()}
                # threshold_high_4_4 = {key: value * 4.4 for key, value in postAmpVrms.items()}
                # threshold_low_4_4 = {key: value * -4.4 for key, value in postAmpVrms.items()}

                # ic(preAmpVrms, postAmpVrms, threshold_high_3_5, threshold_high_4_4)
                # thresholds_high[station_id][3.5] = threshold_high_3_5
                # thresholds_high[station_id][4.4] = threshold_high_4_4
                # thresholds_low[station_id][3.5] = threshold_low_3_5
                # thresholds_low[station_id][4.4] = threshold_low_4_4



                # quit()

            # if simulationSelector.run(evt, station.get_sim_station(), det):
            if True:

                # efieldToVoltageConverter.run(evt, station, det)

                if add_noise:
                    channelGenericNoiseAdder.run(evt, station, det, type='rayleigh', amplitude=preAmpVrms_per_channel[station_id])

                hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)


                if station_id == 52 or station_id == 152:
                    highLowThreshold.run(evt, station, det, threshold_high=thresholds_high[station_id][trigger_sigmas[0]], 
                                        threshold_low=thresholds_low[station_id][trigger_sigmas[0]],
                                        coinc_window = 40*units.ns,
                                        triggered_channels=primary_LPDA_channels_52upward,
                                        number_concidences=2,
                                        trigger_name=f'primary_LPDA_2of4_{trigger_sigmas[0]}sigma')
                else:
                    highLowThreshold.run(evt, station, det, threshold_high=thresholds_high[station_id][trigger_sigmas[0]], 
                                        threshold_low=thresholds_low[station_id][trigger_sigmas[0]],
                                        coinc_window = 40*units.ns,
                                        triggered_channels=primary_LPDA_channels,
                                        number_concidences=2,
                                        trigger_name=f'primary_LPDA_2of4_{trigger_sigmas[0]}sigma')



                if station.get_trigger(f'primary_LPDA_2of4_{trigger_sigmas[0]}sigma').has_triggered():

                    if station_id == 52 or station_id == 152:
                        # For station 52, primary and secondary are reveresed due to ordering of channels that are upward/downward
                        # Upward are primary, downward are secondary
                        highLowThreshold.run(evt, station, det, threshold_high=thresholds_high[station_id][trigger_sigmas[0]], 
                                            threshold_low=thresholds_low[station_id][trigger_sigmas[0]],
                                            coinc_window = 40*units.ns,
                                            triggered_channels=primary_LPDA_channels,
                                            number_concidences=2,
                                            trigger_name=f'secondary_LPDA_2of4_{trigger_sigmas[0]}sigma')
                        for sigma in trigger_sigmas[1:]:
                            highLowThreshold.run(evt, station, det, threshold_high=thresholds_high[station_id][sigma], 
                                                threshold_low=thresholds_low[station_id][sigma],
                                                coinc_window = 40*units.ns,
                                                triggered_channels=primary_LPDA_channels_52upward,
                                                number_concidences=2,
                                                trigger_name=f'primary_LPDA_2of4_{sigma}sigma')

                            highLowThreshold.run(evt, station, det, threshold_high=thresholds_high[station_id][sigma], 
                                                threshold_low=thresholds_low[station_id][sigma],
                                                coinc_window = 40*units.ns,
                                                triggered_channels=primary_LPDA_channels,
                                                number_concidences=2,
                                                trigger_name=f'secondary_LPDA_2of4_{sigma}sigma')


                    else:
                        for sigma in trigger_sigmas[1:]:
                            highLowThreshold.run(evt, station, det, threshold_high=thresholds_high[station_id][sigma], 
                                                threshold_low=thresholds_low[station_id][sigma],
                                                coinc_window = 40*units.ns,
                                                triggered_channels=primary_LPDA_channels,
                                                number_concidences=2,
                                                trigger_name=f'primary_LPDA_2of4_{sigma}sigma')

                    triggerTimeAdjuster.run(evt, station, det)
                    # channelResampler.run(evt, station, det, 1*units.GHz)
                    channelStopFilter.run(evt, station, det, prepend=0*units.ns, append=0*units.ns)
                    correlationDirectionFitter.run(evt, station, det, n_index=1.35, ZenLim=[0*units.deg, 180*units.deg])

                    # Testing
                    # sim_station = station.get_sim_station()
                    # ic(sim_station.get_parameter(stnp.zenith)/units.deg, sim_station.get_parameter(stnp.azimuth)/units.deg)
                    # ic(station.get_parameter(stnp.zenith)/units.deg, station.get_parameter(stnp.azimuth)/units.deg)
                    # quit()
        # Save every event for rate calculation
        eventWriter.run(evt)
        

        # Save every event for proper rate calculation
        # Now every event is saved regardless of if it triggers or not
        # When checking events in nur, now check if station.has_triggered()
        # eventWriter.run(evt, det)


    nevents = eventWriter.end()
    dt = readCoREAS.end()
    ic(f"Finished processing All stations, {nevents} events processed, {dt} seconds elapsed")


if __name__ == "__main__":

    # run_stations(all_stations, mode='direct')
    # run_stations(all_stations_reflected, mode='reflected')
    run_stations(all_stations)