import os
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
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import scipy
import glob
import pickle
from NuRadioReco.modules.base import module
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import stationParameters as stnp
import time
import sys
import numpy as np
import astropy

import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Run CR analysis for specific runids')
parser.add_argument('spacing', type=float, default=1., help='Station spacing')
parser.add_argument('min_run', type=int, default=0, help='Min run to start on')
parser.add_argument('max_run', type=int, default=0, help='Max run to end on, default zero means goes through all') 
parser.add_argument('type', type=str, default='CR', help='CR or Sample footprints')
parser.add_argument('cores', type=int, default=100, help='Number of cores to sim')


args = parser.parse_args()

type = args.type
cores = args.cores
minrun = args.min_run
maxrun = args.max_run

if type == 'CR':
    path = '../../CRFootprints/'
elif type == 'Sample':
    path = '../../SampleFootprints/'
else:
    print('No correct type of footprints to use')
    quit()
input_files = [path + file for file in os.listdir(path)]

#Only import as plt for showing while connected, to make and save have all 3
import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt

# Logging level
import logging
# logger = module.setup_logger(level=logging.INFO)
logger=logging.getLogger("module")
logger.setLevel(logging.WARNING)

galactic_noise_interpolation_frequencies_step = 100

#spacing = float(sys.argv[1]) * units.km
spacing = args.spacing * units.km
threshold = 25.65*units.micro*units.V
Tnoise = 300
max_freq =  2.5 * units.GHz
min_freq = 0
Vrms_thermal_noise = (((scipy.constants.Boltzmann * units.joule / units.kelvin) * Tnoise * (max_freq) * 50 * units.ohm)**0.5)
print(f"noise temperature = {Tnoise} -> Vrms = {Vrms_thermal_noise:.2g}V")

downward_LPDA = [0, 1, 2, 3]
upward_LPDA = [4, 5, 6]

det = generic_detector.GenericDetector(json_filename=f"../configurations/3LPDA_upward_grid_{spacing/units.km:.2f}km.json",
                        assume_inf=False, antenna_by_depth=False, default_station=1) 
det.update(datetime.datetime(2018, 10, 1))

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processung
readCoREAS = NuRadioReco.modules.io.coreas.readCoREASStationGrid.readCoREAS()
readCoREAS.begin(input_files, -spacing, spacing, -spacing, spacing,
                 n_cores=cores, seed=None, log_level=logging.WARNING)
"""
simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
simulationSelector.begin()

electricFieldBandPassFilter = NuRadioReco.modules.electricFieldBandPassFilter.electricFieldBandPassFilter()

efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter(log_level=logging.WARNING)
efieldToVoltageConverter.begin(debug=False)

hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()

channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()

# import NuRadioReco.modules.channelGalacticNoiseAdder
# channelGalacticNoiseAdder = NuRadioReco.modules.channelGalacticNoiseAdder.channelGalacticNoiseAdder()
# channelGalacticNoiseAdder.begin(n_side=4, interpolation_frequencies=np.arange(10, 1100, galactic_noise_interpolation_frequencies_step) * units.MHz)

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()

channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()

triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
"""

station_id = 1
output = {}
# Loop over all events in file as initialized in readCoRREAS and perform analysis

#energies = np.array([])
energies = []
energies_extended = np.array([])
#em_energies = np.array([])
em_energies = []
extra_ss_energies = np.array([])
n_showers = np.array([])
n=0
c=0
for iE, evt in enumerate(readCoREAS.run(detector=det)):
    station = evt.get_station(1)
    station.set_station_time(astropy.time.Time('2019-01-01T00:00:00'))
    ss = station.get_sim_station()
    det.update(station.get_station_time())
    cr_energy = ss[stnp.cr_energy]
    cr_energy_em = ss[stnp.cr_energy_em]
    energies.append(cr_energy)
    em_energies.append(cr_energy_em)
    if cr_energy_em > cr_energy:
        print(f'EM {cr_energy_em} > CR {cr_energy}')
        n +=1
    c +=1
    """
    runid = evt.get_run_number()
#    if runid < minrun:
#        continue
#    if (maxrun != 0) and (runid >= maxrun):
#        break
    sim_shower = evt.get_sim_shower(0)

    if sim_shower[shp.energy] < 10**17.5:
        print(f'First shower energy is {sim_shower[shp.energy]}, printing info')
        energies = np.append(energies, [sim_shower[shp.energy]])
        n_shower = 0
        for shower in evt.get_sim_showers():
            print(f'ss of {shower[shp.energy]}, em energy {shower[shp.electromagnetic_energy]}, primary part {shower[shp.primary_particle]}')
            n_shower += 1
            energies_extended = np.append(energies_extended, [sim_shower[shp.energy]])
            em_energies = np.append(em_energies, [shower[shp.electromagnetic_energy]])
            extra_ss_energies = np.append(extra_ss_energies, [shower[shp.energy]])

        n_showers = np.append(n_showers, [n_shower])
    """
    """
    if(runid not in output):
        output[runid] = {}
        output[runid]['n'] = 0
        output[runid]['n_triggered'] = 0
        output[runid]['n_triggered_upward'] = 0
        output[runid]['n_att_triggered'] = 0
        output[runid]['energy'] = sim_shower[shp.energy]
        output[runid]['ss_energy'] = 0
        output[runid]['zenith'] = sim_shower[shp.zenith]
        output[runid]['azimuth'] = sim_shower[shp.azimuth]
        logger.warning(f"{runid}, {output[runid]['n_triggered']}/{output[runid]['n']} cosmic ray with energy {output[runid]['energy']:.2g}eV zenith = {output[runid]['zenith']/units.deg:.0f}deg, azimuth = {output[runid]['azimuth']/units.deg:.0f}deg")
    
    
    output[runid]['n'] +=1

    triggered = False
    att_triggered = False
    upward_triggered = False

    print(f'Run of {runid}')
    for station in evt.get_stations():
        print('New Station')
        station.set_station_time(astropy.time.Time('2019-01-01T00:00:00'))
        eventTypeIdentifier.run(evt, station, 'forced', 'cosmic_ray')
        efieldToVoltageConverter.run(evt, station, det)
        hardwareResponseIncorporator.run(evt, station, det)
#        channelGenericNoiseAdder.run(evt, station, det, min_freq=0, max_freq=max_freq, type="rayleigh",
#                                     amplitude=Vrms_thermal_noise)
    
#        channelBandPassFilter.run(evt, station, det, passband=[80*units.MHz, 180 *units.MHz],
#                                  filter_type='butter', order=10)
        triggerSimulator.run(evt, station, det, threshold_high=threshold, threshold_low=-threshold, number_concidences=2, triggered_channels=downward_LPDA, trigger_name='downLPDA')
        triggerSimulator.run(evt, station, det, 
                             threshold_high=threshold,
                             threshold_low=-threshold,
                             coinc_window=80 * units.ns,
                             number_concidences=2, triggered_channels=upward_LPDA,
                             trigger_name='upLPDA')

        if station.has_triggered(trigger_name='upLPDA'):
            upward_triggered = True

        print('E field work')
        if (station.has_triggered(trigger_name='downLPDA')) and (not att_triggered):
            triggered = True
            station.remove_triggers()

            station.get_electric_fields_for_channels(downward_LPDA)
            ss = station.get_sim_station()
            output[runid]['ss_energy'] = ss.get_parameter(stnp.cr_energy)
            efields = ss.get_electric_fields()
            ss.set_electric_fields([])
            for field in efields:
                temp_trace = NuRadioReco.framework.base_trace.BaseTrace()
                temp_trace.set_trace(field.get_trace(), field.get_sampling_rate())

                inc_zenith = field.get_parameter(efp.zenith)
                depth_ice = 576 * units.m
                dist_traveled = 2 * depth_ice / np.cos(inc_zenith / units.rad)

                att_length = 400 * units.m
                temp_trace.set_trace(field.get_trace() * np.exp(-dist_traveled/att_length), field.get_sampling_rate())

                field.set_trace(temp_trace.get_trace(), field.get_sampling_rate())
                field.set_parameter(efp.zenith, np.pi * units.rad - inc_zenith)
                ss.add_electric_field(field)

            station.set_sim_station(ss)
            efieldToVoltageConverter.run(evt, station, det)
            hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)
            triggerSimulator.run(evt, station, det, threshold_high=threshold, threshold_low=-threshold, number_concidences=2, triggered_channels=downward_LPDA, trigger_name='attLPDA')

            if station.has_triggered(trigger_name='attLPDA'):
                att_triggered = True

        station.remove_triggers()
        if upward_triggered and triggered and att_triggered:
            break


    if upward_triggered:
        output[runid]['n_triggered_upward'] += 1
    if triggered:
        output[runid]['n_triggered'] += 1
    if att_triggered:
        output[runid]['n_att_triggered'] += 1

#    if runid > 10:
#        break
    """


#with open(f"output/detection_efficiency_{type}_{cores}_{spacing/units.km:.2f}km_{threshold/units.micro/units.V:.0f}muV_id{minrun}_{maxrun}.pkl", 'wb') as fout:
#    pickle.dump(output, fout)


#plt.scatter(np.log10(energies), n_showers)
#plt.title('log10 first shower energy vs num showers in file')
#plt.show()

print(f'Fraction over {n/c}')

plt.scatter(np.log10(energies), np.log10(em_energies))
plt.title('log10 first shower energy vs em energy of all showers')
plt.show()

#plt.scatter(np.log10(energies_extended), np.log10(extra_ss_energies))
#plt.title('log10 first shower energy vs all shower energies')
#plt.show()


