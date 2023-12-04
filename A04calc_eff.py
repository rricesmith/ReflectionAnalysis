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
import astrotools.auger as auger
import time
import sys
import numpy as np
import astropy

#Only import as plt for showing while connected, to make and save have all 3
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

# Logging level
import logging
# logger = module.setup_logger(level=logging.INFO)
logger=logging.getLogger("module")
logger.setLevel(logging.WARNING)

galactic_noise_interpolation_frequencies_step = 100

spacing = float(sys.argv[1]) * units.km
threshold = 25.65*units.micro*units.V
Tnoise = 250
#Tnoise = 300
#max_freq =  2.5 * units.GHz
#min_freq = 0
min_freq = 80 * units.MHz
max_freq = 250 * units.MHz
Vrms_thermal_noise = (((scipy.constants.Boltzmann * units.joule / units.kelvin) * Tnoise * (max_freq-min_freq) * 50 * units.ohm)**0.5)
#Vrms_thermal_noise = (((scipy.constants.Boltzmann * units.joule / units.kelvin) * Tnoise * (max_freq) * 50 * units.ohm)**0.5)
print(f"noise temperature = {Tnoise} -> Vrms = {Vrms_thermal_noise:.2g}V")
quit()

t_start = time.time()
#data_path = "/proj/snic2020-6-162/coreas/southpole/production_202007/output/hdf5"
# data_path = "/Users/cglaser/remote/snic//snic2020-6-162/coreas/southpole/production_202007/output/hdf5"
#input_files = sorted(glob.glob(os.path.join(data_path, "SIM??????.hdf5")))

#path = '../CRFootprints/'
path = '../SampleFootprints/'
input_files = [path + file for file in os.listdir(path)]


det = generic_detector.GenericDetector(json_filename=f"configurations/3LPDA_upward_{spacing/units.km:.2f}km.json",
                        assume_inf=False, antenna_by_depth=False, default_station=1) 
det.update(datetime.datetime(2018, 10, 1))

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processung
readCoREAS = NuRadioReco.modules.io.coreas.readCoREASStationGrid.readCoREAS()
readCoREAS.begin(input_files, -spacing, spacing, -spacing, spacing,
                 n_cores=100, seed=None, log_level=logging.WARNING)

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

station_id = 1
output = {}
# Loop over all events in file as initialized in readCoRREAS and perform analysis
for iE, evt in enumerate(readCoREAS.run(detector=det)):
    runid = evt.get_run_number()
    sim_shower = evt.get_sim_shower(0)
    if(runid not in output):
        output[runid] = {}
        output[runid]['n'] = 0
        output[runid]['n_triggered'] = 0
        output[runid]['n_att_triggered'] = 0
        output[runid]['energy'] = sim_shower[shp.energy]
        output[runid]['zenith'] = sim_shower[shp.zenith]
        output[runid]['azimuth'] = sim_shower[shp.azimuth]
        logger.warning(f"{runid}, {output[runid]['n_triggered']}/{output[runid]['n']} cosmic ray with energy {output[runid]['energy']:.2g}eV zenith = {output[runid]['zenith']/units.deg:.0f}deg, azimuth = {output[runid]['azimuth']/units.deg:.0f}deg")

        
    output[runid]['n'] +=1


    triggered = False
    att_triggered = False

    station = evt.get_station(station_id)
    station.set_station_time(astropy.time.Time('2019-01-01T00:00:00'))
    det.update(station.get_station_time())
    if simulationSelector.run(evt, station.get_sim_station(), det):
        efieldToVoltageConverter.run(evt, station, det)
        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)
        triggerSimulator.run(evt, station, det, number_concidences=2, triggered_channels=[0, 1, 2, 3])

        if station.has_triggered():
            triggered = True
            station.remove_triggers()


            station.get_electric_fields_for_channels([0, 1, 2, 3])
            ss = station.get_sim_station()
            efields = ss.get_electric_fields()
            ss.set_electric_fields([])
            for field in efields:
                temp_trace = NuRadioReco.framework.base_trace.BaseTrace()
                temp_trace.set_trace(field.get_trace(), field.get_sampling_rate())

                inc_zenith = field.get_parameter(efp.zenith)
                depth_ice = 576 * units.m			#Depth of ice at Moores Bay
                dist_traveled = 2 * depth_ice / np.cos(inc_zenith / units.rad)

                att_length = 400 * units.m			#Simple att length for now, freq dep TODO
                temp_trace.set_trace(field.get_trace() * np.exp(-dist_traveled/att_length), field.get_sampling_rate())

                field.set_trace(temp_trace.get_trace(), field.get_sampling_rate())
                field.set_parameter(efp.zenith, np.pi * units.rad - inc_zenith)
                ss.add_electric_field(field)

            station.set_sim_station(ss)
            efieldToVoltageConverter.run(evt, station, det)
            hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)
            triggerSimulator.run(evt, station, det, number_concidences=2, triggered_channels=[0, 1, 2, 3])

            if station.get_trigger('default_high_low').has_triggered():
                att_triggered = True

    """Christian's method, ignore unless looking at code
    logger.debug("processing event {:d} with id {:d}".format(iE, runid))
    triggered = False
    for station in evt.get_stations():
#         logger.info(f"station {station.get_id()}")
        station.set_station_time(astropy.time.Time('2019-01-01T00:00:00'))
        eventTypeIdentifier.run(evt, station, 'forced', 'cosmic_ray')
        efieldToVoltageConverter.run(evt, station, det)
#         channelGalacticNoiseAdder.run(evt, station, det)
        channelGenericNoiseAdder.run(evt, station, det, min_freq=0, max_freq=max_freq, type="rayleigh",
                                     amplitude=Vrms_thermal_noise)
    
        channelBandPassFilter.run(evt, station, det, passband=[80*units.MHz, 180 *units.MHz],
                                  filter_type='butter', order=10)
        
        triggerSimulator.run(evt, station, det, 
                             threshold_high=threshold,
                             threshold_low=-threshold,
                             coinc_window=80 * units.ns,
                             number_concidences=2, triggered_channels=None,
                             trigger_name='highlow_trigger')
        if(station.has_triggered()):
            triggered = True
    """


    if triggered:
        output[runid]['n_triggered'] += 1
    if att_triggered:
        output[runid]['n_att_triggered'] += 1

#    if runid > 50:
#        break


tot_trig = 0
tot_event = 0
tot_att_trig = 0
for runid in output:
    tot_trig += output[runid]['n_triggered']
    tot_event += output[runid]['n']
    tot_att_trig += output[runid]['n_att_triggered']
print(f'total trig {tot_trig} of {tot_event} events')
print(f'att trig of {tot_att_trig}')


energies = np.array([output[runid]['energy'] for runid in output])
zeniths = np.array([output[runid]['zenith'] for runid in output])
n = np.array([output[runid]['n'] for runid in output])
n_trig = np.array([output[runid]['n_triggered'] for runid in output])
n_trig_att = np.array([output[runid]['n_att_triggered'] for runid in output])

n_weighted = n * np.cos(np.pi/2 - zeniths)
n_trig_weighted = n_trig * np.cos(np.pi/2 - zeniths)
n_trig_att_weighted = n_trig_att * np.cos(np.pi/2 - zeniths)

n_bins = 50
#eng_bins = np.logspace(min(energies), max(energies), n_bins)
eng_bins = np.linspace(np.log10(min(energies)), np.log10(max(energies)), n_bins)
print(f'eng bins {len(eng_bins)}')
centered_eng_bins = np.zeros(n_bins - 1)
n_throw_bins = np.zeros(n_bins - 1)
trig_bins = np.zeros(n_bins - 1)
trig_att_bins = np.zeros(n_bins - 1)

for eng in range(len(eng_bins) - 1):
    centered_eng_bins[eng] = eng_bins[eng] + (eng_bins[1] - eng_bins[0])/2
    for i_E in range(len(energies)):
        if eng_bins[eng] <= np.log10(energies[i_E]) < eng_bins[eng+1]:
            trig_bins[eng] += n_trig_weighted[i_E]
            trig_att_bins[eng] += n_trig_att_weighted[i_E]
            n_throw_bins[eng] += n_weighted[i_E]


auger_bins = np.zeros(n_bins-1)
for i_E in range(n_bins - 1):
#    auger_bins[i_E] = auger.event_rate(np.log10(eng_bins[i_E]), np.log10(eng_bins[i_E + 1]), max(zeniths) * 180/np.pi, (spacing ** 2)/units.km2 * trig_att_bins[i_E] / n_throw_bins[i_E])	#If energy changes to need log
#    auger_bins[i_E] = auger.event_rate(eng_bins[i_E], eng_bins[i_E + 1], max(zeniths) * 180/np.pi , (spacing ** 2)/units.km2 * trig_bins[i_E] / n_throw_bins[i_E])				#Unattenuated
    auger_bins[i_E] = auger.event_rate(eng_bins[i_E], eng_bins[i_E + 1], max(zeniths) * 180/np.pi , (spacing ** 2)/units.km2 * trig_att_bins[i_E] / n_throw_bins[i_E])				#Attenuated
    print(f'Auger bin {auger_bins[i_E]}, Emin {eng_bins[i_E]} Emax {eng_bins[i_E + 1]} zen {max(zeniths) * 180/np.pi} Area {(spacing ** 2)/units.km2 * trig_bins[i_E] / n_throw_bins[i_E]}')
    print(f'spacing {(spacing ** 2)/units.km2} trig bins {trig_bins[i_E]} n throw bins {n_throw_bins[i_E]} ratio {trig_bins[i_E] / n_throw_bins[i_E]}')


min_aug = 1
for auger in auger_bins:
    if auger != 0 and auger < min_aug:
        min_aug = auger 

#print(f'eng bins {len(centered_eng_bins)}')
#print(f' auger {len(auger_bins)}')
plt.scatter(centered_eng_bins, auger_bins)
#plt.xscale('log')
plt.title('Reflected Events Per Year of Cosmic Rays at Moores Bay')
plt.yscale('log')
plt.ylim([min_aug / 10, 10 * max(auger_bins)])
plt.xlabel('Energy (log10 eV)')
plt.ylabel('Events/Year/Station')
plt.savefig('EventRateAllCRs.pdf')
#plt.show()

#plt.hist(energies, bins=10)
#plt.show()

#plt.hist(zeniths, bins=10)
#plt.show()


