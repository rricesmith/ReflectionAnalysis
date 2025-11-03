import argparse
from NuRadioMC.simulation import simulation
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.triggerTimeAdjuster
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
import numpy as np
import scipy
from scipy import constants
import matplotlib.pyplot as plt
import logging
import copy
import yaml

from NuRadioMC.utilities import medium, medium_base
from N00_produceEvents import generate_events_square

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runMB")

# 4 channel PA: 2x sampling, fft upsampling, 16 ns window, 11 beams
# Linear fit: -0.2519 x + 0.727
# Quadratic fit: -0.0002581 x x - 0.2393 x + 0.5845
# Noise rate -> threshold
#
# 10000.000 Hz -> 22.74
# 100.000 Hz -> 30.68
# 1.000 Hz -> 38.62
# 0.001 Hz -> 50.53

# 8 Channel PA: 4x sampling, fft upsampling, 16 ns window, 21 beams
# Linear fit: -0.134 x + 1.295
# Quadratic fit: -0.0007308 x x - 0.06398 x - 0.2857
# Noise rate -> threshold
#
# 10000.000 Hz -> 46.98
# 100.000 Hz -> 61.90
# 1.000 Hz -> 76.83
# 0.001 Hz -> 99.22

thresholds_pa = {}
thresholds_pa['8ch'] = {}
thresholds_pa['8ch']['100Hz'] = 61.90
thresholds_pa['8ch']['1Hz'] = 76.83
thresholds_pa['8ch']['1mHz'] = 99.22

thresholds_pa['4ch'] = {}
thresholds_pa['4ch']['100Hz'] = 30.68
thresholds_pa['4ch']['1Hz'] = 38.62
thresholds_pa['4ch']['1mHz'] = 50.53 

# initialize detector sim modules
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
phasedArrayTrigger = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
channelAddCableDelay = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()

triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
triggerTimeAdjuster.begin(pre_trigger_time=200*units.ns)

# DEEP part
# assuming that PA consists out of 8 antennas (channel 0-7)
main_low_angle = np.deg2rad(-59.54968597864437)
main_high_angle = np.deg2rad(59.54968597864437)
phasing_angles_4ant = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))
phasing_angles_8ant = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 21))

shallow_channels = [0, 1, 2, 3]
PA_8ch_channels = [4, 5, 6, 7, 8, 9, 10, 11]
PA_4ch_channels = [8, 9, 10, 11]

passband_low = {}
passband_low_trigger = {}
passband_high = {}
filter_type = {}
order_low = {}
order_high = {}
for channel_id in PA_8ch_channels:
    passband_low[channel_id] = [0 * units.MHz, 1000 * units.MHz]
    passband_low_trigger[channel_id] = [0 * units.MHz, 220 * units.MHz]
    passband_high[channel_id] = [96 * units.MHz, 100 * units.GHz]
    filter_type[channel_id] = 'cheby1'
    order_low[channel_id] = 7
    order_high[channel_id] = 4

thresholds = {
    '2/4_100Hz': 3.9498194908011524,
    '2/4_10mHz': 4.919151494949084,
    'fhigh': 0.15,
    'flow': 0.08}
for channel_id in shallow_channels:
    passband_low[channel_id] = [1 * units.MHz, 1000 * units.MHz]
    passband_low_trigger[channel_id] = [1 * units.MHz, thresholds['fhigh']]
    passband_high[channel_id] = [thresholds['flow'], 800 * units.GHz]
    filter_type[channel_id] = 'butter'
    order_low[channel_id] = 10
    order_high[channel_id] = 5


class TDR_Simulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        # here we apply a larger bandpass filter up to 1GHz
        channelBandPassFilter.run(evt, station, det,
                                  passband=passband_low, filter_type=filter_type, order=order_low, rp=0.1)
        channelBandPassFilter.run(evt, station, det,
                                  passband=passband_high, filter_type=filter_type, order=order_high, rp=0.1)

    def _detector_simulation_trigger(self, evt, station, det):
        # the trigger is calculated on bandpass limited signals, proceduce
        # 1) creat a copy of the station object
        # 2) apply additional lowpass filter
        # 3) calculate new noise RMS for filtered signals
        # 4) calculate trigger
        # 5) set trigger attributes of original station

        # 1) creat a copy of the station object
#        station = copy.deepcopy(station)

        # 2) apply additional lowpass filter
#        channelBandPassFilter.run(evt, station, det,
        channelBandPassFilter.run(evt, station, det,
                                passband=passband_low_trigger, filter_type=filter_type, order=order_low, rp=0.1)

        # 3) calculate new noise RMS for filtered signals
        Vrms_per_channel_copy = copy.deepcopy(self._Vrms_per_channel)

        ff = np.linspace(0, 1 * units.GHz, 10000)
        
        for channel_id in range(station.get_number_of_channels()):
            filt = channelBandPassFilter.get_filter(ff, station.get_id(), channel_id, det,
                                                    passband=passband_low_trigger, filter_type=filter_type, order=order_low, rp=0.1)
            filt *= channelBandPassFilter.get_filter(ff, station.get_id(), channel_id, det,
                                                    passband=passband_high, filter_type=filter_type, order=order_high, rp=0.1)
            filt *= channelBandPassFilter.get_filter(ff, station.get_id(), channel_id, det,
                                                    passband=passband_low, filter_type=filter_type, order=order_low, rp=0.1)
            bandwidth = np.trapz(np.abs(filt) ** 2, ff)
            # the Vrms scales with the squareroot of the bandwidth
            Vrms_per_channel_copy[station.get_id()][channel_id] *= (bandwidth / self._bandwidth_per_channel[station.get_id()][channel_id]) ** 0.5
            if 1:
                print(f"channel {channel_id}: bandwidth = {bandwidth/units.MHz:.1f}MHz, new Vrms = {Vrms_per_channel_copy[station.get_id()][channel_id]/units.micro/units.V:.4g}muV")
                tvrms = np.std(station.get_channel(channel_id).get_trace())
                print(f"\trealized Vrms = {tvrms/units.micro/units.V:.4g}muV")
                quit()

        
        
        # 4) calculate trigger
        # start with the SHALLOW TRIGGER which is always there
        # run a high/low trigger on the 4 downward pointing LPDAs
        threshold_high = {}
        threshold_low = {}
        for channel_id in det.get_channel_ids(station.get_id()):
            threshold_high[channel_id] = thresholds['2/4_100Hz'] * Vrms_per_channel_copy[station.get_id()][channel_id]
            threshold_low[channel_id] = -thresholds['2/4_100Hz'] * Vrms_per_channel_copy[station.get_id()][channel_id]
        highLowThreshold.run(evt, station, det,
                                    threshold_high=threshold_high,
                                    threshold_low=threshold_low,
                                    coinc_window=40 * units.ns,
                                    triggered_channels=shallow_channels,  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_100Hz')

        threshold_high = {}
        threshold_low = {}
        for channel_id in det.get_channel_ids(station.get_id()):
            threshold_high[channel_id] = thresholds['2/4_10mHz'] * Vrms_per_channel_copy[station.get_id()][channel_id]
            threshold_low[channel_id] = -thresholds['2/4_10mHz'] * Vrms_per_channel_copy[station.get_id()][channel_id]
        highLowThreshold.run(evt, station, det,
                                    threshold_high=threshold_high,
                                    threshold_low=threshold_low,
                                    coinc_window=40 * units.ns,
                                    triggered_channels=shallow_channels,  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_10mHz',
                                    set_not_triggered=not station.has_triggered(trigger_name='LPDA_2of4_100Hz')) # only calculate the trigger at a lower threshold if the previous one triggered

        # DEEP TRIGGER
        # check if the station is a hybrid station
        if(station.get_number_of_channels() > 5):
            # get the Vrms of the phased array channels
            Vrms_PA = Vrms_per_channel_copy[station.get_id()][PA_4ch_channels[0]]
            det_channel = det.get_channel(station.get_id(), PA_4ch_channels[0])
            sampling_rate_phased_array = det_channel["trigger_adc_sampling_frequency"]  # the phased array is digitized with a smaller sampling rate
            # run the 8 phased trigger
            # x4 for upsampling
            window_8ant = int(16 * units.ns * sampling_rate_phased_array * 4.0)
            step_8ant = int(8 * units.ns * sampling_rate_phased_array * 4.0)

            # run the 4 phased trigger
            # x2 for upsampling
            window_4ant = int(16 * units.ns * sampling_rate_phased_array * 2.0)
            step_4ant = int(8 * units.ns * sampling_rate_phased_array * 2.0)
            

            phasedArrayTrigger.run(evt, station, det,
                                   Vrms=Vrms_PA,
                                   threshold=thresholds_pa['8ch']['100Hz'] * np.power(Vrms_PA, 2.0),  # see phased trigger module for explanation
                                   triggered_channels=PA_8ch_channels,
                                   phasing_angles=phasing_angles_8ant,
                                   ref_index=1.75,
                                   trigger_name=f'PA_8channel_100Hz',  # the name of the trigger
                                   trigger_adc=True,  # Don't have a seperate ADC for the trigger
                                   adc_output=f'voltage',  # output in volts
                                   trigger_filter=None,
                                   upsampling_factor=4,
                                   window=window_8ant,
                                   step=step_8ant)

            phasedArrayTrigger.run(evt, station, det,
                                   Vrms=Vrms_PA,
                                   threshold=thresholds_pa['4ch']['100Hz'] * np.power(Vrms_PA, 2.0),
                                   triggered_channels=PA_4ch_channels,
                                   phasing_angles=phasing_angles_4ant,
                                   ref_index=1.75,
                                   trigger_name=f'PA_4channel_100Hz',  # the name of the trigger
                                   trigger_adc=True,  # Don't have a seperate ADC for the trigger
                                   adc_output=f'voltage',  # output in volts
                                   trigger_filter=None,
                                   upsampling_factor=2,
                                   window=window_4ant,
                                   step=step_4ant)
            
            phasedArrayTrigger.run(evt, station, det,
                                   Vrms=Vrms_PA,
                                   threshold=thresholds_pa['8ch']['1mHz'] * np.power(Vrms_PA, 2.0),  # see phased trigger module for explanation
                                   triggered_channels=PA_8ch_channels,
                                   phasing_angles=phasing_angles_8ant,
                                   ref_index=1.75,
                                   trigger_name=f'PA_8channel_1mHz',  # the name of the trigger
                                   trigger_adc=True,  # Don't have a seperate ADC for the trigger
                                   adc_output=f'voltage',  # output in volts
                                   trigger_filter=None,
                                   upsampling_factor=4,
                                   window=window_8ant,
                                   step=step_8ant)

            phasedArrayTrigger.run(evt, station, det,
                                   Vrms=Vrms_PA,
                                   threshold=thresholds_pa['4ch']['1mHz'] * np.power(Vrms_PA, 2.0),
                                   triggered_channels=PA_4ch_channels,
                                   phasing_angles=phasing_angles_4ant,
                                   ref_index=1.75,
                                   trigger_name=f'PA_4channel_1mHz',  # the name of the trigger
                                   trigger_adc=True,  # Don't have a seperate ADC for the trigger
                                   adc_output=f'voltage',  # output in volts
                                   trigger_filter=None,
                                   upsampling_factor=2,
                                   window=window_4ant,
                                   step=step_4ant)

        # if(station.has_triggered()):
        #     print("TRIGGERED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # 5) set trigger attributes of original station
        for trigger in station.get_triggers().values():
            station.set_trigger(trigger)
            
        # this module cuts the trace to the record length of the detector
        triggerTimeAdjuster.run(evt, station, det)

        #I think following line causes issues with there being a trigger names, but no 'multiple_triggers' key
#        if not 'trigger_names' in self._mout_attrs:
            # Might have some files which never ever trigger on a hybrid station and would consequently produce output trigger structures (nevents, 2) instead of (nevents, 6)
            # and only the LPDA trigger names. Merging and Veff calculation is then complicated.
            # This hack makes sure all triggers are written to the output hdf5. CAVEAT: make sure to adjust if trigger names above are changed!!!
#            self._mout_attrs['trigger_names'] = ['LPDA_2of4_100Hz', 'LPDA_2of4_10mHz', 'PA_8channel_100Hz', 'PA_4channel_100Hz', 'PA_8channel_1mHz', 'PA_4channel_1mHz']

parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
#parser.add_argument('inputfilename', type=str,
#                    help='path to NuRadioMC input event list')
parser.add_argument('detectordescription', type=str,
                    help='path to file containing the detector description')
parser.add_argument('config', type=str,
                    help='NuRadioMC yaml config file')
parser.add_argument('outputfilename', type=str,
                    help='hdf5 output filename')
parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
parser.add_argument('n_throws', type=int, help='Number of showers to simulate')
parser.add_argument('shower_energy', type=float, help='Shower energy to simulate in log10eV')
parser.add_argument('shower_energy_high', type=float, help='High end of shower energy to simulate, default 0 means only low energy')

parser.add_argument('zen_low', type=float, help='Low zen of bin to sim')
parser.add_argument('zen_high', type=float, help='High zen of bin to sim')

parser.add_argument('location', type=str, help='MB or SP, sets reflective layer depth, dB, and refraction index')
parser.add_argument('depth', type=float, help='Depth of reflective layer')
parser.add_argument('reflection', type=float, help='Reflection coeff in dB')

parser.add_argument('spacing', type=float, help='Spacing of detectors to use')
parser.add_argument('part', type=int, help='Part number to use')
args = parser.parse_args()

R = args.reflection
depth = args.depth

if args.location == 'SP':
    ice = 1.78
    z0 = 77. * units.m
    d_n = 0.423
elif args.location == 'MB':
    depth = 576
    R = 1
    ice = 1.78
    z0 = 37*units.m
    d_n = 0.481
elif args.location == 'GL':
    depth = 3000
    R = 0
    ice = 1.78
    z0 = 37.25 * units.m
    d_n = 0.51
else:
    print(f'{args.location} is not a usable location, use MB or SP')
    quit()

input_file = generate_events_square(n_throws = args.n_throws, min_x = -args.spacing/2, min_y = -args.spacing/2,
                                    max_x = args.spacing/2, max_y = args.spacing/2 , min_z = 5,
                                    shower_energy = args.shower_energy, shower_energy_high = args.shower_energy_high, 
                                    zen_low=args.zen_low, zen_high=args.zen_high, seed=None, depositEnergy=True, part=args.part)


class ice_model_reflection(medium_base.IceModelSimple):
    def __init__(self):
        # from https://doi.org/10.1088/1475-7516/2018/07/055 MB1 model
###     Below is code for mooresbay simple
        # from https://doi.org/10.1088/1475-7516/2018/07/055 MB1 model
        super().__init__(
            n_ice = ice, 
            z_0 = z0, 
            delta_n = d_n,
            )

        # from https://doi.org/10.3189/2015JoG14J214
        self.add_reflective_bottom( 
            refl_z = -depth*units.m, 
            refl_coef = R, 
            refl_phase_shift = 180*units.deg,
            )


ice_model = ice_model_reflection()

print(f'Refl Coeff {ice_model.reflection_coefficient}')
print(f'refl depth {ice_model.reflection}')


sim = TDR_Simulation(inputfilename=input_file,
                            outputfilename=args.outputfilename,
                            detectorfile=args.detectordescription,
                            outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                            config_file=args.config,
                            default_detector_station=1001,
                            file_overwrite=True,
                            ice_model=ice_model,
                            log_level=logging.WARNING)
sim.run()