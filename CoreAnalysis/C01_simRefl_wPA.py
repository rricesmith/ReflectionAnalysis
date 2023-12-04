import argparse
# import detector simulation modules
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.triggerTimeAdjuster
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
from produceEvents import generate_events
from produceEvents import generate_events_square
from radiotools import helper as hp
from NuRadioMC.utilities import medium, medium_base
import numpy as np
import NuRadioReco.modules.phasedarray.triggerSimulator

# initialize detector sim modules
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
phasedArrayTrigger = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
triggerTimeAdjuster.begin(pre_trigger_time=200*units.ns)


# assuming that PA consists out of 8 antennas (channel 0-7)
main_low_angle = np.deg2rad(-59.54968597864437)
main_high_angle = np.deg2rad(59.54968597864437)
phasing_angles_4ant = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 11))
phasing_angles_8ant = np.arcsin(np.linspace(np.sin(main_low_angle), np.sin(main_high_angle), 21))

shallow_channels = [0, 1, 2, 3]
#dipole_channels = []
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
    '2/4_2sigma': 2.0,
    'fhigh': 0.15,
    'flow': 0.08}


thresholds_pa = {}
thresholds_pa['4ch'] = {}
thresholds_pa['4ch']['100Hz'] = 30.68
thresholds_pa['4ch']['1Hz'] = 38.62
thresholds_pa['4ch']['1mHz'] = 50.53 

for channel_id in shallow_channels:
    passband_low[channel_id] = [1 * units.MHz, 1000 * units.MHz]
    passband_low_trigger[channel_id] = [1 * units.MHz, thresholds['fhigh']]
    passband_high[channel_id] = [thresholds['flow'], 800 * units.GHz]
    filter_type[channel_id] = 'butter'
    order_low[channel_id] = 10
    order_high[channel_id] = 5

class mySimulation(simulation.simulation):

    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(evt, station, det,
                                    passband=passband_low, filter_type=filter_type, order=order_low, rp=0.1)
        channelBandPassFilter.run(evt, station, det,
                                    passband=passband_high, filter_type=filter_type, order=order_high, rp=0.1)
        """ Old Method
		channelBandPassFilter.run(evt, station, det,
								  passband=[80 * units.MHz, 1000 * units.GHz], filter_type="butter", order=2)
		channelBandPassFilter.run(evt, station, det,
								  passband=[0, 250 * units.MHz], filter_type="butter", order=10)
#								  passband=[0, 500 * units.MHz], filter_type="butter", order=10)
        

    def _detector_simulation_trigger(self, evt, station, det):
#		simpleThreshold.run(evt, station, det,
#					threshold = 2 * self._Vrms,
#					triggered_channels = [0],
#					number_concidences=1,
#					trigger_name='simple_threshold')
                threshold_high = {}
                threshold_low = {}
                for channel_id in det.get_channel_ids(station.get_id()):
#                    threshold_high[channel_id] = 1 * self._Vrms_per_channel[station.get_id()][channel_id]
#                    threshold_low[channel_id] = -1 * self._Vrms_per_channel[station.get_id()][channel_id]
                    threshold_high[channel_id] = 4.2 * thresh
                    threshold_low[channel_id] = -4.2 * thresh



                highLowThreshold.run(evt, station, det, threshold_high = threshold_high, threshold_low = threshold_low, coinc_window=40 * units.ns, 
                                                                triggered_channels = [0, 1, 2, 3], #select LPDA channels only
                                                                number_concidences = 2, #2/4 majority logic
                                                                trigger_name='LPDA_2of4_3.5sigma')

#                print(f'trigger dipole {2*self._Vrms_per_channel[station.get_id()][12]}')
#                quit()
                simpleThreshold.run(evt, station, det,
                                             threshold = 2.0 * thresh,
#                                             threshold=2.0 * self._Vrms_per_channel[station.get_id()][12],
#                                             threshold=3.0 * self._Vrms_per_channel[station.get_id()][12],
                                             triggered_channels=[12],
                                             number_concidences=1,
                                             trigger_name='dipole_2.0sigma')

                #PA trigger based off gen2 design
                Vrms = self._Vrms_per_channel[station.get_id()][8]
                window_8ant = int(16 * units.ns * self._sampling_rate_detector * 4.0)
                step_8ant = int(8 * units.ns * self._sampling_rate_detector * 4.0)

                phasedArrayTrigger.run(evt, station, det,
                                       Vrms=Vrms,
                                       threshold = 62.15 * np.power(Vrms, 2.0),
                                       triggered_channels=range(4, 12),
                                       phasing_angles=phasing_angles_8ant,
                                       ref_index=1.75,
                                       trigger_name='PA_8channel_100Hz',
                                       trigger_adc=False,
                                       adc_output='voltage',
                                       trigger_filter=None,
                                       upsampling_factor=4,
                                       window=window_8ant,
                                       step=step_8ant)

                window_4ant = int(16 * units.ns * self._sampling_rate_detector * 2.0)
                step_4ant = int(8 * units.ns * self._sampling_rate_detector * 2.0)

                phasedArrayTrigger.run(evt, station, det,
                                       Vrms=Vrms,
                                       threshold = 30.85 * np.power(Vrms, 2.0),
                                       triggered_channels=range(6, 10),
                                       phasing_angles=phasing_angles_4ant,
                                       ref_index=1.75,
                                       trigger_name='PA_4channel_100Hz',
                                       trigger_adc=False,
                                       adc_output='voltage',
                                       trigger_filter=None,
                                       upsampling_factor=2,
                                       window=window_4ant,
                                       step=step_4ant)
            """

        #New method from https://github.com/nu-radio/analysis-scripts/blob/gen2-tdr-2021/gen2-tdr-2021/detsim/D01detector_sim.py
    def _detector_simulation_trigger(self, evt, station, det):
        # the trigger is calculated on bandpass limited signals, proceduce
        # 1) creat a copy of the station object
        # 2) apply additional lowpass filter
        # 3) calculate new noise RMS for filtered signals
        # 4) calculate trigger
        # 5) set trigger attributes of original station

        # 1) creat a copy of the station object
        station_copy = copy.deepcopy(station)

        # 2) apply additional lowpass filter
        channelBandPassFilter.run(evt, station_copy, det,
                                passband=passband_low_trigger, filter_type=filter_type, order=order_low, rp=0.1)

        # 3) calculate new noise RMS for filtered signals
        Vrms_per_channel_copy = copy.deepcopy(self._Vrms_per_channel)

        ff = np.linspace(0, 1 * units.GHz, 10000)
        
        for channel_id in range(station_copy.get_number_of_channels()):
            filt = channelBandPassFilter.get_filter(ff, station_copy.get_id(), channel_id, det,
                                                    passband=passband_low_trigger, filter_type=filter_type, order=order_low, rp=0.1)
            filt *= channelBandPassFilter.get_filter(ff, station_copy.get_id(), channel_id, det,
                                                    passband=passband_high, filter_type=filter_type, order=order_high, rp=0.1)
            filt *= channelBandPassFilter.get_filter(ff, station_copy.get_id(), channel_id, det,
                                                    passband=passband_low, filter_type=filter_type, order=order_low, rp=0.1)
            bandwidth = np.trapz(np.abs(filt) ** 2, ff)
            # the Vrms scales with the squareroot of the bandwidth
            Vrms_per_channel_copy[station_copy.get_id()][channel_id] *= (bandwidth / self._bandwidth_per_channel[station_copy.get_id()][channel_id]) ** 0.5
            if 0:
                print(f"channel {channel_id}: bandwidth = {bandwidth/units.MHz:.1f}MHz, new Vrms = {Vrms_per_channel_copy[station.get_id()][channel_id]/units.micro/units.V:.4g}muV")
                tvrms = np.std(station_copy.get_channel(channel_id).get_trace())
                print(f"\trealized Vrms = {tvrms/units.micro/units.V:.4g}muV")

        
        
        # 4) calculate trigger
        # start with the SHALLOW TRIGGER which is always there
        # run a high/low trigger on the 4 downward pointing LPDAs
        threshold_high = {}
        threshold_low = {}
        for channel_id in det.get_channel_ids(station_copy.get_id()):
            threshold_high[channel_id] = thresholds['2/4_100Hz'] * Vrms_per_channel_copy[station_copy.get_id()][channel_id]
            threshold_low[channel_id] = -thresholds['2/4_100Hz'] * Vrms_per_channel_copy[station_copy.get_id()][channel_id]
        highLowThreshold.run(evt, station_copy, det,
                                    threshold_high=threshold_high,
                                    threshold_low=threshold_low,
                                    coinc_window=40 * units.ns,
                                    triggered_channels=shallow_channels,  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_100Hz')

        threshold_high = {}
        threshold_low = {}
        for channel_id in det.get_channel_ids(station_copy.get_id()):
            threshold_high[channel_id] = thresholds['2/4_10mHz'] * Vrms_per_channel_copy[station_copy.get_id()][channel_id]
            threshold_low[channel_id] = -thresholds['2/4_10mHz'] * Vrms_per_channel_copy[station_copy.get_id()][channel_id]
        highLowThreshold.run(evt, station_copy, det,
                                    threshold_high=threshold_high,
                                    threshold_low=threshold_low,
                                    coinc_window=40 * units.ns,
                                    triggered_channels=shallow_channels,  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_10mHz',
                                    set_not_triggered=not station_copy.has_triggered(trigger_name='LPDA_2of4_100Hz')) # only calculate the trigger at a lower threshold if the previous one triggered

        threshold_high = {}
        threshold_low = {}
        for channel_id in det.get_channel_ids(station_copy.get_id()):
            threshold_high[channel_id] = thresholds['2/4_2sigma'] * Vrms_per_channel_copy[station_copy.get_id()][channel_id]
            threshold_low[channel_id] = -thresholds['2/4_2sigma'] * Vrms_per_channel_copy[station_copy.get_id()][channel_id]
        highLowThreshold.run(evt, station_copy, det,
                                    threshold_high=threshold_high,
                                    threshold_low=threshold_low,
                                    coinc_window=40 * units.ns,
                                    triggered_channels=shallow_channels,  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_2sigma',
                                    set_not_triggered=not station_copy.has_triggered(trigger_name='LPDA_2of4_100Hz')) # only calculate the trigger at a lower threshold if the previous one triggered

        # DEEP TRIGGER
        # check if the station is a hybrid station
        if(station_copy.get_number_of_channels() > 5):
            # get the Vrms of the phased array channels
            Vrms_PA = Vrms_per_channel_copy[station_copy.get_id()][PA_4ch_channels[0]]
            det_channel = det.get_channel(station_copy.get_id(), PA_4ch_channels[0])
            sampling_rate_phased_array = det_channel["trigger_adc_sampling_frequency"]  # the phased array is digitized with a smaller sampling rate
            # run the 8 phased trigger
            # x4 for upsampling
            window_8ant = int(16 * units.ns * sampling_rate_phased_array * 4.0)
            step_8ant = int(8 * units.ns * sampling_rate_phased_array * 4.0)

            # run the 4 phased trigger
            # x2 for upsampling
            window_4ant = int(16 * units.ns * sampling_rate_phased_array * 2.0)
            step_4ant = int(8 * units.ns * sampling_rate_phased_array * 2.0)
            

            #Not running 8-ch PA for now
#            phasedArrayTrigger.run(evt, station_copy, det,
#                                   Vrms=Vrms_PA,
#                                   threshold=thresholds_pa['8ch']['100Hz'] * np.power(Vrms_PA, 2.0),  # see phased trigger module for explanation
#                                   triggered_channels=PA_8ch_channels,
#                                   phasing_angles=phasing_angles_8ant,
#                                   ref_index=1.75,
#                                   trigger_name=f'PA_8channel_100Hz',  # the name of the trigger
#                                   trigger_adc=True,  # Don't have a seperate ADC for the trigger
#                                   adc_output=f'voltage',  # output in volts
#                                   trigger_filter=None,
#                                   upsampling_factor=4,
#                                   window=window_8ant,
#                                   step=step_8ant)

            phasedArrayTrigger.run(evt, station_copy, det,
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
            
#            phasedArrayTrigger.run(evt, station_copy, det,
#                                   Vrms=Vrms_PA,
#                                   threshold=thresholds_pa['8ch']['1mHz'] * np.power(Vrms_PA, 2.0),  # see phased trigger module for explanation
#                                   triggered_channels=PA_8ch_channels,
#                                   phasing_angles=phasing_angles_8ant,
#                                   ref_index=1.75,
#                                   trigger_name=f'PA_8channel_1mHz',  # the name of the trigger
#                                   trigger_adc=True,  # Don't have a seperate ADC for the trigger
#                                   adc_output=f'voltage',  # output in volts
#                                   trigger_filter=None,
#                                   upsampling_factor=4,
#                                   window=window_8ant,
#                                   step=step_8ant)

            phasedArrayTrigger.run(evt, station_copy, det,
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

        # if(station_copy.has_triggered()):
        #     print("TRIGGERED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # 5) set trigger attributes of original station
        for trigger in station_copy.get_triggers().values():
            station.set_trigger(trigger)
            
        # this module cuts the trace to the record length of the detector
        triggerTimeAdjuster.run(evt, station, det)

        if not 'trigger_names' in self._mout_attrs:
            # Might have some files which never ever trigger on a hybrid station and would consequently produce output trigger structures (nevents, 2) instead of (nevents, 6)
            # and only the LPDA trigger names. Merging and Veff calculation is then complicated.
            # This hack makes sure all triggers are written to the output hdf5. CAVEAT: make sure to adjust if trigger names above are changed!!!
            self._mout_attrs['trigger_names'] = ['LPDA_2of4_100Hz', 'LPDA_2of4_10mHz', 'LPDA_2of4_2sigma', 'PA_8channel_100Hz', 'PA_4channel_100Hz', 'PA_8channel_1mHz', 'PA_4channel_1mHz']





parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
#parser.add_argument('inputfilename', type=str,
#					help='path to NuRadioMC input event list')
parser.add_argument('detectordescription', type=str,
					help='path to file containing the detector description')
parser.add_argument('config', type=str,
					help='NuRadioMC yaml config file')
parser.add_argument('outputfilename', type=str,
					help='hdf5 output filename')
parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None,
					help='outputfilename of NuRadioReco detector sim file')
parser.add_argument('n_throws', type=int, help='Number of showers to simulate')
parser.add_argument('shower_energy', type=float, help='Shower energy to simulate')
parser.add_argument('shower_energy_high', type=float, help='High end of energy spectrum to sim, default 0 is just single energy')

parser.add_argument('zen_low', type=float, help='Low zen of bin to sim')
parser.add_argument('zen_high', type=float, help='High zen of bin to sim')

parser.add_argument('location', type=str, help='MB or SP, sets reflective layer depth, dB, and refraction index')
parser.add_argument('depth', type=float, help='Depth of reflective layer')
parser.add_argument('reflection', type=float, help='Reflection coeff in dB')

parser.add_argument('spacing', type=float, help='Length of side of square over which to throw cores')
args = parser.parse_args()

R = args.reflection
depth = args.depth

if args.location == 'SP':
#    depth = -300 * units.m
#    depth = -500 * units.m
#    depth = -800 * units.m
#    depth = -1000 * units.m
#    depth = -1170 * units.m
#    R = 0.01
    ice = 1.78
    z0 = 77. * units.m
    d_n = 0.423
elif args.location == 'MB':
    depth = 576 * units.m
    R = 1
    ice = 1.78
    z0 = 37*units.m
    d_n = 0.481
elif args.location == 'GL':
    depth = 3000 * units.m
    R = 0
    ice = 1.78
    z0 = 37.25 * units.m
    d_n = 0.51
else:
    print(f'{args.location} is not a usable location, use MB or SP')
    quit()

#input_file = generate_events(n_throws=args.n_throws, max_rad = args.spacing, shower_energy=args.shower_energy, shower_energy_high=args.shower_energy_high, zen_low=args.zen_low, zen_high=args.zen_high, seed=None, depositEnergy=True)
input_file = generate_events_square(n_throws = args.n_throws, min_x = -args.spacing/2, min_y = -args.spacing/2,
                                    max_x = args.spacing/2, max_y = args.spacing/2 , min_z = 5,
                                    shower_energy = args.shower_energy, shower_energy_high = args.shower_energy_high, 
                                    zen_low=args.zen_low, zen_high=args.zen_high, seed=None, depositEnergy=True)

#refl_coef = hp.dB_to_linear(-args.reflection * 0.5) # *0.5 because the reflection is specified in power, but we need linear amplitude here
#refl_z = -1 * args.depth * units.m

class ice_model_reflection(medium_base.IceModelSimple):
    def __init__(self):
        # from https://doi.org/10.1088/1475-7516/2018/07/055 MB1 model
        super().__init__(
            n_ice = ice, 
            z_0 = z0, 
            delta_n = d_n,
            )

        # from https://doi.org/10.3189/2015JoG14J214
        self.add_reflective_bottom( 
            refl_z = -depth * units.m, 
            refl_coef = R, 
            refl_phase_shift = 180*units.deg,
            )
ice_model = ice_model_reflection()

print(ice_model.reflection_coefficient)

sim = mySimulation(inputfilename=input_file,
							outputfilename=args.outputfilename,
							detectorfile=args.detectordescription,
							outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
							config_file=args.config,
							default_detector_station=1,
							ice_model=ice_model)
sim.run()

# Example for running this script:
# python simulateNeutrinoEventDetection.py data/tenNeutrinosAt1e19.hdf5 configurations/ARIANNA_4LPDA_1dipole.json configurations/simulateNeutrinosConfig.yaml data/triggeredNeutrinoEvents.hdf5 data/triggeredNeutrinoEvents.nur 100000 18.0 500 40
