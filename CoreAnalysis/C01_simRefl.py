import argparse
# import detector simulation modules
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.triggerTimeAdjuster
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
from N00_produceEvents import generate_events
from N00_produceEvents import generate_events_square
from radiotools import helper as hp
from NuRadioMC.utilities import medium, medium_base

# initialize detector sim modules
#simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

###Currently code runs a trigger on LPDA w/2of4 logic at
###Also runs a deep dipole at 150m to compare to previous simulations

thresholds = {
  '2/4_2sigma': 2.0,
  'fhigh': 0.15,
  'flow': 0.08
  }

passband_low = {}
passband_high = {}
filter_type = {}
order_low = {}
order_high = {}
for channel_id in range(0, 4):
    passband_low[channel_id] = [1 * units.MHz, thresholds['fhigh']]
    passband_high[channel_id] = [thresholds['flow'], 800 * units.GHz]
    filter_type[channel_id] = 'butter'
    order_low[channel_id] = 10
    order_high[channel_id] = 5


passband_low[4] = [1 * units.MHz, 220 * units.MHz]      #PA BW according to latest design
passband_high[4] = [96 * units.MHz, 100 * units.GHz]    #Design found at https://github.com/nu-radio/analysis-scripts/blob/gen2-tdr-2021/gen2-tdr-2021/detsim/D01detector_sim.py
filter_type[4] = 'cheby1'
order_low[4] = 7
order_high[4] = 4

class mySimulation(simulation.simulation):



    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(evt, station, det,
                                  passband=passband_low, filter_type=filter_type, order=order_low, rp=0.1)
        channelBandPassFilter.run(evt, station, det,
                                  passband=passband_high, filter_type=filter_type, order=order_high, rp=0.1)

    def _detector_simulation_trigger(self, evt, station, det):
        #run a high/low trigger on the 4 downward pointing LPDAs
        threshold_high = {}
        threshold_low = {}
#        print(f'v dipole trig {2*self._Vrms_per_channel[station.get_id()][13]}')
        for channel_id in det.get_channel_ids(station.get_id()):
            threshold_high[channel_id] = thresholds['2/4_2sigma'] * self._Vrms_per_channel[station.get_id()][channel_id]
            threshold_low[channel_id] = -thresholds['2/4_2sigma'] * self._Vrms_per_channel[station.get_id()][channel_id]

        print('running trigger of lpda')
        highLowThreshold.run(evt, station, det,
                                    threshold_high=threshold_high,
                                    threshold_low=threshold_low,
                                    coinc_window = 40 * units.ns,
                                    triggered_channels=[0, 1, 2, 3], #select the LPDA channels
                                    number_concidences = 2, #2/4 majority logic
                                    trigger_name = 'LPDA_2of4_2sigma')

        for channel_id in det.get_channel_ids(station.get_id()):
            threshold_high[channel_id] = 3.8 * self._Vrms_per_channel[station.get_id()][channel_id]
            threshold_low[channel_id] = -3.8 * self._Vrms_per_channel[station.get_id()][channel_id]

        highLowThreshold.run(evt, station, det,
                                    threshold_high=threshold_high,
                                    threshold_low=threshold_low,
                                    coinc_window = 40 * units.ns,
                                    triggered_channels=[0, 1, 2, 3], #select the LPDA channels
                                    number_concidences = 2, #2/4 majority logic
                                    trigger_name = 'LPDA_2of4_3.8sigma')

        highLowThreshold.run(evt, station, det,
                                    threshold_high = 2 * self._Vrms_per_channel[station.get_id()][4],
                                    threshold_low = -2 * self._Vrms_per_channel[station.get_id()][4],
                                    triggered_channels = [4],
                                    number_concidences = 1,
                                    trigger_name = 'single_dipole_trigger_2sig')

        highLowThreshold.run(evt, station, det,
                                    threshold_high = 3 * self._Vrms_per_channel[station.get_id()][4],
                                    threshold_low = -3 * self._Vrms_per_channel[station.get_id()][4],
                                    triggered_channels = [4],
                                    number_concidences = 1,
                                    trigger_name = 'single_dipole_trigger_3sig')





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
parser.add_argument('shower_energy', type=float, help='Shower energy to simulate in log10eV')
parser.add_argument('shower_energy_high', type=float, help='High end of shower energy to simulate, default 0 means only low energy')

parser.add_argument('zen_low', type=float, help='Low zen of bin to sim')
parser.add_argument('zen_high', type=float, help='High zen of bin to sim')

parser.add_argument('location', type=str, help='MB or SP, sets reflective layer depth, dB, and refraction index')
parser.add_argument('depth', type=float, help='Depth of reflective layer')
parser.add_argument('reflection', type=float, help='Reflection coeff in dB')

parser.add_argument('spacing', type=float, help='Spacing of detectors to use')
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
                                    zen_low=args.zen_low, zen_high=args.zen_high, seed=None, depositEnergy=True)


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

sim = mySimulation(inputfilename=input_file,
							outputfilename=args.outputfilename,
							detectorfile=args.detectordescription,
							outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
							config_file=args.config,
							default_detector_station=1,
                            file_overwrite=True,        #Testing this, maybe want to make it dynamic in the future
							ice_model=ice_model)                            
sim.run()

# Example for running this script:
# python simulateNeutrinoEventDetection.py data/tenNeutrinosAt1e19.hdf5 configurations/ARIANNA_4LPDA_1dipole.json configurations/simulateNeutrinosConfig.yaml data/triggeredNeutrinoEvents.hdf5 data/triggeredNeutrinoEvents.nur 100000 18.0 500 40
