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
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()

###Currently code runs a trigger on LPDA w/2of4 logic at 100Hz threshold
###Also runs a deep dipole at 200m to compare to previous simulations



passband_low = {}
passband_high = {}
filter_type = {}
order_low = {}
order_high = {}

#1st antenna is Gen2
passband_low[0] = [1 * units.MHz, 220 * units.MHz]
passband_high[0] = [96 * units.MHz, 1000 * units.GHz]
filter_type[0] = 'cheby1'
order_low[0] = 7
order_high[0] = 4

#2nd antenna is RNOG
passband_low[1] = [1 * units.MHz, 300 * units.MHz]
passband_high[1] = [80 * units.MHz, 1000 * units.GHz]
filter_type[1] = 'cheby1'
order_low[1] = 7
order_high[1] = 4

#3rd antenna is ARA
passband_low[2] = [1 * units.MHz, 500 * units.MHz]
passband_high[2] = [80 * units.MHz, 1000 * units.GHz]
filter_type[2] = 'cheby1'
order_low[2] = 7
order_high[2] = 4

#Test 1, 80-500 was old
passband_low[3] = [1 * units.MHz, 500 * units.MHz]
passband_high[3] = [80 * units.MHz, 1000 * units.GHz]
filter_type[3] = 'cheby1'
order_low[3] = 7
order_high[3] = 4

#Test 2, 80-250 was old
passband_low[4] = [1 * units.MHz, 250 * units.MHz]
passband_high[4] = [80 * units.MHz, 1000 * units.GHz]
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

        highLowThreshold.run(evt, station, det,
                                    threshold_high = 2 * self._Vrms_per_channel[station.get_id()][0],
                                    threshold_low = -2 * self._Vrms_per_channel[station.get_id()][0],
                                    triggered_channels = [0],
                                    number_concidences = 1,
                                    trigger_name = 'Gen2_Dipole_2sig')
        highLowThreshold.run(evt, station, det,
                                    threshold_high = 3 * self._Vrms_per_channel[station.get_id()][0],
                                    threshold_low = -3 * self._Vrms_per_channel[station.get_id()][0],
                                    triggered_channels = [0],
                                    number_concidences = 1,
                                    trigger_name = 'Gen2_Dipole_3sig')

        highLowThreshold.run(evt, station, det,
                                    threshold_high = 3 * self._Vrms_per_channel[station.get_id()][1],
                                    threshold_low = -3 * self._Vrms_per_channel[station.get_id()][1],
                                    triggered_channels = [1],
                                    number_concidences = 1,
                                    trigger_name = 'RNOG_3sig')

        highLowThreshold.run(evt, station, det,
                                    threshold_high = 2 * self._Vrms_per_channel[station.get_id()][2],
                                    threshold_low = -2 * self._Vrms_per_channel[station.get_id()][2],
                                    triggered_channels = [2],
                                    number_concidences = 1,
                                    trigger_name = 'ARA_2sig')

        highLowThreshold.run(evt, station, det,
                                    threshold_high = 2 * self._Vrms_per_channel[station.get_id()][3],
                                    threshold_low = -2 * self._Vrms_per_channel[station.get_id()][3],
                                    triggered_channels = [3],
                                    number_concidences = 1,
                                    trigger_name = 'Old_80to500_HighLow_2sig')

        simpleThreshold.run(evt, station, det,
                                    threshold = 2 * self._Vrms_per_channel[station.get_id()][3],
                                    triggered_channels = [3],
                                    number_concidences = 1,
                                    trigger_name = 'Old_80to500_Simple_2sig')

        highLowThreshold.run(evt, station, det,
                                    threshold_high = 2 * self._Vrms_per_channel[station.get_id()][4],
                                    threshold_low = -2 * self._Vrms_per_channel[station.get_id()][4],
                                    triggered_channels = [4],
                                    number_concidences = 1,
                                    trigger_name = 'Old_80to250_HighLow_2sig')

        simpleThreshold.run(evt, station, det,
                                    threshold = 2 * self._Vrms_per_channel[station.get_id()][4],
                                    triggered_channels = [4],
                                    number_concidences = 1,
                                    trigger_name = 'Old_80to250_Simple_2sig')




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

#input_file = generate_events(n_throws=args.n_throws, shower_energy=args.shower_energy)

R = args.reflection
depth = args.depth

if args.location == 'SP':
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
