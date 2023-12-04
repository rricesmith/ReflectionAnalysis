from __future__ import absolute_import, division, print_function
import argparse
# import detector simulation modules
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelAntennaDedispersion
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")

# initialize detector sim modules
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelAntennaDedispersion = NuRadioReco.modules.channelAntennaDedispersion.channelAntennaDedispersion()
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()

class mySimulation(simulation.simulation):

    #400s type trigger config
    def _detector_simulation_filter_amp(self, evt, station, det):
#        channelBandPassFilter.run(evt, station, det, passband=[100 * units.MHz, 1000 * units.MHz], #100s
        channelBandPassFilter.run(evt, station, det, passband=[50 * units.MHz, 1000 * units.MHz], #200s
                                  filter_type='butter', order=2)

    def _detector_simulation_trigger(self, evt, station, det):

        #In sim.py, noise is added before the filter_amp, so we run it here before trigger instead
        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)

        # first run a simple threshold trigger to speed up logic
        simpleThreshold.run(evt, station, det,
                             threshold=2 * self._Vrms,
                             triggered_channels=None,  # run trigger on all channels
                             number_concidences=1,
                             trigger_name='simple_threshold')  # the name of the trigger



        # run a high/low trigger on the 4 downward pointing LPDAs
        highLowThreshold.run(evt, station, det,
                                    threshold_high=4.4 * self._Vrms,
                                    threshold_low=-4.4 * self._Vrms,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_4.4sigma',
                                    set_not_triggered=(not station.has_triggered("simple_threshold")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger

        #Add noise after triggering
        if station.has_triggered(trigger_name="LPDA_2of4_4.4sigma"):
            hardwareResponseIncorporator.run(evt, station, det, sim_to_data=False)
            max_freq = 0.5 / self._dt
            norm = self._bandwidth_per_channel[(self._station.get_id())][0]
            Vrms = self._Vrms / (norm / max_freq) ** 0.5
            print(f'Vrms used is {Vrms}')
            quit()
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=Vrms, min_freq=0 * units.MHz,
                                         max_freq=max_freq, type='rayleigh')
            hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)


parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC input event list')
parser.add_argument('detectordescription', type=str,
                    help='path to file containing the detector description')
parser.add_argument('config', type=str,
                    help='NuRadioMC yaml config file')
parser.add_argument('outputfilename', type=str,
                    help='hdf5 output filename')
parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
parser.add_argument('--part', type=str, default = 'None')
args = parser.parse_args()

if __name__ == "__main__":
    if not args.part == 'None':
        args.inputfilename += '.part' + args.part
        args.outputfilename = args.outputfilename + '.part' + args.part
        args.outputfilenameNuRadioReco = args.outputfilenameNuRadioReco + '.part' + args.part
#        print(f'using input {args.inputfilename} and output {args.outputfilename}')
    sim = mySimulation(inputfilename=args.inputfilename,
                                outputfilename=args.outputfilename,
                                detectorfile=args.detectordescription,
                                outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                                config_file=args.config,
                                file_overwrite=True)
    sim.run()
