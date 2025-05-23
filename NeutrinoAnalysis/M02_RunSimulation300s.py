from __future__ import absolute_import, division, print_function
import argparse
# import detector simulation modules
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.trigger.simpleThreshold
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelAntennaDedispersion
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation
import logging
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import matplotlib.pyplot as plt
from icecream import ic
import numpy as np
from scipy.interpolate import interp1d
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")

# initialize detector sim modules
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelAntennaDedispersion = NuRadioReco.modules.channelAntennaDedispersion.channelAntennaDedispersion()
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()


class mySimulation(simulation.simulation):

    #400s type trigger config
    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(evt, station, det, passband=[1 * units.MHz, 1000 * units.MHz],
                                  filter_type='butter', order=10)
#        channelBandPassFilter.run(evt, station, det, passband=[1*units.MHz, 150 * units.MHz],
#        channelBandPassFilter.run(evt, station, det, passband=[1*units.MHz, 500 * units.MHz],
        # channelBandPassFilter.run(evt, station, det, passband=[1*units.MHz, 150 * units.MHz],
        #                           filter_type='butter', order=10)
        # channelBandPassFilter.run(evt, station, det, passband=[80*units.MHz, 800*units.GHz],
        #                           filter_type='butter', order=5)


    def _detector_simulation_trigger(self, evt, station, det):

        Vrms = 10*units.mV
        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)



        # first run a simple threshold trigger to speed up logic
        # simpleThreshold.run(evt, station, det,
        #                      threshold=1 * Vrms,
        #                      triggered_channels=None,  # run trigger on all channels
        #                      number_concidences=1,
        #                      trigger_name='simple_threshold')  # the name of the trigger


        # run a high/low trigger on the 4 downward pointing LPDAs
        for i in [1.5, 2, 2.5, 3]:
            highLowThreshold.run(evt, station, det,
                                    threshold_high=i * Vrms,
                                    threshold_low=-i * Vrms,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name=f'LPDA_2of4_{i}sigma')

        # highLowThreshold.run(evt, station, det,
        #                             threshold_high=3.8 * Vrms,
        #                             threshold_low=-3.8 * Vrms,
        #                             triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
        #                             number_concidences=2,  # 2/4 majority logic
        #                             trigger_name='LPDA_2of4_3.8sigma')
        #                             # set_not_triggered=(not station.has_triggered("simple_threshold")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger

        # # run a high/low trigger on the 4 downward pointing LPDAs
        # highLowThreshold.run(evt, station, det,
        #                             threshold_high=4.4 * Vrms,
        #                             threshold_low=-4.4 * Vrms,
        #                             triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
        #                             number_concidences=2,  # 2/4 majority logic
        #                             trigger_name='LPDA_2of4_4.4sigma')
        #                             # set_not_triggered=(not station.has_triggered("LPDA_2of4_3.8sigma")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger


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
# parser.add_argument('amp', type=str, nargs='?', default='300', help='300 or 400')
# parser.add_argument('noise', type=bool, nargs='?', default=False, help='True or False')
parser.add_argument('--part', type=str, default = 'None')
args = parser.parse_args()

if __name__ == "__main__":
    if not args.part == 'None':
        args.inputfilename += '.part' + args.part
        args.outputfilename = args.outputfilename + '.part' + args.part
        args.outputfilenameNuRadioReco = args.outputfilenameNuRadioReco + '.part' + args.part
#        print(f'using input {args.inputfilename} and output {args.outputfilename}')
#    plt.draw()
#    print(f'drawn')
    sim = mySimulation(inputfilename=args.inputfilename,
                                outputfilename=args.outputfilename,
                                detectorfile=args.detectordescription,
                                outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                                config_file=args.config,
                                file_overwrite=True)
    sim.run()
