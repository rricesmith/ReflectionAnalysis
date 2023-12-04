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


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")

# initialize detector sim modules
simpleThreshold = NuRadioReco.modules.trigger.simpleThreshold.triggerSimulator()
highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelAntennaDedispersion = NuRadioReco.modules.channelAntennaDedispersion.channelAntennaDedispersion()

set = 0
x = []
y = []
#plt.ion()
#fig = plt.figure()
#h1, =   plt.plot([], [])

class mySimulation(simulation.simulation):

    #400s type trigger config
    def _detector_simulation_filter_amp(self, evt, station, det):
        channelBandPassFilter.run(evt, station, det, passband=[1 * units.MHz, 1000 * units.MHz],
                                  filter_type='butter', order=10)
#        channelBandPassFilter.run(evt, station, det, passband=[1*units.MHz, 150 * units.MHz],
#        channelBandPassFilter.run(evt, station, det, passband=[1*units.MHz, 500 * units.MHz],
        channelBandPassFilter.run(evt, station, det, passband=[1*units.MHz, 150 * units.MHz],
                                  filter_type='butter', order=10)
        channelBandPassFilter.run(evt, station, det, passband=[80*units.MHz, 800*units.GHz],
                                  filter_type='butter', order=5)

    def _detector_simulation_trigger(self, evt, station, det):


        """
        # first run a simple threshold trigger to speed up logic
        simpleThreshold.run(evt, station, det,
                             threshold=1 * self._Vrms,
                             triggered_channels=None,  # run trigger on all channels
                             number_concidences=1,
                             trigger_name='simple_threshold')  # the name of the trigger



        # run a high/low trigger on the 4 downward pointing LPDAs
        highLowThreshold.run(evt, station, det,
                                    threshold_high=1 * self._Vrms,
                                    threshold_low=-1 * self._Vrms,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_1sigma',
                                    set_not_triggered=(not station.has_triggered("simple_threshold")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger

        highLowThreshold.run(evt, station, det,
                                    threshold_high=1.5 * self._Vrms,
                                    threshold_low=-1.5 * self._Vrms,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_1.5sigma',
                                    set_not_triggered=(not station.has_triggered("simple_threshold")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger

        highLowThreshold.run(evt, station, det,
                                    threshold_high=2 * self._Vrms,
                                    threshold_low=-2 * self._Vrms,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_2sigma',
                                    set_not_triggered=(not station.has_triggered("simple_threshold")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger

        highLowThreshold.run(evt, station, det,
                                    threshold_high=3 * self._Vrms,
                                    threshold_low=-3 * self._Vrms,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_3sigma',
                                    set_not_triggered=(not station.has_triggered("simple_threshold")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger

        highLowThreshold.run(evt, station, det,
                                    threshold_high=4 * self._Vrms,
                                    threshold_low=-4 * self._Vrms,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_4sigma',
                                    set_not_triggered=(not station.has_triggered("simple_threshold")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger
        
        """

        #Dedispersion with debug code
        if False:
            #Dedisperse lpda antenna and getting dedispersed 2sigma trigger
            ch1 = station.get_channel(0).get_trace()
            preP2P = max(ch1) - min(ch1)
            print(f'params {station.get_parameters()}')
            print(f'ch parsm {station.get_channel(0).get_parameters()}')
            print(f'ss params {station.get_sim_station().get_parameters()}')
            ss = station.get_sim_station()
            print(f'ssef {ss.get_electric_fields()[0].get_parameters()}')
    #        quit()
            channelAntennaDedispersion.run(evt, station, det, debug=True)
            ch1 = station.get_channel(0).get_trace()
            postP2P = max(ch1) - min(ch1)
            print(f'post P2P {postP2P/preP2P}')
    #        h1.set_xdata(np.append(h1.get_xdata(), set+1))
    #        h1.set_ydata(np.append(h1.get_ydata(), postP2P/preP2P))

    #        plt.plot(set+1, postP2P/preP2P)
    #        plt.draw()
            print(f'draw 2')
    #        plt.show()
        highLowThreshold.run(evt, station, det,
                                    threshold_high=2 * self._Vrms,
                                    threshold_low=-2 * self._Vrms,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_2sigma_dedispersed')

        """
        #Checking dipole trigger
        highLowThreshold.run(evt, station, det,
                                    threshold_high=2 * self._Vrms,
                                    threshold_low=-2 * self._Vrms,
                                    triggered_channels=[4],  # select the LPDA channels
                                    number_concidences=1,  # 2/4 majority logic
                                    trigger_name='DipoleTrigger')

        if station.has_triggered("DipoleTrigger"):
            #Checking dipole + depahsed 2sigma
            highLowThreshold.run(evt, station, det,
                                    threshold_high=1 * self._Vrms,
                                    threshold_low=-1 * self._Vrms,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=1,  # 2/4 majority logic
                                    trigger_name='2of5_Dephased_1sigma',
                                    set_not_triggered=(not station.has_triggered("simple_threshold")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger


            highLowThreshold.run(evt, station, det,
                                    threshold_high=1.5 * self._Vrms,
                                    threshold_low=-1.5 * self._Vrms,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=1,  # 2/4 majority logic
                                    trigger_name='2of5_Dephased_1.5sigma',
                                    set_not_triggered=(not station.has_triggered("simple_threshold")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger


        else:
            #checking case where dipole didn't trigger
            highLowThreshold.run(evt, station, det,
                                    threshold_high=1 * self._Vrms,
                                    threshold_low=-1 * self._Vrms,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='2of5_Dephased_1sigma',
                                    set_not_triggered=(not station.has_triggered("simple_threshold")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger


            highLowThreshold.run(evt, station, det,
                                    threshold_high=1.5 * self._Vrms,
                                    threshold_low=-1.5 * self._Vrms,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='2of5_Dephased_1.5sigma',
                                    set_not_triggered=(not station.has_triggered("simple_threshold")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger

        """



        # run a high/low trigger on the 4 surface dipoles
        """
        highLowThreshold.run(evt, station, det,
                                    threshold_high=3 * self._Vrms,
                                    threshold_low=-3 * self._Vrms,
                                    triggered_channels=[4, 5, 6, 7],  # select the bicone channels
                                    number_concidences=4,  # 4/4 majority logic
                                    trigger_name='surface_dipoles_4of4_3sigma',
                                    set_not_triggered=(not station.has_triggered("simple_threshold")))  # calculate more time consuming ARIANNA trigger only if station passes simple trigger
        """


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
#    plt.draw()
#    print(f'drawn')
    sim = mySimulation(inputfilename=args.inputfilename,
                                outputfilename=args.outputfilename,
                                detectorfile=args.detectordescription,
                                outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                                config_file=args.config,
                                file_overwrite=True)
    sim.run()
