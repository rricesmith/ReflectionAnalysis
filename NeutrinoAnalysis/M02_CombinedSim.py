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
from NuRadioReco.detector.ARIANNA import analog_components
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

    def get_400s_response(self, frequencies):
        #return linear gain per freq
        f = np.array([300000, 5298500, 10297000, 15295500, 20294000, 25292500, 30291000, 35289500, 40288000, 45286500, 50285000, 55283500, 60282000, 65280500, 70279000, 75277500, 80276000, 85274500, 90273000, 95271500, 100270000, 105268500, 110267000, 115265500, 120264000, 125262500, 130261000, 135259500, 140258000, 145256500, 150255000, 155253500, 160252000, 165250500, 170249000, 175247500, 180246000, 185244500, 190243000, 195241500, 200240000, 205238500, 210237000, 215235500, 220234000, 225232500, 230231000, 235229500, 240228000, 245226500, 250225000, 255223500, 260222000, 265220500, 270219000, 275217500, 280216000, 285214500, 290213000, 295211500, 300210000, 305208500, 310207000, 315205500, 320204000, 325202500, 330201000, 335199500, 340198000, 345196500, 350195000, 355193500, 360192000, 365190500, 370189000, 375187500, 380186000, 385184500, 390183000, 395181500, 400180000, 405178500, 410177000, 415175500, 420174000, 425172500, 430171000, 435169500, 440168000, 445166500, 450165000, 455163500, 460162000, 465160500, 470159000, 475157500, 480156000, 485154500, 490153000, 495151500, 500150000, 505148500, 510147000, 515145500, 520144000, 525142500, 530141000, 535139500, 540138000, 545136500, 550135000, 555133500, 560132000, 565130500, 570129000, 575127500, 580126000, 585124500, 590123000, 595121500, 600120000, 605118500, 610117000, 615115500, 620114000, 625112500, 630111000, 635109500, 640108000, 645106500, 650105000, 655103500, 660102000, 665100500, 670099000, 675097500, 680096000, 685094500, 690093000, 695091500, 700090000, 705088500, 710087000, 715085500, 720084000, 725082500, 730081000, 735079500, 740078000, 745076500, 750075000, 755073500, 760072000, 765070500, 770069000, 775067500, 780066000, 785064500, 790063000, 795061500, 800060000, 805058500, 810057000, 815055500, 820054000, 825052500, 830051000, 835049500, 840048000, 845046500, 850045000, 855043500, 860042000, 865040500, 870039000, 875037500, 880036000, 885034500, 890033000, 895031500, 900030000, 905028500, 910027000, 915025500, 920024000, 925022500, 930021000, 935019500, 940018000, 945016500, 950015000, 955013500, 960012000, 965010500, 970009000, 975007500, 980006000, 985004500, 990003000, 995001500, 1000000000]) * units.Hz
        d = [-54.9273077293, -40.0790841512, -18.0426258344, -4.29517325364, 5.64165595293, 13.2553853969, 19.5786452955, 24.9993508902, 29.6906630866, 33.9258564726, 37.5856137851, 40.7434786098, 43.2137424847, 45.0646971531, 46.240047318, 47.0247076053, 47.4557821957, 47.7531649893, 47.8308114071, 47.9299598917, 47.9350171119, 47.9414552248, 47.9166944523, 47.9168285686, 47.8410466534, 47.8330087195, 47.7622830557, 47.6946466538, 47.6323401811, 47.5952981116, 47.5028644752, 47.4531826368, 47.3587197731, 47.3003790359, 47.173564779, 47.1185605575, 47.0343779324, 46.9530296419, 46.8461509669, 46.8134528192, 46.6824029528, 46.6133002215, 46.522999734, 46.4493565219, 46.3164483098, 46.2608681031, 46.1197727431, 46.0675985983, 45.9102815891, 45.8486715479, 45.7091708481, 45.6099201548, 45.4769163884, 45.3752121498, 45.2421769241, 45.151770099, 45.031391931, 44.9169531747, 44.7853931727, 44.718306133, 44.569927632, 44.5125136526, 44.3850752496, 44.2936158643, 44.1737035853, 44.0937949763, 43.9797597924, 43.91027941, 43.8354504375, 43.7601313199, 43.669433607, 43.6086663446, 43.5243972365, 43.4686947601, 43.403243013, 43.3321104185, 43.2593131763, 43.2045263498, 43.1184399104, 43.0710856873, 42.9996662157, 42.9238246833, 42.7990954321, 42.7226878244, 42.6380492625, 42.5827562649, 42.4655550488, 42.4114671102, 42.3336281175, 42.2244157585, 42.1061073188, 42.0053229784, 41.8757744553, 41.7236360338, 41.514871809, 41.3754955968, 41.1661059422, 40.9832016548, 40.7401705865, 40.5322185277, 40.2630606242, 40.0252092839, 39.7188844502, 39.4569899986, 39.1117499226, 38.805442659, 38.4538447327, 38.1155036379, 37.7222594842, 37.3351083628, 36.9282247949, 36.5189203112, 36.0679743807, 35.6282278515, 35.1564960544, 34.7198914601, 34.2506855413, 33.7924946329, 33.3216189376, 32.8759751816, 32.4188582121, 31.9942347248, 31.4867278592, 31.0743435439, 30.6188825094, 30.2081298329, 29.7108683628, 29.2835499602, 28.8071808943, 28.3701393368, 27.8982260278, 27.4254947253, 26.931535156, 26.469390242, 25.9391554754, 25.5021263383, 24.9319012994, 24.4818114543, 23.9450977277, 23.4787200219, 22.9370049828, 22.4737224973, 21.9862900453, 21.524000277, 21.0285819546, 20.5811964355, 20.1321466894, 19.6650638595, 19.2454808952, 18.7734838925, 18.4025323019, 17.9313173981, 17.5566655217, 17.0859186162, 16.6847836212, 16.2416883514, 15.8437830885, 15.3820009268, 14.9902573765, 14.506389251, 14.0872082569, 13.5961684844, 13.1567518606, 12.6738406886, 12.2613387454, 11.7930315386, 11.371871117, 10.9026107344, 10.5140679058, 10.0556555724, 9.64604867506, 9.25016231826, 8.84420889903, 8.48368197223, 8.11607562428, 7.73948878825, 7.37450280104, 7.06496301714, 6.64994364845, 6.34692430004, 5.97495531358, 5.63457444037, 5.24049395801, 4.91114606917, 4.49391050249, 4.15470680843, 3.72580189871, 3.36043790173, 2.90841358944, 2.52841471875, 2.06257386225, 1.65240261665, 1.22921173076, 0.800855756961, 0.343652922866, -0.039812687093, -0.499379548027, -0.88013447221, -1.2808820605, -1.68251193413, -2.07697294371]

        interp = interp1d(f, d, kind='nearest', bounds_error=False, fill_value=0)
        return  10**(interp(frequencies)/20)

    def _detector_simulation_trigger(self, evt, station, det):

        # ic(max(station.get_channel(0).get_trace()))

        dummy_channels = {}

        traces_old = {}
        for channel in station.iter_channels():
            dummy_channels[channel.get_id()] = NuRadioReco.framework.base_trace.BaseTrace()
            dummy_channels[channel.get_id()].set_frequency_spectrum(channel.get_frequency_spectrum(), channel.get_sampling_rate())

            trace_fft = channel.get_frequency_spectrum()
            traces_old[channel.get_id()] = trace_fft
            trace = channel.get_trace()
            frequencies = channel.get_frequencies()
            # plt.plot(frequencies, np.abs(trace_fft), label='preamp')            
            # trace_fft *= self.get_filter(frequencies, station.get_id(), channel.get_id(), det, sim_to_data, phase_only, mode, mingainlin)
            trace_fft *= self.get_400s_response(frequencies)
            # ic(self.get_400s_response(frequencies))
            # zero first bins to avoid DC offset
            trace_fft[0] = 0

            #Test old hardware response
            amp_type = det.get_amplifier_type(station.get_id(), channel.get_id())
            amp_measurement = det.get_amplifier_measurement(station.get_id(), channel.get_id())
            amp_response = analog_components.get_amplifier_response(frequencies, amp_type=amp_type, amp_measurement=amp_measurement)            
            # ic(amp_response)
            # plt.plot(frequencies, np.abs(dummy_channels[channel.get_id()].get_frequency_spectrum()) * np.abs(amp_response), label='300s')


            # hardwareResponse incorporator should always be used in conjunction with bandpassfilter
            # otherwise, noise will be blown up
            channel.set_frequency_spectrum(trace_fft, channel.get_sampling_rate())
            # ic(max(channel.get_trace()))
            # channel.set_frequency_spectrum(dummy_channels[channel.get_id()].get_frequency_spectrum()*amp_response, channel.get_sampling_rate())
            # ic(max(channel.get_trace()))
            # quit()
            # plt.plot(frequencies, np.abs(trace_fft), label='400s')
            # plt.legend()
            # plt.yscale('log')
            # plt.savefig(f'plots/Neutrinos/ManuelThesis/test400s.png')
            # plt.clf()
            # plt.plot(trace, label='preamp')
            # plt.plot(channel.get_trace(), label='postamp')
            # # plt.yscale('log')
            # plt.legend()
            # # plt.savefig(f'test400strace.png')
            # plt.plot(frequencies, self.get_400s_response(frequencies), label='400s')
            # plt.plot(frequencies, amp_response, label='300s')
            # plt.legend()
            # plt.savefig('plots/Neutrinos/ManuelThesis/300s400s.png')
            # plt.clf()
            # plt.plot(frequencies, 20*np.log10(self.get_400s_response(frequencies)), label='400s')
            # plt.plot(frequencies, 20*np.log10(np.abs(amp_response)), label='300s')
            # plt.legend()
            # plt.savefig('plots/Neutrinos/ManuelThesis/300400sdB.png')
            # plt.clf()
            # quit()

        # ic(max(station.get_channel(0).get_trace()))

        Vrms_400 = 2.5*units.mV   #Ask Manuel/Steve. Steve said 2.7, Manuel said 2.5
        # run a high/low trigger on the 4 downward pointing LPDAs
        highLowThreshold.run(evt, station, det,
                                    threshold_high=3.8 * Vrms_400,
                                    threshold_low=-3.8 * Vrms_400,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_3.8sigma') 

        for channel in station.iter_channels():
            channel.set_frequency_spectrum(dummy_channels[channel.get_id()].get_frequency_spectrum(), channel.get_sampling_rate())
        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)
        # ic(max(station.get_channel(0).get_trace()))
        # quit()
        Vrms_300  = 10*units.mV
        # run a high/low trigger on the 4 downward pointing LPDAs
        highLowThreshold.run(evt, station, det,
                                    threshold_high=4.4 * Vrms_300,
                                    threshold_low=-4.4 * Vrms_300,
                                    triggered_channels=[0, 1, 2, 3],  # select the LPDA channels
                                    number_concidences=2,  # 2/4 majority logic
                                    trigger_name='LPDA_2of4_4.4sigma')

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
