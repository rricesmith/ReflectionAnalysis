import os
import datetime
import argparse
import NuRadioReco.modules.io.eventReader
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
from NuRadioReco.detector import generic_detector
import templateCrossCorr as txc
from NuRadioReco.utilities import units
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities.io_utilities import read_pickle
from NuRadioReco.framework.parameters import channelParameters as chp

import NuRadioReco.modules.channelLengthAdjuster
import StationDataAnalysis.Nu_RCR_ChiCut as ChiCut






#ReflCrFiles = ['StationDataAnalysis/data/MB_old_100s_Refl_CRs_500Evts_Noise_False_Amp_False.nur']
	#100s
ReflCrFiles = ['StationDataAnalysis/data/MB_old_100s_Refl_CRs_500Evts_Noise_True_Amp_True.nur', 'StationDataAnalysis/data/MB_old_100s_Refl_CRs_500Evts_Noise_True_Amp_True_part02.nur', 'StationDataAnalysis/data/MB_old_100s_Refl_CRs_500Evts_Noise_True_Amp_True_part03.nur']
NuFiles = ['StationDataAnalysis/data/N02_SimNu_100s_wNoise_wAmp.nur']
templates = '100'
	#200s
# ReflCrFiles = ['StationDataAnalysis/data/MB_old_200s_Refl_CRs_500Evts_Noise_True_Amp_True.nur']
# NuFiles = ['StationDataAnalysis/data/N02_SimNu_200s_wNoise_wAmp.nur']
# templates = '200'


station_id = 1

eventReader = NuRadioReco.modules.io.eventReader.eventReader()
hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelLengthAdjuster = NuRadioReco.modules.channelLengthAdjuster.channelLengthAdjuster()

det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_MB_old_footprint576m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
det.update(datetime.datetime(2018, 10, 1))
#parallelChannels = det.get_parallel_channels(station_id)
parallelChannels = [[4, 6], [5, 7]]


template_sampling_rate = 2*units.GHz
t_zen = np.deg2rad(120)
t_azi = np.deg2rad(30.0)
t_view = np.deg2rad(0.0)
if templates == '100':
    templates_nu = 'StationDataAnalysis/templates/NUdowntemplate_100hp1000lpFilter_100series_SST.pkl'
    templates_nu = read_pickle(templates_nu)[t_zen][t_azi][t_view]
    templates_cr = 'StationDataAnalysis/templates/reflectedCR_template_100series.pkl'
elif templates == '200':
    templates_nu = 'StationDataAnalysis/templates/NUdowntemplate_NoFilter_200series_SST.pkl'
    templates_nu = read_pickle(templates_nu)[t_zen][t_azi][t_view]
    templates_cr = 'StationDataAnalysis/templates/reflectedCR_template_200series.pkl'

templates_cr = read_pickle(templates_cr)
for key in templates_cr:
    temp = templates_cr[key]
templates_cr = temp


CrSim = {}
CrSim['nuXcorr'] = []
CrSim['crXcorr'] = []
CrSim['p2p'] = []
CrSim['arrival'] = []
CrSim['DipP2p'] = []

i = 0
for file in ReflCrFiles:
    eventReader.begin(file)
    print(f'running file {file}')

    for evt in eventReader.run():
        station = evt.get_station(1)
        i = i + 1
        print(f'event {i}')

#        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)
#            channelLengthAdjuster.run(evt, station, channel_ids=parChans)

        """
        simStation = station.get_sim_station()
        print(f'ss params {simStation.get_parameters()}')
        print(f'station params {station.get_parameters()}')
        print(f'ss channels {simStation.get_channel_ids()}')
        for channel in simStation.iter_channels():
            print(f'sim station params {channel.get_parameters()}')
        quit()
        """


        for channel in station.iter_channels(use_channels=[9]):
            CrSim['DipP2p'].append(channel.get_parameter(chp.P2P_amplitude))
#            print(f'channel params {channel.get_parameters()}')
#            quit()

        appendNuCorr = 0
        appendCrCorr = 0
        appendP2P = 0
        appendArrival = 0
        for parChans in parallelChannels:
            nu_xCorr = 0
            cr_xCorr = 0
            Vp2p = 0
            arrival_angle = 0
            for channel in station.iter_channels(use_channels=parChans):
                cTrace = channel.get_trace()
                #If an antenna is outside the star pattern, it is possible the trace is empty with a noiseless simulation. Skip channel in this case
                if not max(cTrace) == 0:
#                    print(f'max ctrace is {max(cTrace)}, length {len(cTrace)} and len times {len(channel.get_times())} and template len {len(templates_nu)}')
                    nu_xCorr += np.abs(txc.get_xcorr_for_channel(cTrace, templates_nu, channel.get_sampling_rate(), template_sampling_rate, times=channel.get_times(), debug=False))		#Chris template
                    cr_xCorr += np.abs(txc.get_xcorr_for_channel(cTrace, templates_cr, channel.get_sampling_rate(), template_sampling_rate, times=channel.get_times(), debug=False))		#CR refl template
                    Vp2p = max(Vp2p, (np.max(cTrace) - np.min(cTrace)))
#                    arrival_angle += channel.get_parameter(chp.zenith)		#Bugged, need to find a way to get this parameter
            nu_xCorr /= len(parChans)
            cr_xCorr /= len(parChans)
            arrival_angle /= len(parChans)

            appendNuCorr = max(appendNuCorr, nu_xCorr)
            appendCrCorr = max(appendCrCorr, cr_xCorr)
            appendP2P = max(appendP2P, Vp2p)
            appendArrival = max(appendArrival, arrival_angle)

        CrSim['nuXcorr'].append(appendNuCorr)
        CrSim['crXcorr'].append(appendCrCorr)
        CrSim['p2p'].append(appendP2P)
        CrSim['arrival'].append(appendArrival)

CrSim['DipP2p'] /= min(CrSim['DipP2p'])
CrDipBins = np.logspace(np.log10(min(CrSim['DipP2p'])), np.log10(max(CrSim['DipP2p'])), 50)
"""
plt.hist(CrSim['DipP2p'], bins=CrDipBins)
plt.xscale('log')
plt.xlabel('R-CR Dipole Amplitude')
plt.show()


plt.scatter(CrSim['p2p'], CrSim['nuXcorr'], label='Nu Chi')
plt.scatter(CrSim['p2p'], CrSim['crXcorr'], label='R-CR Chi')
plt.ylim((0, 1))
plt.legend()
plt.xlabel('Max P2P')
plt.ylabel('Chi')
plt.xscale('log')
plt.title('R-CR Chis')
plt.show()


plt.hist(CrSim['arrival'])
plt.xlabel('Arrival Angle')
plt.show()

plt.scatter(CrSim['nuXcorr'], CrSim['crXcorr'])
plt.xlabel('Nu Chi')
plt.ylabel('R-CR Chi')
plt.ylim((0, 1))
plt.xlim((0, 1))
plt.title('R-CR Chis')
plt.show()
"""

NuSim = {}
NuSim['nuXcorr'] = []
NuSim['crXcorr'] = []
NuSim['p2p'] = []
NuSim['arrival'] = []
NuSim['DipP2p'] = []

det_nu = generic_detector.GenericDetector(json_filename=f'StationDataAnalysis/configs/MB_generic_{templates}s_wDipole.json', assume_inf=False, antenna_by_depth=False, default_station=1)
det_nu.update(datetime.datetime(2018, 10, 1))
#parallelChannels = det_nu.get_parallel_channels(station_id)
parallelChannels = [[0, 2], [1, 3]]

for file in NuFiles:
    eventReader.begin(file)
    print(f'running file {file}')

    for evt in eventReader.run():
        station = evt.get_station(1)


#        hardwareResponseIncorporator.run(evt, station, det_nu, sim_to_data=True)
#            channelLengthAdjuster.run(evt, station, channel_ids=parChans)


        for channel in station.iter_channels(use_channels=[4]):
            NuSim['DipP2p'].append(channel.get_parameter(chp.P2P_amplitude))

        appendNuCorr = 0
        appendCrCorr = 0
        appendP2P = 0
        appendArrival = 0
        for parChans in parallelChannels:
            nu_xCorr = 0
            cr_xCorr = 0
            Vp2p = 0
            arrival_angle = 0
            for channel in station.iter_channels(use_channels=parChans):
                cTrace = channel.get_trace()
                #If an antenna is outside the star pattern, it is possible the trace is empty with a noiseless simulation. Skip channel in this case
                if max(cTrace) == 0:
                    continue
                nu_xCorr += np.abs(txc.get_xcorr_for_channel(cTrace, templates_nu, channel.get_sampling_rate(), template_sampling_rate, times=channel.get_times(), debug=False))		#Chris template
                cr_xCorr += np.abs(txc.get_xcorr_for_channel(cTrace, templates_cr, channel.get_sampling_rate(), template_sampling_rate, times=channel.get_times(), debug=False))		#CR refl template
                Vp2p = max(Vp2p, (np.max(cTrace) - np.min(cTrace)))
#                arrival_angle += channel.get_parameter(chp.zenith)		#Bugged, need to find a way to get this parameter
            nu_xCorr /= len(parChans)
            cr_xCorr /= len(parChans)
            arrival_angle /= len(parChans)


            appendNuCorr = max(appendNuCorr, nu_xCorr)
            appendCrCorr = max(appendCrCorr, cr_xCorr)
            appendP2P = max(appendP2P, Vp2p)
            appendArrival = max(appendArrival, arrival_angle)


        if Vp2p == 0:
            continue

        NuSim['nuXcorr'].append(appendNuCorr)
        NuSim['crXcorr'].append(appendCrCorr)
        NuSim['p2p'].append(appendP2P)
        NuSim['arrival'].append(appendArrival)


NuSim['DipP2p'] /= min(NuSim['DipP2p'])
NuDipBins = np.logspace(np.log10(min(NuSim['DipP2p'])), np.log10(max(NuSim['DipP2p'])), 50)



plt.scatter(NuSim['p2p'], NuSim['nuXcorr'], label='Sim Nu')
plt.scatter(CrSim['p2p'], CrSim['nuXcorr'], label='Sim R-CR')
plt.xlabel('Max P2P')
plt.ylabel('Nu Chi')
plt.xscale('log')
plt.title('Simulated R-CR and Neutrinos, Neutrino matching')
plt.ylim((0, 1))
plt.xlim((0.06, 100))
plt.legend()
plt.show()



plt.scatter(NuSim['p2p'], NuSim['crXcorr'], label='Sim Nu')
plt.scatter(CrSim['p2p'], CrSim['crXcorr'], label='Sim R-CR')
plt.xlabel('Max P2P')
plt.ylabel('R-CR Chi')
plt.xscale('log')
plt.title('Simulated R-CR and Neutrinos, R-CR matching')
plt.ylim((0, 1))
plt.xlim((0.06, 100))
plt.legend()
plt.show()
# quit()
plt.clf()

"""
plt.hist(NuSim['DipP2p'], bins=NuDipBins)
plt.xscale('log')
plt.xlabel('Nu Dipole Amplitude')
plt.show()


plt.scatter(NuSim['p2p'], NuSim['nuXcorr'], label='Nu Chi')
plt.scatter(NuSim['p2p'], NuSim['crXcorr'], label='R-CR Chi')
plt.xlabel('Max P2P')
plt.ylabel('Chi')
plt.xscale('log')
plt.title('Neutrino Chis')
plt.ylim((0, 1))
#plt.xlim((0.06, 100))
plt.legend()
plt.show()


#plt.hist(CrSim['arrival'])
#plt.xlabel('Arrival Angle')
#plt.show()


plt.scatter(NuSim['nuXcorr'], NuSim['crXcorr'])
plt.title('Neutrino Chis')
plt.xlabel('Nu Chi')
plt.ylabel('R-CR Chi')
plt.ylim((0, 1))
plt.xlim((0, 1))
plt.show()


CrHist, _bins = np.histogram(CrSim['DipP2p'], bins=CrDipBins)
NuHist, _bins = np.histogram(NuSim['DipP2p'], bins=NuDipBins)


CrHist = CrHist / max(CrHist)
NuHist = NuHist / max(NuHist)

plt.bar(CrDipBins[:-1], CrHist, label='R-CR', edgecolor='orange')
plt.bar(NuDipBins[:-1], NuHist, label='Nu', edgecolor='blue')

#plt.hist(CrSim['DipP2p'], bins=CrDipBins, density=True, stacked=True, label='R-CR', fill=False, edgecolor='orange')
#plt.hist(NuSim['DipP2p'], bins=NuDipBins, density=True, stacked=True, label='Nu', fill=False, edgecolor='blue')
plt.xscale('log')
plt.xlabel('Dipole Amplitude')
plt.legend()
plt.show()
"""


plt.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), linestyle='--')
plt.scatter(CrSim['nuXcorr'], CrSim['crXcorr'], label='Simulated Reflected Air Showers')
plt.scatter(NuSim['nuXcorr'], NuSim['crXcorr'], label='Simulated Neutrinos')
x = np.linspace(0, 1, 100)
"""
def nuFit(nuChi):
    x1 = 0.34
    x2 = 0.6
    y1 = 0.72
    y2 = 0.85
    if nuChi < x1:
        return y1
    elif nuChi < x2:
        return (nuChi - x2) * (y2-y1)/(x2-x1) + y2
    else:
        return y2 - 3.5*(nuChi - x2)**2
y = []
for i in x:
    y.append(nuFit(i))
nuEff = 0
for iN, nuXcorr in enumerate(NuSim['nuXcorr']):
    if NuSim['crXcorr'][iN] >= nuFit(nuXcorr):
        nuEff += 1
nuEff = 100 * nuEff / len(NuSim['nuXcorr'])
crEff = 0
for iN, nuXcorr in enumerate(CrSim['nuXcorr']):
    if CrSim['crXcorr'][iN] >= nuFit(nuXcorr):
        crEff += 1
crEff = 100 * crEff / len(CrSim['nuXcorr'])
plt.plot(x, y, label=f'{nuEff:.1f}% Nu Efficient\n{crEff:.1f}% R-CR Efficient', color='red', linestyle='-')
"""
nuEff = ChiCut.cutEfficiency(NuSim['nuXcorr'], NuSim['crXcorr'])
crEff = ChiCut.cutEfficiency(CrSim['nuXcorr'], CrSim['crXcorr'])
# plt.plot(x, ChiCut.cutArray(x), label=f'{nuEff:.1f}% Nu Efficient\n{crEff:.1f}% R-CR Efficient', color='red', linestyle='-')
plt.plot(x, ChiCut.cutArray(x), label=f'{nuEff:.1f}% Neutrino Efficient', color='red', linestyle='-')

plt.xlabel('Neutrino Chi')
plt.ylabel('Reflected Air Shower Chi')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.legend()
# plt.show()
plt.savefig(f'plots/Advancement/NuVsCrChi.png', format='png')