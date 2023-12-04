import os
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import particleParameters as parp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
import pickle
import CoreAnalysis.C00_coreAnalysisUtils as CDO_util




nu_arrivals = []
nu_weights = []

nu_files = []
#path = 'NeutrinoAnalysis/output/SP_nu_uniform/'
path = 'NeutrinoAnalysis/output/SP_neutrinos/'
#path = 'NeutrinoAnalysis/output/AttTest_GL3/R12km/'
#path = 'NeutrinoAnalysis/output/'
for file in os.listdir(path):
    if file.endswith('.nur'):
        nu_files.append(os.path.join(path, file))

print(f'files {nu_files}')

reader = NuRadioRecoio.NuRadioRecoio(nu_files)
for i, evt in enumerate(reader.get_events()):
    station = evt.get_station(1)
    if not station.has_triggered():
        continue
    ss = station.get_sim_station()
    ef1 = ss.get_electric_fields()[0]
    zenith = ef1.get_parameter(efp.zenith)

    primary = evt.get_primary()
    weight = primary.get_parameter(parp.weight)

#    print(f'zen {np.rad2deg(zenith)} weight {weight}')

    nu_arrivals.append(np.rad2deg(zenith))
    nu_weights.append(weight)

print(f'len {len(nu_weights)}')

zenBins = np.linspace(0, np.pi, 30)
zenBins = np.rad2deg(zenBins)
#plt.hist(nu_arrivals, bins=zenBins, weights=nu_weights, edgecolor='black', density=True)
#plt.savefig('CoreAnalysis/plots/NuArrivals.png')

#Do cores now
core_input_file = 'data/CoreDataObjects/Gen2_2021_CoreDataObjects_LPDA_2of4_100Hz_below_300mRefl_SP_1R_0.3f_40.0dB_1.7km_1000cores.pkl'
with open(core_input_file, 'rb') as fin:
    CoreObjectsList = pickle.load(fin)

#CDO_util.plotArrivalDirectionHist(CoreObjectsList, cut=[40, 140])
#fig = plt.figure()
plt.hist(nu_arrivals, bins=zenBins, weights=nu_weights, edgecolor='red', histtype='bar', label='Nu', fill=False, density=True)

zenCent = (zenBins[1:] + zenBins[:-1])/2
zenCent = np.rad2deg(zenCent)
arrivalCounts = np.zeros_like(zenCent)
arrivalCutCounts = np.zeros_like(zenCent)
cut = None
if not cut == None:
    binCut = np.digitize(np.deg2rad(cut), zenBins)
for core in CoreObjectsList:
    eventRate = core.totalEventRateCore()
    if eventRate == 0:
        continue
    arrivalHist = core.getZeniths()
#        print(f'arrival hist {arrivalHist}')
    eventRateWeight = eventRate / np.sum(arrivalHist)
    arrivalCounts += arrivalHist * eventRateWeight
#        print(f'arrivalCounts {arrivalCounts}')
    if not cut == None:
        arrivalHist[:binCut[0]] = 0
        arrivalHist[binCut[1]:] = 0
        arrivalCutCounts += arrivalHist * eventRateWeight

arrivalCounts = arrivalCounts / (sum(arrivalCounts) * np.diff(zenBins))
plt.bar(zenBins[:-1], arrivalCounts, width=np.diff(zenBins), edgecolor='blue', fill=False, align='edge', label='Core')
plt.xlabel('Arrival Angle (deg)')
plt.legend()
plt.savefig('CoreAnalysis/plots/NuArrivals.png')
#plt.savefig(f'plots/CoreAnalysis/{savePrefix}_ArrivalHist.png')
plt.clf()