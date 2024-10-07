import numpy as np
import scipy.constants
import pickle
from NuRadioReco.utilities import units
from NuRadioMC.utilities import cross_sections
from NuRadioMC.examples.Sensitivities import E2_fluxes3 as limits
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
from scipy import interpolate
from NuRadioMC.utilities import inelasticities

import CoreAnalysis.coreRateToNuFlux as cToNu
import CoreAnalysis.C00_coreAnalysisUtils as CDO_util

import itertools
color = itertools.cycle(('blue', 'green', 'red'))

energyBinsPerDecade = 1.
plotUnitsEnergy = units.eV
plotUnitsEnergyStr = "eV"
plotUnitsFlux = units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1

n_stations_shallow = 361
n_stations_deep = 164


#coreFilesLabels = [['data/CoreDataObjects/Gen2_2021_CoreDataObjects_LPDA_2of4_100Hz_below_300mRefl_SP_1R_fZenVariablef_40.0dB_1.7km_1000cores.pkl', '300m 40dB']]
coreFilesLabels = [['data/CoreDataObjects/Gen2_2021_CoreDataObjects_LPDA_2of4_100Hz_below_300mRefl_SP_1R_0.3f_40.0dB_1.7km_1000cores.pkl', '300m 40dB']]
dBs = [40, 45, 50]
#fs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
layers = [300, 500, 800]
#dBs = [40, 45, 50]
fs = [0.5]
#layers = [300]

end_energies_layers = {}
max_rate_layers = {}
min_rate_layers = {}

end_energies_layers['all'] = []
max_rate_layers['all'] = []

for layer in layers:
    end_energies = []
    max_rate = []
    min_rate = []

    coreFilesLabels = []
    for dB in dBs:
        for f in fs:
            coreFilesLabels.append([f'data/CoreDataObjects/Gen2_2021_CoreDataObjects_LPDA_2of4_100Hz_below_{layer}mRefl_SP_1R_{f}f_{dB}.0dB_1.7km_1000cores.pkl', layer, dB, f])

#    coreObjects = []
    for file, layer, dB, f in coreFilesLabels:
        with open(file, 'rb') as fin:
            CoreObjectsList = pickle.load(fin)
            fin.close()
#        coreObjects.append(CoreObjectsList)

        energies, rates, rateError = CDO_util.getCoreEnergyEventRate(CoreObjectsList, singleAeff=True)
        energies = f * 10 ** np.array(energies) * units.eV
        rates = np.array(rates) * n_stations_shallow

        nu_e, nu_flux = cToNu.flux_from_num_events_per_bin(energies, rates, dB=dB)
        nu_flux_e2 = nu_flux * nu_e**2

        for iE, e in enumerate(nu_e):
            if e not in end_energies:
                end_energies.append(e)
                max_rate.append(nu_flux_e2[iE])
                min_rate.append(nu_flux_e2[iE])
            else:
                ind = end_energies.index(e)
                if nu_flux_e2 > max_rate[ind]:
                    max_rate[ind] = nu_flux_e2
                elif 0 < nu_flux_e2 < min_rate[ind]:
                    min_rate[ind] = nu_flux_e2


    fig, ax = limits.get_E2_limit_figure(diffuse=True, show_grand_10k=False, show_grand_200k=False, show_RNOG=True, show_ara=True,
                                    show_anita_I_IV_limit=False, show_auger_limit=False,
                                    show_IceCubeGen2_whitepaper=False, show_IceCubeGen2_ICRC2021=True)


    max_rate = [x for _,x in sorted(zip(end_energies,max_rate))]
    min_rate = [x for _,x in sorted(zip(end_energies,min_rate))]
    end_energies = sorted(end_energies)

    end_energies = np.array(end_energies)
    max_rate = np.array(max_rate)
    min_rate = np.array(min_rate)

    peaks, _ = find_peaks(max_rate)
    mins, _ = find_peaks(max_rate*-1)


    labels = []
    """
    _plt, = ax.plot(end_energies / plotUnitsEnergy, max_rate / plotUnitsFlux, linestyle='-', color='black', label=f'Max')
    labels.append(_plt)
    _plt, = ax.plot(end_energies / plotUnitsEnergy, min_rate / plotUnitsFlux, linestyle='--', color='black', label=f'Min')
    labels.append(_plt)
    """
#    _plt, = ax.plot(end_energies[peaks] / plotUnitsEnergy, max_rate[peaks] / plotUnitsFlux, linestyle='-', color='black', label=f'Max')
#    labels.append(_plt)
#    _plt, = ax.plot(end_energies[mins] / plotUnitsEnergy, min_rate[mins] / plotUnitsFlux, linestyle='--', color='black', label=f'Min')
#    labels.append(_plt)
#    _plt, = ax.fill(np.append(end_energies[peaks]/plotUnitsEnergy, end_energies[mins]/plotUnitsEnergy),
#                    np.append(max_rate[peaks]/plotUnitsFlux, max_rate[mins]/plotUnitsFlux), color='green', alpha=0.5, label='300m')
    c = next(color)
    _plt = ax.fill_between(end_energies[peaks]/plotUnitsEnergy, max_rate[peaks]/plotUnitsFlux,
                            color=c, alpha=0.35, label=f'{layer}m')
    labels.append(_plt)

    plt.legend(handles=labels, loc=2)
    plt.savefig(f'CoreAnalysis/plots/gen2CoreBckgrdFlux_{layer}m_all.png')
    plt.clf()

    end_energies_layers[layer] = end_energies[peaks]
    max_rate_layers[layer] = max_rate[peaks]
    if end_energies_layers['all'] == []:
        end_energies_layers['all'] = end_energies
        max_rate_layers['all'] = max_rate
    else:
        print(f'max rate {max_rate}')
        print(max_rate_layers['all'])
#        print(f'shape layers all ' + max_rate_layers['all'].shape + 'and max rate' + max_rate.shape)
        max_rate_layers['all'] += max_rate



fig, ax = limits.get_E2_limit_figure(diffuse=True, show_grand_10k=False, show_grand_200k=False, show_RNOG=True, show_ara=True,
                                show_anita_I_IV_limit=False, show_auger_limit=False,
                                show_IceCubeGen2_whitepaper=False, show_IceCubeGen2_ICRC2021=True)
labels = []
for layer in layers:
    c = next(color)
    _plt = ax.fill_between(end_energies_layers[layer]/plotUnitsEnergy, max_rate_layers[layer]/plotUnitsFlux,
                            alpha=0.35, label=f'{layer}m', color=c)
#                            color='green', alpha=0.35, label=f'{layer}m')
    labels.append(_plt)
plt.legend(handles=labels, loc=2)
plt.savefig(f'CoreAnalysis/plots/gen2CoreBckgrdFlux_Multilayer_all.png')
plt.clf()




peaks, _ = find_peaks(max_rate_layers['all'])

fig, ax = limits.get_E2_limit_figure(diffuse=True, show_grand_10k=False, show_grand_200k=False, show_RNOG=True, show_ara=True,
                                show_anita_I_IV_limit=False, show_auger_limit=False,
                                show_IceCubeGen2_whitepaper=False, show_IceCubeGen2_ICRC2021=True)
labels = []
_plt = ax.fill_between(end_energies_layers['all'][peaks]/plotUnitsEnergy, max_rate_layers['all'][peaks]/plotUnitsFlux,
                        alpha=0.35, label=f'Sum All Layers', color='cyan')
#                            color='green', alpha=0.35, label=f'{layer}m')
labels.append(_plt)
plt.legend(handles=labels, loc=2)
plt.savefig(f'CoreAnalysis/plots/gen2CoreBckgrdFlux_SumLayers_all.png')
plt.clf()

print('Done!')