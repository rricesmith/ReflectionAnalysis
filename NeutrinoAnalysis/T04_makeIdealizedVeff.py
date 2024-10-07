import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import h5py
import glob
from scipy import interpolate
import json
import os
import sys
import matplotlib.lines as mlines

from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioMC.utilities import fluxes
from NuRadioMC.utilities.Veff import get_Veff_Aeff, get_Veff_Aeff_array, get_index, get_Veff_water_equivalent
from NuRadioMC.examples.Sensitivities import E2_fluxes3 as limits
#plt.switch_backend('agg')

if __name__ == "__main__":




    # plot expected limit
    fig, ax = limits.get_E2_limit_figure(diffuse=True, show_grand_10k=True, show_grand_200k=False, show_RNOG=True, show_ara=True,
                                         show_anita_I_IV_limit=True, show_auger_limit=False,
                                         show_IceCubeGen2_whitepaper=False, show_IceCubeGen2_ICRC2021=True)

    fig.savefig("NeutrinoAnalysis/limits_ARI_RNOG_GRAND_Anita_Gen2.png")
    quit()


    energies = [1e16, 3e16,  5e16,  7e16,  1e17,  3e17,  5e17,  7e17,  1e18,  3e18,  5e18,  1e19]
    flux_2sigma =  [7e-8, 2.9e-8, 2e-8, 1.5e-8, 1.2e-8,7e-9, 6.5e-9, 6e-9, 5.5e-9, 5e-9, 5.5e-9, 6e-9]
    flux_4sigma = [2.1e-7,8.5e-8,6e-8,4.5e-8,3.5e-8,2.1e-8,1.9e-8,1.8e-8,1.5e-8,1e-8,1.1e-8,1e-8]
    flux_2sigma = np.array(flux_2sigma)
    flux_2sigmaImprovement = flux_2sigma * 0.5

    print(f'energies {energies}')
    print(f'flux 2sigma {flux_2sigma}')

    n_stations = 1
    livetime = 1*units.year

    labels = []
    label2 = ax.plot(energies, flux_2sigma, linestyle='-', label=r'GRAMMAR(2$\sigma$)', color='black', linewidth=4)
    label2 = ax.plot(energies, flux_4sigma, linestyle='--', label=r'GRAMMAR(4$\sigma$)', color='black', linewidth=4)
#    label3 = ax.plot(energies, flux_2sigmaImprovement, label=r'2$\sigma$ dephased', color='green')

    baseline = mlines.Line2D([], [], color='black', linestyle='-', label=r'GRAMMAR(2$\sigma$)')
    improvement = mlines.Line2D([], [], color='black', linestyle='--', label=r'GRAMMAR(4$\sigma$)')

    handles = [baseline, improvement]
    labels = [h.get_label() for h in handles]

#    print(f'labels {labels}')
#    plt.title(f'{n_stations} Future Stations with {livetime/units.year} years livetime')
#    handles = [label1, label2, label3]
#    labels = [h.get_label() for h in handles]
#    leg = plt.legend(handles=handles, labels=labels, loc=2, prop={'size':8})
#    leg = plt.legend(loc=2, prop={'size':8})

    fig.legend(handles=handles, labels=labels, bbox_to_anchor=(0.5, 0.95))
#    fig.legend(handles=handles, labels=labels, loc='best')
    plt.xlim([1e14,1e19])
    plt.ylim([1e-11,1e-5])
    fig.savefig("NeutrinoAnalysis/limits.png")
    plt.show()




