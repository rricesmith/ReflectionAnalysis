import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import h5py
import glob
from scipy import interpolate
import json
import os
import sys

from NuRadioReco.utilities import units
from NuRadioReco.detector import detector
from NuRadioMC.utilities import fluxes
from NuRadioMC.utilities.Veff import get_Veff_Aeff, get_Veff_Aeff_array, get_index, get_Veff_water_equivalent
from NuRadioMC.examples.Sensitivities import E2_fluxes3 as limits
#plt.switch_backend('agg')

if __name__ == "__main__":

    path_2sigma = 'output/2sigma'
    path_3sigma = 'output/3sigma'
    path_4sigma = 'output/4sigma'
    path_2sigmaGZK = 'output/2sigma_GZK'
    path = path_4sigma

#    SP_path = "output/GL_Bugfig_Orig"
#    GL_path = "output/GL_Bugfig_Dephased"
#    SP_path = 'output/GL_Uniform_with_phasing'
#    SP_path = 'output/SP_Uniform_with_phasing'
#    SP_path = 'output/SP_nu_uniform'
#    SP_path = 'output/SP_neutrinos'
#    MB_path = 'output/MB_neutrinos'

#    SP_path = 'output/GL_Higher_Sampling'
#    GL_path = 'output/GL_Dephased'
#    MB_path = 'output/MB_nu_uniform'
    SP_path = 'output/AttTest_GL3/R12km'
    GL_path = 'output/AttTest_GL2/R12km'
    MB_path = 'output/AttTest_GL1/R12km'

    if(len(sys.argv) == 1):
        print("no path specified, assuming that hdf5 files are in directory 'output'")
    else:
        path = sys.argv[1]

    """
    data = get_Veff_Aeff(path)
    Veffs, energies, energies_low, energies_up, zenith_bins, utrigger_names = get_Veff_Aeff_array(data)
    # calculate the average over all zenith angle bins (in this case only one bin that contains the full sky)
    Veff = np.average(Veffs[:, :, get_index('LPDA_2of4_3.9sigma', utrigger_names), 0], axis=1)
    # we also want the water equivalent effective volume times 4pi
    Veff = get_Veff_water_equivalent(Veff) * 4 * np.pi
    # calculate the uncertainty for the average over all zenith angle bins. The error relative error is just 1./sqrt(N)
    Veff_error = Veff / np.sum(Veffs[:, :, get_index('LPDA_2of4_3.9sigma', utrigger_names), 2], axis=1) ** 0.5
    # plot effective volume
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.errorbar(energies / units.eV, Veff / units.km ** 3 / units.sr,
                yerr=Veff_error / units.km ** 3 / units.sr, fmt='d-')
    ax.semilogx(True)
    ax.semilogy(True)
    ax.set_xlabel("neutrino energy [eV]")
    ax.set_ylabel("effective volume [km$^3$ sr]")
    fig.tight_layout()
    fig.savefig("Veff.pdf")

    # plot expected limit
    fig, ax = limits.get_E2_limit_figure(diffuse=True, show_grand_10k=True, show_grand_200k=False)
    labels = []
    labels = limits.add_limit(ax, labels, energies, Veff,
                              100, 'NuRadioMC example', livetime=3 * units.year, linestyle='-', color='blue', linewidth=3)
    leg = plt.legend(handles=labels, loc=2)
    fig.savefig("limits.pdf")
    plt.show()
    """


    MB_data = get_Veff_Aeff(MB_path)
    SP_data = get_Veff_Aeff(SP_path)
    GL_data = get_Veff_Aeff(GL_path)
    MB_Veffs, MB_energies, MB_energies_low, MB_energies_up, MB_zenith_bins, MB_utrigger_names = get_Veff_Aeff_array(MB_data)
    SP_Veffs, SP_energies, SP_energies_low, SP_energies_up, SP_zenith_bins, SP_utrigger_names = get_Veff_Aeff_array(SP_data)
    GL_Veffs, GL_energies, GL_energies_low, GL_energies_up, GL_zenith_bins, GL_utrigger_names = get_Veff_Aeff_array(GL_data)
#    print(f'SP Veffs {SP_Veffs}')
#    quit()

    GL_Veff_2sigma_avg = np.average(GL_Veffs[:, :, get_index('LPDA_2of4_2sigma_dedispersed', GL_utrigger_names), 0], axis=1)
    MB_Veff_2sigma_avg = np.average(MB_Veffs[:, :, get_index('LPDA_2of4_2sigma_dedispersed', MB_utrigger_names), 0], axis=1)
    SP_Veff_2sigma_avg = np.average(SP_Veffs[:, :, get_index('LPDA_2of4_2sigma_dedispersed', SP_utrigger_names), 0], axis=1)

    """
    MB_Veff_2sigma_avg = np.average(MB_Veffs[:, :, get_index('LPDA_2of4_2sigma', MB_utrigger_names), 0], axis=1)
    MB_Veff_3sigma_avg = np.average(MB_Veffs[:, :, get_index('LPDA_2of4_3sigma', MB_utrigger_names), 0], axis=1)
    MB_Veff_4sigma_avg = np.average(MB_Veffs[:, :, get_index('LPDA_2of4_4sigma', MB_utrigger_names), 0], axis=1)

    SP_Veff_1sigma_avg = np.average(SP_Veffs[:, :, get_index('LPDA_2of4_1sigma', SP_utrigger_names), 0], axis=1)
    SP_Veff_1_5sigma_avg = np.average(SP_Veffs[:, :, get_index('LPDA_2of4_1.5sigma', SP_utrigger_names), 0], axis=1)
    SP_Veff_2sigma_avg = np.average(SP_Veffs[:, :, get_index('LPDA_2of4_2sigma', SP_utrigger_names), 0], axis=1)

#    SP_Veff_1sigma_phased_avg = np.average(SP_Veffs[:, :, get_index('2of5_Dephased_1sigma', SP_utrigger_names), 0], axis=1)
#    SP_Veff_1_5sigma_phased_avg = np.average(SP_Veffs[:, :, get_index('2of5_Dephased_1.5sigma', SP_utrigger_names), 0], axis=1)

#    b4 = SP_Veffs[:, :, get_index('LPDA_2of4_2sigma', SP_utrigger_names), 0]
#    print(f'before average {b4}')
#    print(f'after average {SP_Veff_2sigma_avg}')
    SP_Veff_3sigma_avg = np.average(SP_Veffs[:, :, get_index('LPDA_2of4_3sigma', SP_utrigger_names), 0], axis=1)
    SP_Veff_4sigma_avg = np.average(SP_Veffs[:, :, get_index('LPDA_2of4_4sigma', SP_utrigger_names), 0], axis=1)
    """

    GL_Veff_2sigma = get_Veff_water_equivalent(GL_Veff_2sigma_avg) * 4 * np.pi

    MB_Veff_2sigma = get_Veff_water_equivalent(MB_Veff_2sigma_avg) * 4 * np.pi
#    MB_Veff_3sigma = get_Veff_water_equivalent(MB_Veff_3sigma_avg) * 4 * np.pi
#    MB_Veff_4sigma = get_Veff_water_equivalent(MB_Veff_4sigma_avg) * 4 * np.pi

#    SP_Veff_1sigma = get_Veff_water_equivalent(SP_Veff_1sigma_avg) * 4 * np.pi
#    SP_Veff_1_5sigma = get_Veff_water_equivalent(SP_Veff_1_5sigma_avg) * 4 * np.pi
    SP_Veff_2sigma = get_Veff_water_equivalent(SP_Veff_2sigma_avg) * 4 * np.pi
#    SP_Veff_1sigma_phased = get_Veff_water_equivalent(SP_Veff_1sigma_phased_avg) * 4 * np.pi
#    SP_Veff_1_5sigma_phased = get_Veff_water_equivalent(SP_Veff_1_5sigma_phased_avg) * 4 * np.pi
#    SP_Veff_3sigma = get_Veff_water_equivalent(SP_Veff_3sigma_avg) * 4 * np.pi
#    SP_Veff_4sigma = get_Veff_water_equivalent(SP_Veff_4sigma_avg) * 4 * np.pi

    GL_Veff_error_2sigma = GL_Veff_2sigma / np.sum(GL_Veffs[:, :, get_index('LPDA_2of4_2sigma_dedispersed', GL_utrigger_names), 2], axis=1) ** 0.5

    MB_Veff_error_2sigma = MB_Veff_2sigma / np.sum(MB_Veffs[:, :, get_index('LPDA_2of4_2sigma_dedispersed', MB_utrigger_names), 2], axis=1) ** 0.5
#    MB_Veff_error_3sigma = MB_Veff_3sigma / np.sum(MB_Veffs[:, :, get_index('LPDA_2of4_3sigma', MB_utrigger_names), 2], axis=1) ** 0.5
#    MB_Veff_error_4sigma = MB_Veff_4sigma / np.sum(MB_Veffs[:, :, get_index('LPDA_2of4_4sigma', MB_utrigger_names), 2], axis=1) ** 0.5

#    SP_Veff_error_1sigma = SP_Veff_1sigma / np.sum(SP_Veffs[:, :, get_index('LPDA_2of4_1sigma', SP_utrigger_names), 2], axis=1) ** 0.5
#    SP_Veff_error_1_5sigma = SP_Veff_1_5sigma / np.sum(SP_Veffs[:, :, get_index('LPDA_2of4_1.5sigma', SP_utrigger_names), 2], axis=1) ** 0.5
    SP_Veff_error_2sigma = SP_Veff_2sigma / np.sum(SP_Veffs[:, :, get_index('LPDA_2of4_2sigma_dedispersed', SP_utrigger_names), 2], axis=1) ** 0.5
#    SP_Veff_error_1sigma_phased = SP_Veff_1sigma_phased / np.sum(SP_Veffs[:, :, get_index('2of5_Dephased_1sigma', SP_utrigger_names), 2], axis=1) ** 0.5
#    SP_Veff_error_1_5sigma_phased = SP_Veff_1_5sigma_phased / np.sum(SP_Veffs[:, :, get_index('2of5_Dephased_1.5sigma', SP_utrigger_names), 2], axis=1) ** 0.5
#    print(f'error squared {SP_Veff_error_2sigma**2}')
#    quit()
#    SP_Veff_error_3sigma = SP_Veff_3sigma / np.sum(SP_Veffs[:, :, get_index('LPDA_2of4_3sigma', SP_utrigger_names), 2], axis=1) ** 0.5
#    SP_Veff_error_4sigma = SP_Veff_4sigma / np.sum(SP_Veffs[:, :, get_index('LPDA_2of4_4sigma', SP_utrigger_names), 2], axis=1) ** 0.5

    np.nan_to_num(GL_Veff_error_2sigma, copy=False)

    np.nan_to_num(MB_Veff_error_2sigma, copy=False)
#    np.nan_to_num(MB_Veff_error_3sigma, copy=False)
#    np.nan_to_num(MB_Veff_error_4sigma, copy=False)

#    np.nan_to_num(SP_Veff_error_1sigma, copy=False)
#    np.nan_to_num(SP_Veff_error_1_5sigma, copy=False)
    np.nan_to_num(SP_Veff_error_2sigma, copy=False)
#    np.nan_to_num(SP_Veff_error_1sigma_phased, copy=False)
#    np.nan_to_num(SP_Veff_error_1_5sigma_phased, copy=False)
#    np.nan_to_num(SP_Veff_error_3sigma, copy=False)
#    np.nan_to_num(SP_Veff_error_4sigma, copy=False)



    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.errorbar(GL_energies / units.eV, GL_Veff_2sigma / units.km ** 3 / units.sr,
                yerr=SP_Veff_error_2sigma / units.km ** 3 / units.sr, fmt='d-', label='Dephased', linestyle='-', color='blue', capsize=10)
    ax.errorbar(MB_energies / units.eV, MB_Veff_2sigma / units.km ** 3 / units.sr,
                yerr=MB_Veff_error_2sigma / units.km ** 3 / units.sr, fmt='d-', label='MB 2sigma', linestyle='-', color='red')
#    ax.errorbar(MB_energies / units.eV, MB_Veff_3sigma / units.km ** 3 / units.sr,
#                yerr=MB_Veff_error_3sigma / units.km ** 3 / units.sr, fmt='d-', label='MB 3sigma', linestyle='--', color='blue')
#    ax.errorbar(MB_energies / units.eV, MB_Veff_4sigma / units.km ** 3 / units.sr,
#                yerr=MB_Veff_error_4sigma / units.km ** 3 / units.sr, fmt='d-', label='MB 4sigma', linestyle='-.', color='blue')
#    ax.errorbar(SP_energies / units.eV, SP_Veff_1sigma / units.km ** 3 / units.sr,
#                yerr=SP_Veff_error_1sigma / units.km ** 3 / units.sr, fmt='d-', label='GL 1sigma', linestyle='-', color='blue', capsize=10)
#    ax.errorbar(SP_energies / units.eV, SP_Veff_1_5sigma / units.km ** 3 / units.sr,
#                yerr=SP_Veff_error_1_5sigma / units.km ** 3 / units.sr, fmt='d-', label='SP 1.5sigma', linestyle='--', color='red')
    ax.errorbar(SP_energies / units.eV, SP_Veff_2sigma / units.km ** 3 / units.sr,
                yerr=SP_Veff_error_2sigma / units.km ** 3 / units.sr, fmt='d-', label='2sigma', linestyle='-', color='purple', capsize=10)
#    ax.errorbar(SP_energies / units.eV, SP_Veff_1sigma_phased / units.km ** 3 / units.sr,
#                yerr=SP_Veff_error_1sigma_phased / units.km ** 3 / units.sr, fmt='d-', label='SP 1sigma w/dipole', linestyle='-', color='blue')
#    ax.errorbar(SP_energies / units.eV, SP_Veff_1_5sigma_phased / units.km ** 3 / units.sr,
#                yerr=SP_Veff_error_1_5sigma_phased / units.km ** 3 / units.sr, fmt='d-', label='SP 1.5sigma w/dipole', linestyle='--', color='blue')
#    ax.errorbar(SP_energies / units.eV, SP_Veff_3sigma / units.km ** 3 / units.sr,
#                yerr=SP_Veff_error_3sigma / units.km ** 3 / units.sr, fmt='d-', label='SP 3sigma', linestyle='--', color='red')
#    ax.errorbar(SP_energies / units.eV, SP_Veff_4sigma / units.km ** 3 / units.sr,
#                yerr=SP_Veff_error_4sigma / units.km ** 3 / units.sr, fmt='d-', label='4sigma', linestyle='-', color='red', capsize=10)
    plt.legend()
    ax.semilogx(True)
    ax.semilogy(True)
#    ax.set_ylim(10 ** -3, 10)
    ax.set_ylim(10 ** -6, 10)
    ax.set_xlabel("neutrino energy [eV]")
    ax.set_ylabel("effective volume [km$^3$ sr]")
    fig.tight_layout()
    fig.savefig("Veff.pdf")

    #Doing veff as ratio plot to SP 4sigma
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(GL_energies / units.eV, GL_Veff_2sigma, label='GL2', linestyle='-', color='blue')
    ax.plot(MB_energies / units.eV, MB_Veff_2sigma, label='GL1', linestyle='-', color='red')
#    ax.plot(MB_energies / units.eV, MB_Veff_3sigma / MB_Veff_4sigma, label='MB 3sigma', linestyle='--', color='blue')
#    ax.plot(MB_energies / units.eV, MB_Veff_4sigma / MB_Veff_4sigma, label='MB 4sigma', linestyle='-.', color='blue')
#    ax.plot(SP_energies / units.eV, SP_Veff_1sigma / SP_Veff_4sigma, label='GL 1sigma', linestyle='-', color='blue')
#    ax.plot(SP_energies / units.eV, SP_Veff_1_5sigma / SP_Veff_4sigma, label='SP 1.5sigma', linestyle='--', color='red')
    ax.plot(SP_energies / units.eV, SP_Veff_2sigma, label='GL3', linestyle='-', color='purple')
#    ax.plot(SP_energies / units.eV, SP_Veff_1sigma_phased / SP_Veff_4sigma, label='SP 1sigma w/dip', linestyle='-', color='blue')
#    ax.plot(SP_energies / units.eV, SP_Veff_1_5sigma_phased / SP_Veff_4sigma, label='SP 1.5sigma w/dip', linestyle='--', color='blue')
#    ax.plot(SP_energies / units.eV, SP_Veff_3sigma / MB_Veff_4sigma, label='SP 3sigma', linestyle='--', color='red')
#    ax.plot(SP_energies / units.eV, SP_Veff_4sigma / SP_Veff_4sigma, label='4sigma', linestyle='-', color='red')
    plt.legend()
    ax.semilogx(True)
#    ax.semilogy(True)
#    ax.semilogy(True)
    ax.set_xlabel("neutrino energy [eV]")
    ax.set_ylabel("improvement from MB 4sigma")
    fig.tight_layout()
    fig.savefig("plots/Veff_ratio.png")



#    livetime = 100
    livetime = 10 * units.year
    n_stations = 60
    #Trying N neutrino plot
    print(f'SP energies {SP_energies}')
    print(f'log SP energies {np.log10(SP_energies)}')
    Nnu = fluxes.get_number_of_events_for_flux(SP_energies, limits.ice_cube_nu_fit(SP_energies), SP_Veff_2sigma, n_stations * livetime)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(MB_energies, Nnu)
    ax.semilogx(True)
    fig.tight_layout()
    fig.savefig("plots/Num_neutrinos.png")


    # plot expected limit
    fig, ax = limits.get_E2_limit_figure(diffuse=True, show_grand_10k=False, show_grand_200k=False, show_RNOG=True, show_ara=False,
                                         show_anita_I_IV_limit=False, show_auger_limit=False)

    print(f'energies {SP_energies}')
    print(f'Veff 2sigma {SP_Veff_2sigma}')
    labels = []
#    labels = limits.add_limit(ax, labels, SP_energies, SP_Veff_2sigma,
#                              n_stations, '2sigma', livetime=livetime, linestyle='-', color='purple', linewidth=3)

#    limits.add_limit(ax, labels, GL_energies, GL_Veff_2sigma,
#                              n_stations, r'Dephased 2$\sigma$', livetime=livetime, linestyle='-', color='blue', linewidth=3)

    labels.append(limits.add_limit(ax, labels, SP_energies, SP_Veff_2sigma,
                              n_stations, r'GL3', livetime=livetime, linestyle='-', color='purple', linewidth=3))
    labels.append(limits.add_limit(ax, labels, GL_energies, GL_Veff_2sigma,
                              n_stations, r'GL2', livetime=livetime, linestyle='-', color='red', linewidth=3)[0])
    labels = limits.add_limit(ax, labels, MB_energies, MB_Veff_2sigma,
                              n_stations, 'GL1', livetime=livetime, linestyle='-', color='blue', linewidth=3)

#    labels.append(limits.add_limit(ax, labels, MB_energies, MB_Veff_3sigma,
#                              n_stations, 'MB 3sigma', livetime=livetime, linestyle='--', color='blue', linewidth=3))
#    labels.append(limits.add_limit(ax, labels, MB_energies, MB_Veff_4sigma,
#                              n_stations, 'MB 4sigma', livetime=livetime, linestyle='-.', color='blue', linewidth=3))
#    labels.append(limits.add_limit(ax, labels, SP_energies, SP_Veff_2sigma,
#                              n_stations, r'2$\sigma$', livetime=livetime, linestyle='-', color='purple', linewidth=3)[0])
#    labels.append(limits.add_limit(ax, labels, SP_energies, SP_Veff_3sigma,
#                              n_stations, 'SP 3sigma', livetime=livetime, linestyle='--', color='red', linewidth=3))
#    limits.add_limit(ax, labels, SP_energies, SP_Veff_1sigma,
#                              n_stations, r'1$\sigma$', livetime=livetime, linestyle='-', color='green', linewidth=3)
#    limits.add_limit(ax, labels, SP_energies, SP_Veff_1_5sigma,
 #                             n_stations, r'1.5$\sigma$', livetime=livetime, linestyle='-', color='blue', linewidth=3)
#    labels.append(limits.add_limit(ax, labels, SP_energies, SP_Veff_1sigma_phased,
#                              n_stations, 'SP 1sigma w/dip', livetime=livetime, linestyle='-', color='blue', linewidth=3))
#    labels.append(limits.add_limit(ax, labels, SP_energies, SP_Veff_1_5sigma_phased,
#                              n_stations, 'SP 1.5sigma w/dip', livetime=livetime, linestyle='-.', color='blue', linewidth=3))
#    labels = limits.add_limit(ax, labels, SP_energies, SP_Veff_4sigma,
#                              n_stations, r'4$\sigma$', livetime=livetime, linestyle='-', color='red', linewidth=3)
    print(f'labels {labels}')
#    plt.title(f'{n_stations} Future Stations with {livetime/units.year} years livetime')
#    leg = plt.legend(handles=labels, loc=2)
    leg = plt.legend(loc=2, prop={'size':8})
    fig.savefig("plots/limits.png")
#    plt.show()
    print(f'Done')



