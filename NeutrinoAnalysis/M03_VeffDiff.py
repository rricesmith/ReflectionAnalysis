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

from icecream import ic
from scipy.optimize import curve_fit

if __name__ == "__main__":


    added = 'r6km'
    # path_SP_400 = 'NeutrinoAnalysis/output/MJob/400/SP/'
    # path_SP_300 = 'NeutrinoAnalysis/output/MJob/300/SP/'
    # path_MB_400 = 'NeutrinoAnalysis/output/MJob/400/MB/'
    # path_MB_300 = 'NeutrinoAnalysis/output/MJob/300/MB/'
    # path_SP = f'NeutrinoAnalysis/output/MJob/combined/SP/{added}/'
    # path_SP = f'NeutrinoAnalysis/output/MJob/combined/SP/'
    # path_MB = f'NeutrinoAnalysis/output/MJob/combined/MB/'
    path_SP = f'NeutrinoAnalysis/output/MJob/{added}/SP/'
    path_MB = f'NeutrinoAnalysis/output/MJob/{added}/MB/'

    # SP_400_data = get_Veff_Aeff(path_SP_400)
    # SP_300_data = get_Veff_Aeff(path_SP_300)
    # MB_400_data = get_Veff_Aeff(path_MB_400)
    # MB_300_data = get_Veff_Aeff(path_MB_300)
    SP_data = get_Veff_Aeff(path_SP)
    MB_data = get_Veff_Aeff(path_MB)

    # SP_400_Veffs, SP_400_energies, SP_400_energies_low, SP_400_energies_up, SP_400_zenith_bins, SP_400_utrigger_names = get_Veff_Aeff_array(SP_400_data)
    # SP_300_Veffs, SP_300_energies, SP_300_energies_low, SP_300_energies_up, SP_300_zenith_bins, SP_300_utrigger_names = get_Veff_Aeff_array(SP_300_data)
    # MB_400_Veffs, MB_400_energies, MB_400_energies_low, MB_400_energies_up, MB_400_zenith_bins, MB_400_utrigger_names = get_Veff_Aeff_array(MB_400_data)
    # MB_300_Veffs, MB_300_energies, MB_300_energies_low, MB_300_energies_up, MB_300_zenith_bins, MB_300_utrigger_names = get_Veff_Aeff_array(MB_300_data)
    SP_Veffs, SP_energies, SP_energies_low, SP_energies_up, SP_zenith_bins, SP_utrigger_names = get_Veff_Aeff_array(SP_data)
    MB_Veffs, MB_energies, MB_energies_low, MB_energies_up, MB_zenith_bins, MB_utrigger_names = get_Veff_Aeff_array(MB_data)


    # SP_400_Veff_avg = np.average(SP_400_Veffs[:, :, get_index('LPDA_2of4_3.8sigma', SP_400_utrigger_names), 0], axis=1)
    # SP_300_Veff_avg = np.average(SP_300_Veffs[:, :, get_index('LPDA_2of4_4.4sigma', SP_300_utrigger_names), 0], axis=1)
    # MB_400_Veff_avg = np.average(MB_400_Veffs[:, :, get_index('LPDA_2of4_3.8sigma', MB_400_utrigger_names), 0], axis=1)
    # MB_300_Veff_avg = np.average(MB_300_Veffs[:, :, get_index('LPDA_2of4_4.4sigma', MB_300_utrigger_names), 0], axis=1)
    SP_400_Veff_avg = np.average(SP_Veffs[:, :, get_index('LPDA_2of4_3.8sigma', SP_utrigger_names), 0], axis=1)
    SP_300_Veff_avg = np.average(SP_Veffs[:, :, get_index('LPDA_2of4_4.4sigma', SP_utrigger_names), 0], axis=1)
    MB_400_Veff_avg = np.average(MB_Veffs[:, :, get_index('LPDA_2of4_3.8sigma', MB_utrigger_names), 0], axis=1)
    MB_300_Veff_avg = np.average(MB_Veffs[:, :, get_index('LPDA_2of4_4.4sigma', MB_utrigger_names), 0], axis=1)


    SP_400_Veff_water = get_Veff_water_equivalent(SP_400_Veff_avg) * 4 * np.pi
    SP_300_Veff_water = get_Veff_water_equivalent(SP_300_Veff_avg) * 4 * np.pi
    MB_400_Veff_water = get_Veff_water_equivalent(MB_400_Veff_avg) * 4 * np.pi
    MB_300_Veff_water = get_Veff_water_equivalent(MB_300_Veff_avg) * 4 * np.pi

    # SP_400_Veff_error = SP_400_Veff_water / np.sum(SP_400_Veffs[:, :, get_index('LPDA_2of4_3.8sigma', SP_400_utrigger_names), 2], axis=1) ** 0.5
    # SP_300_Veff_error = SP_300_Veff_water / np.sum(SP_300_Veffs[:, :, get_index('LPDA_2of4_4.4sigma', SP_300_utrigger_names), 2], axis=1) ** 0.5
    # MB_400_Veff_error = MB_400_Veff_water / np.sum(MB_400_Veffs[:, :, get_index('LPDA_2of4_3.8sigma', MB_400_utrigger_names), 2], axis=1) ** 0.5
    # MB_300_Veff_error = MB_300_Veff_water / np.sum(MB_300_Veffs[:, :, get_index('LPDA_2of4_4.4sigma', MB_300_utrigger_names), 2], axis=1) ** 0.5
    SP_400_Veff_error = SP_400_Veff_water / np.sum(SP_Veffs[:, :, get_index('LPDA_2of4_3.8sigma', SP_utrigger_names), 2], axis=1) ** 0.5
    SP_300_Veff_error = SP_300_Veff_water / np.sum(SP_Veffs[:, :, get_index('LPDA_2of4_4.4sigma', SP_utrigger_names), 2], axis=1) ** 0.5
    MB_400_Veff_error = MB_400_Veff_water / np.sum(MB_Veffs[:, :, get_index('LPDA_2of4_3.8sigma', MB_utrigger_names), 2], axis=1) ** 0.5
    MB_300_Veff_error = MB_300_Veff_water / np.sum(MB_Veffs[:, :, get_index('LPDA_2of4_4.4sigma', MB_utrigger_names), 2], axis=1) ** 0.5

    # ic(SP_300_Veff_error)
    # ic(SP_300_Veff_error / SP_300_)
    # quit()

    np.nan_to_num(SP_400_Veff_error, copy=False)
    np.nan_to_num(SP_300_Veff_error, copy=False)
    np.nan_to_num(MB_400_Veff_error, copy=False)
    np.nan_to_num(MB_300_Veff_error, copy=False)

    SP_400_mask = SP_400_Veff_water > 0
    SP_300_mask = SP_300_Veff_water > 0
    MB_400_mask = MB_400_Veff_water > 0
    MB_300_mask = MB_300_Veff_water > 0
    ic(SP_300_Veff_error, SP_400_Veff_error)
    ic(SP_400_mask, SP_300_mask, MB_400_mask, MB_300_mask)


    def func(x, a, b, c):
        return a * np.exp( b * x ) + c
    fit_energies = np.logspace(17, 20, num=100) * units.eV
    # SP_400_fit_coefs = np.polynomial.polynomial.polyfit(SP_energies/units.eV, SP_400_Veff_water/units.km**3/units.sr, 13, full=False, w=SP_400_Veff_error/units.km**3/units.sr)
    # SP_400_popt, SP_400_pcov = curve_fit(func, SP_energies/units.eV, SP_400_Veff_water/units.km**3/units.sr, sigma=SP_400_Veff_error/units.km**3/units.sr)
    deg = 1
    # SP_400_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(SP_energies[SP_400_mask]/units.eV), np.log10(SP_400_Veff_water[SP_400_mask]/units.km**3/units.sr), deg, full=False, w=SP_400_Veff_error[SP_400_mask]/units.km**3/units.sr)
    # SP_300_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(SP_energies[SP_300_mask]/units.eV), np.log10(SP_300_Veff_water[SP_300_mask]/units.km**3/units.sr), deg, full=False, w=SP_300_Veff_error[SP_300_mask]/units.km**3/units.sr)
    # MB_400_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(MB_energies[MB_400_mask]/units.eV), np.log10(MB_400_Veff_water[MB_400_mask]/units.km**3/units.sr), deg, full=False, w=MB_400_Veff_error[MB_400_mask]/units.km**3/units.sr)
    # MB_300_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(MB_energies[MB_300_mask]/units.eV), np.log10(MB_300_Veff_water[MB_300_mask]/units.km**3/units.sr), deg, full=False, w=MB_300_Veff_error[MB_300_mask]/units.km**3/units.sr)
    SP_400_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(SP_energies[SP_400_mask]/units.eV), np.log10(SP_400_Veff_water[SP_400_mask]/units.km**3/units.sr), deg, full=False)
    SP_300_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(SP_energies[SP_300_mask]/units.eV), np.log10(SP_300_Veff_water[SP_300_mask]/units.km**3/units.sr), deg, full=False)
    MB_400_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(MB_energies[MB_400_mask]/units.eV), np.log10(MB_400_Veff_water[MB_400_mask]/units.km**3/units.sr), deg, full=False)
    MB_300_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(MB_energies[MB_300_mask]/units.eV), np.log10(MB_300_Veff_water[MB_300_mask]/units.km**3/units.sr), deg, full=False)

    # SP_400_fit = np.polynomial.polynomial.polyval(fit_energies, SP_400_fit_coefs)
    SP_400_fit_lglg = 10**np.polynomial.polynomial.polyval(np.log10(fit_energies), SP_400_fit_lglg_coeff) * units.km**3/units.sr
    SP_400_fit_lglg[:np.argmin(SP_400_fit_lglg)] = np.NAN
    SP_300_fit_lglg = 10**np.polynomial.polynomial.polyval(np.log10(fit_energies), SP_300_fit_lglg_coeff) * units.km**3/units.sr
    SP_300_fit_lglg[:np.argmin(SP_300_fit_lglg)] = np.NAN
    MB_400_fit_lglg = 10**np.polynomial.polynomial.polyval(np.log10(fit_energies), MB_400_fit_lglg_coeff) * units.km**3/units.sr
    MB_400_fit_lglg[:np.argmin(MB_400_fit_lglg)] = np.NAN
    MB_300_fit_lglg = 10**np.polynomial.polynomial.polyval(np.log10(fit_energies), MB_300_fit_lglg_coeff) * units.km**3/units.sr
    MB_300_fit_lglg[:np.argmin(MB_300_fit_lglg)] = np.NAN

    fig, ax = plt.subplots(1, 2)
    # ax[0].errorbar(SP_400_energies/units.eV, SP_400_Veff_water/units.km**3/units.sr, yerr=SP_400_Veff_error/units.km**3/units.sr, fmt='d-', label='SP 400s', linestyle='-', capsize=10)
    # ax[0].errorbar(SP_300_energies/units.eV, SP_300_Veff_water/units.km**3/units.sr, yerr=SP_300_Veff_error/units.km**3/units.sr, fmt='d-', label='SP 300s', linestyle='-', capsize=10)
    # ax[1].errorbar(MB_400_energies/units.eV, MB_400_Veff_water/units.km**3/units.sr, yerr=MB_400_Veff_error/units.km**3/units.sr, fmt='d-', label='MB 400s', linestyle='-', capsize=10)
    # ax[1].errorbar(MB_300_energies/units.eV, MB_300_Veff_water/units.km**3/units.sr, yerr=MB_300_Veff_error/units.km**3/units.sr, fmt='d-', label='MB 300s', linestyle='-', capsize=10)
    ax[0].errorbar(SP_energies[SP_400_mask]/units.eV, SP_400_Veff_water[SP_400_mask]/units.km**3/units.sr, yerr=SP_400_Veff_error[SP_400_mask]/units.km**3/units.sr, fmt='d-', label='400s+ML Trigger', linestyle='-', capsize=10)
    ax[0].errorbar(SP_energies[SP_300_mask]/units.eV, SP_300_Veff_water[SP_300_mask]/units.km**3/units.sr, yerr=SP_300_Veff_error[SP_300_mask]/units.km**3/units.sr, fmt='d-', label='300s', linestyle='-', capsize=10)
    # ax[0].plot(fit_energies/units.eV, SP_400_fit, linestyle='--', label='polyfit')
    ax[0].plot(fit_energies/units.eV, SP_400_fit_lglg/units.km**3/units.sr, linestyle='--', label='400 fit')
    ax[0].plot(fit_energies/units.eV, SP_300_fit_lglg/units.km**3/units.sr, linestyle='--', label='300 fit')
    # ax[0].plot(fit_energies/units.eV, func(fit_energies/units.eV, *SP_400_popt), 'r-', label='curve_fit') 
    ax[1].errorbar(MB_energies[MB_400_mask]/units.eV, MB_400_Veff_water[MB_400_mask]/units.km**3/units.sr, yerr=MB_400_Veff_error[MB_400_mask]/units.km**3/units.sr, fmt='d-', label='400s+ML Trigger', linestyle='-', capsize=10)
    ax[1].errorbar(MB_energies[MB_300_mask]/units.eV, MB_300_Veff_water[MB_300_mask]/units.km**3/units.sr, yerr=MB_300_Veff_error[MB_300_mask]/units.km**3/units.sr, fmt='d-', label='300s', linestyle='-', capsize=10)
    ax[1].plot(fit_energies/units.eV, MB_400_fit_lglg/units.km**3/units.sr, linestyle='--', label='400 fit')
    ax[1].plot(fit_energies/units.eV, MB_300_fit_lglg/units.km**3/units.sr, linestyle='--', label='300 fit')
    ax[0].set_title('South Pole')
    ax[1].set_title('Moores Bay')
    plt.legend()
    ax[0].semilogx(True)
    ax[1].semilogx(True)
    ax[0].semilogy(True)
    ax[1].semilogy(True)

    ax[0].set_xlabel('Neutrino Energy (eV)')
    ax[1].set_xlabel('Neutrino Energy (eV)')
    ax[0].set_ylabel('Effective Volume (km$^3$ sr)')
    ax[1].set_ylabel('Effective Volume (km$^3$ sr)')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylim(bottom=10**-3)
    ax[1].set_ylim(bottom=10**-3)

    fig.tight_layout()
    fig.savefig(f'plots/Neutrinos/ManuelThesis/{added}Veffs.pdf')
    ic(f'saved plots/Neutrinos/ManuelThesis/{added}Veffs.pdf')
    plt.clf()

    # Plot the fit curves alone
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(fit_energies/units.eV, SP_400_fit_lglg/units.km**3/units.sr, linestyle='--', label='400 fit')
    ax[0].plot(fit_energies/units.eV, SP_300_fit_lglg/units.km**3/units.sr, linestyle='--', label='300 fit')
    ax[1].plot(fit_energies/units.eV, MB_400_fit_lglg/units.km**3/units.sr, linestyle='--', label='400 fit')
    ax[1].plot(fit_energies/units.eV, MB_300_fit_lglg/units.km**3/units.sr, linestyle='--', label='300 fit')
    ax[0].set_title('South Pole')
    ax[1].set_title('Moores Bay')
    plt.legend()
    ax[0].semilogx(True)
    ax[1].semilogx(True)
    ax[0].semilogy(True)
    ax[1].semilogy(True)

    ax[0].set_xlabel('Neutrino Energy (eV)')
    ax[1].set_xlabel('Neutrino Energy (eV)')
    ax[0].set_ylabel('Effective Volume (km$^3$ sr)')
    ax[1].set_ylabel('Effective Volume (km$^3$ sr)')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylim(bottom=10**-3)
    ax[1].set_ylim(bottom=10**-3)

    fig.tight_layout()
    fig.savefig(f'plots/Neutrinos/ManuelThesis/{added}FittedVeffs.pdf')
    ic(f'saved plots/Neutrinos/ManuelThesis/{added}FittedVeffs.pdf')
    plt.clf()


    # Plot the ratio of the data
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    SP_ratio = SP_400_Veff_water[np.logical_and(SP_400_mask, SP_300_mask)] / SP_300_Veff_water[np.logical_and(SP_400_mask, SP_300_mask)]
    MB_ratio = MB_400_Veff_water[np.logical_and(MB_400_mask, MB_300_mask)] / MB_300_Veff_water[np.logical_and(MB_400_mask, MB_300_mask)]
    SP_ratio_error = SP_ratio * np.sqrt(((SP_400_Veff_error[np.logical_and(SP_400_mask, SP_300_mask)] / SP_400_Veff_water[np.logical_and(SP_400_mask, SP_300_mask)])**2 + (SP_300_Veff_error[np.logical_and(SP_400_mask, SP_300_mask)] / SP_300_Veff_water[np.logical_and(SP_400_mask, SP_300_mask)])**2))
    MB_ratio_error = MB_ratio * np.sqrt(((MB_400_Veff_error[np.logical_and(MB_400_mask, MB_300_mask)] / MB_400_Veff_water[np.logical_and(MB_400_mask, MB_300_mask)])**2 + (MB_300_Veff_error[np.logical_and(MB_400_mask, MB_300_mask)] / MB_300_Veff_water[np.logical_and(MB_400_mask, MB_300_mask)])**2))
    SP_ratio_high_mask = SP_ratio < 5
    MB_ratio_high_mask = MB_ratio < 5
    # ax.plot(SP_400_energies/units.eV, SP_ratio, 'd-', label='South Pole', linestyle='-')
    # ax.plot(MB_400_energies/units.eV, MB_ratio, 'd-', label='Moores Bay', linestyle='-')
    # ax.plot(SP_energies[np.logical_and(SP_400_mask, SP_300_mask)][SP_ratio_high_mask]/units.eV, SP_ratio[SP_ratio_high_mask], 'd-', label='South Pole', linestyle='-')
    # ax.plot(MB_energies[np.logical_and(MB_400_mask, MB_300_mask)][MB_ratio_high_mask]/units.eV, MB_ratio[MB_ratio_high_mask], 'd-', label='Moores Bay', linestyle='-')
    ax.errorbar(SP_energies[np.logical_and(SP_400_mask, SP_300_mask)][SP_ratio_high_mask]/units.eV, SP_ratio[SP_ratio_high_mask], yerr=SP_ratio_error[SP_ratio_high_mask], fmt='d-', label='South Pole', linestyle='-')
    ax.errorbar(MB_energies[np.logical_and(MB_400_mask, MB_300_mask)][MB_ratio_high_mask]/units.eV, MB_ratio[MB_ratio_high_mask], yerr=MB_ratio_error[MB_ratio_high_mask], fmt='d-', label='Moores Bay', linestyle='-')
    ax.semilogx(True)
    ax.legend()
    ax.set_xlabel('Neutrino Energy (eV)')
    ax.set_ylabel('Sensitivity Improvement')
    ax.set_ylim(0, 5)
    fig.tight_layout()
    fig.savefig(f'plots/Neutrinos/ManuelThesis/{added}Ratios.pdf')
    ic(f'saved plots/Neutrinos/ManuelThesis/{added}Ratios.pdf')
    plt.clf()

    # Plot the ratio but condense every 3 data points into a single averaged one
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    MB_rolling_energies = []
    SP_rolling_energies = []
    MB_rolling_ratios = []
    SP_rolling_ratios = []
    MB_rolling_error = []
    SP_rolling_error = []
    avg_num = 4
    for i in range(0, len(SP_energies[np.logical_and(SP_400_mask, SP_300_mask)][SP_ratio_high_mask]), avg_num):
        if not i+avg_num > len(SP_energies[np.logical_and(SP_400_mask, SP_300_mask)][SP_ratio_high_mask]):
            ic(i, i+avg_num, SP_energies[np.logical_and(SP_400_mask, SP_300_mask)][SP_ratio_high_mask][i:i+3]/units.eV, np.mean(SP_energies[np.logical_and(SP_400_mask, SP_300_mask)][SP_ratio_high_mask][i:i+3]/units.eV))
            SP_rolling_energies.append(np.mean(SP_energies[np.logical_and(SP_400_mask, SP_300_mask)][SP_ratio_high_mask][i:i+avg_num]/units.eV))
            SP_rolling_ratios.append(np.mean(SP_ratio[SP_ratio_high_mask][i:i+avg_num]))        
            SP_rolling_error.append(np.mean(SP_ratio_error[SP_ratio_high_mask][i:i+avg_num])/np.sqrt(avg_num))
        else:
            SP_rolling_energies.append(np.mean(SP_energies[np.logical_and(SP_400_mask, SP_300_mask)][SP_ratio_high_mask][i:]))
            SP_rolling_ratios.append(np.mean(SP_ratio[SP_ratio_high_mask][i:]))
            SP_rolling_error.append(np.mean(SP_ratio_error[SP_ratio_high_mask][i:])/np.sqrt(len(SP_ratio_error[SP_ratio_high_mask][i:])))
    for i in range(0, len(MB_energies[np.logical_and(MB_400_mask, MB_300_mask)][MB_ratio_high_mask]), avg_num):
        if not i+avg_num > len(MB_energies[np.logical_and(MB_400_mask, MB_300_mask)][MB_ratio_high_mask]):
            MB_rolling_energies.append(np.mean(MB_energies[np.logical_and(MB_400_mask, MB_300_mask)][MB_ratio_high_mask][i:i+avg_num]/units.eV))
            MB_rolling_ratios.append(np.mean(MB_ratio[MB_ratio_high_mask][i:i+avg_num]))        
            MB_rolling_error.append(np.mean(MB_ratio_error[MB_ratio_high_mask][i:i+avg_num])/np.sqrt(avg_num))
        else:
            MB_rolling_energies.append(np.mean(MB_energies[np.logical_and(MB_400_mask, MB_300_mask)][MB_ratio_high_mask][i:]))
            MB_rolling_ratios.append(np.mean(MB_ratio[MB_ratio_high_mask][i:]))
            MB_rolling_error.append(np.mean(MB_ratio_error[MB_ratio_high_mask][i:])/np.sqrt(len(MB_ratio_error[MB_ratio_high_mask][i:])))
    # ax.plot(SP_rolling_energies, SP_rolling_ratios, 'd-', label='South Pole', linestyle='-')
    # ax.plot(MB_rolling_energies, MB_rolling_ratios, 'd-', label='Moores Bay', linestyle='-')
    ax.errorbar(SP_rolling_energies, SP_rolling_ratios, yerr=SP_rolling_error, fmt='d-', label='South Pole', linestyle='', capsize=3, capthick=1, alpha=0.75)
    ax.fill_between(SP_rolling_energies, np.array(SP_rolling_ratios)-np.array(SP_rolling_error), np.array(SP_rolling_ratios)+np.array(SP_rolling_error), alpha=0.25)
    ax.errorbar(MB_rolling_energies, MB_rolling_ratios, yerr=MB_rolling_error, fmt='d-', label='Moores Bay', linestyle='', capsize=3, capthick=1, alpha=0.75)
    ax.fill_between(MB_rolling_energies, np.array(MB_rolling_ratios)-np.array(MB_rolling_error), np.array(MB_rolling_ratios)+np.array(MB_rolling_error), alpha=0.25)
    ax.semilogx(True)
    ax.legend()
    ax.set_ylim(1, 3)
    ax.set_xlabel('Neutrino Energy (eV)')
    ax.set_ylabel('Sensitivity Improvement')
    fig.tight_layout()
    fig.savefig(f'plots/Neutrinos/ManuelThesis/{added}RatiosAverages.pdf')
    ic(f'saved plots/Neutrinos/ManuelThesis/{added}RatiosAverages.pdf')
    plt.clf()

    # Plot the ratio of the fits
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    SP_ratio = SP_400_fit_lglg / SP_300_fit_lglg
    MB_ratio = MB_400_fit_lglg / MB_300_fit_lglg
    ax.plot(fit_energies/units.eV, SP_ratio, label='South Pole', linestyle='-')
    ax.plot(fit_energies/units.eV, MB_ratio, label='Moores Bay', linestyle='-')
    ax.semilogx(True)
    ax.legend()
    ax.set_xlabel('Neutrino Energy (eV)')
    ax.set_ylabel('Sensitivity Improvement')
    fig.tight_layout()
    fig.savefig(f'plots/Neutrinos/ManuelThesis/{added}FittedRatios.pdf')
    ic(f'saved plots/Neutrinos/ManuelThesis/{added}FittedRatios.pdf')
    plt.clf()


    # Plot the flux limits for 30 stations 10 years
    fig, ax = limits.get_E2_limit_figure(diffuse=True, show_grand_10k=False, show_grand_200k=False, show_RNOG=True, show_ara=False, show_anita_I_IV_limit=False, show_auger_limit=False)

    labels = []
    # labels.append(limits.add_limit(ax, labels, SP_400_energies, SP_400_Veff_water, 30, 'New SP', livetime=10*units.year, linestyle='-', color='purple', linewidth=3))
    # labels.append(limits.add_limit(ax, labels, SP_300_energies, SP_300_Veff_water, 30, 'Old SP', livetime=10*units.year, linestyle='--', color='purple', linewidth=3))
    # labels.append(limits.add_limit(ax, labels, MB_400_energies, MB_400_Veff_water, 30, 'New MB', livetime=10*units.year, linestyle='-', color='blue', linewidth=3))
    # labels.append(limits.add_limit(ax, labels, MB_300_energies, MB_300_Veff_water, 30, 'Old MB', livetime=10*units.year, linestyle='--', color='blue', linewidth=3))
    labels.append(limits.add_limit(ax, labels, SP_energies[SP_400_mask], SP_400_Veff_water[SP_400_mask], 30, 'New SP', livetime=10*units.year, linestyle='-', color='purple', linewidth=3))
    labels.append(limits.add_limit(ax, labels, SP_energies[SP_300_mask], SP_300_Veff_water[SP_300_mask], 30, 'Old SP', livetime=10*units.year, linestyle='--', color='purple', linewidth=3))
    labels.append(limits.add_limit(ax, labels, MB_energies[MB_400_mask], MB_400_Veff_water[MB_400_mask], 30, 'New MB', livetime=10*units.year, linestyle='-', color='blue', linewidth=3))
    labels.append(limits.add_limit(ax, labels, MB_energies[MB_300_mask], MB_300_Veff_water[MB_300_mask], 30, 'Old MB', livetime=10*units.year, linestyle='--', color='blue', linewidth=3))
    ax.set_title('Assuming 30 stations 10year livetime')
    leg = plt.legend(loc=2, prop={'size':8})
    fig.savefig(f'plots/Neutrinos/ManuelThesis/{added}limits.pdf')
    ic(f'saved plots/Neutrinos/ManuelThesis/{added}limits.pdf')
    plt.clf()

    # Plot the flux limits for 30 stations 10 years with fits
    fig, ax = limits.get_E2_limit_figure(diffuse=True, show_grand_10k=False, show_grand_200k=False, show_RNOG=True, show_ara=False, show_anita_I_IV_limit=False, show_auger_limit=False)

    labels = []
    labels.append(limits.add_limit(ax, labels, fit_energies, SP_400_fit_lglg, 30, 'New SP', livetime=10*units.year, linestyle='-', color='purple', linewidth=3))
    labels.append(limits.add_limit(ax, labels, fit_energies, SP_300_fit_lglg, 30, 'Old SP', livetime=10*units.year, linestyle='--', color='purple', linewidth=3))
    labels.append(limits.add_limit(ax, labels, fit_energies, MB_400_fit_lglg, 30, 'New MB', livetime=10*units.year, linestyle='-', color='blue', linewidth=3))
    labels.append(limits.add_limit(ax, labels, fit_energies, MB_300_fit_lglg, 30, 'Old MB', livetime=10*units.year, linestyle='--', color='blue', linewidth=3))
    ax.set_title('Assuming 30 stations 10year livetime')
    leg = plt.legend(loc=2, prop={'size':8})
    fig.savefig(f'plots/Neutrinos/ManuelThesis/{added}FittedLimits.pdf')
    ic(f'saved plots/Neutrinos/ManuelThesis/{added}FittedLimits.pdf')
    plt.clf()


    # Find which degree gives best fit
    deg_list = [1, 2, 3, 4, 5]
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(deg_list))))

    # Plot the Veff curves for different deg
    fig, ax = plt.subplots(1, 2)
    for iD, deg in enumerate(deg_list):
        # SP_400_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(SP_energies[SP_400_mask]/units.eV), np.log10(SP_400_Veff_water[SP_400_mask]/units.km**3/units.sr), deg, full=False, w=SP_400_Veff_error[SP_400_mask]/units.km**3/units.sr)
        # SP_300_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(SP_energies[SP_300_mask]/units.eV), np.log10(SP_300_Veff_water[SP_300_mask]/units.km**3/units.sr), deg, full=False, w=SP_300_Veff_error[SP_300_mask]/units.km**3/units.sr)
        # MB_400_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(MB_energies[MB_400_mask]/units.eV), np.log10(MB_400_Veff_water[MB_400_mask]/units.km**3/units.sr), deg, full=False, w=MB_400_Veff_error[MB_400_mask]/units.km**3/units.sr)
        # MB_300_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(MB_energies[MB_300_mask]/units.eV), np.log10(MB_300_Veff_water[MB_300_mask]/units.km**3/units.sr), deg, full=False, w=MB_300_Veff_error[MB_300_mask]/units.km**3/units.sr)
        SP_400_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(SP_energies[SP_400_mask]/units.eV), np.log10(SP_400_Veff_water[SP_400_mask]/units.km**3/units.sr), deg, full=False)
        SP_300_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(SP_energies[SP_300_mask]/units.eV), np.log10(SP_300_Veff_water[SP_300_mask]/units.km**3/units.sr), deg, full=False)
        MB_400_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(MB_energies[MB_400_mask]/units.eV), np.log10(MB_400_Veff_water[MB_400_mask]/units.km**3/units.sr), deg, full=False)
        MB_300_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(MB_energies[MB_300_mask]/units.eV), np.log10(MB_300_Veff_water[MB_300_mask]/units.km**3/units.sr), deg, full=False)

        # SP_400_fit = np.polynomial.polynomial.polyval(fit_energies, SP_400_fit_coefs)
        SP_400_fit_lglg = 10**np.polynomial.polynomial.polyval(np.log10(fit_energies), SP_400_fit_lglg_coeff) * units.km**3/units.sr
        SP_400_fit_lglg[:np.argmin(SP_400_fit_lglg)] = np.NAN
        SP_300_fit_lglg = 10**np.polynomial.polynomial.polyval(np.log10(fit_energies), SP_300_fit_lglg_coeff) * units.km**3/units.sr
        SP_300_fit_lglg[:np.argmin(SP_300_fit_lglg)] = np.NAN
        MB_400_fit_lglg = 10**np.polynomial.polynomial.polyval(np.log10(fit_energies), MB_400_fit_lglg_coeff) * units.km**3/units.sr
        MB_400_fit_lglg[:np.argmin(MB_400_fit_lglg)] = np.NAN
        MB_300_fit_lglg = 10**np.polynomial.polynomial.polyval(np.log10(fit_energies), MB_300_fit_lglg_coeff) * units.km**3/units.sr
        MB_300_fit_lglg[:np.argmin(MB_300_fit_lglg)] = np.NAN

        c = next(color)
        ax[0].plot(fit_energies/units.eV, SP_400_fit_lglg/units.km**3/units.sr, linestyle='-', c=c)
        ax[0].plot(fit_energies/units.eV, SP_300_fit_lglg/units.km**3/units.sr, linestyle='--', c=c)
        ax[1].plot(fit_energies/units.eV, MB_400_fit_lglg/units.km**3/units.sr, linestyle='-', c=c)
        ax[1].plot(fit_energies/units.eV, MB_300_fit_lglg/units.km**3/units.sr, linestyle='--', c=c)
        ax[0].plot([],[], label=f'Degree {deg}', c=c)
    ax[1].plot([],[], linestyle='-', label='400s+ML Trigger', color='black')        
    ax[1].plot([],[], linestyle='--', label='300s', color='black')        
    ax[0].set_title('South Pole')
    ax[1].set_title('Moores Bay')
    plt.legend()
    ax[0].semilogx(True)
    ax[1].semilogx(True)
    ax[0].semilogy(True)
    ax[1].semilogy(True)

    ax[0].set_xlabel('Neutrino Energy (eV)')
    ax[1].set_xlabel('Neutrino Energy (eV)')
    ax[0].set_ylabel('Effective Volume (km$^3$ sr)')
    ax[1].set_ylabel('Effective Volume (km$^3$ sr)')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylim(bottom=10**-3)
    ax[1].set_ylim(bottom=10**-3)

    fig.tight_layout()
    fig.savefig(f'plots/Neutrinos/ManuelThesis/{added}FitSearchVeffs.pdf')
    ic(f'saved plots/Neutrinos/ManuelThesis/{added}FitSearchVeffs.pdf')
    plt.clf()


    # Now tlot the Veff ratios curves for different deg
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(deg_list))))

    fig, ax = plt.subplots(1, 2)
    for iD, deg in enumerate(deg_list):
        # SP_400_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(SP_energies[SP_400_mask]/units.eV), np.log10(SP_400_Veff_water[SP_400_mask]/units.km**3/units.sr), deg, full=False, w=SP_400_Veff_error[SP_400_mask]/units.km**3/units.sr)
        # SP_300_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(SP_energies[SP_300_mask]/units.eV), np.log10(SP_300_Veff_water[SP_300_mask]/units.km**3/units.sr), deg, full=False, w=SP_300_Veff_error[SP_300_mask]/units.km**3/units.sr)
        # MB_400_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(MB_energies[MB_400_mask]/units.eV), np.log10(MB_400_Veff_water[MB_400_mask]/units.km**3/units.sr), deg, full=False, w=MB_400_Veff_error[MB_400_mask]/units.km**3/units.sr)
        # MB_300_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(MB_energies[MB_300_mask]/units.eV), np.log10(MB_300_Veff_water[MB_300_mask]/units.km**3/units.sr), deg, full=False, w=MB_300_Veff_error[MB_300_mask]/units.km**3/units.sr)
        SP_400_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(SP_energies[SP_400_mask]/units.eV), np.log10(SP_400_Veff_water[SP_400_mask]/units.km**3/units.sr), deg, full=False)
        SP_300_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(SP_energies[SP_300_mask]/units.eV), np.log10(SP_300_Veff_water[SP_300_mask]/units.km**3/units.sr), deg, full=False)
        MB_400_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(MB_energies[MB_400_mask]/units.eV), np.log10(MB_400_Veff_water[MB_400_mask]/units.km**3/units.sr), deg, full=False)
        MB_300_fit_lglg_coeff = np.polynomial.polynomial.polyfit(np.log10(MB_energies[MB_300_mask]/units.eV), np.log10(MB_300_Veff_water[MB_300_mask]/units.km**3/units.sr), deg, full=False)

        # SP_400_fit = np.polynomial.polynomial.polyval(fit_energies, SP_400_fit_coefs)
        SP_400_fit_lglg = 10**np.polynomial.polynomial.polyval(np.log10(fit_energies), SP_400_fit_lglg_coeff) * units.km**3/units.sr
        SP_400_fit_lglg[:np.argmin(SP_400_fit_lglg)] = np.NAN
        SP_300_fit_lglg = 10**np.polynomial.polynomial.polyval(np.log10(fit_energies), SP_300_fit_lglg_coeff) * units.km**3/units.sr
        SP_300_fit_lglg[:np.argmin(SP_300_fit_lglg)] = np.NAN
        MB_400_fit_lglg = 10**np.polynomial.polynomial.polyval(np.log10(fit_energies), MB_400_fit_lglg_coeff) * units.km**3/units.sr
        MB_400_fit_lglg[:np.argmin(MB_400_fit_lglg)] = np.NAN
        MB_300_fit_lglg = 10**np.polynomial.polynomial.polyval(np.log10(fit_energies), MB_300_fit_lglg_coeff) * units.km**3/units.sr
        MB_300_fit_lglg[:np.argmin(MB_300_fit_lglg)] = np.NAN

        c = next(color)
        ax[0].plot(fit_energies/units.eV, (SP_400_fit_lglg/units.km**3/units.sr) / (SP_300_fit_lglg/units.km**3/units.sr), linestyle='-', c=c)
        # ax[0].plot(fit_energies/units.eV, SP_300_fit_lglg/units.km**3/units.sr, linestyle='--', c=c)
        ax[1].plot(fit_energies/units.eV, (MB_400_fit_lglg/units.km**3/units.sr) / (MB_300_fit_lglg/units.km**3/units.sr), linestyle='-', c=c)
        # ax[1].plot(fit_energies/units.eV, MB_300_fit_lglg/units.km**3/units.sr, linestyle='--', c=c)
        ax[0].plot([],[], label=f'Degree {deg}', c=c)
        ax[1].plot([],[], label=f'Degree {deg}', c=c)
    # ax[1].plot([],[], linestyle='-', label='400s+ML Trigger', color='black')        
    # ax[1].plot([],[], linestyle='--', label='300s', color='black')        
    ax[0].set_title('South Pole')
    ax[1].set_title('Moores Bay')
    plt.legend()
    ax[0].semilogx(True)
    ax[1].semilogx(True)
    # ax[0].semilogy(True)
    # ax[1].semilogy(True)

    ax[0].set_xlabel('Neutrino Energy (eV)')
    ax[1].set_xlabel('Neutrino Energy (eV)')
    ax[0].set_ylabel('Sensitivity Improvement')
    ax[1].set_ylabel('Sensitivity Improvement')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylim(bottom=1)
    ax[1].set_ylim(bottom=1)

    fig.tight_layout()
    fig.savefig(f'plots/Neutrinos/ManuelThesis/{added}FitSearchVeffsRatios.pdf')
    ic(f'saved plots/Neutrinos/ManuelThesis/{added}FitSearchVeffsRatios.pdf')
    plt.clf()