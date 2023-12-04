from astrotools import auger
import os
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from radiotools import plthelpers as php
import pickle

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

flux = "TA_19"
#flux = "auger_19"
#atm_overburden = 1
#atm_overburden = 680
atm_overburden = 1000


# from https://arxiv.org/abs/1811.04660
def get_L(E):
    logE = np.log10(E)
    if(logE < 18.0):
        return 226.2
    elif(logE < 18.2):
        return 227.6
    elif(logE < 18.5):
        return 229.1
    elif(logE < 18.8):
        return 231.4
    elif(logE < 19.2):
        return 233.3
    elif(logE >= 19.2):
        return 238.3


def get_R(E):
    logE = np.log10(E)
    if(logE < 18.0):
        return 0.26
    elif(logE < 18.2):
        return 0.244
    elif(logE < 18.5):
        return 0.252
    elif(logE < 18.8):
        return 0.267
    elif(logE < 19.2):
        return 0.264
    elif(logE >= 19.2):
        return 0.264


def gh(X, Xmax, E):
    X2 = X - Xmax
    R = get_R(E)
    L = get_L(E)
    return (1 + R * X2 / L) ** (R ** -2) * np.exp(-X2 / R / L)


key_year = 'moments%s' % 19
d = auger.DXMAX['%s' % key_year]

log10e = d['meanLgEnergy']
m_x = d['meanXmax']
s_x = d['sigmaXmax']


get_xmax = interp1d(log10e, m_x, fill_value="extrapolate")
get_var_xmax = interp1d(log10e, s_x, fill_value="extrapolate")

    

#Below plots the xmax distribution for energies of given year auger, in this case 2019
plt.errorbar(log10e, m_x, yerr=s_x)
plt.savefig('plots/shower_rate/Xmax_dist.png')
plt.clf()

logEs_bin_edges = np.arange(17, 20.01, 0.1)
logEs = 0.5 * (logEs_bin_edges[1:] + logEs_bin_edges[:-1])
plt.errorbar(logEs, get_xmax(logEs), yerr=get_var_xmax(logEs), fmt='o', elinewidth=3, capsize=4)
plt.xlabel('$log_{10}$($E_{cr}$ (eV))', fontsize=16)
plt.ylabel('Mean Xmax (g/$cm^{2}$)')
plt.ylim([545, 855])
plt.xlim([17, 20])
plt.savefig('plots/shower_rate/Xmax_interp_dist.png')
plt.clf()

if 0:
    # test GH
    xx = np.linspace(0, 1500, 1000)
    Xmax = 800
    frac = quad(gh, 0, 1300, args=(Xmax, 1e18))[0] / quad(gh, 0, np.infty, args=(Xmax, 1e18))[0]
    print(1 - frac)
    yy = gh(xx, Xmax, 1e18)

    fig, ax = plt.subplots(1, 1)
    ax.plot(xx, yy)
    plt.show()


def spectrum_analytic(log10e, type="auger_19"):
    """
    Returns a analytic parametrization of the Auger energy spectrum
    units are 1/(eV km^2 sr yr) for cosmic-ray energy in log10(E / eV)

    :param log10e: Input energies (in log10(E / eV))
    :param year: take ICRC 15, 17 or 19 data
    :return: analytic parametrization of spectrum for given input energies
    """
    # noinspection PyTypeChecker
    energy = 10 ** log10e  # type: np.ndarray
    if type == "auger_17":
        p = auger.SPECTRA_DICT_ANA[17]  # type: np.ndarray
        return np.where(energy < p[1],
                        p[0] * (energy / p[1]) ** (-p[3]),
                        p[0] * (energy / p[1]) ** (-p[4]) * (1 + (p[1] / p[2]) ** p[5])
                        * (1 + (energy / p[2]) ** p[5]) ** -1)
    elif type == "auger_19":
        p = auger.SPECTRA_DICT_ANA[19]  # type: np.ndarray
        return (energy / p[0]) ** (-p[5]) * \
               (1 + (energy / p[1]) ** p[5]) / (1 + (energy / p[1]) ** p[6]) * \
               (1 + (energy / p[2]) ** p[6]) / (1 + (energy / p[2]) ** p[7]) * \
               (1 + (energy / p[3]) ** p[7]) / (1 + (energy / p[3]) ** p[8]) * \
               (1 + (energy / p[4]) ** p[8]) / (1 + (energy / p[4]) ** p[9])
    elif type == "TA_19":

        p1 = -3.28
        p2 = -2.68
        p3 = -4.84
        E1 = 10 ** 18.69
        E2 = 10 ** 19.81
        c = 2.24e-30
        c1 = c * (E1 / 1e18) ** p1
        c2 = c1 * (E2 / E1) ** p2
        yy = np.where(energy < E1,
                 c * (energy / 1e18) ** p1,
                 np.where(energy < E2,
                          c1 * (energy / E1) ** p2,
                          c2 * (energy / E2) ** p3))
        # convert 1/m**2 to 1/km**2 abd second to year
        return yy * 1e6 * 3.154 * 10 ** 7


def event_rate(log10e_min, log10e_max=21, dCos=1, area=1, type="auger_17"):
    """
    Cosmic ray event rate in specified energy range assuming a detector with area
    'area' and maximum zenith angle cut 'zmax'. Uses AUGERs energy spectrum.

    :param log10e_min: lower energy for energy range, in units log10(energy/eV)
    :param log10e_max: upper energy for energy range, in units log10(energy/eV)
    :param zmax: maximum zenith angle in degree (default: 60)
    :param area: detection area in square kilometer (default: Auger, 3000 km^2)
    :param year: take ICRC 15 or 17 data
    :return: event rate in units (1 / year)
    """

    def flux(x):
        """ Bring parametrized energy spectrum in right shape for quad() function """
        return spectrum_analytic(np.log10(np.array([x])), type)[0]

    # integradted flux in units 1 / (sr km^2 year)
    integrated_flux = quad(flux, 10 ** log10e_min, 10 ** log10e_max)[0]
    omega = 2 * np.pi * dCos  # solid angle in sr

    return integrated_flux * omega * area

def plotXmaxDist():
    Xmean_SP = []
    Xmean_MB = []
    Xstd = []
    for Ebin in logEs_bin_edges:
        Xmean_SP.append( 680 - get_xmax(Ebin))
        Xmean_MB.append( 1000 - get_xmax(Ebin))
        Xstd.append( get_var_xmax(Ebin))

#    plt.plot(logEs_bin_edges, Xmean, label='X mean')
#    plt.plot(logEs_bin_edges, Xstd, label='X std')
    plt.errorbar(logEs_bin_edges, Xmean_SP, yerr=Xstd, label='SP')
    plt.errorbar(logEs_bin_edges, Xmean_MB, yerr=Xstd, label='MB')

    plt.xlabel(f'Log E')
    plt.ylabel('Xice-Xmean (g/cm^2)')
    plt.savefig('plots/CoreAnalysis/XmeanTest.png')
    plt.clf()

plotXmaxDist()
quit()

if __name__ == "__main__":
    shower_E_bins = np.arange(16, 20.01, 0.1)

#     n_per_year = event_rate(log10e_min, log10e_max, zmax, area, flux)
#     print(f"{flux} spectrum: {10**log10e_min:.3g}-{10**log10e_max:.3g} up to {zmax:.0f}deg for area = {area:.1f}km^2 -> {n_per_year:.2f} events/year")
    # n_per_year = event_rate(log10e_min, log10e_max, zmax, area, "TA_19")
    # print(f"TA ICRC19 spectrum: {10**log10e_min:.3g}-{10**log10e_max:.3g} up to {zmax:.0f}deg for area = {area:.1f}km^2 -> {n_per_year:.2f} events/year")

    n_xmax = 2000
    dCos = 0.05
    coszen_bin_edges = np.arange(np.cos(np.deg2rad(60)), 1.01, dCos)
    coszens = 0.5 * (coszen_bin_edges[1:] + coszen_bin_edges[:-1])
    logEs_bin_edges = np.arange(17, 20.01, 0.1)
    logEs = 0.5 * (logEs_bin_edges[1:] + logEs_bin_edges[:-1])

    # translate CR energy to shower energy
    shower_energies = np.zeros((len(logEs), len(coszens), n_xmax))
    shower_xmax = np.zeros((len(logEs), len(coszens), n_xmax))
    weights_shower_energies = np.ones((len(logEs), len(coszens), n_xmax))
    for iE, logE in enumerate(logEs):
        E = 10 ** logE
        logE1 = logEs_bin_edges[iE]
        logE2 = logEs_bin_edges[iE + 1]
        mean_Xmax = get_xmax(logE)
        std_Xmax = get_var_xmax(logE)
        print(f"E = {E:.4g} -> mean xmax {mean_Xmax:.0f} +- {std_Xmax:.1f} g/cm^2")
        for iC, coszen in enumerate(coszens):
            n_events_year_area_dcos = event_rate(logE1, logE2, dCos, area=1, type=flux) * coszen
            zen = np.arccos(coszen)
            atm = atm_overburden / coszen
            print(f"cos(zen) = {coszen:.2f}, zen = {np.rad2deg(zen):.0f}deg -> atmostphere = {atm:.0f}")
            for iXmax, xmax in enumerate(np.random.normal(mean_Xmax, std_Xmax, n_xmax)):
                #Frac is the integration of energy lost from upper atmosphere to the surface of the ice
                #for a given Xmax and energy
                #Divided by ???
                frac = quad(gh, 0, atm, args=(xmax, E))[0] / quad(gh, 0, atm_overburden * 5, args=(xmax, E))[0]
                Eshower = (1 - frac) * E
#                 print(f"{xmax:.0f} g/cm^2, energy deposit in atm = {frac*100:.1f}% -> shower energy {Eshower:.4g}eV")
                shower_energies[iE, iC, iXmax] = Eshower
                shower_xmax[iE, iC, iXmax] = xmax
                weights_shower_energies[iE, iC, iXmax] = 1. / n_xmax * n_events_year_area_dcos
            if 0:
                fout = f"plots/shower_rate/indiv/shower_rate_year_area_zen_{atm_overburden:.0f}gcm2_{flux}_{iE}_{iC}.png"
                if(not os.path.exists(fout)):
                    yy = shower_energies[iE, iC].flatten()
                    mask = ~np.isnan(yy)
                    ww = weights_shower_energies[iE, iC].flatten()
                    fig, ax = php.get_histogram(np.log10(shower_energies[iE, iC][mask]), weights=weights_shower_energies[iE, iC][mask], bins=shower_E_bins,
                                                xlabel=r"Core Energy [$log_{10}eV$]", title=f"Ecr = {E:.4g}eV ({logEs_bin_edges[iE]:.2f}..{logEs_bin_edges[iE+1]:.2f}), zen = {np.rad2deg(zen):.0f}deg, coszen {coszen_bin_edges[iC]:.2f}..{coszen_bin_edges[iC+1]:.2f}",
                                                ylabel=r"Normalied Rate of Cores per $E_{CR} \theta_{zen}$ bin")
                    fig.savefig(fout)
                    plt.close(fig)
    #         plt.show()
        plt.close("all")
        yy = shower_energies[iE].flatten()
        mask = ~np.isnan(yy)
        yy = yy[mask]
        weights = weights_shower_energies[iE].flatten()[mask]
        if 0:
            fout = f"plots/shower_rate/shower_rate_year_area_{atm_overburden:.0f}gcm2_{flux}_{iE}.png"
            if(not os.path.exists(fout)):
                fig, ax = php.get_histogram(np.log10(yy), bins=shower_E_bins, weights=weights,
                                            xlabel="shower energy [eV]", title=f"Ecr = {E:.4g}eV ({logEs_bin_edges[iE]:.2f}..{logEs_bin_edges[iE+1]:.2f})",
                                            ylabel="in-ice showers per year, 1km^2, bin")
                fig.savefig(fout)
                plt.close("all")

#    with open(f"output_{flux}_{atm_overburden}_{n_xmax}xmax.pkl", "wb") as fout:
    with open(f"output_{flux}_{atm_overburden}.pkl", "wb") as fout:
        pickle.dump([shower_energies, weights_shower_energies, shower_xmax], fout)
    # fold per energy histograms with flux
    yy = shower_energies.flatten()
    mask = ~np.isnan(yy)
    ww = weights_shower_energies.flatten()
    fig, ax = php.get_histogram(np.log10(yy[mask]), bins=np.arange(16, 20, 0.1), weights=ww[mask],
                                    xlabel="shower energy [eV]", title=f"{flux} flux, averaged over all angles up to 60deg, 1e17-1e20",
                                    ylabel="events per year, km^2, bin")
    fig.savefig(f"plots/shower_rate/shower_energies.png")
#     plt.show()
