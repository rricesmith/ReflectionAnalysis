import numpy as np
import scipy.constants
from NuRadioReco.utilities import units
from NuRadioMC.utilities import cross_sections
from NuRadioMC.examples.Sensitivities import E2_fluxes3 as limits

import matplotlib.pyplot as plt
from scipy import interpolate
from NuRadioMC.utilities import inelasticities


energyBinsPerDecade = 1.
plotUnitsEnergy = units.eV
plotUnitsEnergyStr = "eV"
plotUnitsFlux = units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1






def get_Veff_from_e2(u1, energy, 
                    livetime=10 * units.year, 
                    signalEff=1.00, 
                    energyBinsPerDecade=1.000, 
                    upperLimOnEvents=2.44, 
                    nuCrsScn='ctw',
                    inttype="total"):
    """
    Calculate the Veff of a sim from its E2 limit plot
    Reverse of the function NuRadioMC.utilities.fluxes.get_limit_e2_flux

    Parameters
    ----------
    u1: array of floats
        Flux * E^2 sensitivity
    energy: array of floats
        neutrino energy
    livetime: float
        time used
    signalEff: float
        efficiency of signal reconstruction
    energyBinsPerDecade: float
        1 for decade bins, 2 for half-decade bins, etc.
    upperLimOnEvents: float
         2.3 for Neyman UL w/ 0 background,
         2.44 for F-C UL w/ 0 background, etc
    nuCrsScn: str
        type of neutrino cross-section

    Returns
    -------
    Veff_sr: float
        the Veff per steradian
    """

    u1 = u1 / energy ** 2
    u1 = u1 * energy
    evtsPerFluxEnergy = (upperLimOnEvents / u1) * (energyBinsPerDecade / np.log(10))
    veff_sr = (evtsPerFluxEnergy * cross_sections.get_interaction_length(energy, cross_section_type=nuCrsScn, inttype=inttype)) / (signalEff * livetime)
    return veff_sr


def flux_from_num_events_per_bin(number_of_events_per_energy, energies, Veff,
                                livetime=10 * units.year,
                                nuCrsScn='ctw'):
    """
    calculates the number of expected neutrinos for a certain flux assumption

    Parameters
    ----------
    energies: array of floats
        energies (the bin centers), the binning in log10(E) must be equidistant!
    Veff: array of floats
        the effective volume per energy logE
    livetime: float
        the livetime of the detector (including signal efficiency)

    Returns
    -------
    flux: array of floats
        the flux at energy logE    
    """

    logE = np.log10(energies)
    dLogE = logE[1] - logE[0]
    flux = (number_of_events_per_energy * cross_sections.get_interaction_length(energies, cross_section_type=nuCrsScn)) / (np.log(10) * livetime * energies * Veff * dLogE)

    return flux



if __name__ == "__main__":


    #Get Gen2 flux and convert to Veff
    gen2_E, gen2_flux = np.loadtxt("../NuRadioMC/NuRadioMC/examples/Sensitivities/data/Gen2radio_sensitivity_ICRC2021.txt")
    gen2_E *= units.eV
    gen2_flux *= units.GeV * units.cm ** -2 * units.second ** -1 * units.sr ** -1

    veff_gen2 = get_Veff_from_e2(gen2_flux, gen2_E)
    veff_interp = interpolate.interp1d(gen2_E, veff_gen2, kind='linear')

#    print(f'energies log10eV {np.log10(gen2_E)}')
    #energies log10eV [16.5 17.  17.5 18.  18.5 19.  19.5 20. ]

    if 0:
        #Test of the gen2 recreation
        print(f'veff {veff_gen2 / units.km**3}')

        #Test it works in reverse again by plotting
        fig, ax = limits.get_E2_limit_figure(diffuse=True, show_grand_10k=False, show_grand_200k=False, show_RNOG=True, show_ara=True,
                                            show_anita_I_IV_limit=False, show_auger_limit=False,
                                            show_IceCubeGen2_whitepaper=False, show_IceCubeGen2_ICRC2021=True)

        labels = []
#        labels = limits.add_limit(ax, labels, gen2_E, veff_gen2, n_stations=1, livetime=10 * units.year, label='Gen2 recreation')
        print(f'orig energies {gen2_E} and veff {veff_gen2/units.km**3}')
        test_energies = np.linspace(10**16.5, 10**20, 2000) * units.eV
        print(f'energies {test_energies} veff interp {veff_interp(test_energies)/units.km**3}')
        labels = limits.add_limit(ax, labels, test_energies, veff_interp(test_energies), n_stations=1, livetime=10*units.year, label='Gen2 interp')


        plt.legend(handles=labels, loc=2)
        plt.savefig('CoreAnalysis/plots/gen2RecreationTest.png')
    

    #Have to account for inelasticities of neutrinos, and using a mean to get the average expected neutrino energy from the pulse energy seen
    inelastics = inelasticities.get_neutrino_inelasticity(1000)
    mean_ine = 1 / np.mean(inelastics)
    if 0:
        #Testing inelasticities
        plt.hist(inelastics)
        plt.savefig('CoreAnalysis/plots/inelasticsDist.png')
        plt.clf()

        plt.hist(np.log(1/inelastics))
        plt.axvline(x=np.log10(mean_ine), label=f'Mean {np.log10(mean_ine):.1f}', color='red', linestyle='--')
        plt.legend()
        plt.xlabel('log10(1/inelasticity)')
        plt.savefig('CoreAnalysis/plots/inverseInelast.png')
        quit()


    #dummy approximate events per energy bin
    n_stations = 361
#    evts_per_eBin = np.array([0, 0, 0, 0, 0, 5*10**-5, 8*10**-3, 2*10**-3]) * n_stations   #3.8 sigma shallow
#    evts_per_eBin = np.array([0, 0, 0, 0, 10**-5, 2*10**-4, 3.2*10**-2, 8*10**-3]) * n_stations    #Testing approximate increase to 2sigma
#    evts_per_eBin = np.array([0, 10**-4, 10**-3, 2*10**-4, 10**-3, 10**-2, 10**-4, 0]) * n_stations    #Testing Alans results
#    dummy_flux = flux_from_num_events_per_bin(evts_per_eBin, gen2_E, veff_gen2)
#    e2_dummy_flux = dummy_flux * gen2_E ** 2
#    print(f'dummy flux {e2_dummy_flux / plotUnitsFlux}')

    #More in depth Alan numbers for LPDA 100Hz
    alan_energies = 10**np.array([17, 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 17.9, 18, 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7, 18.8, 18.9, 19.0, 19.1, 19.2, 19.3]) * units.eV
    alan_events = np.array([10**-4, 3*10**-4, 4*10**-4, 4.5*10**-4, 4.4*10**-4, 4.3*10**-4, 4.2*10**-4, 4*10**-4, 3.5*10**-4, 2.2*10**-4, 1.9*10**-4, 
                            1.6*10**-4, 1.3*10**-4, 1.1*10**-4, 2*10**-4, 6*10**-4, 1.7*10**-3, 3*10**-3, 4*10**-3, 5*10**-3, 4.5*10**-3, 3.8*10**-3, 2*10**-3, 6*10**-4]) * n_stations

    dB = 40
    alan_energies *= 10**(-dB/20) * mean_ine
    while alan_energies[0] < gen2_E[0]:
        alan_energies = np.delete(alan_energies, 0)
        alan_events = np.delete(alan_events, 0)

    alan_flux = flux_from_num_events_per_bin(alan_events, alan_energies, veff_interp(alan_energies))
    alan_flux_e2 = alan_flux * alan_energies**2

    fig, ax = limits.get_E2_limit_figure(diffuse=True, show_grand_10k=False, show_grand_200k=False, show_RNOG=True, show_ara=True,
                                    show_anita_I_IV_limit=False, show_auger_limit=False,
                                    show_IceCubeGen2_whitepaper=False, show_IceCubeGen2_ICRC2021=True)

    labels = []
#    _plt, = ax.plot(gen2_E / plotUnitsEnergy, e2_dummy_flux / plotUnitsFlux, linestyle='--', color='black', label='300m Background')
    _plt, = ax.plot(alan_energies / plotUnitsEnergy, alan_flux_e2 / plotUnitsFlux, linestyle='--', color='black', label=f'300m {dB}dB')
    labels.append(_plt)

    plt.legend(handles=labels, loc=2)
    plt.savefig('CoreAnalysis/plots/gen2CoreBackgroundFlux.png')
