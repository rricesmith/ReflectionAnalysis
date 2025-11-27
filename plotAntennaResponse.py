import NuRadioReco.detector.antennapattern
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import numpy as np
from icecream import ic
from scipy import optimize
import os
from NuRadioReco.utilities import fft

def inverse_func(x, a, b, c):
    return a/(x + b) + c

provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()

theta = 90 * units.deg
ff = np.linspace(50 * units.MHz, 1 * units.GHz, 500)
LPDA_antenna = provider.load_antenna_pattern("createLPDA_100MHz_InfFirn")
# bicone_n14 = provider.load_antenna_pattern("bicone_v8_inf_n1.4")
# bicone_air = provider.load_antenna_pattern("bicone_v8_InfAir")

# bicone_XFDTD = provider.load_antenna_pattern("XFDTD_Vpol_CrossFeed_150mmHole_n1.78")

#inc_zen = 0*units.deg
# inc_azi = 45*units.deg
inc_azis = [0*units.deg, 15*units.deg, 30*units.deg, 45*units.deg, 90*units.deg, 128.9*units.deg, 180*units.deg, 180.65*units.deg, 236.1*units.deg, 237.25*units.deg, 238.4*units.deg, 240.1*units.deg]

zen_range = [0, 30, 41.8, 44.1, 44.5, 44.7, 60, 61.9, 80]
for inc_azi in inc_azis:
    for inc_zen in zen_range:
        inc_zen = inc_zen * units.deg

        orientation_theta_phi = [np.deg2rad(0), np.deg2rad(0)]
        rotation_theta_phi = [np.deg2rad(90), np.deg2rad(0)]

        fig, (ax) = plt.subplots(1, 1, sharey=True)

        VELs = LPDA_antenna.get_antenna_response_vectorized(ff, inc_zen, inc_azi,
                                                    orientation_theta_phi[0], orientation_theta_phi[1], rotation_theta_phi[0], rotation_theta_phi[1])
        VELs['theta'] = VELs['theta'] / np.max(np.abs(VELs['theta']))
        VELs['phi'] = VELs['phi'] / np.max(np.abs(VELs['phi']))
        ax.plot(ff / units.MHz, np.abs(VELs['theta']), label=f'eTheta LPDA {inc_zen/units.deg:.0f}deg', color='red')
        # ax.plot(ff / units.MHz, np.abs(VELs['phi']), label=f'ePhi LPDA {inc_zen/units.deg:.0f}deg')

        # Limit fit to sensitive region of LPDA
        fitmask = (ff > 70*units.MHz) & (ff < 210*units.MHz)

        # Linear fit
        # fit = np.polyfit(ff[fitmask]/units.MHz, np.abs(VELs['theta'])[fitmask], 1)
        # ax.plot(ff[fitmask] / units.MHz, fit[0]*ff[fitmask]/units.MHz + fit[1], label=f'Fit {fit[0]:.5f}x+{fit[1]:.2f}', color='red', linestyle='--')

        # Inverse fit
        try:
            fit, cov = optimize.curve_fit(inverse_func, ff[fitmask]/units.MHz, np.abs(VELs['theta'])[fitmask], p0=[100, 2, 0], maxfev=10000)
        except RuntimeError:
            ic(f'Failed to fit {inc_zen/units.deg}, continuing')
            continue
        ax.plot(ff[fitmask] / units.MHz, inverse_func(ff[fitmask]/units.MHz, *fit), label=f'Fit {fit[0]:.5f}/(x+{fit[1]:.2f})+{fit[2]:.2f}', color='red', linestyle='--')

        VELs = LPDA_antenna.get_antenna_response_vectorized(ff, 180*units.deg-inc_zen, inc_azi,
                                                    orientation_theta_phi[0], orientation_theta_phi[1], rotation_theta_phi[0], rotation_theta_phi[1])
        VELs['theta'] = VELs['theta'] / np.max(np.abs(VELs['theta']))
        VELs['phi'] = VELs['phi'] / np.max(np.abs(VELs['phi']))
        ax.plot(ff / units.MHz, np.abs(VELs['theta']), label=f'eTheta LPDA {(180*units.deg-inc_zen)/units.deg:.0f}deg', color='blue')
        # ax.plot(ff / units.MHz, np.abs(VELs['phi']), label=f'ePhi LPDA {(180*units.deg-inc_zen)/units.deg:.0f}deg')

        # Linear fit
        # fit = np.polyfit(ff[fitmask]/units.MHz, np.abs(VELs['theta'])[fitmask], 1)
        # ax.plot(ff[fitmask] / units.MHz, fit[0]*ff[fitmask]/units.MHz + fit[1], label=f'Fit {fit[0]:.5f}x+{fit[1]:.2f}', color='blue', linestyle='--')

        # Inverse fit
        try:
            fit, cov = optimize.curve_fit(inverse_func, ff[fitmask]/units.MHz, np.abs(VELs['theta'])[fitmask], p0=[100, 2, 0], maxfev=10000)
        except RuntimeError:
            ic(f'Failed to fit {inc_zen/units.deg}, continuing')
            continue
        ax.plot(ff[fitmask] / units.MHz, inverse_func(ff[fitmask]/units.MHz, *fit), label=f'Fit {fit[0]:.5f}/(x+{fit[1]:.2f})+{fit[2]:.2f}', color='blue', linestyle='--')


        # VELs = bicone_XFDTD.get_antenna_response_vectorized(
        #     ff,
        #     90 * units.deg,
        #     np.deg2rad(0),
        #     np.deg2rad(180),
        #     0,
        #     np.deg2rad(90),
        #     np.deg2rad(0)
        # )
        # ax.plot(ff / units.MHz, np.abs(VELs['theta']), '--', label='eTheta bicone n=1.78 (XFDTD)')
        # ax.plot(ff / units.MHz, np.abs(VELs['phi']), '--', label='ePhi bicone (old ARA)')

        # VELs = bicone_air.get_antenna_response_vectorized(ff, 90 * units.deg, np.deg2rad(0),
        #                                                   np.deg2rad(180), 0, np.deg2rad(90), np.deg2rad(0))
        # ax.plot(ff / units.MHz, np.abs(VELs['theta']), '--', label='eTheta bicone (air)')
        # ax.plot(ff / units.MHz, np.abs(VELs['phi']), '--', label='ePhi bicone (air)')

        ax.set_title(f'LPDA Response at {inc_azi/units.deg:.0f}deg Azimuth, {inc_zen/units.deg:.0f}deg Zenith above/below')
        ax.legend()
        ax.set_ylabel("Normalized Heff")
        ax.set_xlabel("frequency [MHz]")
        ax.set_xlim(0, 500)

        savename = f'plots/AntennaResponse_Zen{inc_zen/units.deg:.0f}deg_Azi{inc_azi/units.deg:.0f}deg.png'
        plt.savefig(savename)
        plt.close(fig)
        ic(f'Saved {savename}')

quit()

templates2016folder = f'StationDataAnalysis/templates/confirmed2016Templates/'
for file in os.listdir(templates2016folder):
    template = np.load(templates2016folder + file)
    trace = template[0]
    for i in range(1, len(template)):
        if max(template[i]) > max(trace):
            trace = template[i]

    tracefft = fft.time2freq(trace, 2*units.GHz)
    freq = np.fft.rfftfreq(len(trace), d=1/(2*units.GHz)) * units.GHz
    ic(freq)

    tracefft = np.abs(tracefft)
    tracefft = tracefft / np.max(np.abs(tracefft))
    fig, (ax) = plt.subplots(1, 1, sharey=True)

    ax.plot(freq/units.MHz, tracefft, color='blue', linestyle='-', label='Event')    

    fitmask = (freq > 70*units.MHz) & (freq < 210*units.MHz)
    fit, cov = optimize.curve_fit(inverse_func, freq[fitmask]/units.MHz, np.abs(tracefft)[fitmask], p0=[100, 2, 0], maxfev=90000)
    ax.plot(freq[fitmask]/units.MHz, inverse_func(freq[fitmask]/units.MHz, *fit), label=f'Fit {fit[0]:.5f}/(x+{fit[1]:.2f})+{fit[2]:.2f}', color='red', linestyle='--')

    linfit = np.polyfit(freq[fitmask]/units.MHz, tracefft[fitmask], 1)
    ax.plot(freq[fitmask]/units.MHz, linfit[0]*freq[fitmask]/units.MHz + linfit[1], label=f'Fit {linfit[0]:.5f}x+{linfit[1]:.2f}', color='green', linestyle='--')

    ax.legend()
    ax.set_title(file)
    ax.set_ylabel("Normalized Amplitude")
    ax.set_xlabel("frequency [MHz]")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1)
    savename = f'plots/{file}.png'
    plt.savefig(savename)
    ic(f'Saved {savename}')

