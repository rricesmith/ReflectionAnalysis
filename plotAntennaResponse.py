import NuRadioReco.detector.antennapattern
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import numpy as np
from icecream import ic
from scipy import optimize
import os
from NuRadioReco.utilities import fft
from plotAttenuationLengthsMB import calculate_attenuation_length

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
inc_azis = [0*units.deg, 30*units.deg, 45*units.deg, 90*units.deg, 240.1*units.deg]

zen_range = [0, 30, 45]
for inc_azi in inc_azis:
    for inc_zen in zen_range:
        inc_zen = inc_zen * units.deg

        orientation_theta_phi = [np.deg2rad(0), np.deg2rad(0)]
        rotation_theta_phi = [np.deg2rad(90), np.deg2rad(0)]

        fig, (ax) = plt.subplots(1, 1, sharey=True)

        VELs_front = LPDA_antenna.get_antenna_response_vectorized(ff, inc_zen, inc_azi,
                                                    orientation_theta_phi[0], orientation_theta_phi[1], rotation_theta_phi[0], rotation_theta_phi[1])
        
        VELs_back = LPDA_antenna.get_antenna_response_vectorized(ff, 180*units.deg-inc_zen, inc_azi,
                                                    orientation_theta_phi[0], orientation_theta_phi[1], rotation_theta_phi[0], rotation_theta_phi[1])

        # Normalize to the larger of the two sets
        norm_factor = max(np.max(np.abs(VELs_front['theta'])), np.max(np.abs(VELs_back['theta'])))

        VELs = VELs_front
        VELs['theta'] = VELs['theta'] / norm_factor
        VELs['phi'] = VELs['phi'] / np.max(np.abs(VELs['phi']))
        ax.plot(ff / units.MHz, np.abs(VELs['theta']), label=f'Frontlobe', color='red')
        # ax.plot(ff / units.MHz, np.abs(VELs['phi']), label=f'ePhi LPDA {inc_zen/units.deg:.0f}deg')

        # Limit fit to sensitive region of LPDA
        fitmask = (ff > 70*units.MHz) & (ff < 210*units.MHz)

        # Linear fit
        # fit = np.polyfit(ff[fitmask]/units.MHz, np.abs(VELs['theta'])[fitmask], 1)
        # ax.plot(ff[fitmask] / units.MHz, fit[0]*ff[fitmask]/units.MHz + fit[1], label=f'Fit {fit[0]:.5f}x+{fit[1]:.2f}', color='red', linestyle='--')

        # Inverse fit
        # try:
        #     fit, cov = optimize.curve_fit(inverse_func, ff[fitmask]/units.MHz, np.abs(VELs['theta'])[fitmask], p0=[100, 2, 0], maxfev=10000)
        # except RuntimeError:
        #     ic(f'Failed to fit {inc_zen/units.deg}, continuing')
        #     continue
        # ax.plot(ff[fitmask] / units.MHz, inverse_func(ff[fitmask]/units.MHz, *fit), label=f'Fit {fit[0]:.5f}/(x+{fit[1]:.2f})+{fit[2]:.2f}', color='red', linestyle='--')

        VELs = VELs_back
        VELs['theta'] = VELs['theta'] / norm_factor
        VELs['phi'] = VELs['phi'] / np.max(np.abs(VELs['phi']))
        ax.plot(ff / units.MHz, np.abs(VELs['theta']), label=f'Backlobe', color='blue')
        # ax.plot(ff / units.MHz, np.abs(VELs['phi']), label=f'ePhi LPDA {(180*units.deg-inc_zen)/units.deg:.0f}deg')

        # Linear fit
        # fit = np.polyfit(ff[fitmask]/units.MHz, np.abs(VELs['theta'])[fitmask], 1)
        # ax.plot(ff[fitmask] / units.MHz, fit[0]*ff[fitmask]/units.MHz + fit[1], label=f'Fit {fit[0]:.5f}x+{fit[1]:.2f}', color='blue', linestyle='--')

        # Inverse fit
        # try:
        #     fit, cov = optimize.curve_fit(inverse_func, ff[fitmask]/units.MHz, np.abs(VELs['theta'])[fitmask], p0=[100, 2, 0], maxfev=10000)
        # except RuntimeError:
        #     ic(f'Failed to fit {inc_zen/units.deg}, continuing')
        #     continue
        # ax.plot(ff[fitmask] / units.MHz, inverse_func(ff[fitmask]/units.MHz, *fit), label=f'Fit {fit[0]:.5f}/(x+{fit[1]:.2f})+{fit[2]:.2f}', color='blue', linestyle='--')


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

        ax.set_title(f'LPDA Response')
        ax.legend()
        
        # Add text below legend
        textstr = '\n'.join((
            r'Zenith $%.0f^\circ$' % (inc_zen/units.deg, ),
            r'Azimuth $%.0f^\circ$' % (inc_azi/units.deg, )))
        
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.95, 0.85, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        ax.set_ylabel("Normalized Heff")
        ax.set_xlabel("frequency [MHz]")
        ax.set_xlim(0, 500)

        savename = f'plots/AntennaResponse_Zen{inc_zen/units.deg:.0f}deg_Azi{inc_azi/units.deg:.0f}deg.png'
        plt.savefig(savename)
        plt.close(fig)
        ic(f'Saved {savename}')

# Polar plots
plot_azimuths = [0*units.deg, 45*units.deg, 90*units.deg]
plot_freqs = np.array([100, 200, 300, 400]) * units.MHz
polar_angles_deg = np.arange(0, 361, 5)
polar_angles = polar_angles_deg * units.deg

orientation_theta_phi = [np.deg2rad(0), np.deg2rad(0)]
rotation_theta_phi = [np.deg2rad(90), np.deg2rad(0)]

for plot_azi in plot_azimuths:
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    responses = np.zeros((len(plot_freqs), len(polar_angles)))
    
    idx_180 = np.where(polar_angles_deg == 180)[0][0]

    for i, angle in enumerate(polar_angles):
        if i <= idx_180:
            zen = angle
            azi = plot_azi
            
            VELs = LPDA_antenna.get_antenna_response_vectorized(plot_freqs, zen, azi,
                                                        orientation_theta_phi[0], orientation_theta_phi[1], rotation_theta_phi[0], rotation_theta_phi[1])
            responses[:, i] = 20 * np.log10(np.abs(VELs['theta']))
        else:
            mirror_idx = idx_180 - (i - idx_180)
            responses[:, i] = responses[:, mirror_idx]

        # Fix 270 by making it same as 90
        if zen == 270*units.deg:
            responses[:, i] = responses[:, idx_180 - (idx_180 - np.where(polar_angles_deg == 90)[0][0])]

    ax.set_theta_zero_location("S")
    ax.set_theta_direction(-1)
    
    for i, freq in enumerate(plot_freqs):
        ax.plot(np.deg2rad(polar_angles_deg), responses[i, :], label=f'{freq/units.MHz:.0f} MHz')
        
    ax.set_title(f'LPDA Response Azimuth {plot_azi/units.deg:.0f} deg')
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0))
    
    savename = f'plots/AntennaResponse_Polar_Azi{plot_azi/units.deg:.0f}deg.png'
    plt.savefig(savename, bbox_inches='tight')
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

