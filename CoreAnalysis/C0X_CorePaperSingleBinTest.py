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
import hdf5AnalysisUtils as hdau
from coreDataObjects import coreStatistics
import pickle
from NuRadioReco.modules.io import NuRadioRecoio
from DeepLearning.D04B_reprocessNurPassingCut import pT
from NuRadioReco.modules import channelResampler
import NuRadioReco.modules.triggerTimeAdjuster
from NuRadioReco.detector import detector
from NuRadioReco.detector import generic_detector


channelResampler = channelResampler.channelResampler()
channelResampler.begin(debug=False)
triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()

detector_filename = 'configurations/gen2_hybrid_2021.json'
det = generic_detector.GenericDetector(json_filename=detector_filename, assume_inf=False, antenna_by_depth=False, default_station=1001)

eV = 15.7
# trigger_antennas = [ ['LPDA_2of4_100Hz', [0, 1, 2, 3]], ['PA_8ch_100Hz', [8, 9, 10, 11]]]
trigger_name = 'LPDA_2of4_100Hz'
antennas = [0, 1, 2, 3]
# trigger_name = 'PA_8ch_100Hz'
# antennas = [8, 9, 10, 11]
station = 'station_1001'
triggerTimeAdjuster.begin(trigger_name=trigger_name)


# Plot a few triggers from noise triggered events
eV_array = [15.1, 15.5, 15.7, 16.1, 16.3, 16.4, 16.5, 16.8]
n_save = 10
for eV in eV_array:
    # path = f'run/CoreRefl1.00km/{eV}log10eV/noise/'
    path = f'run/CoreRefl1.00km/{eV}log10eV/noiseless/'
    saved = 0
    for file in glob.glob(path + '/*.nur'):
        if saved >= n_save:
            break
        template = NuRadioRecoio.NuRadioRecoio(file)
        for i, evt in enumerate(template.get_events()):
            station = evt.get_station(1001)
            if not station.has_triggered(trigger_name=trigger_name):
                continue
            # triggerTimeAdjuster.run(evt, station, det)
            channelResampler.run(evt, station, det, 2.4*units.GHz)
            traces = []
            times = []
            for iC, channel in enumerate(station.iter_channels(use_channels=antennas)):                        
                y = channel.get_trace()
                traces.append(y)
                times.append(channel.get_times())
            # pT(traces, f'{eV}log10eV #{saved}', saveLoc=f'plots/CoreAnalysis/CorePaper/NoiseTriggers/{eV}eV_{saved}.png')
            
            trigger = station.get_trigger(trigger_name)
            trigger_time = trigger.get_trigger_time()
            # ic(trigger_time, times)

            fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True)
            for iC in range(4):
                axs[iC].vlines(times[iC][0]+200, -1, 1, linestyle='--', color='green')
                axs[iC].plot(times[iC], traces[iC])
                axs[iC].set_ylabel('Voltage [μV]')
                # axs[iC].set_xlabel('Time [ns]')
                axs[iC].set_ylim(min(traces[iC])*1.1, max(traces[iC])*1.1)
            axs[-1].set_xlabel('Time [ns]')
            axs[0].set_title(f'{eV}log10eV #{saved}')
            fig.tight_layout()
            # savename = f'plots/CoreAnalysis/CorePaper/NoiseTriggers/{eV}eV_{saved}.png'
            savename = f'plots/CoreAnalysis/CorePaper/NoiselessTriggers/{eV}eV_{saved}.png'
            fig.savefig(savename)
            print(f'saved {savename}')
            plt.close(fig)
            saved += 1
            if saved >= n_save:
                break

quit()

eV_array = [15.1, 15.5, 15.7, 16.1, 16.3, 16.4, 16.5, 16.8]
n_trig = {}
n_throw = {}
for eV in eV_array:
    n_trig[eV] = {}
    n_throw[eV] = {}
    noise_path = f'run/CoreRefl1.00km/{eV}log10eV/noise/'
    noiseless_path = f'run/CoreRefl1.00km/{eV}log10eV/noiseless/'
    paths = [noise_path, noiseless_path]
    for path in paths:
        n_trig[eV][path] = 0
        n_throw[eV][path] = 0

        throws = coreStatistics(eV)       #New line testing changing core statistics
        n_throw[eV][path] = throws * 50


        for file in glob.glob(path + '/*.hdf5'):
        # for file in glob.glob(noiseless_path + '/*.hdf5'):

            fin = h5py.File(file, 'r')
            ic(file)
            # for key in fin:
            #     print(f'key {key}')
            # for key in fin.attrs.keys():
            #     print(f'attrs key {key}')
            try:
                for key in fin.attrs['trigger_names']:
                    ic(key)
            except:
                ic('no trigger names')
            # for key in fin['station_1001']:
            #     print(f'station key {key}')
            # triggered = fin['station_1001']['triggered']
            # ic(triggered)

            # throws = coreStatistics(eV)       #New line testing changing core statistics
            # n_throw[eV][path] += throws


            cores = fin.attrs['n_events']
            keys = fin.attrs.keys()
            triggered = 'trigger_names' in keys
            multi_triggered = 'multiple_triggers' in keys
            if not triggered:
                print(f'no triggers in {file}')
                continue
            trig_mask, trig_index = hdau.trigger_mask(fin, trigger = trigger_name, station = station)
            if trig_index == []:
                print(f'no triggers of {trigger_name}')
                continue
            ra_mask, rb_mask, da_mask, db_mask, arrival_z, launch_z = hdau.multi_array_direction_masks(fin, antennas, station, trig_index)

            refl_mask = ra_mask | rb_mask

            ic(np.sum(refl_mask))
            n_trig[eV][path] += np.sum(refl_mask)

lowercut_eV_array = [14.7, 14.8, 14.9, 15.0, 15.1]
for eV in lowercut_eV_array:
    if not eV in n_trig:
        n_trig[eV] = {}
        n_throw[eV] = {}
    path = f'run/CoreRefl1.00km/{eV}log10eV/noise.25Vrms/'
    n_trig[eV][path] = 0
    n_throw[eV][path] = 0

    throws = coreStatistics(eV)       #New line testing changing core statistics
    n_throw[eV][path] = throws * 50


    for file in glob.glob(path + '/*.hdf5'):
    # for file in glob.glob(noiseless_path + '/*.hdf5'):

        fin = h5py.File(file, 'r')
        ic(file)
        # for key in fin:
        #     print(f'key {key}')
        # for key in fin.attrs.keys():
        #     print(f'attrs key {key}')
        try:
            for key in fin.attrs['trigger_names']:
                ic(key)
        except:
            ic('no trigger names')
        # for key in fin['station_1001']:
        #     print(f'station key {key}')
        # triggered = fin['station_1001']['triggered']
        # ic(triggered)

        # throws = coreStatistics(eV)       #New line testing changing core statistics
        # n_throw[eV][path] += throws


        cores = fin.attrs['n_events']
        keys = fin.attrs.keys()
        triggered = 'trigger_names' in keys
        multi_triggered = 'multiple_triggers' in keys
        if not triggered:
            print(f'no triggers in {file}')
            continue
        trig_mask, trig_index = hdau.trigger_mask(fin, trigger = trigger_name, station = station)
        if trig_index == []:
            print(f'no triggers of {trigger_name}')
            continue
        ra_mask, rb_mask, da_mask, db_mask, arrival_z, launch_z = hdau.multi_array_direction_masks(fin, antennas, station, trig_index)

        refl_mask = ra_mask | rb_mask

        ic(np.sum(refl_mask))
        n_trig[eV][path] += np.sum(refl_mask)

ic(n_trig)
ic(n_throw)
# quit()

eV_array = [14.7, 14.8, 14.9, 15.0, 15.1, 15.5, 15.7, 16.1, 16.3, 16.4, 16.5, 16.8]
labels = ['noise', 'noiseless', 'noise.25Vrms']
fig, ax = plt.subplots(1, 1)
aeff = {}
yerr = {}
eVs = {}
for label in labels:
    aeff[label] = []
    yerr[label] = []
    eVs[label] = []
    for eV in eV_array:
        path = f'run/CoreRefl1.00km/{eV}log10eV/{label}/'
        if not path in n_trig[eV]:
            continue
        eVs[label].append(eV)
        aeff[label].append( n_trig[eV][path] / n_throw[eV][path] )
        yerr[label].append( (np.sqrt(n_trig[eV][path]) / n_throw[eV][path]) )

    ax.errorbar(eVs[label], aeff[label], yerr=yerr[label], label=label, linestyle='none', marker='o', capsize=6)

ax.set_xlabel('Energy [lgeV]')
ax.set_ylabel('Aeff / km^2')
ax.set_xlim(14.6, 18)
ax.set_ylim(10**-7, 10)
ax.set_yscale('log')
ax.legend()
savename = f'plots/CoreAnalysis/CorePaper/AeffVsEnergy'
# ax.patch.set_alpha(0)
fig.savefig(f'{savename}.png')
print(f'saved {savename}.png')
plt.clf()


with open(f'{savename}.pkl', 'wb') as fout:
    pickle.dump([labels, eV_array, aeff, yerr], fout)
    fout.close()
print(f'saved {savename}.pkl')


quit()

noise_data = get_Veff_Aeff(noise_path)
noiseless_data = get_Veff_Aeff(noiseless_path)


noise_Veffs, noise_energies, noise_energies_low, noise_energies_up, noise_zenith_bins, noise_utrigger_names = get_Veff_Aeff_array(noise_data)
ic(1)
noiseless_Veffs, noiseless_energies, noiseless_energies_low, noiseless_energies_up, noiseless_zenith_bins, noiseless_utrigger_names = get_Veff_Aeff_array(noiseless_data)
ic(2)

noise_Veff_avg = np.average(noise_Veffs[:, :, get_index('LPDA_2of4_100Hz', noise_utrigger_names), 0], axis=1)
noiseless_Veff_avg = np.average(noiseless_Veffs[:, :, get_index('LPDA_2of4_100Hz', noiseless_utrigger_names), 0], axis=1)

noise_Veff_water = get_Veff_water_equivalent(noise_Veff_avg) * 4 * np.pi
noiseless_Veff_water = get_Veff_water_equivalent(noiseless_Veff_avg) * 4 * np.pi

noise_Veff_error = noise_Veff_water / np.sum(noise_Veffs[:, :, get_index('LPDA_2of4_100Hz', noise_utrigger_names), 0], axis=1) ** 0.5
noiseless_Veff_error = noiseless_Veff_water / np.sum(noiseless_Veffs[:, :, get_index('LPDA_2of4_100Hz', noiseless_utrigger_names), 0], axis=1) ** 0.5

fig, ax = plt.subplots(1, 1)
ax.errorbar(noiseless_energies/units.eV, noiseless_Veff_water/units.km**3/units.sr, yerr=noiseless_Veff_error/units.km**3/units.sr, fmt='d-', label='Noiseless', linestyle='-', capsize=10)
ax.errorbar(noise_energies/units.eV, noise_Veff_water/units.km**3/units.sr, yerr=noise_Veff_error/units.km**3/units.sr, fmt='d-', label='Noise', linestyle='-', capsize=10)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Energy [eV]')
ax.set_ylabel('Veff [km^3 sr]')
ax.legend()
fig.savefig(f'plots/CoreAnalysis/CorePaper/NoiseVsNoiseless15.5eV.png')
