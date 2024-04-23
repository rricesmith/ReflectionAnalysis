import os
import numpy as np
import astrotools.auger as auger
from NuRadioReco.modules.io import NuRadioRecoio
import matplotlib.pyplot as plt
from icecream import ic
from NuRadioReco.utilities import units
from pathlib import Path



date = '4.22.24'
sim_folder = f'SimpleFootprintSimulation/output/{date}/'
plot_folder = f'plots/SimpleFootprintSimulation/{date}/'
Path(plot_folder).mkdir(parents=True, exist_ok=True)

# Livetime of station data
station_livetime = (46 + (10 + (57 + 27.6/60)/60)/24) / 364.25 # 46 days, 10 hours, 57 minutes, 27.6 seconds
ic(station_livetime)

max_distance = 2
n_cores = 100
num_icetop = 30
min_energy = 16.0
max_energy = 18.6
# sin2Val = -0.1
e_range = np.arange(min_energy, max_energy, 0.1)
sin2Val = np.arange(0, 1.01, 0.1)
angle_bins = np.rad2deg(np.arcsin(np.sqrt(sin2Val)))



total_cores = n_cores * num_icetop
trigger_names = ['direct_LPDA_2of3_3.5sigma', 'direct_LPDA_3of3_3.5sigma', 'direct_LPDA_3of3_5sigma']
colors = ['b', 'g', 'r']
trig_rate_per_bin = {}
for trigger in trigger_names:
    trig_rate_per_bin[trigger] = np.zeros((len(sin2Val),len(e_range)-1))



for iE in range(len(e_range)-1):
    for iS in range(len(sin2Val)-1):
        nurFiles= []

        # file = f'SimpleFootprintSimulation/output/4.16.24/Stn51_IceTop_{e_range[iE]:.1f}-{e_range[iE+1]:.1f}eV_{sin2:.1f}sin2_{n_cores}cores.nur'
        for file in os.listdir(sim_folder):
            # ic(file, f'{e_range[iE]:.1f}-{e_range[iE+1]:.1f}eV')
            if file.endswith('.nur') and (f'{e_range[iE]:.1f}-{e_range[iE+1]:.1f}eV_{sin2Val[iS]:.1f}sin2' in file):
                # ic(True)
                nurFiles.append(os.path.join(sim_folder, file))
        if nurFiles == []:
            continue

        ic(nurFiles)
        eventReader = NuRadioRecoio.NuRadioRecoio(nurFiles)
        n_trig_2of3_3_5sig = 0
        n_trig_3of3_3_5sig = 0
        n_trig_3of3_5sig = 0
        n_throw = 0
        for i, evt in enumerate(eventReader.get_events()):
            n_throw += 1
            station_ids = evt.get_station_ids()
            for stn_id in station_ids:
                station = evt.get_station(stn_id)
                if not station.has_triggered():
                    continue
                for trigger in trigger_names:
                    if station.has_triggered(trigger_name=trigger):
                        trig_rate_per_bin[trigger][iS][iE] += 1
    
        for trigger in trigger_names:
            trig_rate_per_bin[trigger][iS][iE] *= 1 / n_throw


# trig_rate_per_bin is now n_trig/n_throw per energy bin
aeff_per_bin = {}
for trigger in trigger_names:
    ic((max_distance)**2 / np.pi)
    aeff_per_bin[trigger] = trig_rate_per_bin[trigger] * (max_distance)**2 / np.pi


# Plot each trigger separately, showing different sin bins
for iT, trigger in enumerate(trigger_names):
    for iS in range(len(sin2Val)-1):
        # plt.bar(e_range[:-1], aeff_per_bin[trigger][iS], width=0.1, alpha=0.5, label=f'{angle_bins[iS]:.1f}-{angle_bins[iS+1]:.1f}sin2')
        plt.hist(e_range[:-1], weights=aeff_per_bin[trigger][iS], bins=e_range, histtype='step', label=f'{angle_bins[iS]:.1f}-{angle_bins[iS+1]:.1f} deg')
    plt.xlabel('Energy [eV]')
    plt.ylabel('Effective Area [km^2]')
    plt.yscale('log')
    plt.legend(loc='upper left', prop={'size': 8})
    plt.savefig(plot_folder+f'Stn51_Aeff_{trigger}.png')
    plt.clf()

# Plot all triggers together, sum of all sin bins
aeff_sin_sum = {}
for iT, trigger in enumerate(trigger_names):
    aeff_sin_sum[trigger] =  np.zeros(len(e_range)-1)
    for iS in range(len(sin2Val)-1):
        aeff_sin_sum[trigger] += aeff_per_bin[trigger][iS]
    # plt.bar(e_range[:-1], aeff_sin_sum[trigger], width=0.1, alpha=0.5, label=f'{trigger}')
    plt.hist(e_range[:-1], weights=aeff_sin_sum[trigger], bins=e_range, histtype='step', label=f'{trigger}, {np.sum(aeff_sin_sum[trigger]):.0f} total Aeff', color=colors[iT])
plt.xlabel('Energy [eV]')
plt.ylabel('Effective Area [km^2]')
plt.yscale('log')
plt.legend(loc='upper left', prop={'size': 8})
plt.savefig(plot_folder+f'Stn51_Aeff_AllTriggers.png')
plt.clf()


rate_per_bin = {}
rate_sin_sum = {}
for iT, trigger in enumerate(trigger_names):
    rate_per_bin[trigger] = aeff_per_bin[trigger]
    rate_sin_sum[trigger] = np.zeros(len(e_range)-1)
    for iE in range(len(e_range)-1):
        for iS in range(len(sin2Val)-1):
            ic(aeff_per_bin[trigger][iS][iE] )
            high_flux = auger.event_rate(e_range[iE], e_range[iE+1], zmax=angle_bins[iS+1], area=aeff_per_bin[trigger][iS][iE])
            low_flux = auger.event_rate(e_range[iE], e_range[iE+1], zmax=angle_bins[iS], area=aeff_per_bin[trigger][iS][iE])
            # flux_in_bin = auger.event_rate(e_range[iE], e_range[iE+1], zmax=90, area=aeff_per_bin[trigger][iE] / units.km**2)
            rate_per_bin[trigger][iS][iE] = high_flux - low_flux

    for iS in range(len(sin2Val)-1):
        rate_sin_sum[trigger] += rate_per_bin[trigger][iS]


# Plot event rate showing different angle bins per trigger
for iT, trigger in enumerate(trigger_names):
    for iS in range(len(sin2Val)-1):
        # plt.bar(e_range[:-1], rate_per_bin[trigger][iS], width=0.1, alpha=0.5, label=f'{angle_bins[iS]:.1f}-{angle_bins[iS+1]:.1f}sin2')
        plt.hist(e_range[:-1], weights=rate_per_bin[trigger][iS], bins=e_range, histtype='step', label=f'{angle_bins[iS]:.1f}-{angle_bins[iS+1]:.1f} deg')
    plt.xlabel('Energy [eV]')
    plt.ylabel('Events / Year')
    plt.yscale('log')
    plt.legend(loc='upper left', prop={'size': 8})
    plt.savefig(plot_folder+f'Stn51_EventRate_{trigger}.png')
    plt.clf()

for iT, trigger in enumerate(trigger_names):
    # plt.bar(e_range[:-1], rate_sin_sum[trigger], width=0.1, alpha=0.5, label=trigger)
    evts_in_data = station_livetime * np.sum(rate_sin_sum[trigger])
    plt.hist(e_range[:-1], weights=rate_sin_sum[trigger], bins=e_range, histtype='step', label=f'{trigger}, {evts_in_data:.1f} events in data', color=colors[iT])
plt.xlabel('Energy [eV]')
plt.ylabel('Events / Year')
plt.yscale('log')
plt.legend(loc='upper left', prop={'size': 8})
plt.savefig(plot_folder+f'Stn51_EventRate_AllTriggers.png')
plt.clf()


# Calc rate after cutting at ~40deg. Doing cut at 39.2deg since that's bin edge
rate_sin_sum_cut = {}
for trigger in trigger_names:
    rate_sin_sum_cut[trigger] = np.zeros(len(e_range)-1)
    for iS in range(len(sin2Val)-1):
        if angle_bins[iS] > 39:
            rate_sin_sum_cut[trigger] += rate_per_bin[trigger][iS]

for iT, trigger in enumerate(trigger_names):
    # plt.bar(e_range[:-1], rate_sin_sum[trigger], width=0.1, alpha=0.5, label=trigger)
    evts_in_data = station_livetime * np.sum(rate_sin_sum[trigger])
    evts_in_data_cut = station_livetime * np.sum(rate_sin_sum_cut[trigger])
    plt.hist(e_range[:-1], weights=rate_sin_sum[trigger], bins=e_range, histtype='step', label=f'{trigger}, {evts_in_data:.1f} events in data', color=colors[iT])
    plt.hist(e_range[:-1], weights=rate_sin_sum_cut[trigger], bins=e_range, histtype='step', label=f'39deg cut, {evts_in_data_cut:.1f} events in data', color=colors[iT], linestyle='--')
plt.xlabel('Energy [eV]')
plt.ylabel('Events / Year')
plt.yscale('log')
plt.legend(loc='upper left', prop={'size': 8})
plt.savefig(plot_folder+f'Stn51_EventRate_AllTriggers_40degCut.png')
plt.clf()
