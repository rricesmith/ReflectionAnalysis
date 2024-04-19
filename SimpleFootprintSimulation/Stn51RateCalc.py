import os
import numpy as np
import astrotools.auger as auger
from NuRadioReco.modules.io import NuRadioRecoio
import matplotlib.pyplot as plt
from icecream import ic
from NuRadioReco.utilities import units



sim_folder = 'SimpleFootprintSimulation/output/4.17.24/'
plot_folder = 'plots/SimpleFootprintSimulation/'

max_distance = 2
n_cores = 100
num_icetop = 30
min_energy = 16.0
max_energy = 18.6
e_range = np.arange(min_energy, max_energy, 0.1)


total_cores = n_cores * num_icetop
trigger_names = ['direct_LPDA_2of3_3.5sigma', 'direct_LPDA_3of3_3.5sigma', 'direct_LPDA_3of3_5sigma']
trig_rate_per_bin = {}
for trigger in trigger_names:
    trig_rate_per_bin[trigger] = np.zeros(len(e_range)-1)


for iE in range(len(e_range)-1):
    nurFiles= []

    file = f'SimpleFootprintSimulation/output/4.16.24/Stn51_IceTop_{e_range[iE]:.1f}-{e_range[iE+1]:.1f}eV_{n_cores}cores.nur'
    for file in os.listdir(sim_folder):
        ic(file, f'{e_range[iE]:.1f}-{e_range[iE+1]:.1f}eV')
        if file.endswith('.nur') and (f'{e_range[iE]:.1f}-{e_range[iE+1]:.1f}eV' in file):
            ic(True)
            nurFiles.append(os.path.join(sim_folder, file))
    if nurFiles == []:
        continue

    ic(nurFiles)
    eventReader = NuRadioRecoio.NuRadioRecoio(nurFiles)
    n_trig_2of3_3_5sig = 0
    n_trig_3of3_3_5sig = 0
    n_trig_3of3_5sig = 0
    for i, evt in enumerate(eventReader.get_events()):
        station_ids = evt.get_station_ids()
        for stn_id in station_ids:
            station = evt.get_station(stn_id)
            for trigger in trigger_names:
                if station.has_triggered(trigger_name=trigger):
                    trig_rate_per_bin[trigger][iE] += 1

    
for trigger in trigger_names:
    trig_rate_per_bin[trigger] = trig_rate_per_bin[trigger] / total_cores

# trig_rate_per_bin is now n_trig/n_throw per energy bin
aeff_per_bin = {}
for trigger in trigger_names:
    ic((max_distance)**2 / np.pi)
    aeff_per_bin[trigger] = trig_rate_per_bin[trigger] * (max_distance)**2 / np.pi


for trigger in trigger_names:
    plt.bar(e_range[:-1], aeff_per_bin[trigger], width=0.1, alpha=0.5, label=trigger)
plt.xlabel('Energy [eV]')
plt.ylabel('Effective Area [km^2]')
plt.yscale('log')
plt.legend()
plt.savefig(plot_folder+'Stn51_Aeff.png')
plt.clf()


rate_per_bin = {}
for trigger in trigger_names:
    rate_per_bin[trigger] = aeff_per_bin[trigger]
    for iE in range(len(e_range)-1):
        ic(aeff_per_bin[trigger][iE] )
        flux_in_bin = auger.event_rate(e_range[iE], e_range[iE+1], zmax=90, area=aeff_per_bin[trigger][iE] / units.km**2)
        rate_per_bin[trigger][iE] = flux_in_bin

for trigger in trigger_names:
    plt.bar(e_range[:-1], rate_per_bin[trigger], width=0.1, alpha=0.5, label=trigger + f', {sum(rate_per_bin[trigger]):.2f}/yr')
plt.xlabel('Energy [log10eV]')
plt.ylabel('Events per year')
plt.yscale('log')
plt.legend()
plt.savefig(plot_folder+'Stn51_EventRate.png')
plt.clf()