import os
import numpy as np

from NuRadioReco.utilities import units, fft
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.utilities.io_utilities import read_pickle

import matplotlib.pyplot as plt
from icecream import ic


def getEvenlySpacedPoints(data_1d, num=20):
    # returns points evently spaced in dimension
    return np.linspace(min(data_1d), max(data_1d), num=num)

def getEvenlySpacedFromData2d(x_data, y_data, num_per_direction=20):
    # From points (x, y) returns num^2 points maximizing distance between them
    x_spaced = getEvenlySpacedPoints(x_data, num=num_per_direction)
    y_spaced = getEvenlySpacedPoints(y_data, num=num_per_direction)
    ic(x_spaced, y_spaced)

    return_x = []
    return_y = []
    return_ind = []

    for x in x_spaced:
        min_ind = np.argmin(np.abs(x_data - x))
        for y in y_spaced:
            d = np.inf
            min_x = 0
            min_y = 0
            for i in range(-3, 3):
                if min_ind + i < 0 or min_ind + i > len(x_data):
                    continue
                dist = np.sqrt((x_data[min_ind+i] - x)**2 + (y_data[min_ind+i] - y)**2)
                if dist < d:
                    d = dist
                    min_x = min_ind + i
                    min_y = min_ind + i
            return_x.append(x_data[min_x])
            return_y.append(y_data[min_y])
            return_ind.append(min_x)

    # ic(return_x, return_y, return_ind)

    return np.array(return_x), np.array(return_y), np.array(return_ind)


def plotOldTemplate(ax):
    templates_RCR = 'StationDataAnalysis/templates/reflectedCR_template_200series.pkl'
    templates_RCR = read_pickle(templates_RCR)
    for key in templates_RCR:
        temp = templates_RCR[key]
    templates_RCR = temp

    ax.plot(templates_RCR, label='RCR')

# fig, ax = plt.subplots()
# plotOldTemplate(ax)
# fig.savefig(f'DeepLearning/plots/testTraces/templates/OldSingleRCRTemplate_200s.png')


series = '200s'     #Alternative is 200s
simdate = '10.1.24'
# station_files_path = 'FootprintAnalysis/output/'
# station_files_path = f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/simulatedRCRs/{series}_2.9.24/'
station_files_path = f'DeepLearning/data/5thpass/'

SimRCRFiles = []
SimRCRParams = []
for filename in os.listdir(station_files_path):
    if filename.startswith(f'SimRCR_{series}'):
        SimRCRFiles.append(os.path.join(station_files_path, filename))
    if filename.startswith(f'SimParams_SimRCR_{series}'):
        SimRCRParams.append(os.path.join(station_files_path, filename))


# New method for getting templates from processed numpy files from D01
ic(SimRCRFiles)
ic(SimRCRParams)

for file in SimRCRFiles:
    traces = np.load(file, allow_pickle=True)
for file in SimRCRParams:
    params = np.load(file, allow_pickle=True)   # energy in log10eV, zenith in deg, azimuth in deg
ic(len(traces))
ic(len(params))


energies = []
zeniths = []
azimuths = []
for iT, trace in enumerate(traces):
    energies.append(params[iT][0])
    zeniths.append(params[iT][1])
    azimuths.append(params[iT][2])

energies = np.array(energies)
zeniths = np.array(zeniths)
azimuths = np.array(azimuths)

plt.scatter(energies, zeniths)
plt.ylabel('Zenith')
plt.xlabel('Energy')
# plt.xlim(16, 20)
plt.ylim(0, 90)
plt.title('Zenith vs Energy')
savename = f'DeepLearning/plots/testTraces/templates/{simdate}/{series}_ZenithEnergy.png'
plt.savefig(savename)
# plt.clf()
print(f'Saved {savename}')

# Test background plot
x_spaced = getEvenlySpacedPoints(energies, num=20)
y_spaced = getEvenlySpacedPoints(zeniths, num=20)
for x in x_spaced:
    for y in y_spaced:
        plt.scatter(x, y, color='red', alpha=0.5, s=20)

spaced_x, spaced_y, spaced_ind = getEvenlySpacedFromData2d(energies, zeniths, num_per_direction=20)
plt.scatter(spaced_x, spaced_y, marker='*',label='selected', edgecolors='black', facecolor='black', s=120)
plt.legend()
savename=f'DeepLearning/plots/testTraces/templates/{simdate}/{series}_ZenithEnergy_selected.png'
plt.savefig(savename)
plt.clf()
print(f'Saved {savename}')



for ind in np.unique(spaced_ind):
    fig, ax = plt.subplots(nrows=4, ncols=1)

    max_trace = 0
    for i in range(4):
        if max(traces[ind][i]) > max(traces[ind][max_trace]):
            max_trace = i

    for i in range(4):
        if i == max_trace:
            ax[i].plot(traces[ind][i], color='red', label='chosen')
            ax[i].legend()
        else:
            ax[i].plot(traces[ind][i], color='blue')

    # plt.plot(traces[ind][max_trace])
    fig.suptitle(f'{series} {energies[ind]:.1f}log10eV, {zeniths[ind]:.1f}Deg Zen, {azimuths[ind]:.1f}Deg Azi, {ind} Event')
    savename = f'DeepLearning/plots/testTraces/templates/{simdate}/MaxTrace_{series}_{ind}_{energies[ind]:.1f}eV_{zeniths[ind]:.1f}Zen.png'
    fig.savefig(savename)
    fig.clf()
    plt.close(fig)
    print(f'Saved {savename}')

    savetemplate = f'DeepLearning/templates/RCR/{simdate}/{series}_{ind}_{energies[ind]:.1f}eV_{zeniths[ind]:.1f}Zen.npy'
    np.save(savetemplate, traces[ind][max_trace])
    print(f'Saved {savetemplate}')

quit()

# Old method for creating template from nur files directly

# saveChannels = [4, 5, 6, 7]
# template = NuRadioRecoio.NuRadioRecoio(SimRCRFiles)
# for i, evt in enumerate(template.get_events()):

#     station = evt.get_station(1)

#     sim_shower = evt.get_sim_shower(0)
#     sim_energy = sim_shower[shp.energy]
#     sim_zen = sim_shower[shp.zenith]
#     sim_azi = sim_shower[shp.azimuth]

#     zeniths.append(sim_zen)
#     energies.append(sim_energy)

    # template = [0]
    # for ChId, channel in enumerate(station.iter_channels(use_channels=saveChannels)):

    #     y = channel.get_trace()
    #     if max(y) > max(template):
    #         template = y

    # if True:
    #     plt.plot(template)
    #     plt.title(f'{series} {np.log10(sim_energy):.1f}log10eV, {np.rad2deg(sim_zen):.1f}Deg Zen, {np.rad2deg(sim_azi):.1f}Deg Azi, {i} Event')
    #     plt.savefig(f'DeepLearning/plots/testTraces/templates/{series}_{i}_{np.log10(sim_energy):.1f}eV_{np.rad2deg(sim_zen):.1f}Zen.png')
    #     plt.clf()

