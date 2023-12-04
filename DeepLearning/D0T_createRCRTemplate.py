import os
import numpy as np

from NuRadioReco.utilities import units, fft
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.modules.io import NuRadioRecoio

import matplotlib.pyplot as plt



series = '200s'     #Alternative is 200s
station_files_path = 'FootprintAnalysis/output/'




SimRCRFiles = []
for filename in os.listdir(station_files_path):
    if filename.startswith(f'MB_old_{series}') and filename.endswith('.nur'):
        SimRCRFiles.append(os.path.join(station_files_path, filename))

saveChannels = [4, 5, 6, 7]

template = NuRadioRecoio.NuRadioRecoio(SimRCRFiles)

for i, evt in enumerate(template.get_events()):

    station = evt.get_station(1)

    sim_shower = evt.get_sim_shower(0)
    sim_energy = sim_shower[shp.energy]
    sim_zen = sim_shower[shp.zenith]
    sim_azi = sim_shower[shp.azimuth]

    template = [0]
    for ChId, channel in enumerate(station.iter_channels(use_channels=saveChannels)):

        y = channel.get_trace()
        if max(y) > max(template):
            template = y

    if True:
        plt.plot(template)
        plt.title(f'{series} {np.log10(sim_energy):.1f}log10eV, {np.rad2deg(sim_zen):.1f}Deg Zen, {np.rad2deg(sim_azi):.1f}Deg Azi, {i} Event')
        plt.savefig(f'DeepLearning/plots/testTraces/templates/{series}_{i}_{np.log10(sim_energy):.1f}eV_{np.rad2deg(sim_zen):.1f}Zen.png')
        plt.clf()

