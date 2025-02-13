from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import eventParameters as evtp
from NuRadioReco.framework.parameters import showerParameters as shp
import NuRadioReco.modules.io.eventReader
from icecream import ic
import os
import numpy as np
import astrotools.auger as auger
import matplotlib.colors
import matplotlib.pyplot as plt
import HRASimulation.HRAAnalysis as HRAAnalysis
import configparser
from HRASimulation.HRAAnalysis import HRAevent

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('HRASimulation/config.ini')
    sim_folder = config['FOLDERS']['sim_folder']
    numpy_folder = config['FOLDERS']['numpy_folder']
    save_folder = config['FOLDERS']['save_folder']


    os.makedirs(save_folder, exist_ok=True)

    if os.path.exists(f'{numpy_folder}HRAeventList.npy'):        
        HRAeventList = np.load(f'{numpy_folder}HRAeventList.npy', allow_pickle=True)
        direct_trigger_rate_dict, reflected_trigger_rate_dict, combined_trigger_rate, e_bins, z_bins = np.load(f'{numpy_folder}trigger_rate_dict.npy', allow_pickle=True)
        direct_event_rate, reflected_event_rate, combined_event_rate = np.load(f'{numpy_folder}event_rate_dict.npy', allow_pickle=True)
    else:
        ic('No numpy file found.  Please run HRAAnalysis.py first')
        quit()


    if HRAeventList[0][0].weight == np.nan:
        ic('Weights not yet set, run HRAAnalysis.py first')
        quit()


    # Plot the area of the HRA
    save_folder = f'{save_folder}Area/'
    os.makedirs(save_folder, exist_ok=True)


    bad_stations = [32, 52, 132, 152]
    savename = f'{save_folder}AreaReflected.png'
    x, y, weight = HRAAnalysis.getXYWeights(HRAeventList, bad_stations=bad_stations)
    HRAAnalysis.histAreaRate(x, y, weight, save_folder)
    ic(f'Saved {savename}')

    bad_stations = [32, 52, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    savename = f'{save_folder}AreaDirect.png'
    x, y, weight = HRAAnalysis.getXYWeights(HRAeventList, bad_stations=bad_stations)
    HRAAnalysis.histAreaRate(x, y, weight, save_folder)
    ic(f'Saved {savename}')

    bad_stations = [32, 132, 152]
    savename = f'{save_folder}AreaReflected_w52.png'
    x, y, weight = HRAAnalysis.getXYWeights(HRAeventList, bad_stations=bad_stations)
    HRAAnalysis.histAreaRate(x, y, weight, save_folder)
    ic(f'Saved {savename}')

    bad_stations = [32, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    savename = f'{save_folder}AreaDirect_w52.png'
    x, y, weight = HRAAnalysis.getXYWeights(HRAeventList, bad_stations=bad_stations)
    HRAAnalysis.histAreaRate(x, y, weight, save_folder)
    ic(f'Saved {savename}')

