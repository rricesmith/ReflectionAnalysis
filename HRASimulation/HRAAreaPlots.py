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


    # Testing if one station, 13, has weights set
    if HRAeventList[0].getWeight(13) == np.nan:
        ic('Weights not yet set, run HRAAnalysis.py first')
        quit()

    dir_trig, refl_trig = HRAAnalysis.getDirectReflTriggered(HRAeventList)
    relf_trig = np.array(refl_trig) - 100 # Subtract 100 to get the station number

    # Plot the area of the HRA
    save_folder = f'{save_folder}Area/'
    os.makedirs(save_folder, exist_ok=True)


    bad_stations = [32, 52, 132, 152]
    savename = f'{save_folder}AreaReflected.png'
    x, y, weight = HRAAnalysis.getXYWeights(HRAeventList, weight_name='combined_reflected')
    HRAAnalysis.histAreaRate(x, y, weight, "Combined Reflected", savename, dir_trig=dir_trig, refl_trig=refl_trig, exclude=bad_stations)
    ic(f'Saved {savename}')

    bad_stations = [32, 52, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    savename = f'{save_folder}AreaDirect.png'
    x, y, weight = HRAAnalysis.getXYWeights(HRAeventList, weight_name='combined_direct')
    HRAAnalysis.histAreaRate(x, y, weight, "Combined Direct", savename, dir_trig=dir_trig, refl_trig=refl_trig, exclude=bad_stations)
    ic(f'Saved {savename}')


    # Plot the area of the HRA for coinc reflections
    for i in [2, 3, 4, 5, 6, 7]:
        weight_name=f'{i}_coincidence_wrefl'
        if not HRAeventList[0].hasWeight(weight_name):
            ic(f'Weight {weight_name} not found')
            continue
        savename = f'{save_folder}AreaDirect_coinc{i}.png'
        x, y, weight = HRAAnalysis.getXYWeights(HRAeventList, weight_name=weight_name)
        HRAAnalysis.histAreaRate(x, y, weight, f'{i} Coinc w/Refl',savename, dir_trig=dir_trig, refl_trig=refl_trig, exclude=bad_stations)
        ic(f'Saved {savename}')

    # Same but without reflections
    for i in [2, 3, 4, 5, 6, 7]:
        weight_name=f'{i}_coincidence_norefl'
        if not HRAeventList[0].hasWeight(weight_name):
            ic(f'Weight {weight_name} not found')
            continue
        savename = f'{save_folder}AreaDirect_coinc{i}.png'
        x, y, weight = HRAAnalysis.getXYWeights(HRAeventList, weight_name=weight_name)
        HRAAnalysis.histAreaRate(x, y, weight, f'{i} Coinc w/o Refl', savename, dir_trig=dir_trig, refl_trig=refl_trig, exclude=bad_stations)
        ic(f'Saved {savename}')

    # Plots with reflection and station 52
    for i in [2, 3, 4, 5, 6, 7]:
        weight_name=f'{i}_coincidence_52up_wrefl'
        if not HRAeventList[0].hasWeight(weight_name):
            ic(f'Weight {weight_name} not found')
            continue
        savename = f'{save_folder}Area_52up_wrefl_coinc{i}.png'
        x, y, weight = HRAAnalysis.getXYWeights(HRAeventList, weight_name=weight_name)
        HRAAnalysis.histAreaRate(x, y, weight, f'{i} Coinc w/52up w/Refl',savename, dir_trig=dir_trig, refl_trig=refl_trig, exclude=bad_stations)
        ic(f'Saved {savename}')

    # Plots without reflection and station 52
    for i in [2, 3, 4, 5, 6, 7]:
        weight_name=f'{i}_coincidence_52up_norefl'
        if not HRAeventList[0].hasWeight(weight_name):
            ic(f'Weight {weight_name} not found')
            continue
        savename = f'{save_folder}Area_52up_norefl_coinc{i}.png'
        x, y, weight = HRAAnalysis.getXYWeights(HRAeventList, weight_name=weight_name)
        HRAAnalysis.histAreaRate(x, y, weight, f'{i} Coinc w/52up w/o Refl',savename, dir_trig=dir_trig, refl_trig=refl_trig, exclude=bad_stations)
        ic(f'Saved {savename}')
