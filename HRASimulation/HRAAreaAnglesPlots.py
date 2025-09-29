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
from HRASimulation.HRAEventObject import HRAevent
from HRASimulation.S02_HRANurToNpy import loadHRAfromH5

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('HRASimulation/config.ini')
    sim_folder = config['FOLDERS']['sim_folder']
    numpy_folder = config['FOLDERS']['numpy_folder']
    save_folder = config['FOLDERS']['save_folder']
    diameter = config['SIMPARAMETERS']['diameter']
    max_distance = float(diameter)/2*units.km


    os.makedirs(save_folder, exist_ok=True)

    HRAeventList = loadHRAfromH5(f'{numpy_folder}HRAeventList.h5')
    # direct_trigger_rate_dict, reflected_trigger_rate_dict, combined_trigger_rate, e_bins, z_bins = np.load(f'{numpy_folder}trigger_rate_dict.npy', allow_pickle=True)
    # direct_event_rate, reflected_event_rate, combined_event_rate = np.load(f'{numpy_folder}event_rate_dict.npy', allow_pickle=True)


    # Testing if one station, 13, has weights set
    if HRAeventList[0].getWeight(13) == np.nan:
        ic('Weights not yet set, run HRAAnalysis.py first')
        quit()

    dir_trig, refl_trig = HRAAnalysis.getDirectReflTriggered(HRAeventList)
    refl_trig = np.array(refl_trig)

    # Plot the area of the HRA
    save_folder = f'{save_folder}AreaAngles/'
    os.makedirs(save_folder, exist_ok=True)


    for station in dir_trig:
        savename = f'{save_folder}AreaAnglesDirect_{station}.png'
        x, y, weights = HRAAnalysis.getXYWeights(HRAeventList, weight_name=station)
        mask = weights != 0
        zenith, recon_zenith, azimuth, recon_azimuth, weights = HRAAnalysis.getAnglesReconWeights(HRAeventList, station, station)
        HRAAnalysis.plotAreaAziZenArrows(x[mask], y[mask], zenith[mask], azimuth[mask], weights[mask], title=f'{station} Direct', savename=savename, dir_trig=dir_trig, refl_trig=refl_trig-100, max_distance=max_distance)

    for station in refl_trig:
        savename = f'{save_folder}AreaAnglesReflected_{station}.png'
        x, y, weights = HRAAnalysis.getXYWeights(HRAeventList, weight_name=station)
        mask = weights != 0
        zenith, recon_zenith, azimuth, recon_azimuth, weightss = HRAAnalysis.getAnglesReconWeights(HRAeventList, station, station)
        HRAAnalysis.plotAreaAziZenArrows(x[mask], y[mask], zenith[mask], azimuth[mask], weights[mask], title=f'{station} Reflected', savename=savename, dir_trig=dir_trig, refl_trig=refl_trig-100, max_distance=max_distance)

    relf_trig = np.array(refl_trig) - 100 # Subtract 100 to get the station number for plot numbers
    ic(dir_trig, refl_trig)

    # Plot combined reflected
    bad_stations = [32, 52, 132, 152]
    savename = f'{save_folder}AreaAnglesReflectedCombined.png'
    x, y, weights = HRAAnalysis.getXYWeights(HRAeventList, weight_name='combined_reflected')
    mask = weights != 0
    zenith, recon_zenith, azimuth, recon_azimuth, weights = HRAAnalysis.getAnglesReconWeights(HRAeventList, 'combined_reflected', [113, 114, 115, 117, 118, 119, 130])
    HRAAnalysis.plotAreaAziZenArrows(x[mask], y[mask], zenith[mask], azimuth[mask], weights[mask], title="Combined Reflected", savename=savename, dir_trig=dir_trig, refl_trig=refl_trig, exclude=bad_stations, max_distance=max_distance)
    ic(f'Saved {savename}')

    # Plot weighted histogram of zenith and azimuths
    savename = f'{save_folder}ReflectedCombined_Hist.png'
    HRAAnalysis.histAngleRecon(zenith[mask], azimuth[mask], recon_zenith[mask], recon_azimuth[mask], weights[mask], title="Combined Reflected", savename=savename)

    # Plot combined direct
    bad_stations = [32, 52, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    savename = f'{save_folder}AreaAnglesDirectCombined.png'
    x, y, weights = HRAAnalysis.getXYWeights(HRAeventList, weight_name='combined_direct')
    mask = weights != 0
    zenith, recon_zenith, azimuth, recon_azimuth, weights = HRAAnalysis.getAnglesReconWeights(HRAeventList, 'combined_direct', [13, 14, 15, 17, 18, 19, 13])
    HRAAnalysis.plotAreaAziZenArrows(x[mask], y[mask], zenith[mask], azimuth[mask], weights[mask], title="Combined Direct", savename=savename, dir_trig=dir_trig, refl_trig=refl_trig, exclude=bad_stations, max_distance=max_distance)
    ic(f'Saved {savename}')

    # Plot weighted histogram of zenith and azimuths
    savename = f'{save_folder}DirectCombined_Hist.png'
    HRAAnalysis.histAngleRecon(zenith[mask], azimuth[mask], recon_zenith[mask], recon_azimuth[mask], weights[mask], title="Combined Direct", savename=savename)


    # Now plot coincidences

    # Coinc all w/refl
    bad_stations = [32, 52, 132, 152]
    good_stations = [13, 14, 15, 17, 18, 19, 30, 113, 114, 115, 117, 118, 119, 130]
    for i in [2, 3, 4, 5, 6, 7]:
        weight_name=f'{i}_coincidence_wrefl'
        if not HRAeventList[0].hasWeight(weight_name):
            ic(f'Weight {weight_name} not found')
            continue
        savename = f'{save_folder}AreaAnglesDirect_coinc{i}.png'
        x, y, weights = HRAAnalysis.getXYWeights(HRAeventList, weight_name=weight_name)
        mask = weights != 0
        zenith, recon_zenith, azimuth, recon_azimuth, weights = HRAAnalysis.getAnglesReconWeights(HRAeventList, weight_name, good_stations)
        HRAAnalysis.plotAreaAziZenArrows(x[mask], y[mask], zenith[mask], azimuth[mask], weights[mask], title=f'{i} Coinc w/Refl',savename=savename, dir_trig=dir_trig, refl_trig=refl_trig, exclude=bad_stations, max_distance=max_distance)
        ic(f'Saved {savename}')

    # Coinc all w/o refl
    bad_stations = [32, 52, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    good_stations = [13, 14, 15, 17, 18, 19, 30]
    for i in [2, 3, 4, 5, 6, 7]:
        weight_name=f'{i}_coincidence_norefl'
        if not HRAeventList[0].hasWeight(weight_name):
            ic(f'Weight {weight_name} not found')
            continue
        savename = f'{save_folder}AreaAnglesDirect_norefl_coinc{i}.png'
        x, y, weights = HRAAnalysis.getXYWeights(HRAeventList, weight_name=weight_name)
        mask = weights != 0
        zenith, recon_zenith, azimuth, recon_azimuth, weights = HRAAnalysis.getAnglesReconWeights(HRAeventList, weight_name, good_stations)
        HRAAnalysis.plotAreaAziZenArrows(x[mask], y[mask], zenith[mask], azimuth[mask], weights[mask], title=f'{i} Coinc w/o Refl',savename=savename, dir_trig=dir_trig, refl_trig=refl_trig, exclude=bad_stations, max_distance=max_distance)
        ic(f'Saved {savename}')

    # Coinc with reflection and station 52
    bad_stations = [32, 132, 152]
    good_stations = [13, 14, 15, 17, 18, 19, 30, 52, 113, 114, 115, 117, 118, 119, 130]
    for i in [2, 3, 4, 5, 6, 7]:
        weight_name=f'{i}_coincidence_52up_wrefl'
        if not HRAeventList[0].hasWeight(weight_name):
            ic(f'Weight {weight_name} not found')
            continue
        savename = f'{save_folder}AreaAnglesDirect_coinc{i}_52up.png'
        x, y, weights = HRAAnalysis.getXYWeights(HRAeventList, weight_name=weight_name)
        mask = weights != 0
        zenith, recon_zenith, azimuth, recon_azimuth, weights = HRAAnalysis.getAnglesReconWeights(HRAeventList, weight_name, good_stations)
        HRAAnalysis.plotAreaAziZenArrows(x[mask], y[mask], zenith[mask], azimuth[mask], weights[mask], title=f'{i} Coinc w/Refl 52up',savename=savename, dir_trig=dir_trig, refl_trig=refl_trig, exclude=bad_stations, max_distance=max_distance)
        ic(f'Saved {savename}')

    # Coinc without reflection and with station 52
    bad_stations = [32, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    good_stations = [13, 14, 15, 17, 18, 19, 30, 52]
    for i in [2, 3, 4, 5, 6, 7]:
        weight_name=f'{i}_coincidence_52up_norefl'
        if not HRAeventList[0].hasWeight(weight_name):
            ic(f'Weight {weight_name} not found')
            continue
        savename = f'{save_folder}AreaAnglesDirect_coinc{i}_52up_norefl.png'
        x, y, weights = HRAAnalysis.getXYWeights(HRAeventList, weight_name=weight_name)
        mask = weights != 0
        zenith, recon_zenith, azimuth, recon_azimuth, weights = HRAAnalysis.getAnglesReconWeights(HRAeventList, weight_name, good_stations)
        HRAAnalysis.plotAreaAziZenArrows(x[mask], y[mask], zenith[mask], azimuth[mask], weights[mask], title=f'{i} Coinc w/o Refl 52up',savename=savename, dir_trig=dir_trig, refl_trig=refl_trig, exclude=bad_stations, max_distance=max_distance)
        ic(f'Saved {savename}')