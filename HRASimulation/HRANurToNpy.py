from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import eventParameters as evtp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.modules.io.eventReader
from icecream import ic
import os
import numpy as np
import astrotools.auger as auger
import matplotlib.colors
import matplotlib.pyplot as plt
import configparser
import HRASimulation.HRAAnalysis as HRAA
import h5py
import pickle

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('HRASimulation/config.ini')
    sim_folder = config['FOLDERS']['sim_folder']
    numpy_folder = config['FOLDERS']['numpy_folder']
    save_folder = config['FOLDERS']['save_folder']
    plot_sigma = float(config['PLOTPARAMETERS']['trigger_sigma'])
    ic(plot_sigma)

    os.makedirs(save_folder, exist_ok=True)

    if not os.path.exists(numpy_folder):
        os.makedirs(numpy_folder)

    HRAeventList = HRAA.getHRAeventsFromDir(sim_folder)
    direct_trigger_rate_dict, reflected_trigger_rate_dict, combined_trigger_rate, e_bins, z_bins = HRAA.getBinnedTriggerRate(HRAeventList)
    direct_event_rate = {}
    for station_id in direct_trigger_rate_dict:
        ic(station_id)
        direct_event_rate[station_id] = HRAA.getEventRate(direct_trigger_rate_dict[station_id], e_bins, z_bins)
        HRAA.setHRAeventListRateWeight(HRAeventList, direct_trigger_rate_dict[station_id], weight_name=station_id)
    reflected_event_rate = {}
    for station_id in reflected_trigger_rate_dict:
        ic(station_id)
        reflected_event_rate[station_id] = HRAA.getEventRate(reflected_trigger_rate_dict[station_id], e_bins, z_bins)
        HRAA.setHRAeventListRateWeight(HRAeventList, reflected_trigger_rate_dict[station_id], weight_name=station_id)

    combined_event_rate = {}
    combined_event_rate['direct'] = HRAA.getEventRate(combined_trigger_rate['direct'], e_bins, z_bins)
    combined_event_rate['reflected'] = HRAA.getEventRate(combined_trigger_rate['reflected'], e_bins, z_bins)
    HRAA.setHRAeventListRateWeight(HRAeventList, combined_trigger_rate['direct'], weight_name='combined_direct')
    HRAA.setHRAeventListRateWeight(HRAeventList, combined_trigger_rate['reflected'], weight_name='combined_reflected')


    np.save(f'{numpy_folder}trigger_rate_dict.npy', [direct_trigger_rate_dict, reflected_trigger_rate_dict, combined_trigger_rate, e_bins, z_bins])
    np.save(f'{numpy_folder}event_rate_dict.npy', [direct_event_rate, reflected_event_rate, combined_event_rate])


    # Coincidence plots with reflections
    bad_stations = [32, 52, 132, 152]
    trigger_rate_coincidence = HRAA.getCoincidencesTriggerRates(HRAeventList, bad_stations)
    event_rate_coincidence = {}
    for i in trigger_rate_coincidence:
        if not np.any(trigger_rate_coincidence[i] > 0):
            ic(f'No events for {i} coincidences')
            continue
        HRAA.setHRAeventListRateWeight(HRAeventList, trigger_rate_coincidence[i], weight_name=f'{i}_coincidence_wrefl')

    # Coincidence plots without reflections
    bad_stations = [32, 52, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    trigger_rate_coincidence = HRAA.getCoincidencesTriggerRates(HRAeventList, bad_stations)
    event_rate_coincidence = {}
    for i in trigger_rate_coincidence:
        if not np.any(trigger_rate_coincidence[i] > 0):
            ic(f'No events for {i} coincidences')
            continue
        HRAA.setHRAeventListRateWeight(HRAeventList, trigger_rate_coincidence[i], weight_name=f'{i}_coincidence_norefl')


    # Coincidence with reflection and station 52 upwards LPDA
    bad_stations = [32, 132, 152]
    trigger_rate_coincidence = HRAA.getCoincidencesTriggerRates(HRAeventList, bad_stations, use_secondary=False, force_station=52)
    event_rate_coincidence = {}
    for i in trigger_rate_coincidence:
        if not np.any(trigger_rate_coincidence[i] > 0):
            ic(f'No events for {i} coincidences')
            continue
        HRAA.setHRAeventListRateWeight(HRAeventList, trigger_rate_coincidence[i], weight_name=f'{i}_coincidence_52up_wrefl')


    # Coincidence without reflection and station 52 upwards LPDA
    bad_stations = [32, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    trigger_rate_coincidence = HRAA.getCoincidencesTriggerRates(HRAeventList, bad_stations, use_secondary=False, force_station=52)
    event_rate_coincidence = {}
    for i in trigger_rate_coincidence:
        if not np.any(trigger_rate_coincidence[i] > 0):
            ic(f'No events for {i} coincidences')
            continue
        HRAA.setHRAeventListRateWeight(HRAeventList, trigger_rate_coincidence[i], weight_name=f'{i}_coincidence_52up_norefl')

    # np.save(f'{numpy_folder}HRAeventList.npy', HRAeventList)

    # Attempt to save as h5py file instead
    with h5py.File(f'{numpy_folder}HRAeventList.h5', 'w') as hf:
        for i, obj in enumerate(HRAeventList):
            # Serialize if needed
            if not isinstance(obj, np.ndarray):
                obj_bytes = pickle.dumps(obj)
                dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                dset = hf.create_dataset(f'object_{i}', (1,), dtype=dt)
                dset[0] = np.frombuffer(obj_bytes, dtype='uint8')
            else:
                hf.create_dataset(f'object_{i}', data=obj)  