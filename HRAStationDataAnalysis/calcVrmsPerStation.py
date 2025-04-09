from icecream import ic
from NuRadioReco.detector import detector
import NuRadioReco.modules.channelSignalReconstructor
from NuRadioReco.modules.io import NuRadioRecoio
import datetime
import numpy as np
from HRAStationDataAnalysis.HRADataConvertToNpy import inBlackoutTime, getBlackoutTimes
import os
from NuRadioReco.framework.parameters import channelParameters as chp

def getVrms(nurFiles, save_chans, station_id, det, blackoutTimes, max_check=1000):
    # Calculate the average Vrms for given channels based on forced triggers

    channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
    template = NuRadioRecoio.NuRadioRecoio(nurFiles)


    Vrms_sum = 0
    num_avg = 0

    for i, evt in enumerate(template.get_events()):
        station = evt.get_station(station_id)
        stationtime = station.get_station_time().unix
        if inBlackoutTime(stationtime, blackoutTimes):
            continue

        # Check if the event is a forced trigger
        if station.has_triggered():
            continue

        channelSignalReconstructor.run(evt, station, det)
        for ChId, channel in enumerate(station.iter_channels(use_channels=save_chans)):
            Vrms_sum += channel[chp.noise_rms]
            ic(f'Vrms for channel {ChId} in station {station_id}: {channel[chp.noise_rms]}')
            num_avg += 1
        
        if num_avg >= max_check:
            break

    ic(f'Vrms sum: {Vrms_sum}, num_avg: {num_avg}')

    return Vrms_sum / num_avg

if __name__ == "__main__":


    stations = [13, 14, 15, 17, 18, 19, 30]
    save_channels = [0, 1, 2, 3]
    det = detector.Detector(f"HRASimulation/HRAStationLayoutForCoREAS.json")
    blackoutTimes = getBlackoutTimes()

    vrms_per_station = {}

    for station_id in stations:

        HRAdataPath = f"/dfs8/sbarwick_lab/ariannaproject/station_nur/station_{station_id}/"
        nurFiles = []
        for file in os.listdir(HRAdataPath):
            if file.endswith('_statDatPak.root.nur'):
                continue
            else:
                nurFiles.append(HRAdataPath + file)


        ic('calculating Vrms')
        Vrms = getVrms(nurFiles, save_channels, station_id, det, blackoutTimes)
        ic(f'normalizing to {Vrms} vrms')

        vrms_per_station[station_id] = Vrms

    # Save in a human-readable and importable format
    with open('HRAStationDataAnalysis/vrms_per_station.txt', 'w') as f:
        for station_id, vrms in vrms_per_station.items():
            f.write(f'{station_id}: {vrms}\n')
    ic(vrms_per_station)