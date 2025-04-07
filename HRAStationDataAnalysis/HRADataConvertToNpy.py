import datetime
import numpy as np
from NuRadioReco.modules.io import NuRadioRecoio
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.channelSignalReconstructor
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units
import argparse
import os
import json
import DeepLearning.D00_helperFunctions as D00_helperFunctions
from HRAStationDataAnalysis.calculateChi import getMaxChi, getMaxAllChi
from icecream import ic

def inBlackoutTime(time, blackoutTimes):
    #This check removes data that have bad datetime format. No events should be recorded before 2013 season when the first stations were installed
    if datetime.datetime.fromtimestamp(time) > datetime.datetime(2019, 3, 31):
        return True
    #This check removes events happening during periods of high noise
    for blackouts in blackoutTimes:
        if blackouts[0] < time and time < blackouts[1]:
            return True
    return False

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

        channelSignalReconstructor.run(evt, station, det)
        for ChId, channel in enumerate(station.iter_channels(use_channels=save_chans)):
            Vrms_sum += channel[chp.noise_rms]
            num_avg += 1
        
        if num_avg >= max_check:
            break


    return Vrms_sum / num_avg

def calcSNR(traces, Vrms):
    # Calculate SNR from the average of two highest traces

    SNRs = []
    for trace in traces:
        p2p = np.max(trace) - np.min(trace)
        SNRs.append(p2p/(2*Vrms))

    SNRs.sort(reverse=True)
    SNR = (SNRs[0] + SNRs[1]) / 2
    return SNR


def convertHRANurToNpy(nurFiles, save_channels, save_folder, station_id, prefix):
    # Pass in list of nur files, and save data to numpy files in folder

    # Proccessing should save the following:
    # 1. Traces
    # 2. Times
    # 3. SNR
    # 4. Chi
    # 5. Azimuth
    # 6. Zenith
    

    # Limit numpy size for memory issues
    max_events = 500000
    count = 0
    part = 0

    save_traces = np.zeros((max_events, 4, 256))
    save_times = np.zeros((max_events, 1))
    save_snr = np.zeros((max_events, 1))
    save_chi_2016 = np.zeros((max_events, 1))
    save_chi_RCR = np.zeros((max_events, 1))
    save_chi_RCR_bad = np.zeros((max_events, 1))
    # No calculation for Chi, as that is done in separate script depending upon templates desired
    save_azi = np.zeros((max_events, 1))
    save_zen = np.zeros((max_events, 1))


    file_reader = NuRadioRecoio.NuRadioRecoio(nurFiles)

    det = detector.Detector(f"HRASimulation/HRAStationLayoutForCoREAS.json")

    correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
    correlationDirectionFitter.begin(debug=False)

    templates_2016 = D00_helperFunctions.loadMultipleTemplates(100, template_date='2016')    # selection of 2016 events that are presumed to all be backlobes
    template_series_100 = D00_helperFunctions.loadMultipleTemplates(100)                         # selection of 'good' RCR simulated events for templates
    template_series_bad_100 = D00_helperFunctions.loadMultipleTemplates(100, bad=True)           # selection of 'bad' RCR simulated events for templates
    template_series_200 = D00_helperFunctions.loadMultipleTemplates(200)                         # selection of 'good' RCR simulated events for templates
    template_series_bad_200 = D00_helperFunctions.loadMultipleTemplates(200, bad=True)           # selection of 'bad' RCR simulated events for templates    
    stations_100s = [13, 15, 18, 32]
    stations_200s = [14, 17, 19, 30]

    blackoutFile = open('DeepLearning/BlackoutCuts.json')
    blackoutData = json.load(blackoutFile)
    blackoutFile.close()

    blackoutTimes = []

    for iB, tStart in enumerate(blackoutData['BlackoutCutStarts']):
        tEnd = blackoutData['BlackoutCutEnds'][iB]
        blackoutTimes.append([tStart, tEnd])

    # Get Vrm from forced triggers
    ic('calculating Vrms')
    Vrms = getVrms(nurFiles, save_channels, station_id, det, blackoutTimes)
    ic(f'normalizing to {Vrms} vrms')


    for i, evt in enumerate(file_reader.get_events()):

        station = evt.get_station(station_id)
        stationtime = station.get_station_time().unix
        det.update(station.get_station_time())

        if station in stations_100s:
            use_templates = template_series_100
            use_templates_bad = template_series_bad_100
        elif station in stations_200s:
            use_templates = template_series_200
            use_templates_bad = template_series_bad_200

        # Untriggered events are forced triggers, and should be ignored. They are only used for calculating Vrm of the station
        if not station.has_triggered():
            continue

        # Check if the event is in the blackout times previously found in earlier analysis, which mostly correspond to periods of high noise
        if inBlackoutTime(stationtime, blackoutTimes):
            continue

        if i % 1000 == 0:
            ic(f'{i} events processed...')

        # Save if limit reached
        count = i - max_events * part
        if count >= max_events:
            savename = f'{save_folder}/{prefix}_Station{station_id}'
            savesuffix = f'_Part{part}.npy'

            np.save(savename + '_Traces' + savesuffix, save_traces)
            np.save(savename + '_Times' + savesuffix, save_times)
            np.save(savename + '_SNR' + savesuffix, save_snr)
            np.save(savename + '_Chi2016' + savesuffix, save_chi_2016)
            np.save(savename + '_ChiRCR' + savesuffix, save_chi_RCR)
            np.save(savename + '_ChiBad' + savesuffix, save_chi_RCR_bad)
            np.save(savename + '_Azi' + savesuffix, save_azi)
            np.save(savename + '_Zen' + savesuffix, save_zen)
            ic(f'Saved {savename} to {save_folder}')
            part += 1

            save_traces = np.zeros((max_events, 4, 256))
            save_times = np.zeros((max_events, 1))
            save_snr = np.zeros((max_events, 1))
            save_chi_2016 = np.zeros((max_events, 1))
            save_chi_RCR = np.zeros((max_events, 1))
            save_chi_RCR_bad = np.zeros((max_events, 1))
            save_azi = np.zeros((max_events, 1))
            save_zen = np.zeros((max_events, 1))

        i = i - max_events * part

        save_times[i] = stationtime

        traces = []
        for chId, channel in enumerate(station.iter_channels(use_channels=save_channels)):
            # Get the traces from the channel
            trace = channel.get_trace()
            save_traces[i][chId] = trace
            traces.append(trace)

        # Calculate the SNR from the traces
        SNR = calcSNR(traces, Vrms)
        save_snr[i] = SNR

        save_chi_2016[i] = getMaxAllChi(traces, 2*units.GHz, templates_2016, 2*units.GHz)
        save_chi_RCR[i] = getMaxAllChi(traces, 2*units.GHz, use_templates, 2*units.GHz)
        save_chi_RCR_bad[i] = getMaxAllChi(traces, 2*units.GHz, use_templates_bad, 2*units.GHz)

        # Calculate the azimuth and zenith angles from the correlation fitter only for high Chi events
        if save_chi_2016[i] > 0.65 or save_chi_RCR[i] > 0.65:
            correlationDirectionFitter.run(evt, station, det, n_index=1.35)

            save_azi[i] = station.get_parameter(stnp.azimuth)   # Saved in radians
            save_zen[i] = station.get_parameter(stnp.zenith)

    # Remove empty spots from arrays
    save_traces = save_traces[:i]
    save_times = save_times[:i]
    save_snr = save_snr[:i]
    save_chi_2016 = save_chi_2016[:i]
    save_chi_RCR = save_chi_RCR[:i]
    save_chi_RCR_bad = save_chi_RCR_bad[:i]
    save_azi = save_azi[:i]
    save_zen = save_zen[:i]
    # Save the last part
    savename = f'{save_folder}/{prefix}_Station{station_id}'
    savesuffix = f'_Part{part}.npy'
    np.save(savename + '_Traces' + savesuffix, save_traces)
    np.save(savename + '_Times' + savesuffix, save_times)
    np.save(savename + '_SNR' + savesuffix, save_snr)
    np.save(savename + '_Chi2016' + savesuffix, save_chi_2016)
    np.save(savename + '_ChiRCR' + savesuffix, save_chi_RCR)
    np.save(savename + '_ChiBad' + savesuffix, save_chi_RCR_bad)
    np.save(savename + '_Azi' + savesuffix, save_azi)
    np.save(savename + '_Zen' + savesuffix, save_zen)
    print(f'Saved {savename} to {save_folder}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert HRA Nur files to numpy files')
    parser.add_argument('stnID', type=int)
    parser.add_argument('date', type=str)

    args = parser.parse_args()
    station_id = args.stnID
    date = args.date

    save_channels = [0, 1, 2, 3]
    save_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'
    os.makedirs(save_folder, exist_ok=True)

    nurFiles = []
    HRAdataPath = f"/dfs8/sbarwick_lab/ariannaproject/station_nur/station_{station_id}/"
    for file in os.listdir(HRAdataPath):
        if file.endswith('_statDatPak.root.nur'):
            continue
        else:
            nurFiles.append(HRAdataPath + file)

    convertHRANurToNpy(nurFiles, save_channels, save_folder, station_id, date)