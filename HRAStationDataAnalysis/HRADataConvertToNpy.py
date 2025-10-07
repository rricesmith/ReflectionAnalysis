import datetime
import numpy as np
from NuRadioReco.modules.io import NuRadioRecoio
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.detector import detector
from NuRadioReco.utilities import units
import argparse
import os
import json
import DeepLearning.D00_helperFunctions as D00_helperFunctions
from HRAStationDataAnalysis.calculateChi import getMaxChi, getMaxAllChi
from icecream import ic
from HRAStationDataAnalysis.batchHRADataConversion import loadStationNurFiles

def inBlackoutTime(time, blackoutTimes):
    #This check removes data that have bad datetime format. No events should be recorded before 2013 season when the first stations were installed
    if datetime.datetime.fromtimestamp(time) > datetime.datetime(2019, 3, 31):
        return True
    #This check removes events happening during periods of high noise
    for blackouts in blackoutTimes:
        if blackouts[0] < time and time < blackouts[1]:
            return True
    return False

def calcSNR(traces, Vrms):
    # Calculate SNR from the average of two highest traces

    SNRs = []
    for trace in traces:
        p2p = np.max(trace) - np.min(trace)
        SNRs.append(p2p/(2*Vrms))

    SNRs.sort(reverse=True)
    if len(traces) > 1:
        SNR = (SNRs[0] + SNRs[1]) / 2
    else:
        SNR = SNRs[0]
    return SNR

def getVrms(station_id):
    # This loads the Vrms from the file

    with open(f'HRAStationDataAnalysis/vrms_per_station.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(f'{station_id}:'):
                Vrms = float(line.split(':')[1].strip())
                ic(f'Vrms for station {station_id} is {Vrms}')
                return float(Vrms) * units.V

    ic(f'Vrms for station {station_id} not found')
    return None

def getBlackoutTimes():
    blackoutFile = open('DeepLearning/BlackoutCuts.json')
    blackoutData = json.load(blackoutFile)
    blackoutFile.close()

    blackoutTimes = []

    for iB, tStart in enumerate(blackoutData['BlackoutCutStarts']):
        tEnd = blackoutData['BlackoutCutEnds'][iB]
        blackoutTimes.append([tStart, tEnd])
    return blackoutTimes

def convertHRANurToNpy(nurFiles, save_channels, save_folder, station_id, prefix, file_id=0):
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
    save_times = np.zeros((max_events))
    save_snr = np.zeros((max_events))
    save_chi_2016 = np.zeros((max_events))
    save_chi_RCR = np.zeros((max_events))
    save_chi_RCR_bad = np.zeros((max_events))
    # No calculation for Chi, as that is done in separate script depending upon templates desired
    save_azi = np.zeros((max_events))
    save_zen = np.zeros((max_events))
    save_eventIDs = np.zeros((max_events))


    file_reader = NuRadioRecoio.NuRadioRecoio(nurFiles)

    det = detector.Detector(f"HRASimulation/HRAStationLayoutForCoREAS.json")

    # Initialize bandpass filter module
    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
    channelBandPassFilter.begin()
    
    correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
    correlationDirectionFitter.begin(debug=False)

    templates_2016 = D00_helperFunctions.loadMultipleTemplates(100, date='2016')    # selection of 2016 events that are presumed to all be backlobes
    template_series_100 = D00_helperFunctions.loadMultipleTemplates(100)                         # selection of 'good' RCR simulated events for templates
    template_series_bad_100 = D00_helperFunctions.loadMultipleTemplates(100, bad=True)           # selection of 'bad' RCR simulated events for templates
    template_series_200 = D00_helperFunctions.loadMultipleTemplates(200)                         # selection of 'good' RCR simulated events for templates
    template_series_bad_200 = D00_helperFunctions.loadMultipleTemplates(200, bad=True)           # selection of 'bad' RCR simulated events for templates    
    stations_100s = [13, 15, 18, 32]
    stations_200s = [14, 17, 19, 30]

    blackoutTimes = getBlackoutTimes()

    # Get Vrm from forced triggers
    # ic('calculating Vrms')
    # Vrms = getVrms(nurFiles, save_channels, station_id, det, blackoutTimes)
    # ic(f'normalizing to {Vrms} vrms')
    Vrms = getVrms(station_id)
    if Vrms is None:
        ic('Vrms not found, exiting')
        quit()

    n_event = 0
    for i, evt in enumerate(file_reader.get_events()):
        n_event += 1
        
        station = evt.get_station(station_id)
        station_id = station.get_id()
        stationtime = station.get_station_time().unix
        det.update(station.get_station_time())
        
        # Apply bandpass filter (50 MHz to 1000 MHz) to all channels
        channelBandPassFilter.run(evt, station, det, passband=[50 * units.MHz, 1000 * units.MHz], filter_type='butter', order=5)

        if station_id in stations_100s:
            use_templates = template_series_100
            use_templates_bad = template_series_bad_100
        elif station_id in stations_200s:
            use_templates = template_series_200
            use_templates_bad = template_series_bad_200
        else:
            ic(f'{station_id} not in {stations_100s} or {stations_200s}')
            quit()

        # Untriggered events are forced triggers, and should be ignored. They are only used for calculating Vrm of the station
        if not station.has_triggered():
            continue

        # Check if the event is in the blackout times previously found in earlier analysis, which mostly correspond to periods of high noise
        if inBlackoutTime(stationtime, blackoutTimes):
            continue

        if i % 1000 == 0:
            ic(f'{i} events processed...')

        # Save if limit reached
        if n_event >= max_events:
            # Remove empty spots from arrays that were skipped due to blackout times or no trigger
            mask_empty = save_times == 0

            savename = f'{save_folder}/{prefix}_Station{station_id}'
            savesuffix = f'_fileID{file_id}_{(~mask_empty).sum()}evts_Part{part}.npy'

            np.save(savename + '_Traces' + savesuffix, save_traces[~mask_empty])
            np.save(savename + '_Times' + savesuffix, save_times[~mask_empty])
            np.save(savename + '_SNR' + savesuffix, save_snr[~mask_empty])
            np.save(savename + '_Chi2016' + savesuffix, save_chi_2016[~mask_empty])
            np.save(savename + '_ChiRCR' + savesuffix, save_chi_RCR[~mask_empty])
            np.save(savename + '_ChiBad' + savesuffix, save_chi_RCR_bad[~mask_empty])
            np.save(savename + '_Azi' + savesuffix, save_azi[~mask_empty])
            np.save(savename + '_Zen' + savesuffix, save_zen[~mask_empty])
            np.save(savename + '_EventIDs' + savesuffix, save_eventIDs[~mask_empty])
            ic(f'Saved {savename} to {save_folder}')
            part += 1

            save_traces = np.zeros((max_events, 4, 256))
            save_times = np.zeros((max_events))
            save_snr = np.zeros((max_events))
            save_chi_2016 = np.zeros((max_events))
            save_chi_RCR = np.zeros((max_events))
            save_chi_RCR_bad = np.zeros((max_events))
            save_azi = np.zeros((max_events))
            save_zen = np.zeros((max_events))

            n_event = 0


        save_times[n_event] = stationtime

        traces = []
        for chId, channel in enumerate(station.iter_channels(use_channels=save_channels)):
            # Get the traces from the channel
            trace = channel.get_trace()
            save_traces[n_event][chId] = trace
            traces.append(trace)

        # Calculate the SNR from the traces
        SNR = calcSNR(traces, Vrms)
        save_snr[n_event] = SNR
        save_eventIDs[n_event] = evt.get_id()
        save_chi_2016[n_event] = getMaxAllChi(traces, 2*units.GHz, templates_2016, 2*units.GHz)
        save_chi_RCR[n_event] = getMaxAllChi(traces, 2*units.GHz, use_templates, 2*units.GHz)
        save_chi_RCR_bad[n_event] = getMaxAllChi(traces, 2*units.GHz, use_templates_bad, 2*units.GHz)

        # Calculate the azimuth and zenith angles from the correlation fitter only for high Chi events
        # if save_chi_2016[n_event] > 0.65 or save_chi_RCR[n_event] > 0.65:
        if False: # Skipping for now to speedup, can do calculations later if needed
            # Some events have bad datetimes, so check if the time is valid
            # Change the date to one that falls within HRA livetime if it isn't, but the bad datetime is saved still so data can be culled later
            try:
                correlationDirectionFitter.run(evt, station, det, n_index=1.35)
            except LookupError:
                ic(f'LookupError for event {i}, station {station_id}, time {stationtime}')
                det.update(datetime.datetime(2018, 12, 31))
                correlationDirectionFitter.run(evt, station, det, n_index=1.35)


            save_azi[n_event] = station.get_parameter(stnp.azimuth)   # Saved in radians
            save_zen[n_event] = station.get_parameter(stnp.zenith)

    # Remove empty spots from arrays
    save_traces = save_traces[:n_event]
    save_times = save_times[:n_event]
    save_snr = save_snr[:n_event]
    save_chi_2016 = save_chi_2016[:n_event]
    save_chi_RCR = save_chi_RCR[:n_event]
    save_chi_RCR_bad = save_chi_RCR_bad[:n_event]
    save_azi = save_azi[:n_event]
    save_zen = save_zen[:n_event]
    save_eventIDs = save_eventIDs[:n_event]
    # Remove empty spots from arrays that were skipped due to blackout times or no trigger
    mask_empty = save_times == 0
    # Save the last part
    savename = f'{save_folder}/{prefix}_Station{station_id}'
    savesuffix = f'_fileID{file_id}_{(~mask_empty).sum()}evts_Part{part}.npy'
    np.save(savename + '_Traces' + savesuffix, save_traces[~mask_empty])
    np.save(savename + '_Times' + savesuffix, save_times[~mask_empty])
    np.save(savename + '_SNR' + savesuffix, save_snr[~mask_empty])
    np.save(savename + '_Chi2016' + savesuffix, save_chi_2016[~mask_empty])
    np.save(savename + '_ChiRCR' + savesuffix, save_chi_RCR[~mask_empty])
    np.save(savename + '_ChiBad' + savesuffix, save_chi_RCR_bad[~mask_empty])
    np.save(savename + '_Azi' + savesuffix, save_azi[~mask_empty])
    np.save(savename + '_Zen' + savesuffix, save_zen[~mask_empty])
    np.save(savename + '_EventIDs' + savesuffix, save_eventIDs[~mask_empty])
    print(f'Saved {savename} to {save_folder}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert HRA Nur files to numpy files')
    parser.add_argument('stnID', type=int)
    parser.add_argument('date', type=str)
    parser.add_argument('--start_file', type=int, default=0, help='Start file number for processing')
    parser.add_argument('--end_file', type=int, default=0, help='End file number for processing')

    args = parser.parse_args()
    station_id = args.stnID
    date = args.date
    start_file = args.start_file
    end_file = args.end_file

    save_channels = [0, 1, 2, 3]
    save_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'
    os.makedirs(save_folder, exist_ok=True)

    nurFiles = loadStationNurFiles(station_id)

    ic(len(nurFiles), start_file, end_file, nurFiles[start_file:end_file])
    if not end_file == 0:
        nurFiles = nurFiles[start_file:end_file]
    elif end_file == 0 and start_file > 0:
        nurFiles = nurFiles[start_file:]

    ic(f'Files {nurFiles}')
    convertHRANurToNpy(nurFiles, save_channels, save_folder, station_id, date, file_id=start_file)