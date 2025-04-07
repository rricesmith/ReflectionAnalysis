import os
import configparser
import templateCrossCorr as txc
import numpy as np
from NuRadioReco.utilities import units
from icecream import ic
import DeepLearning.D00_helperFunctions as D00_helperFunctions


def getMaxChi(traces, sampling_rate, template_trace, template_sampling_rate, parallelChannels=[[0, 2], [1, 3]], use_average=False):
    #Parallel channels should be index corresponding to the channel in traces

    maxCorr = []
    if use_average:
        for parChans in parallelChannels:
            parCorr = 0
            for chan in parChans:
                xCorr = txc.get_xcorr_for_channel(traces[chan], template_trace, sampling_rate, template_sampling_rate)
                parCorr += np.abs(xCorr)
            maxCorr.append(parCorr / len(parChans))
    else:
        for trace in traces:
            xCorr = txc.get_xcorr_for_channel(trace, template_trace, sampling_rate, template_sampling_rate)
            maxCorr.append(np.abs(xCorr))

    return max(maxCorr)

def getMaxAllChi(traces, sampling_rate, template_traces, template_sampling_rate, parallelChannels=[[0, 2], [1, 3]], exclude_match=None):

    maxCorr = []
    for key in template_traces:
        # ic(key, exclude_match, key == str(exclude_match))
        if key == str(exclude_match):
            continue
        trace = template_traces[key]
        maxCorr.append(getMaxChi(traces, sampling_rate, trace, template_sampling_rate, parallelChannels=parallelChannels))

    return max(maxCorr)






if __name__ == "__main__":


    config = configparser.ConfigParser()
    config.read('HRAStationDataAnalysis/config.ini')
    date = config['PARAMETERS']['date']


    data_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'

    stations_100s = [13, 15, 18, 32]
    stations_200s = [14, 17, 19, 30]
    stations = {100: stations_100s, 200: stations_200s}


    for series in stations.keys():
        templates_2016 = D00_helperFunctions.loadMultipleTemplates(series, date='2016')    # selection of 2016 events that are presumed to all be backlobes
        template_series = D00_helperFunctions.loadMultipleTemplates(series)                         # selection of 'good' RCR simulated events for templates
        template_series_bad = D00_helperFunctions.loadMultipleTemplates(series, bad=True)           # selection of 'bad' RCR simulated events for templates
        for station_id in stations[series]:
            for file in os.listdir(data_folder):
                if file.startswith(f'{date}_Station{station_id}_Traces'):
                    traces_array = np.load(data_folder+file, allow_pickle=True)

                    chi_2016 = np.zeros((len(traces_array)))
                    chi_RCR = np.zeros((len(traces_array)))
                    chi_RCR_bad = np.zeros((len(traces_array)))

                    for iT, traces in traces_array:

                        chi_2016[iT] = getMaxAllChi(traces, 2*units.GHz, templates_2016, 2*units.GHz)
                        chi_RCR[iT] = getMaxAllChi(traces, 2*units.GHz, template_series, 2*units.GHz)
                        chi_RCR_bad[iT] = getMaxAllChi(traces, 2*units.GHz, template_series_bad, 2*units.GHz)

                    # Save the chi values
                    np.save(data_folder+file.replace('Traces', 'Chi_2016'), chi_2016)
                    print(f'Saved {file.replace("Traces", "Chi_2016")}')
                    np.save(data_folder+file.replace('Traces', 'Chi_RCR'), chi_RCR)
                    print(f'Saved {file.replace("Traces", "Chi_RCR")}')
                    np.save(data_folder+file.replace('Traces', 'Chi_RCR_bad'), chi_RCR_bad)
                    print(f'Saved {file.replace("Traces", "Chi_RCR_bad")}')
