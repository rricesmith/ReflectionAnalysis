import configparser
import templateCrossCorr as txc
import numpy as np
from NuRadioReco.utilities import units
from icecream import ic


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


    save_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'


