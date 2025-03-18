from NuRadioReco.utilities.io_utilities import read_pickle
import os
import numpy as np


def loadSingleTemplate(series):
    # Series should be 200 or 100
    # Loads the first version of a template made for an average energy/zenith
    templates_RCR = f'StationDataAnalysis/templates/reflectedCR_template_{series}series.pkl'
    templates_RCR = read_pickle(templates_RCR)
    for key in templates_RCR:
        temp = templates_RCR[key]
    templates_RCR = temp

    return templates_RCR

def loadMultipleTemplates(series, date='9.16.24', addSingle=True):
    # Dates - 9.16.24 (noise included), 10.1.24 (no noise)
    #       - 2016 : found backlobe events from 2016

    # 10.1.24 has issues with the templates, so use 9.16.24
    # Series should be 200 or 100
    # Loads all the templates made for an average energy/zenith
    if not date == '2016':
        template_series_RCR_location = f'DeepLearning/templates/RCR/{date}/' 
        template_series_RCR = []
        for filename in os.listdir(template_series_RCR_location):
            if filename.startswith(f'{series}s'):
                temp = np.load(os.path.join(template_series_RCR_location, filename))
                template_series_RCR.append(temp)
    else:
        templates_2016_location = f'StationDataAnalysis/templates/confirmed2016Templates/'
        template_series_RCR = []
        for filename in os.listdir(templates_2016_location):
            temp = np.load(os.path.join(templates_2016_location, filename))
            template_series_RCR.append(temp)

    if addSingle:
        template_series_RCR.append(loadSingleTemplate(series))

    return template_series_RCR