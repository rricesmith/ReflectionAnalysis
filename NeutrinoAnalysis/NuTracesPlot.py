from NuRadioReco.utilities.io_utilities import read_pickle
import numpy as np
from icecream import ic


template_nu = 'StationDataAnalysis/templates/NUdowntemplate_NoFilter_200series_SST.pkl'
templates_nu = read_pickle(template_nu)
ic(templates_nu)
for key in templates_nu:
    temp = templates_nu[key]
templates_nu = temp

ic(templates_nu)

