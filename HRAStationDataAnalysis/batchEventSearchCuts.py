import A00_SlurmUtil
import numpy as np
from pathlib import Path
import configparser
import os
from icecream import ic


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('HRAStationDataAnalysis/config.ini')
    date = config['PARAMETERS']['date']


    stations = [13, 14, 15, 17, 18, 19, 30]

    for station in stations:
        cmd = f'python HRAStationDataAnalysis/eventSearchCuts.py {station} {date}'
        A00_SlurmUtil.makeAndRunJob(cmd, f'{station}_eventSearchCuts', runDirectory='run/eventSearchCuts', partition='standard', n_cpus=2)

