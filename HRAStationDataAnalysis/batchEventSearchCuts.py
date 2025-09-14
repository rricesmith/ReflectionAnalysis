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
    date_processing = config['PARAMETERS']['date_processing']


    stations = [13, 14, 15, 17, 18, 19, 30]

    for station in stations:
        cmd = f'python HRAStationDataAnalysis/C00_eventSearchCuts.py --stnID {station} --date {date} --date_processing {date_processing}'
        A00_SlurmUtil.makeAndRunJob(cmd, f'{station}_eventSearchCuts', runDirectory='run/eventSearchCuts', partition='standard', n_cpus=2, n_tasks=3)

    # Also run without a station ID to process time overlaps
    quit()

    cmd = f'python HRAStationDataAnalysis/C00_eventSearchCuts.py --date {date} --date_processing {date_processing}'
    A00_SlurmUtil.makeAndRunJob(cmd, 'ALL_eventSearchCuts', runDirectory='run/ALL_eventSearchCuts', partition='standard', n_cpus=2, n_tasks=3)
