import A00_SlurmUtil
import numpy as np
from pathlib import Path
import configparser
import os
from icecream import ic

config = configparser.ConfigParser()
config.read('HRAStationDataAnalysis/config.ini')
date = config['PARAMETERS']['date']


# stations = [13, 14, 15, 17, 18, 19, 30]
stations = [13]

n_slurm_jobs = 20

for station_id in stations:

    HRAdataPath = f"/dfs8/sbarwick_lab/ariannaproject/station_nur/station_{station_id}/"
    nurFiles = []
    # for file in os.listdir(HRAdataPath):
    #     if file.endswith('_statDatPak.root.nur'):
    #         continue
    #     else:
    #         nurFiles.append(HRAdataPath + file)
    nurFiles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    n_files = len(nurFiles)
    n_files_per_job = n_files // n_slurm_jobs
    for i in range(n_slurm_jobs):
        if i == n_slurm_jobs - 1:
            cmd = f'python HRAStationDataAnalysis/HRADataConvertToNpy.py {station_id} {date} {i*n_files_per_job} {0}'
        else:
            cmd = f'python HRAStationDataAnalysis/HRADataConvertToNpy.py {station_id} {date} {i*n_files_per_job} {(i+1)*n_files_per_job}'

        A00_SlurmUtil.makeAndRunJob(cmd, f'HRAData_{station_id}', runDirectory='run/HRAData', partition='standard')
