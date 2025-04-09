import A00_SlurmUtil
import numpy as np
from pathlib import Path
import configparser
import os
from icecream import ic


def loadStationNurFiles(station_id):
    # Load the station nur files
    nurFiles = []
    HRAdataPath = f"/dfs8/sbarwick_lab/ariannaproject/station_nur/station_{station_id}/"
    for file in os.listdir(HRAdataPath):
        if file.endswith('_statDatPak.root.nur'):
            continue
        else:
            nurFiles.append(HRAdataPath + file)

    return nurFiles


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('HRAStationDataAnalysis/config.ini')
    date = config['PARAMETERS']['date']


    # stations = [13, 14, 15, 17, 18, 19, 30]
    stations = [13]

    n_slurm_jobs = 100

    for station_id in stations:

        nurFiles = loadStationNurFiles(station_id)

        
        n_files = len(nurFiles)
        n_files_per_job = n_files // n_slurm_jobs
        if n_slurm_jobs > n_files:
            n_slurm_jobs = n_files
        if n_files_per_job < 1:
            n_files_per_job = 1
        ic(n_files, n_files_per_job)
        for i in range(n_slurm_jobs):
            if i == n_slurm_jobs - 1:
                cmd = f'python HRAStationDataAnalysis/HRADataConvertToNpy.py {station_id} {date} --start_file {i*n_files_per_job} --end_file {0}'
            else:
                cmd = f'python HRAStationDataAnalysis/HRADataConvertToNpy.py {station_id} {date} --start_file {i*n_files_per_job} --end_file {(i+1)*n_files_per_job}'

            A00_SlurmUtil.makeAndRunJob(cmd, f'{station_id}_{i}_HRA', runDirectory='run/HRAData', partition='standard')
