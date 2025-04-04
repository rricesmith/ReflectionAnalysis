import A00_SlurmUtil
import numpy as np
from pathlib import Path
import configparser



config = configparser.ConfigParser()
config.read('HRAStationDataAnalysis/config.ini')
date = config['PARAMETERS']['date']


# stations = [13, 14, 15, 17, 18, 19, 30]
stations = [13]

for station in stations:

    cmd = f'python HRAStationDataAnalysis/HRADataConvertToNpy.py {station} {date}'

    A00_SlurmUtil.makeAndRunJob(cmd, f'HRAData_{station}', runDirectory='run/HRAData', partition='standard')
