import A00_SlurmUtil
import numpy as np
from pathlib import Path
import configparser


config = configparser.ConfigParser()
config.read('HRASimulation/config.ini')

# Normal sim loc
sim_folder = config['FOLDERS']['sim_folder']
# Area sim loc
# sim_folder = config['FOLDERS']['area_sim_folder']


# n_cores = 1000   #Decenty sensitivity to RCRs, so don't overdo
n_cores = 10000   #Doing for area sensitivity, so can do more cores b/c only 1 footprint each
distance = 12 #km, diameter of throws. 5km has triggers at edges still, so can go farther out
add_noise = True
# date_sim = '2.25.25'
# output_folder = f'HRASimulation/output/HRA/{date_sim}/'
output_folder = sim_folder    # Saving to shared folder due to size of simulations
output_filename = f'HRA_Noise{add_noise}_{distance}km'

# Make directory if it doesn't exist
Path(output_folder).mkdir(parents=True, exist_ok=True)

min_file = 0
max_file = 1000     #For MB up to 4000, 1000 is reduced/broad for MB
num_sims = int(n_cores/2)   # How many simulations to break up into
if num_sims < 50:
    num_sims = 50
num_sims = 100 # Override for testing

file_range = np.linspace(min_file, max_file, num_sims)


# We will use the lower file number as a seed so that there is variation among the simulated events
for iF in range(len(file_range)-1):
    lower_file = file_range[iF]
    upper_file = file_range[iF+1]
    cmd = f'python HRASimulation/S01A_HRASim.py {output_folder}{output_filename}_files{lower_file:.0f}-{upper_file:.0f}_{n_cores}cores {n_cores} --min_file {lower_file:.0f} --max_file {upper_file:.0f} --add_noise {add_noise} --distance {distance} --seed {int(lower_file)}'
    # cmd = f'python HRASimulation/S01B_HRASimAreaSensitivity.py {output_folder}{output_filename}_files{lower_file:.0f}-{upper_file:.0f}_{n_cores}cores {n_cores} --min_file {lower_file:.0f} --max_file {upper_file:.0f} --add_noise {add_noise} --distance {distance} --seed {int(lower_file)}'

    A00_SlurmUtil.makeAndRunJob(cmd, f'HRA_{lower_file:.0f}-{upper_file:.0f}', runDirectory='run/HRASimulation', partition='standard')
