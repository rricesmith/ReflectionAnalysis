import A00_SlurmUtil
import numpy as np
from pathlib import Path


date_sim = '10.24.24'
n_cores = 100   #Decenty sensitivity to RCRs, so don't overdo
distance = 10 #km, want a large enough area that all stations will be covered and then some
add_noise = False
output_folder = f'SimpleFootprintSimulation/output/HRA/{date_sim}/'
output_filename = f'HRA_Noise{add_noise}_{distance}km'

# Make directory if it doesn't exist
Path(output_folder).mkdir(parents=True, exist_ok=True)

min_file = 0
max_file = 1000     #For MB up to 4000, 1000 is reduced/broad for MB
num_sims = 20       # How many simulations to break up into

file_range = np.linspace(min_file, max_file, num_sims)


# We will use the lower file number as a seed so that there is variation among the simulated events
for iF in range(len(file_range)-1):
    lower_file = file_range[iF]
    upper_file = file_range[iF+1]
    cmd = f'python SimpleFootprintSimulation/HRASimulation.py {output_folder}{output_filename}_files{lower_file:.0f}-{upper_file:.0f}_{n_cores}cores {n_cores} --min_file {lower_file:.0f} --max_file {upper_file:.0f} --add_noise {add_noise} --distance {distance} --seed {int(lower_file)}'

    A00_SlurmUtil.makeAndRunJob(cmd, f'HRA_{lower_file:.0f}-{upper_file:.0f}', runDirectory='run/SimpleFootprintSimulation', partition='standard')
