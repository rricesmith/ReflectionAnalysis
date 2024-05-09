import A00_SlurmUtil
import numpy as np
from pathlib import Path


n_cores = 1000 # Low sensitivity to BL, so 1k per file is fine
loc = 'MB'  #Or SP, not setup yet though
amp = True
amp_type = 100
add_noise = True
output_folder = f'SimpleFootprintSimulation/output/Backlobe/5.7.24/{amp_type}s/'
output_filename = f'Backlobe_{loc}'

# Make directory if it doesn't exist
Path(output_folder).mkdir(parents=True, exist_ok=True)

min_file = 0
max_file = 1000     #For MB up to 4000, 1000 is reduced/broad for MB. For SP use IceTop (needs to be added)
num_sims = 10

file_range = np.linspace(min_file, max_file, num_sims)


for iF in range(len(file_range)-1):
    lower_file = file_range[iF]
    upper_file = file_range[iF+1]
    cmd = f'python SimpleFootprintSimulation/BacklobeSimulation.py {output_folder}{output_filename}_files{lower_file:.0f}-{upper_file:.0f}_{n_cores}cores.nur {n_cores} --loc {loc} --min_file {lower_file:.0f} --max_file {upper_file:.0f} --sim_amp {amp} --amp_type {amp_type} --add_noise {add_noise}'

    A00_SlurmUtil.makeAndRunJob(cmd, f'BL_{lower_file:.0f}-{upper_file:.0f}', runDirectory='run/SimpleFootprintSimulation', partition='standard')