import A00_SlurmUtil
import numpy as np
from pathlib import Path


date = '3.28.25'
n_cores = 5000   #Decenty sensitivity to RCRs, so don't overdo
loc = 'MB'  #Or SP, not setup yet though
# Need to change default distance and layer depth for SP, but for MB its fine right now
if loc == 'MB':
    distance = 5 #km
    depthLayer = 576 #m
    dB = 0 #Assume perfect reflector, although realistically it isn't quite that for MB
amp = True
amp_type = 200
add_noise = False
output_folder = f'SimpleFootprintSimulation/output/RCR/{date}/{amp_type}s/'
output_filename = f'RCR_{loc}_{depthLayer}m_{dB}dB_{amp_type}s_Noise{add_noise}'

# Make directory if it doesn't exist
Path(output_folder).mkdir(parents=True, exist_ok=True)

min_file = 0
max_file = 1000     #For MB up to 4000, 1000 is reduced/broad for MB. For SP use IceTop (needs to be added)
num_sims = 100

file_range = np.linspace(min_file, max_file, num_sims)


for iF in range(len(file_range)-1):
    lower_file = file_range[iF]
    upper_file = file_range[iF+1]
    cmd = f'python SimpleFootprintSimulation/ReflectedSimulation.py {output_folder}{output_filename}_files{lower_file:.0f}-{upper_file:.0f}_{n_cores}cores.nur {n_cores} --loc {loc} --min_file {lower_file:.0f} --max_file {upper_file:.0f} --sim_amp {amp} --amp_type {amp_type} --add_noise {add_noise} --distance {distance} --depthLayer {depthLayer} --dB {dB}'

    A00_SlurmUtil.makeAndRunJob(cmd, f'RCR_{lower_file:.0f}-{upper_file:.0f}', runDirectory='run/SimpleFootprintSimulation', partition='standard')
