import A00_SlurmUtil
import numpy as np
from pathlib import Path


n_cores = 10 #1000
num_icetop = 10 #30
amp = True
add_noise = False
output_folder = 'SimpleFootprintSimulation/output/7.13.25/'
output_filename = 'Stn51_IceTop'

# Make directory if it doesn't exist
Path(output_folder).mkdir(parents=True, exist_ok=True)

min_energy = 16.0
max_energy = 18.6

e_range = np.arange(min_energy, max_energy, 0.1)
sin2Val = np.arange(0, 1.01, 0.1)


for e in e_range:
    for sin2 in sin2Val:
        # e = 18.4
        # sin2 = 0.0
        cmd = f'python SimpleFootprintSimulation/Stn51Simulation.py {output_folder}{output_filename}_{e:.1f}-{e+0.1:.1f}eV_{sin2:.1f}sin2_{n_cores}cores.nur {n_cores} --min_energy {e:.1f} --max_energy {e+0.1:.1f} --sin2 {sin2:.1f} --num_icetop {num_icetop} --sim_amp {amp} --add_noise {add_noise}'

        A00_SlurmUtil.makeAndRunJob(cmd, f'Stn51_{e:.1f}_{sin2:.1f}sin2', runDirectory='run/SimpleFootprintSimulation', partition='standard')
        # quit()
