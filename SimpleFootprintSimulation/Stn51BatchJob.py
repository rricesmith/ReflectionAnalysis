import A00_SlurmUtil
import numpy as np



n_cores = 100
num_icetop = 30
amp = True
add_noise = True
output_folder = 'SimpleFootprintSimulation/output/4.17.24/'
output_filename = 'Stn51_IceTop'

min_energy = 16.0
max_energy = 18.6

e_range = np.arange(min_energy, max_energy, 0.1)


for e in e_range:
    cmd = f'python SimpleFootprintSimulation/Stn51Simulation.py {output_folder}{output_filename}_{e:.1f}-{e+0.1:.1f}eV_{n_cores}cores.nur {n_cores} --min_energy {e:.1f} --max_energy {e+0.1:.1f} --num_icetop {num_icetop} --sim_amp {amp} --add_noise {add_noise}'

    A00_SlurmUtil.makeAndRunJob(cmd, f'Stn51_{e:.1f}to{e+0.1:.1f}', runDirectory='run/SimpleFootprintSimulation', partition='standard')