import glob
import os
import argparse
import numpy as np
import subprocess
from NuRadioReco.utilities import units
from os.path import exists


def makeAndRunFile(loc, energy, n_nu, part=0 ,parts=False, part_max=9):

    working_dir = os.path.join(f"NeutrinoAnalysis/run/")
    filename = f'RunSim_{loc}_E{energy}_N{n_nu}_part{part}'


    amp = '300'
    # amp = 'r6km'
    save_prefix = "MJob"
    extra_stat = 'EventForAndrew'

    cmd = f'python NeutrinoAnalysis/M02_RunSimulation{amp}s.py '
    # cmd = f'python NeutrinoAnalysis/M02_CombinedSim.py '
#    neutrino_file = f'NeutrinoAnalysis/GeneratedEvents/AddedStats_{loc}_{energy:.4e}_n{n_nu:.4e}.hdf5'
    neutrino_file = f'NeutrinoAnalysis/GeneratedEvents/{save_prefix}_{loc}_{energy:.4e}_n{n_nu:.4e}.hdf5'
    cmd += neutrino_file
    if parts == True:
        cmd += f'.part{part:04d}'
    # cmd += f' NeutrinoAnalysis/station_configs/gen2_{loc}_infirn.json '
    cmd += 'configurations/station61.json'
    cmd += f'NeutrinoAnalysis/{loc}_config.yaml '

    #Output, can add prefix here
#    folder = "GL_Higher_Sampling/"
#    folder = "GL_Uniform_with_phasing/"
#    folder = "GL_Bugfig_Orig/"
#    folder = "GL_Bugfig_Dephased/"
    # folder = "AttTest_GL3/"
    folder = f'MJob/{amp}/{loc}/'
    cmd += f'NeutrinoAnalysis/output/{folder}{save_prefix}_{loc}_Allsigma_{energy}_n{n_nu}_{extra_stat}'
    if parts == True:
        cmd += f'_part{part:04d}'
    cmd += f'.hdf5 '
    cmd += f'NeutrinoAnalysis/output/{folder}{save_prefix}_{loc}_Allsigma_{energy}_n{n_nu}_{extra_stat}'
    if parts == True:
        cmd += f'_part{part:04d}'
    cmd += '.nur '
#        cmd += f'--part 0009
    #now adding settings for slurm scheduler
    header = "#!/bin/bash\n"
    header += f"#SBATCH --job-name={amp}{loc}{energy}_{part}      ##Name of the job.\n"
    header += "#SBATCH -A sbarwick_lab                  ##Account to charge to\n"
    header += "#SBATCH -p standard                          ##Partition/queue name\n"
#    header += "#SBATCH -p free                          ##Partition/queue name\n"
    header += "#SBATCH --time=3-00:00:00                ##Max runtime D-HH:MM:SS, 3 days free maximum\n"
    header += "#SBATCH --nodes=1                        ##Nodes to be used\n"
    header += "#SBATCH --ntasks=1                       ##Numer of processes to be launched\n"
    header += "#SBATCH --cpus-per-task=1                ##Cpu's to be used\n"
    header += "#SBATCH --mem-per-cpu=6G"		##6GB memory per job
    header += "#SBATCH --output={}\n".format(os.path.join(working_dir, 'logs', f'{filename}.out'))
    header += "#SBATCH --error={}\n".format(os.path.join(working_dir, 'logs', f'{filename}.err'))
    header += "#SBATCH --mail-type=fail\n"
    header += "#SBATCH --mail-user=rricesmi@uci.edu\n"

    #Add software to the python path
    header += "export PYTHONPATH=$NuM:$PYTHONPATH\n"
    header += "export PYTHONPATH=$Nu:$PYTHONPATH\n"
    header += "export PYTHONPATH=$Radio:$PYTHONPATH\n"
 
    header += "module load python/3.8.0\n"
    header += "cd $ReflectiveAnalysis\n"

    slurm_name = os.path.join(working_dir, os.path.basename(filename)) + ".sh"
#    with open(os.path.join('run/SPCRFootprints/', os.path.basename(filename) + ".sh"), 'w') as fout:
    with open(slurm_name, 'w') as fout:
        fout.write(header)
        fout.write(cmd)
    fout.close()


    slurm_name = 'sbatch ' + slurm_name
    print(f'cmd running {cmd}')
    print(f'running {slurm_name}, error ' + os.path.join(working_dir, 'logs', f'{filename}.err'))
    process = subprocess.Popen(slurm_name.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


    if parts==True and part < part_max:
        new_part = part + 1
        print(f'checking if {neutrino_file}.part{new_part:04d} exists')
        if exists( neutrino_file + f'.part{new_part:04d}'):
            print(f'it exists, iterating...')
            makeAndRunFile(loc, energy, n_nu, part=new_part, parts=parts, part_max=part_max)



# loc = ['MB', 'SP']
loc = ['SP']
#energy_number = [['3e16', '1e6'], ['5e16', '1e6'], ['1e17', '1e5']]
#energy_number = [['1e17', '1e6'], ['3e16', '1e6']]
#energy_number = [['1e16', '1e7'], ['3e17', '1e6'], ['1e17', '5e6']]
#energy_number = [['1e17', '5e6']]

parts = False # Break up the jobs into parts if True
part_max = 299
part = 0

# n_bins = 12
n_bins = 20
energies = np.logspace(17, 20, num=n_bins)

for l in loc:
    for energy in energies:
        # if energy < 5*1e17:
        #     num = 1e6
        # if energy < 1e18:
        #     num = 5e6
        # else:
        #     continue
        # elif energy > 1e19:
        #     num = 1e4
        #     parts=False
        # else:
        #     num = 1e5
        if not energy > 1e19:
            continue
        num = 1e3
        makeAndRunFile(l, energy, num, part=0, parts=parts, part_max=part_max)


#for energy, n_nu in energy_number:

#    makeAndRunFile(loc, energy, n_nu, part=part, parts=parts, part_max=part_max)

