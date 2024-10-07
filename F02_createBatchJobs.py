import glob
import os
import argparse
import numpy as np
import subprocess
from NuRadioReco.utilities import units

partition = 'standard'

# CoREAS_mode = 'direct'
CoREAS_mode = 'refracted'
#spacing = 5	#diameter of circle to throw over
spacing = 2
depth = 300
dB = 40
#ncores = 10
ncores = 1000
# noise = False
noise = True

#type = 'MB'
type = 'IceTop'
#type = 'SP'
#type = 'GL'

#config = 'SP'
config = 'MB_old'
#config = 'MB_future'
#config = 'GL'
#config = 'TriggerTest'
#config = 'BacklobeTest'
# config = 'Stn51'

#100 - RCR only and some upward current stations, 200 - some upward current stations, 300 - SP current
#All Future set amp False type future
#amp = False
amp = True
#amp_type = 'future'
#amp_type = 'TriggerTest'
#amp_type = '100'
amp_type = '200'
# amp_type = '300'
#amp_type = None

antenna = 'lpda'
#antenna = 'dipole'
#antenna = 'PA'

if type == 'SP':
    lowid = 0
    highid = 9
#    lowid = 1000
#    highid = 1049
#    highid = 1099
    limit = 2199	#max SP file
elif type == 'IceTop':
    lowid = 16.0
    highid = 16.1
    limit = 18.7    
    numIceTop = 30  #Max is ~34, often use 10 for lower statistics
    iceTopSin = 0
#-1 makes a file with all sin bins
#    iceTopSin = -1

elif type == 'MB':
    depth = 576
    dB = 0

    lowid = 0
    highid = 4
#    highid = 49
#    highid = 99
    limit = 1000	#limited MB footprint pool, covers all needs, Anna's footprint selection
#    limit = 4000	#max safe MB file

elif type == 'GL':
    depth = 300
    dB = 0

    lowid = 0
    highid = 9
    limit = 601


iter = (highid - lowid) + 1
if type == 'IceTop':
    iter = highid - lowid


while highid < limit:

    if type == 'IceTop':
        lowid = round(lowid, 1)
        highid = round(highid, 1)


    #specify a working directory for this specific simulation run
    working_dir = os.path.join(f"run/FootprintDirRefl_{depth}m/")


    filename = f'RefractedFootprint_{type}_Layer{depth}m_{dB}dB_Area{spacing}km_{ncores}cores_{lowid}lowid'

    cmd = f'python FootprintAnalysis/F01_FootprintSimulation.py {spacing} {ncores} {CoREAS_mode} --dB {dB} --depthLayer {depth} --type {type} --min_file {lowid} --max_file {highid} --config {config} --antenna {antenna}'
    if not noise:
        cmd += ' --no_noise'
    if amp:
        cmd += ' --add_amp'
    if not amp_type == None:
        cmd += f' --amp_type {amp_type}'
    if type == 'IceTop':
        cmd += f' --num_icetop {numIceTop:.0f} --icetop_sin {iceTopSin:.1f}'

    #now adding settings for slurm scheduler
    header = "#!/bin/bash\n"
    header += f"#SBATCH --job-name={lowid}_{type}      ##Name of the job.\n"
    header += "#SBATCH -A sbarwick_lab                  ##Account to charge to\n"
    header += f"#SBATCH -p {partition}                          ##Partition/queue name\n"
#    header += "#SBATCH -p free                          ##Partition/queue name\n"

    header += "#SBATCH --time=3-00:00:00                ##Max runtime D-HH:MM:SS, 3 days free maximum\n"
    header += "#SBATCH --nodes=1                        ##Nodes to be used\n"
    header += "#SBATCH --ntasks=1                       ##Numer of processes to be launched\n"
    header += "#SBATCH --cpus-per-task=1                ##Cpu's to be used\n"
    header += "#SBATCH --mem-per-cpu=6G"		##6GB memory per job
    header += "#SBATCH --output={}\n".format(os.path.join('run', 'logs', f'refl_{filename}.out'))
    header += "#SBATCH --error={}\n".format(os.path.join('run', 'logs', f'refl_{filename}.err'))
    header += "#SBATCH --mail-type=fail\n"
    header += "#SBATCH --mail-user=rricesmi@uci.edu\n"

    #Add software to the python path
    header += "export PYTHONPATH=$NuM:$PYTHONPATH\n"
    header += "export PYTHONPATH=$Nu:$PYTHONPATH\n"
    header += "export PYTHONPATH=$Radio:$PYTHONPATH\n"
 
    header += "module load python/3.8.0\n"
    header += "cd $ReflectiveAnalysis\n"

    slurm_name = os.path.join(f'run/FootprintDirRefl_{depth}m', os.path.basename(filename)) + ".sh"
#    with open(os.path.join('run/SPCRFootprints/', os.path.basename(filename) + ".sh"), 'w') as fout:
    with open(slurm_name, 'w') as fout:
        fout.write(header)
        fout.write(cmd)
    fout.close()


    slurm_name = 'sbatch ' + slurm_name
    print(f'cmd running {cmd}')
    print(f'running {slurm_name}')
    process = subprocess.Popen(slurm_name.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    if type != 'IceTop':
        lowid += iter
        highid += iter
    else:
        if iceTopSin == -1:
            lowid += iter
            highid += iter
        else:
            if iceTopSin >= 1:
                iceTopSin = 0
                lowid += iter
                highid += iter
            else:
                iceTopSin += 0.1

