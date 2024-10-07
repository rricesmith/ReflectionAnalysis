import glob
import os
import argparse
import numpy as np
from NuRadioReco.utilities import units
import subprocess
from coreDataObjects import coreStatistics
from icecream import ic

type = 'SP'
skip_repeats = False
partition = 'standard'      #Alternative is free
dipoles = False
noise = True

if not dipoles:
#    identifier = 'Gen2_AllTriggers_DipFix'
#    detector_sim = 'CoreAnalysis/C01_simRefl.py'
    identifier = 'Gen2_2021'
    detector_sim = 'CoreAnalysis/C01_simRefl_Gen2Design.py'
else:
    identifier = 'DipoleTesting'
    detector_sim = 'CoreAnalysis/C01_simRefl_ManyDipoles.py'



if type == 'MB':
    config_file = 'configurations/reflectionConfigMoores.yaml'
    detector_filename = f'configurations/gen2_MB_CoreConfig.json'
elif type == 'SP':
#    config_file = 'configurations/reflectionConfigSouthPole.yaml'
    if not noise:
        config_file = 'configurations/gen2_2021_SP.yaml'
    elif noise:
        # config_file = 'configurations/gen2_2021_SP_noise.yaml'
        config_file = 'configurations/gen2_2021_SP_noise.25RMS.yaml'
    if not dipoles:
#        detector_filename = 'configurations/gen2_SP_CoreConfig.json'
        detector_filename = 'configurations/gen2_hybrid_2021.json'
    else:
        detector_filename = 'configurations/gen2_SP_CoreConfig_ManyDipoles.json'
elif type =='GL':
    print(f'GL not setup yet')
    quit()
else:
    print('wrong type')
    quit()


#Pairs of depth layer in m and area of cores thrown in km
#depth_area = [ [300, 1.7], [500, 1.7], [800, 1.7], [1000, 1.7], [1170, 1.7] ]
#depth_area = [ [300, 1.7], [500, 1.7], [800, 1.7] ]
depth_area = [ [300, 1.0] ]
# depth_area = [ [500, 1.7], [800, 1.7] ]
#depth_area = [ [576, 1.7] ]
#depth_area = [ [300, 1.7] ]


throws = 2000       
repeats = 50

#e_bins = np.arange(17, 20.01, 0.1)	#old number of bins before using R=1
#Using L after sim, L~10^-3 and lower. So the energy bins here correspond to approximately cores of 10^3 higher
# e_bins = np.arange(14, 17.5, 0.1)      #Normally use this
e_bins = np.array([14.7, 14.8, 14.9, 15.0, 15.1, 15.2])

refl_coef = 1

#Zenith bins to simulate
dCos = 0.05
coszen_bin_edges = np.arange(np.cos(np.deg2rad(60)), 1.01, dCos)
coszen_bin_edges = np.flip(coszen_bin_edges)
cos_edges = np.arccos(coszen_bin_edges)
cos_edges[np.isnan(cos_edges)] = 0

#Just do a single bin only
cos_edges = np.array([cos_edges[0], cos_edges[1]])

#for iS, space in enumerate(spacing):
for depth, space in depth_area:

    #specify a working directory for this specific simulation run
#    working_dir = os.path.join(f"run/{detector_label}/{config_name}/{detector_sim_label}/{depth:.0f}m_{dB:.0f}dB")
    working_dir = os.path.join(f"run/CoreRefl{space:.2f}km")

    #run and output directories are created automatically if not yet present
    if not os.path.exists(os.path.join(working_dir, "output")):
        os.makedirs(os.path.join(working_dir, "output"))
    if not os.path.exists(os.path.join(working_dir, "run", 'logs')):
        os.makedirs(os.path.join(working_dir, "run", 'logs'))

    print(f"config: {config_file}")
    print(f"detector description: {detector_filename}")
    print(f"detector simulation {detector_sim}")
    print(f"working dir {working_dir}")

    for iE in range(len(e_bins)-1):
        for iC in range(len(cos_edges)-1):
            for iR in range(repeats):
                Estring = f"{e_bins[iE]:.1f}log10eV"
                zString = f'coszen{coszen_bin_edges[iC]:.3f}'
    #            zString = f'coszen{coszen_bin_edges[iC]:.2f}'
                eLow = e_bins[iE]
                eHigh = e_bins[iE+1]
                cLow = cos_edges[iC] * units.rad
                cHigh = cos_edges[iC+1] * units.rad
                t1 = os.path.join(working_dir, Estring)
                if not os.path.exists(t1):
                    os.makedirs(t1)


                throws = coreStatistics(eLow)       #New line testing changing core statistics
                filename = f"{identifier}_Cores_Gen2Design_{depth}mRefl_{type}_{refl_coef}R_{space:.2f}km_{Estring}_{zString}_cores{throws}_noise{noise}_part{iR}"
                output_filename = os.path.join(working_dir, Estring, filename+".hdf5")
                if skip_repeats:
                    if os.path.exists(output_filename):
                        print(f'output hdf5 already exists, skipping')
                        continue
                output_filename_nur = os.path.join(working_dir, Estring, filename+".nur")
                cmd = f"python {detector_sim} {detector_filename} {config_file} {output_filename} {output_filename_nur} {throws} {eLow} {eHigh} {cLow} {cHigh} {type} {depth} {refl_coef} {space} {iR}\n"

                #now adding settings for slurm scheduler
                header = "#!/bin/bash\n"
                header += f"#SBATCH --job-name={Estring}_{zString}_{filename}      ##Name of the job.\n"
                header += "#SBATCH -A sbarwick_lab                  ##Account to charge to\n"
                
                header += f"#SBATCH -p {partition}                          ##Partition/queue name\n"

                header += "#SBATCH --nodes=1                        ##Nodes to be used\n"
                header += "#SBATCH --ntasks=1                       ##Numer of processes to be launched\n"
                header += "#SBATCH --cpus-per-task=1                ##Cpu's to be used\n"
                header += "#SBATCH --output={}\n".format(os.path.join(working_dir, 'run', 'logs', f'refl_{filename}.out'))
                header += "#SBATCH --error={}\n".format(os.path.join(working_dir, 'run', 'logs', f'refl_{filename}.err'))
    #            header += "#SBATCH --mail-type=fail,end\n"
    #            header += "#SBATCH --mail-user=rricesmi@uci.edu\n"
        
                #Add software to the python path
                header += "export PYTHONPATH=$ARIANNAanalysis/../NuRadioMC:$PYTHONPATH\n"
                header += "export PYTHONPATH=$ARIANNAanalysis/../MuRadioReco:$PYTHONPATH\n"
                header += "export PYTHONPATH=$ARIANNAanalysis/../radiotools:$PYTHONPATH\n"
            
                header += "module load python/3.8.0\n"
                header += "cd $ReflectiveAnalysis\n"

                slurm_name = os.path.join(f'run/CoreRefl{space:.2f}km', os.path.basename(filename) + ".sh")
                with open(slurm_name, 'w') as fout:
                    fout.write(header)
                    fout.write(cmd)
                fout.close()


                slurm_name = 'sbatch ' + slurm_name
                print(f'cmd running {cmd}')
                print(f'running {slurm_name}')
                process = subprocess.Popen(slurm_name.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()

