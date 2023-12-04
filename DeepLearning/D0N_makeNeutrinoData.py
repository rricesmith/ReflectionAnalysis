import os
import subprocess
import numpy as np
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder




###
#This file is meant to be run while sitting inside the folder before DeepLearning
#Additionally, there needs to be the folders DeepLearning/data/simulatedNeutrinos , DeepLearning/data/generatedNeutrinos ,
#						Also DeepLearning/run and DeepLearning/run/logs
#It will create simulated neutrino events
#then will simulate these events on an existing station configuration that has been preset
#By making a series of batch jobs that will be ran in parallel
###



def makeAndRunJob(stationNum, simulationFile, nuFilename, stationConfig, locConfig, outputDirectory, filePart=0,runDirectory='DeepLearning/run/'):

    cmd = f'python {simulationFile}'
    cmd += f' {nuFilename}'
    cmd += f' {stationConfig}'
    cmd += f' {locConfig}'

    #Name to save file with
    filenameToSave = f'Station{stationNum}_Nu_part{filePart}'
    cmd += f' {outputDirectory}' + filenameToSave + '.hdf5'
    cmd += f' {outputDirectory}' + filenameToSave + '.nur'


    #now adding settings for slurm scheduler
    header = "#!/bin/bash\n"
    header += f"#SBATCH --job-name=Stn{stationNum}_{filePart}      ##Name of the job.\n"
    header += "#SBATCH -A sbarwick_lab                  ##Account to charge to\n"
    header += "#SBATCH -p standard                          ##Partition/queue name\n"
#    header += "#SBATCH -p free                          ##Partition/queue name\n"
    header += "#SBATCH --time=3-00:00:00                ##Max runtime D-HH:MM:SS, 3 days free maximum\n"
    header += "#SBATCH --nodes=1                        ##Nodes to be used\n"
    header += "#SBATCH --ntasks=1                       ##Numer of processes to be launched\n"
    header += "#SBATCH --cpus-per-task=1                ##Cpu's to be used\n"
    header += "#SBATCH --mem-per-cpu=6G		            ##6GB memory per job\n"
    header += "#SBATCH --output={}\n".format(os.path.join(runDirectory, 'logs', f'{filenameToSave}.out'))
    header += "#SBATCH --error={}\n".format(os.path.join(runDirectory, 'logs', f'{filenameToSave}.err'))
    header += "#SBATCH --mail-type=fail\n"
    header += "#SBATCH --mail-user=rricesmi@uci.edu\n"

    #Add software to the python path
    header += "export PYTHONPATH=$NuM:$PYTHONPATH\n"
    header += "export PYTHONPATH=$Nu:$PYTHONPATH\n"
    header += "export PYTHONPATH=$Radio:$PYTHONPATH\n"
 
    header += "module load python/3.8.0\n"
    header += "cd $ReflectiveAnalysis\n"

    slurm_name = os.path.join(runDirectory, os.path.basename(filenameToSave)) + ".sh"
    print(f'running cmd {cmd}')
    with open(slurm_name, 'w') as fout:
        fout.write(header)
        fout.write(cmd)
    fout.close()


    slurm_name = 'sbatch ' + slurm_name
    print(f'running {slurm_name}')
    process = subprocess.Popen(slurm_name.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    return




#Define simulation volume
volume = {
#'fiducial_zmin':-2.7 * units.km,  # the ice sheet at South Pole is 2.7km deep
'fiducial_zmin':-0.575 * units.km,  # the ice sheet at Moores Bay is 576mkm deep
'fiducial_zmax': 0 * units.km,
'fiducial_rmin': 0 * units.km,
'fiducial_rmax': 2 * units.km}


Emin = 1e18 * units.eV
Emax = 1e20 * units.eV
numNeutrinos = 1e7
#numNeutrinos = 5*1e5   #Got about 6k triggers on 100/200s with this number for 1e17-1e20eV Nu's
numNeutrinos = 2*1e6

pathToSaveTo = 'DeepLearning/data/generatedNeutrinos/'

neutrinos_exist=True
if not neutrinos_exist:
    generate_eventlist_cylinder(pathToSaveTo + f'neutrinos_{Emin:.4e}_to_{Emax:.4e}_Num{numNeutrinos:.4e}.hdf5', numNeutrinos, Emin, Emax, volume, n_events_per_file=5*1e4)

pathToSimulationFile = 'DeepLearning/D0S_simulateNeutrinos.py'
pathToSaveSimTo = 'DeepLearning/data/simulatedNeutrinos/'

station_num = 19


detectorConfig = f'DeepLearning/station_configs/station{station_num}.json'
locConfig = 'DeepLearning/MB_config.yaml'

filePart = 0
for neutrinoFilename in os.listdir(pathToSaveTo):

    makeAndRunJob(station_num, pathToSimulationFile, pathToSaveTo + neutrinoFilename, detectorConfig,
                    locConfig, pathToSaveSimTo, filePart=filePart)
    filePart += 1
#    if filePart == 2:
#        quit()
