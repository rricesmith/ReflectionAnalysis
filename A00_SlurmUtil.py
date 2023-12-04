import os
import subprocess


def makeAndRunJob(commandToRun, jobName='job', partition='free', runDirectory='run/'):

    if not (partition == 'free' or partition == 'standard'):
        print(f'Partition {partition} does not exist')
        quit()

    cmd = f'{commandToRun}'

    #now adding settings for slurm scheduler
    header = "#!/bin/bash\n"
    header += f"#SBATCH --job-name={jobName}            ##Name of the job.\n"
    header += "#SBATCH -A sbarwick_lab                  ##Account to charge to\n"
    header += f"#SBATCH -p {partition}                          ##Partition/queue name\n"
#    header += "#SBATCH -p free                          ##Partition/queue name\n"
    header += "#SBATCH --time=3-00:00:00                ##Max runtime D-HH:MM:SS, 3 days free maximum\n"
    header += "#SBATCH --nodes=1                        ##Nodes to be used\n"
    header += "#SBATCH --ntasks=1                       ##Numer of processes to be launched\n"
    header += "#SBATCH --cpus-per-task=1                ##Cpu's to be used\n"
    header += "#SBATCH --mem-per-cpu=6G		            ##6GB memory per job\n"
    header += "#SBATCH --output={}\n".format(os.path.join(runDirectory, 'logs', f'{jobName}.out'))
    header += "#SBATCH --error={}\n".format(os.path.join(runDirectory, 'logs', f'{jobName}.err'))
    header += "#SBATCH --mail-type=fail\n"
    header += "#SBATCH --mail-user=rricesmi@uci.edu\n"

    #Add software to the python path
    header += "export PYTHONPATH=$NuM:$PYTHONPATH\n"
    header += "export PYTHONPATH=$Nu:$PYTHONPATH\n"
    header += "export PYTHONPATH=$Radio:$PYTHONPATH\n"
 
    header += "module load python/3.8.0\n"
    header += "cd $ReflectiveAnalysis\n"

    slurm_name = os.path.join(runDirectory, os.path.basename(jobName)) + ".sh"
    print(f'running cmd {cmd}')
    with open(slurm_name, 'w') as fout:
        fout.write(header)
        fout.write(cmd)
    fout.close()


    slurm_name = 'sbatch ' + slurm_name
    print(f'running {slurm_name}')
    errLoc = os.path.join(runDirectory, 'logs', f'{jobName}.err')
    print(f'Logs at {errLoc}')
    process = subprocess.Popen(slurm_name.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    return

if __name__ == "__main__":

    cmd = 'python DeepLearning/angularReconstructionProcessing.py'
    station_id = 30

    station_path = f"../../../ariannaproject/station_nur/station_{station_id}/"

    for iF, filename in enumerate(os.listdir(station_path)):
        if filename.endswith('_statDatPak.root.nur'):
            continue
        else:
            makeAndRunJob(cmd + f' {os.path.join(station_path, filename)} {station_id}', jobName=f'ang{iF}', partition='standard')