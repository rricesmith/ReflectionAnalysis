import glob
import os
import subprocess



def write_batch_beginning(i, station_id, queue='free'):
    header = "#!/bin/bash\n"
    header += f"#SBATCH --job-name=Stn{station_id}_Nu{i}      ##Name of the job.\n"
    header += "#SBATCH -A sbarwick_lab                  ##Account to charge to\n"
    header += f"#SBATCH -p {queue}                          ##Partition/queue name\n"
    header += "#SBATCH --time=3-00:00:00                ##Max runtime D-HH:MM:SS, 3 days free maximum\n"
    header += "#SBATCH --nodes=1                        ##Nodes to be used\n"
    header += "#SBATCH --ntasks=1                       ##Numer of processes to be launched\n"
    header += "#SBATCH --cpus-per-task=1                ##Cpu's to be used\n"
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

    return header

def submitJobs(header, working_dir, filename):
    slurm_name = os.path.join(working_dir, os.path.basename(filename)) + ".sh"
    with open(slurm_name, 'w') as fout:
        fout.write(header)
    fout.close()

    slurm_name = 'sbatch ' + slurm_name
    print(f'running {slurm_name}')
    process = subprocess.Popen(slurm_name.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


s100 = [13, 15, 18]
s200 = [14, 17, 19, 30]


station_id = 30
num_per_job = 10

if station_id in s100:
    amp = 100
elif station_id in s200:
    amp = 200



working_dir = os.path.join(f"run/NurProcessing/")
#filename = f'reanalyze_station_{station_id}'
#cmd = f'python StationDataAnalysis/reanalyzeChrisStationData.py '

filename = f'forced_triggers_station_{station_id}'
cmd = f'python StationDataAnalysis/saveForcedTriggers.py '


i=0
header = write_batch_beginning(i=i, station_id=station_id)
num_in_job = 0
while i < 99999:
    file = f'../../../../pub/arianna/leshanz/data_for_others/2022_reflected_cr_search/data/station_{station_id}/station_{station_id}_run_{i:05d}.root.nur'
    if os.path.exists(file):
        num_in_job += 1
        header += cmd + file + f' --station {station_id} --templates_nu {amp} --templates_cr {amp}\n'

        if num_in_job == num_per_job:
            submitJobs(header, working_dir, filename)
            header = write_batch_beginning(i=i, station_id=station_id)
            num_in_job = 0

    i += 1

submitJobs(header, working_dir, filename)


