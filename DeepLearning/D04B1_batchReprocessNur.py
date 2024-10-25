import A00_SlurmUtil as slurm
import os


base_folder = '7thpass'

stations_100s = [13, 15, 18, 32]
stations_200s = [14, 17, 19, 30]
stations_300s = [52]


for station in stations_100s:
    amp = '100s'
    folder = f'{base_folder}/Station{station}'
    station_path = f"/dfs8/sbarwick_lab/ariannaproject/station_nur/station_{station}/"
    i = 0
    for file in os.listdir(station_path):
        if file.endswith('_statDatPak.root.nur'):
            continue    
        else:
            filename = os.path.join(station_path, file)
            cmd = f'python DeepLearning/D04B_reprocessNurPassingCut.py {station} --folder {folder} --single_file {filename} --amp {amp}'
            print(f'cmd {cmd}')
            slurm.makeAndRunJob(cmd, jobName=f'Stn{station}_{i}', partition='standard', runDirectory='run/')
            i += 1

quit()

for station  in stations_200s:
    amp = '200s'
    folder = f'{base_folder}/Station{station}'
    station_path = f"/dfs8/sbarwick_lab/ariannaproject/station_nur/station_{station}/"
    i = 0
    for file in os.listdir(station_path):
        if file.endswith('_statDatPak.root.nur'):
            continue    
        else:
            filename = os.path.join(station_path, file)
            cmd = f'python DeepLearning/D04B_reprocessNurPassingCut.py {station} --folder {folder} --single_file {filename} --amp {amp}'
            slurm.makeAndRunJob(cmd, jobName=f'Stn{station}_{i}', partition='standard', runDirectory='run/')
            i += 1

for station in stations_300s:
    amp = '300s'
    folder = f'{base_folder}/Station{station}'
    station_path = f"/dfs8/sbarwick_lab/ariannaproject/leshanz_backup/arianna/station_{station}/data/"
    i = 0
    for file in os.listdir(station_path):
        if file.endswith('_statDatPak.root.nur'):
            continue    
        else:
            filename = os.path.join(station_path, file)
            cmd = f'python DeepLearning/D04B_reprocessNurPassingCut.py {station} --folder {folder} --single_file {filename} --amp {amp}'
            slurm.makeAndRunJob(cmd, jobName=f'Stn{station}_{i}', partition='standard', runDirectory='run/')
            i += 1
