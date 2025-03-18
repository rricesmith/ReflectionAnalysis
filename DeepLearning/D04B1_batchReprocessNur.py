import A00_SlurmUtil as slurm
import os
import configparser



config = configparser.ConfigParser()
config.read('StationDataAnalysis/config.ini')
base_folder = config['BASEFOLDER']['base_folder']
single_file = config['BASEFOLDER']('single_file') == 'True'
template_date = config['TEMPLATE']['template']

stations_100s = [13, 15, 18, 32]
stations_200s = [14, 17, 19, 30]
# stations_100s = []
# stations_200s = []
stations_300s = [52]


for station in stations_100s:
    amp = '100s'
    folder = f'{base_folder}/Station{station}'
    station_path = f"/dfs8/sbarwick_lab/ariannaproject/station_nur/station_{station}/"
    i = 0
    if single_file:
        for file in os.listdir(station_path):
            if file.endswith('_statDatPak.root.nur'):
                continue    
            else:
                filename = os.path.join(station_path, file)
                cmd = f'python DeepLearning/D04B_reprocessNurPassingCut.py {station} --folder {folder} --single_file {filename} --amp {amp} --template_date {template_date}'
                print(f'cmd {cmd}')
                slurm.makeAndRunJob(cmd, jobName=f'Stn{station}_{i}', partition='standard', runDirectory='run/')
                i += 1
    else:
        cmd = f'python DeepLearning/D04B_reprocessNurPassingCut.py {station} --folder {folder} --amp {amp} --template_date {template_date}'
        print(f'cmd {cmd}')
        slurm.makeAndRunJob(cmd, jobName=f'Stn{station}_{i}', partition='standard', runDirectory='run/')

# quit()

for station  in stations_200s:
    amp = '200s'
    folder = f'{base_folder}/Station{station}'
    station_path = f"/dfs8/sbarwick_lab/ariannaproject/station_nur/station_{station}/"
    i = 0
    if single_file:
        for file in os.listdir(station_path):
            if file.endswith('_statDatPak.root.nur'):
                continue    
            else:
                filename = os.path.join(station_path, file)
                cmd = f'python DeepLearning/D04B_reprocessNurPassingCut.py {station} --folder {folder} --single_file {filename} --amp {amp} --template_date {template_date}'
                slurm.makeAndRunJob(cmd, jobName=f'Stn{station}_{i}', partition='standard', runDirectory='run/')
                i += 1
    else:
        cmd = f'python DeepLearning/D04B_reprocessNurPassingCut.py {station} --folder {folder} --amp {amp} --template_date {template_date}'
        print(f'cmd {cmd}')
        slurm.makeAndRunJob(cmd, jobName=f'Stn{station}_{i}', partition='standard', runDirectory='run/')

for station in stations_300s:
    amp = '300s'
    first_ch = 0
    folder = f'{base_folder}/Station{station}/FirstCh{first_ch}'
    station_path = f"/dfs8/sbarwick_lab/ariannaproject/leshanz_backup/arianna/station_{station}/data/"
    i = 0
    if single_file:
        for file in os.listdir(station_path):
            if file.endswith('_statDatPak.root.nur'):
                continue    
            else:
                filename = os.path.join(station_path, file)
                cmd = f'python DeepLearning/D04B_reprocessNurPassingCut.py {station} --folder {folder} --single_file {filename} --amp {amp} --first_ch {first_ch} --template_date {template_date}'
                slurm.makeAndRunJob(cmd, jobName=f'Stn{station}_ch{first_ch}_{i}', partition='standard', runDirectory='run/')
                i += 1
    else:
        cmd = f'python DeepLearning/D04B_reprocessNurPassingCut.py {station} --folder {folder} --amp {amp} --first_ch {first_ch} --template_date {template_date}'
        print(f'cmd {cmd}')
        slurm.makeAndRunJob(cmd, jobName=f'Stn{station}_ch{first_ch}_{i}', partition='standard', runDirectory='run/')

    first_ch = 4 # 4 is first channel for upward facing LPDAs, only in stn 52
    folder = f'{base_folder}/Station{station}/FirstCh{first_ch}'
    if single_file:
        i = 0
        for file in os.listdir(station_path):
            if file.endswith('_statDatPak.root.nur'):
                continue    
            else:
                filename = os.path.join(station_path, file)
                cmd = f'python DeepLearning/D04B_reprocessNurPassingCut.py {station} --folder {folder} --single_file {filename} --amp {amp} --first_ch {first_ch} --template_date {template_date}'
                slurm.makeAndRunJob(cmd, jobName=f'Stn{station}_ch{first_ch}_{i}', partition='standard', runDirectory='run/')
                i += 1
    else:
        cmd = f'python DeepLearning/D04B_reprocessNurPassingCut.py {station} --folder {folder} --amp {amp} --first_ch {first_ch} --template_date {template_date}'
        print(f'cmd {cmd}')
        slurm.makeAndRunJob(cmd, jobName=f'Stn{station}_ch{first_ch}_{i}', partition='standard', runDirectory='run/')

