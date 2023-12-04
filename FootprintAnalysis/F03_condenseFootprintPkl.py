import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units


def saveFile(condensed_output, filename, suffix):
    save_file_as = filename + suffix + '.pkl'
    print(f'saving file')
    with open(save_file_as, 'wb') as fout:
        pickle.dump(condensed_output, fout, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'saved file as {save_file_as}')
    fout.close()




condensed_output = {}

CoREAS_mode = 'direct'
#CoREAS_mode = 'refracted'
type = 'MB'
#type = 'SP'
#type = 'IceTop'
#type = 'GL'

#config = 'MB_future'
#config = 'MB_old'
#config = 'SP'
#config = 'GL'
#config = 'TriggerTest'
config = 'BacklobeTest'

#noise = True
noise = False
amp = True
#amp = False
#amp_type = 'future'
#amp_type = 'TriggerTest'
amp_type = '100'
noise_figure = 1.4
noise_temp = 400

antenna = 'lpda'
#antenna = 'dipole'
#antenna = 'PA'

depthLayer = 300.0
dB = 40.0

#cores = 1000
#spacing = 10
cores = 1000
spacing = 4

#amp_type = None


if type == 'SP':
    low = 0
    high = 9
#    low = 1000
#    high = 1049
    limit = 2199
elif type == 'MB':
    depthLayer = 576.0
    dB = 0.0

    low = 0
    high = 4
#    high = 24
#    high = 49
#    high = 99
    limit = 1000
#    limit = 4000
elif type == 'IceTop':
    cores = 100
    low = 16.0
    high = 16.1
    limit = 18.7    
    numIceTop = 10
    iceTopSin = 0
#-1 makes a file with all sin bins
#    iceTopSin = -1
elif type == 'GL':
    depth = 300
    dB = 0.0

    low = 0
    high = 9
    limit = 601


iter = (high - low) + 1
if type == 'IceTop':
    iter = high - low


passband_high = 500
if True:
    if amp_type == '100':
        noise_figure = 1.4
        noise_temp = 400 * units.kelvin
        passband_low = 100
    elif amp_type == '200':
        noise_figure = 1.4
        noise_temp = 400 * units.kelvin
        passband_low = 50
    elif amp_type == 'future':
        noise_figure = 1
        noise_temp = 350 * units.kelvin
        passband_low = 50
        passband_high = 150
    elif amp_type == 'TriggerTest':
#        noise_figure = 1.4
        noise_figure = 1.0
        noise_temp = 350
#        noise_temp = 400
    else:
        noise_figure = 1.4
        noise_temp = 400 * units.kelvin
        passband_low = 50


PA_sigma = []
if config == 'SP' or config == 'MB_future' or config == 'GL':
    if not amp_type == 'future':
        print(f'Future stations only use future, not {amp_type}')
        quit()
    lpda_sigma = [[2, '2sigma'], [4, '4sigma'], [6, '6sigma'], [8, '8sigma'], [10, '10sigma']]
#    lpda_sigma = [[3.5, '3.5sigma'], [3.9498194908011524, '100Hz'], [4.919151494949084, '10mHz']]
    PA_sigma = [[30.68, '100Hz'], [38.62, '1Hz'], [50.53, '1mHz']]
elif config == 'MB_old' or config == 'BacklobeTest':
    lpda_sigma = [[4.4, '4.4sigma']]
elif config == 'SP_old':
    if not amp_type == '300':
        print(f'SP_old only is 300s')
        quit()
    lpda_sigma = [[5, '5sigma']]
elif config == 'Stn51':
    lpda_sigma = [[5, '5sigma']]
elif config == 'TriggerTest':
    lpda_sigma = [[3.5, '3.5sigma'], [3.9498194908011524, '100Hz'], [4.4, '4.4sigma'], [4.919151494949084, '10mHz']]
else:
    print(f'no config of {config} used, use SP, MB_old, or MB_future')




if False:
    lpda_coinc = 2
    channels = 4
    config += f'_{lpda_coinc}of{channels}'

if noise:
    config += '_wNoise'
if amp:
    config += f'_wAmp'
config += f'{amp_type}s'

config += '_ProposalFlux'

#if type == 'IceTop':
#    config += '_InfAir'

if not type == 'IceTop' and False:
    config += f'_{noise_figure:.1f}NF_{noise_temp:.0f}KTemp'
    config += '_Refract'
#    config += f'_LowPass{passband_high}'
    config += f'_{antenna}'

#config += f'_500MHzLowpassTest'
#config += f'_RateCheck300'
#config += f'_TotalChange'
#config += f'_ThreshChange'
#config += '_Temp350K'
#config += '_1.0NF'
#config += '_150MHzLowpass'
#config += '_250MHzLowpass'
#config += '_300MHzLowpass'
#config += '_500MHzLowpass'
#config += '_2of3Trigger6m'
#config += '_2of3Trigger13m'
#config += '_SPTest'
#config += '_ProposalFlux'

save_identifier = '_ProposalFlux'

save_file_as = f'FootprintAnalysis/data/CoREAS{save_identifier}_{CoREAS_mode}_{type}_{config}_Layer{depthLayer}m_{dB}dB_Area{spacing:.2f}_{cores}cores'
suffix = '_part'
split = True
split_iD = 0

new_runid = 0
#while high < 4000:	#MB
#while high < 2099:	#SP
while high < limit:
    if type == 'IceTop':
        low = round(low, 1)
        high = round(high, 1)

###Do I need this anymore?
#   if low == 1000:
#        low += iter
#        high += iter
#        continue


    to_open = f'output/CRFootprintRates/CoREAS_{CoREAS_mode}_{config}'
    if type == 'IceTop':
#        to_open += f'_InfAir'
        to_open += f'_{iceTopSin:.1f}Sin_{numIceTop}PerBin'
#        to_open += f'_{noise_figure:.1f}NF_{noise_temp:.0f}KTemp'
#        to_open += '_Refract'
#        to_open += f'_LowPass{passband_high}'
        to_open += '_ProposalFlux'

    to_open += f'_Layer{depthLayer}m_{dB}dB_Area{spacing:.2f}_{cores}cores_id{low}_{high}.pkl'

#    with open(f'output/CRFootprintRates/CoREAS_{CoREAS_mode}_{config}_Layer{depthLayer}m_{dB}dB_Area{spacing:.2f}_{cores}cores_id{low}_{high}.pkl', 'rb') as fin:
    if not os.path.exists(to_open):
        print(f'File does not exist, {to_open}, skipping')
        if type != 'IceTop':
            low += iter
            high += iter
        else:
            if iceTopSin == -1:
                low += iter
                high += iter
            else:
                if iceTopSin >= 1:
                    iceTopSin = 0
                    low += iter
                    high += iter
                else:
                    iceTopSin += 0.1
        continue
    with open(to_open, 'rb') as fin:
        output = pickle.load(fin)

    print(f'runid of {new_runid}')
    for runid in output:
        print(runid)
        
        condensed_output[new_runid] = {}
        condensed_output[new_runid]['n'] = output[runid]['n']
        condensed_output[new_runid]['n_dip_dir'] = output[runid]['n_dip_dir']
        condensed_output[new_runid]['n_dip_refl'] = output[runid]['n_dip_refl']
#        condensed_output[new_runid]['n_lpda_refl'] = output[runid]['n_lpda_refl']
#        condensed_output[new_runid]['n_lpda_dir'] = output[runid]['n_lpda_dir']
        condensed_output[new_runid]['energy'] = output[runid]['energy']
        condensed_output[new_runid]['zenith'] = output[runid]['zenith']
        condensed_output[new_runid]['azimuth'] = output[runid]['azimuth']
        condensed_output[new_runid]['x_dir_lpda'] = output[runid]['x']
        condensed_output[new_runid]['y_dir_lpda'] = output[runid]['y']
        condensed_output[new_runid]['dip_dir_mask'] = output[runid]['dip_dir_mask']
        condensed_output[new_runid]['dip_refl_mask'] = output[runid]['dip_refl_mask']
#        condensed_output[new_runid]['lpda_refl_mask'] = output[runid]['lpda_refl_mask']
#        condensed_output[new_runid]['lpda_dir_mask'] = output[runid]['lpda_dir_mask']
        condensed_output[new_runid]['ant_zen'] = output[runid]['ant_zen']
        condensed_output[new_runid]['dip_dir_SNR'] = output[runid]['dip_dir_SNR']
        condensed_output[new_runid]['dip_refl_SNR'] = output[runid]['dip_refl_SNR']
#        condensed_output[new_runid]['lpda_dir_SNR'] = output[runid]['lpda_dir_SNR']
#        condensed_output[new_runid]['lpda_refl_SNR'] = output[runid]['lpda_refl_SNR']
        for sigma, name in lpda_sigma:
            condensed_output[new_runid][name] = output[runid][name]
        for sigma, name in PA_sigma:
            if not name in condensed_output[new_runid]:
                condensed_output[new_runid][name] = output[runid][name]
#        for run in runid:
        """
        n.append(output[runid]['n'])
        n_trig_up.append(output[runid]['n_triggered_upward'])
        n_trig.append(output[runid]['n_triggered'])
        n_trig_att.append(output[runid]['n_att_triggered'])
        energy.append(output[runid]['energy'])
        ss_energy.append(output[runid]['ss_energy'])
        zenith.append(output[runid]['zenith'])
        azimuth.append(output[runid]['azimuth'])
        x.append(output[runid]['x'])
        y.append(output[runid]['y'])
        up_mask.append(output[runid]['up_mask'])
        refl_mask.append(output[runid]['refl_mask'])
        """
        new_runid += 1

    if type != 'IceTop':
        low += iter
        high += iter
    else:
        if iceTopSin == -1:
            low += iter
            high += iter
        else:
            if iceTopSin >= 1:
                iceTopSin = 0
                low += iter
                high += iter
            else:
                iceTopSin += 0.1

    if sys.getsizeof(condensed_output) > 5000:
        saveName = save_file_as
#        saveName += f'_{noise_figure:.1f}NF_{noise_temp:.0f}KTemp'
#        saveName += '_Refract'
#        saveName += f'_LowPass{passband_high}'
        saveFile(condensed_output, saveName, suffix+f'{split_iD}')
        split_iD += 1
        condensed_output = {}


print(f'ended and saving')
saveName = save_file_as
#saveName += f'_{noise_figure:.1f}NF_{noise_temp:.0f}KTemp'
#saveName += '_Refract'
#saveName += f'_LowPass{passband_high}'
saveFile(condensed_output, saveName, suffix+f'{split_iD}')
quit()

print(f'size of dict is {sys.getsizeof(condensed_output)}')

save_file_as = f'FootprintAnalysis/data/CoREAS_{CoREAS_mode}_{config}_Layer{depthLayer}m_{dB}dB_Area{spacing:.2f}_{cores}cores.pkl'

with open(save_file_as, 'wb') as fout:
    print(f'saved file as {save_file_as}')
    pickle.dump(condensed_output, fout, protocol=pickle.HIGHEST_PROTOCOL)

