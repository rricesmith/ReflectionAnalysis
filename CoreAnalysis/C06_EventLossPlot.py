import matplotlib.pyplot as plt
import pickle
import matplotlib.colors
import itertools
plt.style.use('plotsStyle.mplstyle')
import numpy as np
from os.path import exists


color = itertools.cycle(('blue', 'black', 'red'))
textcolor = itertools.cycle(('green', 'orange', 'purple'))
hatch = itertools.cycle(('/', '|', '-'))

#depths_area = [ [300, 1.7], [500, 1.7], [800, 1.7] ]
depths_area = [ [300, 1.7] ]
f_values = [0.5, 0.4, 0.3, 0.2, 0.1]
dB_values = [40, 45, 50]

type = 'SP'

mainFolder = 'plots/CoreAnalysis/LossRunning'
saveFolder = '12_6_23'

dipole_check = False
#identifier = 'Gen2_AllTriggers'
identifier = 'Gen2_AllTriggers_DipFix'
lpda_triggers = ['LPDA_2of4_3.8sigma', 'LPDA_2of4_2sigma']
dipole_triggers = ['single_dipole_trigger_3sig', 'single_dipole_trigger_2sig']

"""
dipole_check = True
identifier = 'DipoleTesting'
lpda_triggers = [ 'Gen2_Dipole_2sig', 'Gen2_Dipole_3sig']
dipole_triggers = [ 'RNOG_3sig', 'ARA_2sig', 'Old_80to500_HighLow_2sig', 'Old_80to250_HighLow_2sig']
f_values = [0.5, 0.4, 0.3, 0.2, 0.1]
"""
#Various less often changed values
refl_coef = 1
cores = 1000
#direction = 'below'
direction = 'above'
#reSave = True
reSave = False


#save_suffix = 'Unscaled'
#save_suffix = 'AboveUnscaled'
save_suffix = f'{direction}Scaled'
#save_suffix = f'{direction}Unscaled'
#save_suffix = ''

def getErateArray(depths_area, lpda_triggers, dipole_triggers, f_values, dB_values):
    LPDA_stations = 361
    dipole_stations = 164
#    LPDA_stations = 1
#    dipole_stations = 1
    E_rate = {}
    for depth, area in depths_area:
        E_rate[depth] = {}
        for trigger in lpda_triggers:
            print(f'Trigger {trigger}')
            E_rate[depth][trigger] = []
            for f in f_values:
                print(f'{f}f')
                for dB in dB_values:
                    print(f'{dB}dB')
                    core_input_file = f'data/CoreDataObjects/{identifier}_CoreDataObjects_{trigger}_{direction}_{depth}mRefl_{type}_{refl_coef}R_{f}f_{dB:.1f}dB_{area}km_{cores}cores.pkl'

                    with open(core_input_file, 'rb') as fin:
                        CoreObjectsList = pickle.load(fin)

                    total_event_rate = 0
                    for core in CoreObjectsList:
                        total_event_rate += core.totalEventRateCore()

                    total_event_rate = total_event_rate * LPDA_stations
                    E_rate[depth][trigger].append(total_event_rate)

            E_rate[depth][trigger] = np.array(E_rate[depth][trigger])[L_order]

        for trigger in dipole_triggers:
            print(f'Trigger {trigger}')
            E_rate[depth][trigger] = []
            for f in f_values:
                print(f'{f}f')
                for dB in dB_values:
                    print(f'{dB}dB')
                    core_input_file = f'data/CoreDataObjects/{identifier}_CoreDataObjects_{trigger}_{direction}_{depth}mRefl_{type}_{refl_coef}R_{f}f_{dB:.1f}dB_{area}km_{cores}cores.pkl'

                    with open(core_input_file, 'rb') as fin:
                        CoreObjectsList = pickle.load(fin)

                    total_event_rate = 0
                    for core in CoreObjectsList:
                        total_event_rate += core.totalEventRateCore()

                    total_event_rate = total_event_rate * dipole_stations
                    E_rate[depth][trigger].append(total_event_rate)

            E_rate[depth][trigger] = np.array(E_rate[depth][trigger])[L_order]

    return E_rate   

def plotText(f_values, dB_values, L_array, y_location=0.11):
    #If direction is above, x-axis label should be f
    if direction == 'above':
        return
    dist_between_L = (max(L_array) - min(L_array))/60
    L_plotted = []
    for dB in dB_values:
        c = next(textcolor)
        for f in f_values:
            L = f * 10 ** (-dB/20)
            #Want to skip L's that are too close to already plotted L values
            #Because of this want to have dB ordered low->high and f ordered high->low
            skip = False
            for Lp in L_plotted:
                if abs(L - Lp) < dist_between_L:
                    skip = True
            if not skip:
                L_plotted.append(L)
                va = 'center'
                if L == max(L_array):
                    va = 'bottom'
                elif L == min(L_array):
                    va = 'top'
                    L = L * 1.05
                plt.text(L, y_location, f'{dB}dB {f}f', rotation=90, color=c, rotation_mode='anchor', verticalalignment=va, horizontalalignment='left', fontsize=16)

def plotPredictionLine(dB=40, f=0.3, y_loc=80, color='black', linestyle='-', direction='below'):
    L = f * 10 ** (-dB/20)
    label = f'{dB}dB {f}f'
    if dB == 0:
        label = f'{f}f'
    if direction == 'above':
        L = f
        label = f'{f} f'
#    plt.plot([L, L], [10**-6, 10**6], color=color, linestyle=linestyle)
    plt.axvline(x=L, color=color, linestyle=linestyle)

    plt.text(L * 0.95, y_loc, label, rotation=90, va='center', fontsize=16, weight='bold')

#For direct cores, no reflector contributing to plot
if direction == 'above':
    dB_values = [0]

#L will be x axis
L_array = []
if direction == 'below':
    for f in f_values:
        for dB in dB_values:
            #L is combined loss of f and dB
            L = f * 10**(-dB/20)
            L_array.append(L)
else:
    L_array = f_values
print(f'L values {L_array}')

L_order = np.argsort(L_array)
L_array = np.array(L_array)[L_order]


saveName = f'data/CoreDataObjects/{saveFolder}/LossRunning_{identifier}.pkl'
if exists(saveName) and not reSave:
    with open(saveName, 'rb') as fin:
        L_array, E_rate = pickle.load(fin)
else:
    E_rate = getErateArray(depths_area, lpda_triggers, dipole_triggers, f_values, dB_values)
    print(f'saving data to {saveName}')
    with open(saveName, 'wb') as fout:
        pickle.dump([L_array, E_rate], fout)
    print(f'Saved!')


if dipole_check == True:
    #Nu22 poster data for reference
    old_data_L = [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]
    old_data_rate = [1.5*10**-2, 2*10**-2, 3*10**-2, 4*10**-2, 5.5*10**-2, 7*10**-2, 8*10**-2]
#    plt.plot(old_data_L, old_data_rate, label='Nu22 Poster', color='black', alpha=1)

    #General plot of all triggers
    for depth, area in depths_area:

        for trigger in lpda_triggers:
            plt.plot(L_array, E_rate[depth][trigger], linestyle='--', label=f'{depth}m {trigger}')
        for trigger in dipole_triggers:
            plt.plot(L_array, E_rate[depth][trigger], linestyle='--', label=f'{depth}m {trigger}')
        break


    plotText(f_values, dB_values, L_array)
    plt.xlim((min(L_array), max(L_array)))
#    plt.ylim((10**-5, 10**0))
    plt.yscale('log')
    plt.xlabel('L')
    plt.ylabel('Evts/Stn/Yr')
    plt.legend(loc='lower right')
    plt.savefig(f'{mainFolder}/{saveFolder}/TriggersCheck{save_suffix}.png')
    plt.clf()

    #Plot triggers of same depth/sigma, BW restriction difference only
#    plt.plot(old_data_L, old_data_rate, label='Nu22 80-500MHz', color='black')
    plt.plot(L_array, E_rate[depth][dipole_triggers[2]], label='80-500MHz')
    plt.plot(L_array, E_rate[depth][dipole_triggers[3]], label='80-250MHz')
    plt.plot(L_array, E_rate[depth][lpda_triggers[0]], label='80-150MHz')
    plotText(f_values, dB_values, L_array)
    plt.xlim((min(L_array), max(L_array)))
#    plt.ylim((10**-5, 10**0))
    plt.yscale('log')
    plt.xlabel('L')
    plt.ylabel('Evts/Stn/Yr')
    plt.legend(loc='lower right')
    plt.savefig(f'{mainFolder}/{saveFolder}/DipoleBWCheck_{save_suffix}.png')
    plt.clf()

    #Plot triggers of all experimental dipoles
    plt.plot(L_array, E_rate[depth][dipole_triggers[0]], label='RNOG')
    plt.plot(L_array, E_rate[depth][dipole_triggers[1]], label='ARA')
#    plt.plot(L_array, E_rate[depth][lpda_triggers[0]], label='Gen2 2sig')
#    plt.plot(L_array, E_rate[depth][lpda_triggers[1]], label='Gen2 3sig')
    #Need new sim for gen2 2sig 3sig of dipoles
    plotText(f_values, dB_values, L_array, y_location=30)
    plt.xlim((min(L_array), max(L_array)))
#    plt.ylim((10**-5, 10**0))
#    plt.yscale('log')
    plt.xlabel('f')
    plt.ylabel('Evts/Stn/Yr')
    plt.legend(loc='upper left')
    plt.savefig(f'{mainFolder}/{saveFolder}/DetectorDipoles_{save_suffix}.png')
    plt.clf()

    #Plot RNOG expected events, with 35 deep detectors per https://arxiv.org/pdf/2010.12279.pdf
    RNOG_deep_stations =  35
    RNOG_scaled = [RNOG_deep_stations * rate for rate in E_rate[depth][dipole_triggers[0]]]
    plt.plot(L_array, RNOG_scaled)
#    plotText(f_values, dB_values, L_array, y_location=0.2)
    plt.xlim((min(L_array), max(L_array)))
#    plt.ylim((10**-5, 10**0))
#    plt.yscale('log')
#    plt.xlabel('L')
    plt.xlabel('f')
    plt.ylabel('Evts/Yr Over RNOG Array')
#    plt.legend(loc='lower right')
    plt.savefig(f'{mainFolder}/{saveFolder}/RNOG_Rate_{save_suffix}.png')
    plt.clf()


    quit()

#Loss running error bar plot
for depth, area in depths_area:
    c = next(color)
    h = next(hatch)
    a = 0.7
    plt.fill_between(L_array, E_rate[depth][lpda_triggers[0]], E_rate[depth][lpda_triggers[1]], edgecolor=c, linestyle='-', label=f'{depth}m', alpha=a, hatch=h, facecolor='none', linewidth=3)
    plt.fill_between(L_array, E_rate[depth][dipole_triggers[0]], E_rate[depth][dipole_triggers[1]], edgecolor=c, linestyle='--', alpha=a, hatch='x', facecolor='none', linewidth=3)

plotText(f_values, dB_values, L_array)
if not save_suffix == 'Unscaled':
    plt.ylim((0.1, 200))
plt.xlim((min(L_array), max(L_array)))
plt.yscale('log')
plt.xlabel(r'L = f * $10^{-dB/20}$')
plt.ylabel('Evts/Stn/Yr')
plt.legend(loc='upper left')
plt.savefig(f'{mainFolder}/{saveFolder}/LossRunning{save_suffix}.png')
plt.clf()

#Loss running plot bar plot
color_depths = []
for depth, area in depths_area:
    c = next(color)
    plt.plot(L_array, E_rate[depth][lpda_triggers[0]], color=c, linestyle='--')
    plt.plot(L_array, E_rate[depth][lpda_triggers[1]], color=c, linestyle='-')
    color_depths.append([c, f'{depth}m'])

#    plt.plot(L_array, E_rate[depth][dipole_triggers[0]], color=c, linestyle='.', label=f'{depth}m LPDA 3sig')
#    plt.plot(L_array, E_rate[depth][dipole_triggers[1]], color=c, linestyle='-.', label=f'{depth}m LPDA 2sig')
leg1 = plt.legend(handles=[plt.plot([], [], linestyle='-', color=color_label, label=depth_label)[0] for color_label, depth_label in color_depths],
                  loc='upper left')
plt.gca().add_artist(leg1)
leg2 = plt.legend(handles=[plt.plot([], [], linestyle='-', color='black', label=r'2$\sigma$')[0], 
                           plt.plot([], [], linestyle='--', color='black', label=r'3.8$\sigma$')[0]],
                           loc='upper left', bbox_to_anchor=(0, 0.8))
plt.gca().add_artist(leg2)

plotText(f_values, dB_values, L_array)
plotPredictionLine()
if not save_suffix == 'Unscaled':
    plt.ylim((0.1, 200))
plt.xlim((min(L_array), max(L_array)))
plt.yscale('log')
plt.xlabel(r'L = f * $10^{-dB/20}$')
if 'Unscaled' in save_suffix:
    plt.ylabel('Evts/Stn/Yr')
else:
    plt.ylabel('Evts/Yr')
plt.savefig(f'{mainFolder}/{saveFolder}/LossRunningLinesLPDA{save_suffix}.png')
plt.clf()


#Loss running plot bar plot
color_depths = []
for depth, area in depths_area:
#    c = next(color)
    c = 'black'

    if direction == 'above':
        label_prefix = ''
    else:
        label_prefix = f'{depth}m '
    plt.plot(L_array, E_rate[depth][dipole_triggers[0]], color=c, linestyle='--')
    plt.plot(L_array, E_rate[depth][dipole_triggers[1]], color=c, linestyle='-')
    color_depths.append([c, f'{depth}m'])

if direction == 'below':
    leg1 = plt.legend(handles=[plt.plot([], [], linestyle='-', color=color_label, label=depth_label)[0] for color_label, depth_label in color_depths],
                    loc='upper left')
    plt.gca().add_artist(leg1)
    leg2 = plt.legend(handles=[plt.plot([], [], linestyle='-', color='black', label=r'2$\sigma$')[0], 
                            plt.plot([], [], linestyle='--', color='black', label=r'3$\sigma$')[0]],
                            loc='upper left', bbox_to_anchor=(0, 0.8))
    plt.gca().add_artist(leg2)
else:
    leg2 = plt.legend(handles=[plt.plot([], [], linestyle='-', color='black', label=r'2$\sigma$')[0], 
                            plt.plot([], [], linestyle='--', color='black', label=r'3$\sigma$')[0]],
                            loc='upper left')
    plt.gca().add_artist(leg2)


plotText(f_values, dB_values, L_array)
if direction == 'below':
    plotPredictionLine(f=0.32)
else:
    plotPredictionLine(dB=0, f=0.22, y_loc=5*10**3, color='blue')
if not save_suffix == 'Unscaled' and direction == 'below':
    plt.ylim((0.1, 200))
else:
    plt.ylim((3*10**3, 1.5*10**5))
plt.xlim((min(L_array), max(L_array)))
plt.yscale('log')
if direction == 'below':
    plt.xlabel(r'L = f * $10^{-dB/20}$')
else:
    plt.xlabel('f')
if 'Unscaled' in save_suffix:
    plt.ylabel('Evts/Stn/Yr')
else:
    plt.ylabel('Evts/Yr')
plt.savefig(f'{mainFolder}/{saveFolder}/LossRunningLinesDipole{save_suffix}.png')
plt.clf()

for depth, area in depths_area:
    c = next(color)
    plt.plot(L_array, E_rate[depth][lpda_triggers[0]], color=c, linestyle='--', label=f'{depth}m LPDA 3.8sig')
    plt.plot(L_array, E_rate[depth][lpda_triggers[1]], color=c, linestyle='-', label=f'{depth}m LPDA 2sig')

    plt.plot(L_array, E_rate[depth][dipole_triggers[0]], color=c, linestyle='-.', label=f'{depth}m Dip 3sig')
    plt.plot(L_array, E_rate[depth][dipole_triggers[1]], color=c, linestyle=':', label=f'{depth}m Dip 2sig')

plotText(f_values, dB_values, L_array)
if not save_suffix == 'Unscaled':
    plt.ylim((0.1, 200))
plt.xlim((min(L_array), max(L_array)))
plt.yscale('log')
plt.xlabel(r'L = f * $10^{-dB/20}$')
plt.ylabel('Evts/Stn/Yr')
plt.legend(loc='upper left')
plt.savefig(f'{mainFolder}/{saveFolder}/LossRunningLines{save_suffix}.png')
plt.clf()



for depth, area in depths_area:
    c = next(color)
    plt.plot(L_array, E_rate[depth][lpda_triggers[0]], color=c, linestyle='--', label=f'{depth}m')
    plt.plot(L_array, E_rate[depth][lpda_triggers[1]], color=c, linestyle='-')

#    plt.fill_between(L_array, E_rate[depth][lpda_triggers[0]], E_rate[depth][lpda_triggers[1]], color=c, linestyle='-', label=f'{depth}m LPDA', alpha=0.3)
    plt.fill_between(L_array, E_rate[depth][dipole_triggers[0]], E_rate[depth][dipole_triggers[1]], color=c, linestyle='-', alpha=1)

plotText(f_values, dB_values, L_array)
if not save_suffix == 'Unscaled':
    plt.ylim((0.1, 200))
plt.xlim((min(L_array), max(L_array)))
plt.yscale('log')
plt.xlabel(r'L = f * $10^{-dB/20}$')
plt.ylabel('Evts/Year')
plt.legend(loc='upper left')
plt.savefig(f'{mainFolder}/{saveFolder}/LossRunningCombo{save_suffix}.png')
plt.clf()


print(f'Done!')

