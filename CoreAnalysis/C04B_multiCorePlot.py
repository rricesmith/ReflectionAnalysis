from NuRadioReco.utilities import units
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import coreDataObjects as CDO
import pickle
import CoreAnalysis.C00_coreAnalysisUtils as CDO_util
import itertools
plt.style.use('plotsStyle.mplstyle')
import CoreAnalysis.C00_plotCoreSpectra as C00_plotCoreSpectra



save_location = 'plots/CoreAnalysis/SP/CombinedPlots/'
save_prefix = 'May29_MultiLayer_3.8sigma_40dB_0.25f_wMB'
save_location = save_location + save_prefix
#files_comments = [['data/CoreDataObjects/May23/SP_Cores/80to150MHz_2sigma_CoreDataObjects_LPDA_below_300mRefl_SP_1R_0.3f_40.0dB_1.7km_1000cores.pkl', 'Future, 300m'],
#                  ['data/CoreDataObjects/May23/SP_Cores/80to150MHz_2sigma_CoreDataObjects_LPDA_below_500mRefl_SP_1R_0.3f_40.0dB_1.7km_1000cores.pkl', 'Future, 500m'],
#                  ['data/CoreDataObjects/May23/SP_Cores/80to150MHz_2sigma_CoreDataObjects_LPDA_below_800mRefl_SP_1R_0.3f_40.0dB_1.7km_1000cores.pkl', 'Future, 800m'],
#                  ['data/CoreDataObjects/May23/SP_Cores/80to500MHZ_4.4sigma_CoreDataObjects_LPDA_below_300mRefl_SP_1R_0.3f_40.0dB_1.7km_1000cores.pkl', 'Current, 300m']
#                    ]
"""
files_comments = [  ['data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_2sigma_below_300mRefl_SP_1R_0.2f_40.0dB_1.7km_1000cores.pkl', f'300m 2sigma 0.2f 40dB'],
                    ['data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_2sigma_below_300mRefl_SP_1R_0.4f_40.0dB_1.7km_1000cores.pkl', f'300m 2sigma 0.4f 40dB'],
                    ['data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_2sigma_below_300mRefl_SP_1R_0.2f_50.0dB_1.7km_1000cores.pkl', f'300m 2sigma 0.2f 50dB'],
                    ['data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_2sigma_below_300mRefl_SP_1R_0.4f_50.0dB_1.7km_1000cores.pkl', f'300m 2sigma 0.4f 50dB'],
                    ['data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_3.8sigma_below_300mRefl_SP_1R_0.2f_40.0dB_1.7km_1000cores.pkl', f'300m 3.8sigma 0.2f 40dB'],
                    ['data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_3.8sigma_below_300mRefl_SP_1R_0.4f_40.0dB_1.7km_1000cores.pkl', f'300m 3.8sigma 0.4f 40dB'],
                    ['data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_3.8sigma_below_300mRefl_SP_1R_0.4f_50.0dB_1.7km_1000cores.pkl', f'300m 3.8sigma 0.2f 50dB'],
                    ['data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_3.8sigma_below_300mRefl_SP_1R_0.4f_50.0dB_1.7km_1000cores.pkl', f'300m 3.8sigma 0.4f 50dB'],
                    ]
"""
files_comments = [  ['data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_3.8sigma_below_300mRefl_SP_1R_0.25f_40.0dB_1.7km_1000cores.pkl', f'300m'],
                    ['data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_3.8sigma_below_500mRefl_SP_1R_0.25f_40.0dB_1.7km_1000cores.pkl', f'500m'],
                    ['data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_3.8sigma_below_800mRefl_SP_1R_0.25f_40.0dB_1.7km_1000cores.pkl', f'800m'],
                    ['data/CoreDataObjects/Gen2_AllTriggers_DipFix_CoreDataObjects_LPDA_2of4_3.8sigma_below_576mRefl_MB_1R_0.05f_0.0dB_1.7km_1000cores.pkl', f'Moores Bay']
                    ]

cores_comments = []            

print(f'1')

for file, legend_comment in files_comments:
    with open(file, 'rb') as fin:
        CoreObjectsList = pickle.load(fin)
        fin.close()
    cores_comments.append([CoreObjectsList, legend_comment])



if __name__ == '__main__':
    color = itertools.cycle(('blue', 'black', 'red', 'green', 'purple', 'orange'))
    print(f'2')

    #Making plot of multiple layer event rate per core energy
    for CoreObjectList, legend_comment in cores_comments:
        c = next(color)
        p1 = CDO_util.plotCoreEnergyEventRate(CoreObjectList, error=False, legend_comment=legend_comment, type='TA')
        p2 = CDO_util.plotCoreEnergyEventRate(CoreObjectList, error=False, legend_comment=legend_comment, type='Auger')
        p1[0].set_color(c)
        p2[0].set_color(c)
        p1[0].set_linestyle('-')
        p2[0].set_linestyle('--')

    color = itertools.cycle(('blue', 'black', 'red', 'green', 'purple', 'orange'))
    legend1 = plt.legend(handles=[plt.plot([],[], color=next(color), label=comment)[0] for coreObjectList, comment in cores_comments], 
                         loc='upper left', prop={'size':18})
    plt.gca().add_artist(legend1)
    legend2 = plt.legend(handles=[plt.plot([], [], color='black', linestyle='-', label='TA')[0],
                                 plt.plot([], [], color='black', linestyle='--', label='Auger')[0]],
                                 loc='lower left', prop={'size':18})
    plt.gca().add_artist(legend2)
#    plotCoreSpectra.plotCoreCRFluxBackground(text=True)
#    plt.xlim((18.5, 20))
#    plt.ylim(bottom=10**-4)
    plt.yscale('log')
    plt.savefig(f'{save_location}_Core_Energy_Erate.png')
    plt.clf()

    #same as above but for CR energy
    color = itertools.cycle(('blue', 'black', 'red', 'green', 'purple', 'orange'))
    print(f'3')
    for CoreObjectList, legend_comment in cores_comments:
        c = next(color)
        p1 = CDO_util.plotCREnergyEventRate(CoreObjectList, error=False, legend_comment=legend_comment, type='TA')
        p1[0].set_color(c)
        p1[0].set_linestyle('-')

        #For custom plot, remove
        y_data = np.array(p1[0].get_ydata())
        ind = 0
        for iD, data in enumerate(y_data):
            if data < 2*10**-6:
                ind = iD
        if ind == len(y_data) or legend_comment == 'Moores Bay':
            ind = 0
        y_data[:ind] = 10**-7
        p1[0].set_ydata(y_data)
        plt.draw()


        p2 = CDO_util.plotCREnergyEventRate(CoreObjectList, error=False, legend_comment=legend_comment, type='Auger')
        p2[0].set_color(c)
        p2[0].set_linestyle('--')

        #For custom plot, remove
        y_data = np.array(p2[0].get_ydata())
        ind = 0
        for iD, data in enumerate(y_data):
            if data < 2*10**-6:
                ind = iD
        if ind == len(y_data) or legend_comment == 'Moores Bay':
            ind = 0
        y_data[:ind] = 10**-7
        p2[0].set_ydata(y_data)
        plt.draw()


    color = itertools.cycle(('blue', 'black', 'red', 'green', 'purple', 'orange'))
    legend1 = plt.legend(handles=[plt.plot([],[], color=next(color), label=comment)[0] for coreObjectList, comment in cores_comments], 
                         loc='upper left', prop={'size':18})
    plt.gca().add_artist(legend1)
    legend2 = plt.legend(handles=[plt.plot([], [], color='black', linestyle='-', label='TA')[0],
                                 plt.plot([], [], color='black', linestyle='--', label='Auger')[0]],
                                 loc='lower left', prop={'size':18})
    plt.gca().add_artist(legend2)
#    plotCoreSpectra.plotCoreCRFluxBackground(text=True)
    plt.xlim((18.5, 20))
    plt.ylim(bottom=10**-6)
    plt.yscale('log')
    plt.savefig(f'{save_location}_CR_Energy_Erate.png')
    plt.clf()

    print(f'Done!')
    print(f'saved to {save_location}')
