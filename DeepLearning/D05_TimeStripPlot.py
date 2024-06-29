from DeepLearning.D03_processData import plotTimeStrip, getTimeClusterMask
import numpy as np
from glob import glob
import os
from icecream import ic
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plotTimeHist(times, vals, title, saveLoc, timeCut = 1, yearStart=2014):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=4)    #Originally 5 and 10
    plt.gca().xaxis.set_major_locator(locator)
#    plt.locator_params(axis='x', nbins=3)
#    plt.scatter(times,vals, alpha=0.5)

    timeMin = datetime.datetime(yearStart, 11, 1)
    timeMax = datetime.datetime(2019, 6, 1)
    delta_years = int(2019-yearStart)
    # Bins of 1 day width over range
    delta = (timeMax - timeMin).days
    bins = np.arange(timeMin.date(), timeMax.date() + datetime.timedelta(days=1), datetime.timedelta(days=1))
    print(f'bins 0, 1, and last {bins[0]}, {bins[1]}, {bins[-1]}')

    highMask = vals > 0.95
    fig, axs = plt.subplots(1, delta_years, sharey=True, facecolor='w')
#    plt.hist(times[highMask], bins=bins, label='>0.95', stacked=True)    
#    plt.hist(times[~highMask], bins=bins, label='<0.95', stacked=True)    
    for iE, ax in enumerate(axs):
        if iE < (len(axs)-1):
            ax.hist(times[~highMask], bins=bins, stacked=False)
            n, b, _ = ax.hist(times[highMask], bins=bins, stacked=False)
        else:
            ax.hist(times[~highMask], bins=bins, stacked=False, label='Failed ML Cut')
            n, b, _ = ax.hist(times[highMask], bins=bins, stacked=False, label='Passed ML Cut')
    # ax2.hist(times[~highMask], bins=bins, stacked=False)
    # ax2.hist(times[highMask], bins=bins, stacked=False)
    # ax3.hist(times[~highMask], bins=bins, stacked=False)
    # ax3.hist(times[highMask], bins=bins, stacked=False)
    # ax4.hist(times[~highMask], bins=bins, stacked=False)
    # ax4.hist(times[highMask], bins=bins, stacked=False)
    # ax5.hist(times[~highMask], bins=bins, label='<.95', stacked=False)
    # n, b, _ = ax5.hist(times[highMask], bins=bins, label='>.95', stacked=False)


    for iN, num in enumerate(n):
        if num > 10**2:
            print(f'{num} in bin edges {bins[iN]} and {bins[iN+1]}')
#            print(f'printing in stf' + bins[iN].strftime("%m/%d/%Y") + ' to ' + bins[iN+1].strftime("%m/%d/%Y") )
#            print(f'{num} events in days ' + datetime.datetime.fromtimestamp(bins[iN]).strftime("%m/%d/%Y") + ' - ' + datetime.datetime.fromtimestamp(bins[iN+1]).strftime("%m/%d/%Y") )

    plt.gcf().autofmt_xdate()
#Lower limit set to install date of earliest station to remove error/bad times
#Need to remove to see the effect of bad datetimes

    """
    years_plot = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for year in years_plot:
        for month in months:
            if not month == 12:
                plt.xlim(left=datetime.datetime(year, month, 1), right=datetime.datetime(year, month+1, 1)-datetime.timedelta(days=1))   
            else:
                plt.xlim(left=datetime.datetime(year, month, 1), right=datetime.datetime(year+1, 1, 1)-datetime.timedelta(days=1))   
            plt.title(title + f' {year}-{month}')
            plt.savefig(saveLoc + f'{year}_{month}.png', format='png')
    """

#    plt.xlim(left=timeMin, right=timeMax)
    for iE, ax in enumerate(axs):
        ax.set_xlim(left=datetime.datetime(yearStart+iE, 11, 1), right=datetime.datetime(yearStart+1+iE, 6, 1))
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    axs[0].spines['left'].set_visible(True)
    axs[-1].spines['right'].set_visible(True)
    # ax1.set_xlim(left=timeMin, right=datetime.datetime(2015, 6, 1))
    # ax2.set_xlim(left=datetime.datetime(2015, 11, 1), right=datetime.datetime(2016, 6, 1))
    # ax3.set_xlim(left=datetime.datetime(2016, 11, 1), right=datetime.datetime(2017, 6, 1))
    # ax4.set_xlim(left=datetime.datetime(2017, 11, 1), right=datetime.datetime(2018, 6, 1))
    # ax5.set_xlim(left=datetime.datetime(2018, 11, 1), right=timeMax)

    # ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax3.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax4.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax5.xaxis.set_major_locator(plt.MaxNLocator(3))

    # ax1.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax3.spines['left'].set_visible(False)
    # ax3.spines['right'].set_visible(False)
    # ax4.spines['left'].set_visible(False)
    # ax4.spines['right'].set_visible(False)
    # ax5.spines['left'].set_visible(False)

    d = 0.015
    kwargs = dict(transform=axs[0].transAxes, color='k', clip_on=False)
    for iE, ax in enumerate(axs):
        kwargs.update(transform=ax.transAxes)
        ax.plot((1-d, 1+d), (-d, +d), **kwargs)
        ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
        if not (iE==0 or iE ==len(axs)-1):
            ax.plot((-d, +d), (-d, +d), **kwargs)
            ax.plot((-d, +d), (1-d, 1+d), **kwargs)

    # kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    # ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    # ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    # kwargs.update(transform=ax2.transAxes)
    # ax2.plot((1-d, 1+d), (-d, +d), **kwargs)
    # ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    # ax2.plot((-d, +d), (-d, +d), **kwargs)
    # ax2.plot((-d, +d), (1-d, 1+d), **kwargs)

    # kwargs.update(transform=ax3.transAxes)
    # ax3.plot((1-d, 1+d), (-d, +d), **kwargs)
    # ax3.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    # ax3.plot((-d, +d), (-d, +d), **kwargs)
    # ax3.plot((-d, +d), (1-d, 1+d), **kwargs)

    # kwargs.update(transform=ax4.transAxes)
    # ax4.plot((1-d, 1+d), (-d, +d), **kwargs)
    # ax4.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    # ax4.plot((-d, +d), (-d, +d), **kwargs)
    # ax4.plot((-d, +d), (1-d, 1+d), **kwargs)

    # kwargs.update(transform=ax5.transAxes)
    # ax5.plot((-d, +d), (-d, +d), **kwargs)
    # ax5.plot((-d, +d), (1-d, 1+d), **kwargs)


    plt.yscale('log')
    plt.legend()
    # plt.title(title + f' Hist')
    fig.suptitle(title)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(saveLoc + f'HistStrip.png', format='png')
    plt.clf()


    fig, ax = plt.subplots(1, 1, sharey=True, facecolor='w')
    ax.hist(times[~highMask], bins=bins, stacked=False, label='Failed ML Cut')
    n, b, _ = ax.hist(times[highMask], bins=bins, stacked=False, label='Passed ML Cut')
    ax.set_xlim(left=datetime.datetime(2015, 12, 1), right=datetime.datetime(2016, 3, 1))
    plt.gcf().autofmt_xdate()
    plt.yscale('log')
    plt.legend()
    plt.savefig(saveLoc + f'SingleWeekHistStrip.png', format='png')

    ax.axhline(1, color='black', linestyle='--')
    plt.savefig(saveLoc + f'SingleWeekCutHistStrip.png', format='png')

    plt.clf()


    return

prefix = 'Filtered'
station = 15
forced = False
plotPath = f'plots/DeepLearning/4.19.24/'
# round = '3rdpass'
round = '4thpass'
path = f'DeepLearning/data/{round}/'

# DateTime_files = glob(os.path.join(path, f"DateTime_{prefix}Station{station}_Data_*_part*.npy"))
DateTime_files = glob(os.path.join(path, f"DateTime_{prefix}Station{station}_Data_forced{forced}_*_part*.npy"))


all_datetimes = []
for file in DateTime_files:
    all_datetimes.append(np.load(file))
all_datetimes = np.array(all_datetimes).flatten()
all_datetimes = np.vectorize(datetime.datetime.fromtimestamp)(all_datetimes)
all_prob_Noise, all_max_amps, timeMask, ampMask = np.load(f'DeepLearning/data/3rdpass/modelOutputs_{prefix}Station{station}.npy')

ic(all_datetimes.shape, all_prob_Noise.shape)

timeCut = 1
timeMask, timeEff, livetime, goodDays = getTimeClusterMask(all_datetimes, all_prob_Noise, cutVal=timeCut)
print(f'shape datetimes {np.shape(all_datetimes)} and prob noise {np.shape(all_prob_Noise)}')
# plotTimeStrip(all_datetimes, all_prob_Noise, f'Station {station}, {livetime} livetime, {goodDays} pass cut, {goodDays/livetime*100:.0f}% Eff', saveLoc=plotPath+f'TimeStrips/{prefix}TimeStrip')
plotTimeHist(all_datetimes, all_prob_Noise, f'Station {station}, {livetime} livetime, {goodDays} pass cut, {goodDays/livetime*100:.0f}% Eff', saveLoc=plotPath+f'{prefix}HistTimeStrip', timeCut=timeCut, yearStart=2015)

