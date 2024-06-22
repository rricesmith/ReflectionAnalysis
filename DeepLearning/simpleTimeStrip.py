import os
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
import datetime
import matplotlib.dates as mdates
from glob import glob

def plotTimeHist(times, vals, title, saveLoc, timeCut = 1):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=4)    #Originally 5 and 10
    plt.gca().xaxis.set_major_locator(locator)
#    plt.locator_params(axis='x', nbins=3)
#    plt.scatter(times,vals, alpha=0.5)

    timeMax = datetime.datetime(2019, 5, 1)
    timeMin = datetime.datetime(2014, 12, 1)
    # Bins of 1 day width over range
    delta = (timeMax - timeMin).days
    bins = np.arange(timeMin.date(), timeMax.date() + datetime.timedelta(days=1), datetime.timedelta(days=1))
    print(f'bins 0, 1, and last {bins[0]}, {bins[1]}, {bins[-1]}')

    highMask = vals > 0.95
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, facecolor='w')
#    plt.hist(times[highMask], bins=bins, label='>0.95', stacked=True)    
#    plt.hist(times[~highMask], bins=bins, label='<0.95', stacked=True)    
    ax1.hist(times[~highMask], bins=bins, stacked=False)
    ax1.hist(times[highMask], bins=bins, stacked=False)
    ax2.hist(times[~highMask], bins=bins, stacked=False)
    ax2.hist(times[highMask], bins=bins, stacked=False)
    ax3.hist(times[~highMask], bins=bins, stacked=False)
    ax3.hist(times[highMask], bins=bins, stacked=False)
    ax4.hist(times[~highMask], bins=bins, stacked=False)
    ax4.hist(times[highMask], bins=bins, stacked=False)
    ax5.hist(times[~highMask], bins=bins, label='w/forced', stacked=False)
    n, b, _ = ax5.hist(times[highMask], bins=bins, label='w/o forced', stacked=False)


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
    ax1.set_xlim(left=timeMin, right=datetime.datetime(2015, 6, 1))
    ax2.set_xlim(left=datetime.datetime(2015, 11, 1), right=datetime.datetime(2016, 6, 1))
    ax3.set_xlim(left=datetime.datetime(2016, 11, 1), right=datetime.datetime(2017, 6, 1))
    ax4.set_xlim(left=datetime.datetime(2017, 11, 1), right=datetime.datetime(2018, 6, 1))
    ax5.set_xlim(left=datetime.datetime(2018, 11, 1), right=timeMax)

    ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax3.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax4.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax5.xaxis.set_major_locator(plt.MaxNLocator(3))

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax5.spines['left'].set_visible(False)

    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax3.transAxes)
    ax3.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax3.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    ax3.plot((-d, +d), (-d, +d), **kwargs)
    ax3.plot((-d, +d), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax4.transAxes)
    ax4.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax4.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    ax4.plot((-d, +d), (-d, +d), **kwargs)
    ax4.plot((-d, +d), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax5.transAxes)
    ax5.plot((-d, +d), (-d, +d), **kwargs)
    ax5.plot((-d, +d), (1-d, 1+d), **kwargs)


    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.title(title + f' Hist')
    plt.savefig(saveLoc + f'HistStrip.png', format='png')


    plt.clf()
    
    return

path = f'DeepLearning/data/4thpass/'
prefix = 'Filtered'
station = 15
forced = False
new_DateTime_files = glob(os.path.join(path, f"DateTime_{prefix}Station{station}_Data_forced{forced}_*_part*.npy"))

vals = []

all_datetimes = []
for file in new_DateTime_files:
    all_datetimes.append(np.load(file))
all_datetimes = np.array(all_datetimes).flatten()
all_datetimes = np.vectorize(datetime.datetime.fromtimestamp)(all_datetimes)

all_vals = np.ones_like(all_datetimes)

path = f'DeepLearning/data/3rdpass/'
old_DateTime_files = glob(os.path.join(path, f"DateTime_{prefix}Station{station}_Data_*_part*.npy"))


old_datetimes = []
for file in old_DateTime_files:
    old_datetimes.append(np.load(file))
old_datetimes = np.array(old_datetimes).flatten()
old_datetimes = np.vectorize(datetime.datetime.fromtimestamp)(old_datetimes)
old_vals = np.zeros_like(old_datetimes)

all_datetimes = np.append(all_datetimes, old_datetimes)
all_vals = np.append(all_vals, old_vals)

# all_vals = new_vals
# all_vals.append(old_vals)

plotTimeHist(all_datetimes, all_vals, f'Station {station}', f'plots/DeepLearning/4.19.24/TimeStrip_ForcedvsUnforced.png') #1= new time, zero is old
