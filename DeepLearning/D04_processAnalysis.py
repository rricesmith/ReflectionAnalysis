import os
import numpy as np
import matplotlib.dates as mdates
import datetime
from glob import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units, fft

import argparse
from DeepLearning.D04B_reprocessNurPassingCut import loadTemplate, getMaxSNR, getMaxChi


def plotTrace(traces, title, saveLoc, sampling_rate=2, show=False):
    #Sampling rate should be in GHz

    num = len(traces[0])
    x = np.linspace(1, int(256 / sampling_rate), num=num)
    x_freq = np.fft.rfftfreq(len(x), d=(1 / sampling_rate*units.GHz)) / units.MHz

#    print(f'shape traces {np.shape(traces)}')

    fig, axs = plt.subplots(nrows=4, ncols=2, sharex=False)
    fmax = 0
    vmax = 0
    for chID, trace in enumerate(traces):
        trace = trace.reshape(len(trace))
        freqtrace = np.abs(fft.time2freq(trace, sampling_rate*units.GHz))
        axs[chID][0].plot(x, trace)
#        print(f'shape trace {np.shape(trace)}')
#        print(f'shape fft trace {np.shape(np.abs(fft.time2freq(trace, sampling_rate*units.GHz)))}')
#        print(f'trace {trace}')
#        print(f'fft {np.abs(fft.time2freq(trace, sampling_rate*units.GHz))}')
        axs[chID][1].plot(x_freq, freqtrace)
        fmax = max(fmax, max(freqtrace))
        vmax = max(vmax, max(trace))

    axs[3][0].set_xlabel('time [ns]',fontsize=18)
    axs[3][1].set_xlabel('Frequency [MHz]',fontsize=18)

    for chID, trace in enumerate(traces):
        axs[chID][0].set_ylabel(f'ch{chID}',labelpad=10,rotation=0,fontsize=13)
        # axs[i].set_ylim(-250,250)
        axs[chID][0].set_xlim(-3, 260 / sampling_rate)
        axs[chID][1].set_xlim(-3, 1000)
        axs[chID][0].tick_params(labelsize=13)
        axs[chID][1].tick_params(labelsize=13)

        axs[chID][0].set_ylim(-vmax * 1.1, vmax * 1.1)
        axs[chID][1].set_ylim(-0.05, fmax * 1.1)

    axs[0][0].tick_params(labelsize=13)
    axs[0][1].tick_params(labelsize=13)
    axs[0][0].set_ylabel(f'ch{0}',labelpad=3,rotation=0,fontsize=13)
    axs[chID][0].set_xlim(-3, 260 / sampling_rate)
    axs[chID][1].set_xlim(-3, 1000)



    fig.text(0.03, 0.5, 'voltage [V]', ha='center', va='center', rotation='vertical',fontsize=18)
    plt.xticks(size=13)
    plt.suptitle(title)

    if show:
        plt.show()
    else:
        plt.savefig(saveLoc, format='png')
    plt.clf()
    plt.close()
    return


def plotTimeStrip(times, vals, title, saveLoc):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    plt.gca().xaxis.set_major_locator(locator)
    plt.scatter(times,vals, alpha=0.5)
    plt.gcf().autofmt_xdate()
#Lower limit set to install date of earliest station to remove error/bad times
#Need to remove to see the effect of bad datetimes

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

    plt.xlim(left=datetime.datetime(2017, 3, 27), right=datetime.datetime(2017, 3, 28))   
    plt.title(title + f' 2017-3-27')
    plt.savefig(saveLoc + f'GoldenDay.png', format='png')

    plt.clf()
    
    return

def plotTimeHist(times, vals, title, saveLoc):
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
    ax1.hist(times[highMask], bins=bins, stacked=True)
    ax1.hist(times[~highMask], bins=bins, stacked=True)
    ax2.hist(times[highMask], bins=bins, stacked=True)
    ax2.hist(times[~highMask], bins=bins, stacked=True)
    ax3.hist(times[highMask], bins=bins, stacked=True)
    ax3.hist(times[~highMask], bins=bins, stacked=True)
    ax4.hist(times[highMask], bins=bins, stacked=True)
    ax4.hist(times[~highMask], bins=bins, stacked=True)
    n, b, _ = ax5.hist(times[highMask], bins=bins, label='>.95', stacked=True)
    ax5.hist(times[~highMask], bins=bins, label='<.95', stacked=True)

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
    plt.legend()
    plt.title(title + f' Hist')
    plt.savefig(saveLoc + f'HistStrip.png', format='png')


    plt.clf()
    
    return

def getTimeClusterMask(times, cutVal=10**3):
    #Pass in times array, and make a cut on a day-by-day basis depending upon how many triggers are on a given day
    return


def data_generator(files, chunk_size=50000):
    for file in files:
        # Load the noise data from the current file
        data_file = np.load(file)
        num_chunks = data_file.shape[0] // chunk_size
        if data_file.shape[0] % chunk_size:
            num_chunks += 1

        for chunk in range(num_chunks):
            start = chunk * chunk_size
            end = min(start + chunk_size, data_file.shape[0])

            # Yield the chunk from the data
            yield data_file[start:end]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Analysis on particular station')
    parser.add_argument('station', type=int, default=19, help='Station to run on')
    args = parser.parse_args()
    station_id = args.station

    round = '3rdpass'
    prefix = 'Filtered'
#    station_id = 17
    eventsPerDayCut = 1
    ampEff = 0.95

    timeCutTimes, ampCutTimes, deepLearnCutTimes, allCutTimes = np.load(f'DeepLearning/data/{round}/timesPassedCuts_{prefix}Station{station_id}_TimeCut_{eventsPerDayCut}perDay_Amp{ampEff}%.npy', allow_pickle=True)
    all_prob_Noise, all_max_amps, timeMask, ampMask = np.load(f'DeepLearning/data/{round}/modelOutputs_{prefix}Station{station_id}.npy')

    print(f'len time cut {len(allCutTimes)}')
#    for date in allCutTimes:
#        print(date.strftime("%m-%d-%Y, %H:%M:%S"))

    path = f'DeepLearning/data/{round}/'
    Noise_files = glob(os.path.join(path, f"{prefix}Station{station_id}_Data_*_part*.npy"))
    DateTime_files = glob(os.path.join(path, f"DateTime_{prefix}Station{station_id}_Data_*_part*.npy"))
    noise_generator = data_generator(Noise_files)
    datetime_generator = data_generator(DateTime_files)


    RCR_template = loadTemplate()
    PassingCut_SNRs = []
    PassingCut_RCR_Chi = []


    print(f'starting chunk')
    # Predict in chunks and process the data immediately to free memory
    for datetime_chunk, noise_chunk in zip(datetime_generator, noise_generator):

        noise_chunk = np.reshape(noise_chunk, (noise_chunk.shape[0], noise_chunk.shape[1], noise_chunk.shape[2]))
        datetime_chunk = np.vectorize(datetime.datetime.fromtimestamp)(datetime_chunk)
        print(f'going through')

        for iE, date in enumerate(datetime_chunk):
#            print(f'got a date')
#            print(date)

            #For some reason, lots of events in 2019, in march, pass cut even though there shouldn't be multiple per day
            #So we skip them
            if date > datetime.datetime(2019, 1, 1):
                continue
            for goodTime in allCutTimes:
#                print(date.date())
#                print(goodTime.date())
#                quit()
                if date == goodTime:
                    chi = getMaxChi(noise_chunk[iE], 2*units.GHz, RCR_template, 2*units.GHz)
                    SNR = getMaxSNR(noise_chunk[iE])

                    PassingCut_SNRs.append(SNR)
                    PassingCut_RCR_Chi.append(chi)

                    plotTrace(noise_chunk[iE], f'Stn19 DeepLearn & {eventsPerDayCut} Evt/Day Cut, ' + date.strftime("%m-%d-%Y, %H:%M:%S") + f', Chi {chi:.2f} SNR {SNR:.1f}',
                              f'DeepLearning/plots/Station_{station_id}/TracesPassingCuts/Stn{station_id}_Candidate_{eventsPerDayCut}PerDay_DeepLearn2Layer_{iE}_Chi{chi:.2f}_SNR{SNR:.1f}.png')
                    continue


    plt.scatter(PassingCut_SNRs, PassingCut_RCR_Chi, label=f'{len(PassingCut_RCR_Chi)} Events Passing All Cuts', facecolor='none', edgecolor='red')
    plt.xlim((3, 100))
    plt.ylim((0, 1))
    plt.xlabel('SNR')
    plt.ylabel('RCR Avg Chi Highest Parallel Channels')

    plt.xscale('log')
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.tick_params(axis='y', which='minor', left=True)
    plt.grid(which='both', axis='both')
    plt.title(f'Station {station_id} RCR Correlations Passing Cuts')
    plt.savefig(f'DeepLearning/plots/Station_{station_id}/ChiSNR_PassedCuts_Stnd{station_id}.png')
    plt.clf()
    plt.close()
