import os
import numpy as np
from numpy import save, load
import keras
import time
#can do tensorflow.keras if they have tensorflow
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import random
import datetime
import pandas as pd
from glob import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units, fft
import argparse

def plotTrace(traces, title, saveLoc, sampling_rate=2, show=False):
    #Sampling rate should be in GHz

    x = np.linspace(1, int(256 / sampling_rate), num=256)
    x_freq = np.fft.rfftfreq(len(x), d=(1 / sampling_rate*units.GHz)) / units.MHz

#    print(f'shape traces {np.shape(traces)}')

    fig, axs = plt.subplots(nrows=4, ncols=2, sharex=False)
    for chID, trace in enumerate(traces):
        trace = trace.reshape(len(trace))
        axs[chID][0].plot(x, trace)
#        print(f'shape trace {np.shape(trace)}')
#        print(f'shape fft trace {np.shape(np.abs(fft.time2freq(trace, sampling_rate*units.GHz)))}')
#        print(f'trace {trace}')
#        print(f'fft {np.abs(fft.time2freq(trace, sampling_rate*units.GHz))}')
        axs[chID][1].plot(x_freq, np.abs(fft.time2freq(trace, sampling_rate*units.GHz)))

    axs[3][0].set_xlabel('time [ns]',fontsize=18)
    axs[3][1].set_xlabel('Frequency [MHz]',fontsize=18)

    for chID, trace in enumerate(traces):
        axs[chID][0].set_ylabel(f'ch{chID}',labelpad=10,rotation=0,fontsize=13)
        # axs[i].set_ylim(-250,250)
        axs[chID][0].set_xlim(-3,260 / sampling_rate)
        axs[chID][1].set_xlim(-3, 500)
        axs[chID][0].tick_params(labelsize=13)
        axs[chID][1].tick_params(labelsize=13)
    axs[0][0].tick_params(labelsize=13)
    axs[0][1].tick_params(labelsize=13)
    axs[0][0].set_ylabel(f'ch{0}',labelpad=3,rotation=0,fontsize=13)
    axs[chID][0].set_xlim(-3,260 / sampling_rate)
    axs[chID][1].set_xlim(-3, 500)

    fig.text(0.03, 0.5, 'voltage [V]', ha='center', va='center', rotation='vertical',fontsize=18)
    plt.xticks(size=13)
    plt.suptitle(title)

    if show:
        plt.show()
    else:
        plt.savefig(saveLoc, format='png')
    plt.clf()
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
    ax5.hist(times[~highMask], bins=bins, label='<.95', stacked=False)
    n, b, _ = ax5.hist(times[highMask], bins=bins, label='>.95', stacked=False)


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

def getTimeClusterMask(times, vals, cutVal=10**3):
    #Pass in times array, and make a cut on a day-by-day basis depending upon how many triggers are on a given day
    timeMax = datetime.datetime(2019, 5, 1)
    timeMin = datetime.datetime(2014, 12, 1)
    # Bins of 1 day width over range
    bins = np.arange(timeMin.date(), timeMax.date() + datetime.timedelta(days=1), datetime.timedelta(days=1))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=4)    #Originally 5 and 10
    plt.gca().xaxis.set_major_locator(locator)
    f, ax = plt.subplots(1, 1, sharey=True, facecolor='w')
    nTimes, b, _ = ax.hist(times[vals > 0.95], bins=bins)
    plt.clf()


    badTimes = []
    livetime = 0
    badStart = -1
    badEnd = 0
    for iN, n in enumerate(nTimes):
        if n > cutVal:
            if badStart == -1:
                badStart = iN
                badEnd = iN + 1
            elif badEnd == iN:
                badEnd = iN + 1
            else:
                badTimes.append([badStart, badEnd])
                badStart = iN
                badEnd = iN + 1
#        else:
#            print(f'good times')
#            print(bins[iN])
 #           print(bins[iN+1])
        if n > 0:
            livetime += 1



    timesMask = np.ones_like(times, dtype=bool)
    for iT, time in enumerate(times):
        if time < timeMin:
            timesMask[iT] = False
            continue
        for bad in badTimes:
            if bins[bad[0]] < time < bins[bad[1]]:
 #               print(f'bad time')
 #               print(bins[bad[0]])
 #               print(time)
 #               print(bins[bad[1]])
                timesMask[iT] = False
                continue

    goodDays = livetime - len(badTimes)
    print(f'Livetime {livetime} days and good days {goodDays}')
    print(f'bad times {badTimes}')
    print(f'len times sent in and vals {len(times)} {len(vals)}')
    if livetime == 0:
        livetime = 1
    eff = goodDays / livetime
    print(f'Total efficiency {eff}')
#    quit()
 #   print(f'eff {eff}')
 #   quit()

    return timesMask, eff, livetime, goodDays

def findMaxAmpCut(max_amps, eff=0.95):
    sorted = max_amps
    sorted.sort()
    length = len(max_amps)

    return sorted[ int(length * eff)]

def getMaxAmpMask(max_amps_RCR, max_amps_Data, eff=0.95):
    cutVal = findMaxAmpCut(max_amps_RCR, eff)

    ampMask = np.ones_like(max_amps_Data, dtype=bool)
    for iA, amp in enumerate(max_amps_Data):
        if amp > cutVal:
            ampMask[iA] = False

    return ampMask, cutVal

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
    station = args.station

    if station == 14 or station == 17 or station == 19 or station == 30:
        amp = '200s'
    elif station == 13 or station == 15 or station ==18:
        amp = '100s'

    # input path and file names and station ID.
    round = '3rdpass'
    path = f'DeepLearning/data/{round}/'
    filtered = True
#    station = 17  # Change this value to match the station you are working with
    plotPath = f'DeepLearning/plots/Station_{station}/'

    # Change this value to control how many times the simulation file is used
    simulation_multiplier = 1  

    prefix = ''
    if filtered:
        prefix = 'Filtered'

    # Get a list of all the Noise files
    #Noise_files = glob(os.path.join(path, f"Station{station}_Data_*_part*.npy"))
    Noise_files = glob(os.path.join(path, f"{prefix}Station{station}_Data_*_part*.npy"))
    # Get a list of all the DateTime files
    #DateTime_files = glob(os.path.join(path, f"DateTime_Station{station}_Data_*_part*.npy"))
    DateTime_files = glob(os.path.join(path, f"DateTime_{prefix}Station{station}_Data_*_part*.npy"))

    # Create generators
    noise_generator = data_generator(Noise_files)
    datetime_generator = data_generator(DateTime_files)

    # Load the model
    #model = keras.models.load_model(f'DeepLearning/h5_models/{round}_trained_CNN_2l-20-4-10-10-1-10_do0.5_fltn_sigm_valloss_p4_measNoise0-20k_0-5ksigNU-Scaled_shuff_monitortraining_0_{simulation_multiplier}.h5')
    model = keras.models.load_model(f'DeepLearning/h5_models/{prefix}_{amp}_{round}_trained_CNN_2l-20-4-10-10-1-10_do0.5_fltn_sigm_valloss_p4_measNoise0-20k_0-5ksigNU-Scaled_shuff_monitortraining_0_{simulation_multiplier}.h5')


    # Collectors for plot values
    all_max_amps = []
    all_prob_Noise = []
    all_datetimes = []

    # Predict in chunks and process the data immediately to free memory
    for datetime_chunk, noise_chunk in zip(datetime_generator, noise_generator):
        noise_chunk = np.reshape(noise_chunk, (noise_chunk.shape[0], noise_chunk.shape[1], noise_chunk.shape[2], 1))
        prob_Noise = model.predict(noise_chunk)
        datetime_chunk = np.vectorize(datetime.datetime.fromtimestamp)(datetime_chunk)

        # Process the prediction immediately
        prob_Noise = 1 - prob_Noise
        plotHighOutput = False
        if plotHighOutput == True:
            for iE, Noi in enumerate(noise_chunk):
                output = prob_Noise[iE][0]
                if output > 0.95:
                    print(f'output {output}')
                    plotTrace(Noi, f"Noise {iE}, Output {output:.2f}, " + datetime_chunk[iE].strftime("%m-%d-%Y, %H:%M:%S"),
                            f"{plotPath}/Noise/{prefix}Noise_{iE}_Output_{output:.2f}_Station{station}.png")

        goldenSearch = False
        if goldenSearch and station == 19:
            for iE, Noi in enumerate(noise_chunk):
                output = prob_Noise[iE][0]
    #            if output > 0.95 and datetime.datetime(2017, 3, 27) < datetime_chunk[iE] < datetime.datetime(2017, 3, 28):
                if output > 0.95 and datetime.datetime(2017, 3, 20, 1) < datetime_chunk[iE] < datetime.datetime(2017, 4, 3, 1):
                    plotTrace(Noi, f"Noise {iE}, Output {output:.2f}, " + datetime_chunk[iE].strftime("%m-%d-%Y, %H:%M:%S"),
                            f"{plotPath}/GoldenDay/GoldenEvent_{prefix}_{iE}_Output_{output:.2f}_Station{station}.png")
                    


        max_amps = np.zeros(len(prob_Noise))
        for iC, trace in enumerate(noise_chunk):
            max_amps[iC] = np.max(trace)

        # Add max_amps and prob_Noise to collectors
        all_prob_Noise.extend(prob_Noise.flatten())
        all_max_amps.extend(max_amps.tolist())
        all_datetimes.extend(datetime_chunk.tolist())

        # Clear memory
        del noise_chunk

    all_prob_Noise = np.array(all_prob_Noise)
    all_max_amps = np.array(all_max_amps)
    all_datetimes = np.array(all_datetimes)


    # Load and predict the RCR data
    RCR_files = glob(os.path.join(path, f"{prefix}SimRCR_{amp}*.npy"))
    RCR_files = []
    print(f'checking path')
    for filename in os.listdir(path):
        print(f'checking filename {filename}')
        if f'{prefix}SimRCR_{amp}' in filename:
            print(f'found filename')
            RCR_files.append(os.path.join(path, filename))
    print(f'rcr files {RCR_files} in path {path}')

    RCR = np.empty((0, 4, 256))
    for file in RCR_files:
        RCR_data = np.load(file)[5000:, 0:4]
        RCR_data = np.vstack([RCR_data] * simulation_multiplier)  # Stack the data multiple times
        RCR = np.concatenate((RCR, RCR_data))

    RCR = np.reshape(RCR, (RCR.shape[0], RCR.shape[1], RCR.shape[2], 1))
    print(f'shape RCR {RCR.shape}')
    prob_RCR = model.predict(RCR)
    prob_RCR = 1 - prob_RCR

    # Process the RCR prediction
    plotRandom = False
    if plotRandom == True:
        for iE, rcr in enumerate(RCR):
            if iE % 1000 == 0:
    #            output = 1 - prob_RCR[iE][0]
                output = prob_RCR[iE][0]
                if output > 0.95:
                    plotTrace(rcr, f"RCR {iE}, Output {output:.2f}",f"{plotPath}/RCR/{prefix}RCR_{iE}_Output_{output:.2f}_Station{station}.png")

    max_amps_RCR = np.zeros(len(prob_RCR))
    for iC, trace in enumerate(RCR):
        max_amps_RCR[iC] = np.max(trace)

    #prob_RCR = 1 - prob_RCR
    #all_prob_Noise = 1 - all_prob_Noise

    plt.scatter(all_max_amps, all_prob_Noise, color='blue', label='Noise')
    plt.scatter(max_amps_RCR, prob_RCR, color='orange', label='SimRCR')
    plt.ylabel('Network Output - 1 = RCR')
    plt.xlabel('Max amp of channels')
    plt.legend()
    plt.title(f'Station {station} Training')
    plt.grid(True)
    plt.savefig(f'{plotPath}/{prefix}MaxAmpsOutputStation{station}_M{simulation_multiplier}.png', format='png')
    plt.clf()


    #Make a histogram of the max amps for data/RCRs
    #Separate potentials vs non-potentials
    eff = 0.95
    ampMask, ampCutVal = getMaxAmpMask(max_amps_RCR, all_max_amps, eff=eff)

    bins = np.linspace(0, max(max_amps_RCR), num=25)
    plt.axvline(x=findMaxAmpCut(max_amps_RCR), label='95% RCR Efficiency', linestyle='--')
    plt.axvline(x=findMaxAmpCut(max_amps_RCR, eff=0.9), label='90% RCR Efficiency', linestyle='-.')
    plt.axvline(x=findMaxAmpCut(max_amps_RCR, eff=0.8), label='80% RCR Efficiency', linestyle=':')
    plt.hist(all_max_amps[all_prob_Noise > 0.95], bins=bins, label='>0.95 Data', fill=None, edgecolor='red', histtype='step', density=True)
    plt.hist(all_max_amps[all_prob_Noise < 0.95], bins=bins, label='<0.95 Data', fill=None, edgecolor='blue', histtype='step', density=True)
    plt.hist(max_amps_RCR, bins=bins, label='RCR', fill=None, edgecolor='green', histtype='step', density=True)
    plt.xlabel('Max amp')
    #plt.yscale('log')
    plt.legend()
    plt.savefig(f'{plotPath}/{prefix}MaxAmpHistogram_Station{station}_M{simulation_multiplier}.png', format='png')
    plt.clf()

    haveTimes = True
    if haveTimes:
        timeCut = 1
        timeMask, timeEff, livetime, goodDays = getTimeClusterMask(all_datetimes, all_prob_Noise, cutVal=timeCut)
        print(f'shape datetimes {np.shape(all_datetimes)} and prob noise {np.shape(all_prob_Noise)}')
        plotTimeStrip(all_datetimes, all_prob_Noise, f'Station {station}, {livetime} livetime, {goodDays} pass cut, {eff*100:.0f}% Eff', saveLoc=plotPath+f'TimeStrips/{prefix}TimeStrip')
        plotTimeHist(all_datetimes, all_prob_Noise, f'Station {station}, {livetime} livetime, {goodDays} pass cut, {eff*100:.0f}% Eff', saveLoc=plotPath+f'{prefix}HistTimeStrip', timeCut=timeCut)


    fig = plt.figure()
    ax = fig.add_subplot(111)

    dense_val = False
    ax.hist(all_prob_Noise, bins=20, range=(0, 1), histtype='step', color='red', linestyle='solid', label=f'Station_{station} Data', density=dense_val)
    ax.hist(prob_RCR, bins=20, range=(0, 1), histtype='step',color='blue', linestyle='solid',label='SimRCR',density = dense_val)

    plt.xlabel('network output', fontsize=18)
    plt.ylabel('events', fontsize=18)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.yscale('log')
    plt.ylim(bottom=1)
    plt.title(f'Station {station}')
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles]
    plt.legend(loc='upper center', handles=new_handles, labels=labels, fontsize=18)
    plt.savefig(plotPath+f'/{prefix}Station{station}AnalysisOutput_M{simulation_multiplier}.png', format='png')
    plt.clf()


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(all_prob_Noise, bins=20, range=(0, 1), histtype='step', color='red', linestyle='solid', label='Station19 Output', density=dense_val, alpha=0.5)
    ax.hist(all_prob_Noise[ampMask], bins=20, range=(0, 1), histtype='step', color='blue', linestyle='solid', label=f'Amp Cut, {eff}% Eff', density=dense_val, alpha=0.5)
    ax.hist(all_prob_Noise[timeMask], bins=20, range=(0, 1), histtype='step', color='green', linestyle='solid', label=f'Time Cut, {timeEff*100:.0f}% Eff', density=dense_val, alpha=0.5)
    duoMask = np.all([ampMask, timeMask], axis=0)
    ax.hist(all_prob_Noise[duoMask], bins=20, range=(0, 1), histtype='step', color='orange', linestyle='solid', label=f'Time & Amp Cut, {timeEff * eff * 100:.0f}% Eff', density=dense_val, alpha=0.5)

    plt.xlabel('network output', fontsize=18)
    plt.ylabel('events', fontsize=18)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.yscale('log')
    plt.ylim(bottom=1)
    plt.title(f'Station {station}')
#    plt.legend(loc='upper center', handles=new_handles, labels=labels, fontsize=18)
    plt.legend()
    print(f'saving to ' + plotPath+f'/{prefix}_TimeAmpCuts_Station{station}AnalysisOutput_M{simulation_multiplier}.png')
    plt.savefig(plotPath+f'/{prefix}_TimeAmpCuts_Station{station}AnalysisOutput_M{simulation_multiplier}.png', format='png')
    plt.clf()

    timeCutTimes = all_datetimes[timeMask]
    ampCutTimes = all_datetimes[ampMask]
    deepLearnCutTimes = all_datetimes[all_prob_Noise > 0.95]
    allCutTimes = all_datetimes[np.all([timeMask, ampMask, all_prob_Noise > 0.95], axis=0)]
    np.save(f'DeepLearning/data/3rdpass/timesPassedCuts_{prefix}Station{station}_TimeCut_{timeCut}perDay_Amp{eff}%.npy', [timeCutTimes, ampCutTimes, deepLearnCutTimes, allCutTimes])
    np.save(f'DeepLearning/data/3rdpass/modelOutputs_{prefix}Station{station}.npy', [all_prob_Noise, all_max_amps, timeMask, ampMask])
