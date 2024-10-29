import datetime
from NuRadioReco.utilities import units
from NuRadioReco.modules import channelResampler as CchannelResampler
from NuRadioReco.modules.ARIANNA import hardwareResponseIncorporator as ChardwareResponseIncorporator
from NuRadioReco.modules import channelTimeWindow as cTWindow
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelStopFilter
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.triggerTimeAdjuster
import NuRadioReco.modules.channelLengthAdjuster
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.modules.io import NuRadioRecoio
import numpy as np
import os
from NuRadioReco.detector import generic_detector
from NuRadioReco.detector import detector
import datetime
import json
from NuRadioReco.utilities import units
from NuRadioReco.utilities import units, fft
from glob import glob
from icecream import ic
import argparse

import templateCrossCorr as txc
from NuRadioReco.utilities.io_utilities import read_pickle
import DeepLearning.D00_helperFunctions as D00_helperFunctions

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


#####
#This code takes output from D03/D04, and then rerun's through the original events
#Finds events that passed cuts, and then makes plots with them
#####



channelResampler = CchannelResampler.channelResampler()
channelResampler.begin(debug=False)
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
hardwareResponseIncorporator = ChardwareResponseIncorporator.hardwareResponseIncorporator()
hardwareResponseIncorporator.begin(debug=False)
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
correlationDirectionFitter.begin(debug=False)
channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()
cTW = cTWindow.channelTimeWindow()
cTW.begin(debug=False)
triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
triggerTimeAdjuster.begin(pre_trigger_time=30*units.ns)
channelLengthAdjuster = NuRadioReco.modules.channelLengthAdjuster.channelLengthAdjuster()
channelLengthAdjuster.begin()
# det = detector_sys_uncertainties.DetectorSysUncertainties(source='sql', assume_inf=False)  # establish mysql connection


#Need blackout times for high-rate noise regions
def inBlackoutTime(time, blackoutTimes):
    #This check removes data that have bad datetime format. No events should be recorded before 2013 season when the first stations were installed
    if datetime.datetime.fromtimestamp(time) > datetime.datetime(2019, 3, 31):
        return True
    #This check removes events happening during periods of high noise
    for blackouts in blackoutTimes:
        if blackouts[0] < time and time < blackouts[1]:
            return True
    return False

blackoutFile = open('DeepLearning/BlackoutCuts.json')
blackoutData = json.load(blackoutFile)
blackoutFile.close()

blackoutTimes = []

for iB, tStart in enumerate(blackoutData['BlackoutCutStarts']):
    tEnd = blackoutData['BlackoutCutEnds'][iB]
    blackoutTimes.append([tStart, tEnd])


def plotTrace(traces, title, saveLoc, show=False):
    f, ax = plt.subplots(4,1)
    for chID, trace in enumerate(traces):
        ax[chID].plot(trace)
    plt.suptitle(title)
    if show:
        plt.show()
    else:
        plt.savefig(saveLoc, format='png')
    return

def pT(traces, title, saveLoc, sampling_rate=2, show=False, average_fft_per_channel=[]):
    #Sampling rate should be in GHz
    print(f'printing')
    x = np.linspace(1, int(256 / sampling_rate), num=256)
    x_freq = np.fft.rfftfreq(len(x), d=(1 / sampling_rate*units.GHz)) / units.MHz

    """ #Method for plotting a single plot
    #fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False)
    plt.plot(x, traces[0]*100, color='orange')
    plt.plot(x, traces[1]*100, color='blue')
    plt.plot(x, traces[2]*100, color='purple')
    plt.plot(x, traces[3]*100, color='green')
    plt.xlabel('time [ns]',fontsize=18)
    plt.ylabel('Amplitude (mV)')
    plt.xlim(-3,260 / sampling_rate)
    plt.title(title)
    plt.savefig(saveLoc + '_Traces.png', format='png')
    plt.clf()
    plt.close()


    freqs = []
    for trace in traces:
        freqtrace = np.abs(fft.time2freq(trace, sampling_rate*units.GHz))
        freqs.append(freqtrace)
    plt.plot(x_freq/1000, freqs[0], color='orange', label='Channel 0')
    plt.plot(x_freq/1000, freqs[1], color='blue', label='Channel 1')
    plt.plot(x_freq/1000, freqs[2], color='purple', label='Channel 2')
    plt.plot(x_freq/1000, freqs[3], color='green', label='Channel 3')
    plt.xlabel('Frequency [GHz]',fontsize=18)
    plt.ylabel('Amplitude')
#    axs[0][1].set_ylabel('Amplitude')
    plt.xlim(-0.003, 1.050)
    plt.xticks(size=13)
    plt.title(title)
    plt.savefig(saveLoc + '_Freqs.png', format='png')
#    plt.savefig(f'DeepLearning/plots/Station_19/GoldenDay/NuSearchFreqs_{title}.png', format='png')
    plt.clf()
    plt.close()
    return
    """

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
        if len(average_fft_per_channel) > 0:
            axs[chID][1].plot(x_freq, average_fft_per_channel[chID], color='gray', linestyle='--')
        axs[chID][1].plot(x_freq, freqtrace)
        fmax = max(fmax, max(freqtrace))
        vmax = max(vmax, max(trace))

    axs[3][0].set_xlabel('time [ns]',fontsize=18)
    axs[3][1].set_xlabel('Frequency [MHz]',fontsize=18)

    for chID, trace in enumerate(traces):
        axs[chID][0].set_ylabel(f'ch{chID}',labelpad=10,rotation=0,fontsize=13)
        # axs[i].set_ylim(-250,250)
        axs[chID][0].set_xlim(-3,260 / sampling_rate)
        axs[chID][1].set_xlim(-3, 1000)
        axs[chID][0].tick_params(labelsize=13)
        axs[chID][1].tick_params(labelsize=13)

        axs[chID][0].set_ylim(-vmax * 1.1, vmax * 1.1)
        axs[chID][1].set_ylim(-0.05, fmax * 1.1)

    axs[0][0].tick_params(labelsize=13)
    axs[0][1].tick_params(labelsize=13)
    axs[0][0].set_ylabel(f'ch{0}',labelpad=3,rotation=0,fontsize=13)
    axs[chID][0].set_xlim(-3,260 / sampling_rate)
    axs[chID][1].set_xlim(-3, 1000)

    fig.text(0.03, 0.5, 'voltage [V]', ha='center', va='center', rotation='vertical',fontsize=18)
    plt.xticks(size=13)
    plt.suptitle(title)

    if show:
        plt.show()
    else:
        print(f'saving to {saveLoc}')
        plt.savefig(saveLoc, format='png')
    plt.clf()
    plt.close(fig)

    return

def getVrms(nurFile, save_chans, station_id, det, check_forced=False, max_check=1000, plot_avg_trace=False, saveLoc='plots/'):
    template = NuRadioRecoio.NuRadioRecoio(nurFile)


    Vrms_sum = 0
    num_avg = 0
    if plot_avg_trace:
        trace_sum = []

    for i, evt in enumerate(template.get_events()):
        station = evt.get_station(station_id)
        stationtime = station.get_station_time().unix
        if inBlackoutTime(stationtime, blackoutTimes):
            continue

        channelSignalReconstructor.run(evt, station, det)
        for ChId, channel in enumerate(station.iter_channels(use_channels=save_chans)):
            Vrms_sum += channel[chp.noise_rms]
            num_avg += 1
            if plot_avg_trace:
                if trace_sum == []:
                    trace_sum = channel.get_trace()
                else:
                    trace_sum += channel.get_trace()
        
        if num_avg >= max_check:
            break

    if plot_avg_trace:
        plt.plot(trace_sum/num_avg)
        plt.xlabel('sample', label=f'{Vrms_sum/num_avg:.2f} Average Vrms')
        plt.legend()
        plt.savefig(saveLoc + f'stn{station_id}_average_trace.png')

    return Vrms_sum / num_avg


        
def getMaxChi(traces, sampling_rate, template_trace, template_sampling_rate, parallelChannels=[[0, 2], [1, 3]]):
    #Parallel channels should be index corresponding to the channel in traces


    maxCorr = []
    for parChans in parallelChannels:
        parCorr = 0
        for chan in parChans:
            xCorr = txc.get_xcorr_for_channel(traces[chan], template_trace, sampling_rate, template_sampling_rate)
            parCorr += np.abs(xCorr)
        maxCorr.append(parCorr / len(parChans))

    return max(maxCorr)

def getMaxAllChi(traces, sampling_rate, template_traces, template_sampling_rate, parallelChannels=[[0, 2], [1, 3]]):

    maxCorr = []
    for trace in template_traces:
        maxCorr.append(getMaxChi(traces, sampling_rate, trace, template_sampling_rate, parallelChannels=parallelChannels))

    return max(maxCorr)

def getMaxSNR(traces, noiseRMS=22.53 * units.mV):
    # traces should be a list or numpy array of channels
    # iterate through each channel and return the highest SNR of all channels

    SNRs = []
    for trace in traces:
        p2p = (np.max(trace) + np.abs(np.min(trace))) * units.V
        SNRs.append(p2p / (2*noiseRMS))

    return max(SNRs)

def loadTemplate(type='RCR', amp='200s'):
    if type == 'RCR':
        if amp == '200s':
                templates_RCR = 'StationDataAnalysis/templates/reflectedCR_template_200series.pkl'
                templates_RCR = read_pickle(templates_RCR)
                for key in templates_RCR:
                    temp = templates_RCR[key]
                templates_RCR = temp
                return templates_RCR

    print(f'{type} {amp} not implemented')
    quit()

def converter(nurFile, folder, savename, save_chans, station_id = 1, det=None, plot=False, 
              filter=False, BW=[80*units.MHz, 500*units.MHz], normalize=False, saveTimes=False, timeAdjust=True, cut=True):
    if cut:
        import DeepLearning.D04C_CutInBacklobeRCR as D04C_CutInBacklobeRCR
    count = 0
    part = 0
    max_events = 500000
    ary = np.zeros((max_events, 4, 256))
    if saveTimes:
        art = np.zeros(max_events)
    template = NuRadioRecoio.NuRadioRecoio(nurFile)

    cutPath = f'DeepLearning/data/{folder}/timesPassedCuts_FilteredStation{station_id}_TimeCut_1perDay_Amp0.95%.npy'
    if os.path.isfile(cutPath):
        timeCutTimes, ampCutTimes, deepLearnCutTimes, allCutTimes = np.load(cutPath, allow_pickle=True)
    else:
        print(f'No cut times found, ignoring')
        timeCutTimes = []
        ampCutTimes = []
        deepLearnCutTimes = []
        allCutTimes = []


    #Load the average fft for plotting
    # average_fft, average_fft_per_channel = np.load(f'DeepLearning/data/Station{station_id}_NoiseFFT_Filtered.npy', allow_pickle=True)


    #200s Noise
    noiseRMS = 22.53 * units.mV


    #Normalizing will save traces with values of sigma, rather than voltage
    if normalize:
        Vrms = getVrms(nurFile, save_chans, station_id, det)
        print(f'normalizing to {Vrms} vrms')

    #Load 200s template
    templates_RCR = D00_helperFunctions.loadSingleTemplate('200')
    # templates_RCR = 'StationDataAnalysis/templates/reflectedCR_template_200series.pkl'
    # templates_RCR = read_pickle(templates_RCR)
    # for key in templates_RCR:
    #     temp = templates_RCR[key]
    # templates_RCR = temp

    # Load series of templates
    # template_series_RCR_location = 'DeepLearning/templates/RCR/'
    # template_series_RCR = []
    # for filename in os.listdir(template_series_RCR_location):
    #     if filename.startswith('200s'):
    #         temp = np.load(os.path.join(template_series_RCR_location, filename))
    #         template_series_RCR.append(temp)
    template_series_RCR = D00_helperFunctions.loadMultipleTemplates('200')
    template_series_RCR.append(templates_RCR)

    PassingCut_SNRs = []
    PassingCut_RCR_Chi = []
    PassingCut_Zen = []
    PassingCut_Azi = []
    PassingCut_Traces = []
    PassingCut_Times = []

    PassingChiCut_SNRs = []
    PassingChiCut_RCR_Chi = []
    PassingChiCut_Zen = []
    PassingChiCut_Azi = []
    PassingChiCut_Traces = []
    PassingChiCut_Times = []

    All_SNRs = []
    All_RCR_Chi = []
    # All_Zen = []
    # All_Azi = []
    All_Times = []
#    All_Traces = []
    forcedMask = []


#    i = 0  #Maybe I need to do iteration outside of enumerate because we skip events in blackoutTime?
    for i, evt in enumerate(template.get_events()):
        #If in a blackout region, skip event
        station = evt.get_station(station_id)
        try:
            stationtime = station.get_station_time().unix
        except AttributeError:
            print(f'Error on event {i}, skipping')
            continue
        if inBlackoutTime(stationtime, blackoutTimes):
            continue
        det.update(station.get_station_time())
        # det.update(datetime.datetime(2018, 12, 10))
        forcedMask.append(station.has_triggered())

        #Checking if event on Chris' golden day
        if i % 1000 == 0:
            print(f'{i} events processed...')
        traces = []
        channelBandPassFilter.run(evt, station, det, passband=[1*units.Hz, 1000*units.MHz], filter_type='butter', order=10)
        for ChId, channel in enumerate(station.iter_channels(use_channels=save_chans)):
            y = channel.get_trace()
            traces.append(y)
        # Moving appending to after correlation in case there is an error in time lookup

        #Skipping this for now, moving to processing phase. Takes hours per file alone
        # try:
        #     correlationDirectionFitter.run(evt, station, det, n_index=1.35)
        # except LookupError:
        #     print(f'Error for date {datetime.datetime.fromtimestamp(stationtime)}, skipping')
        #     continue

        All_SNRs.append(getMaxSNR(traces, noiseRMS=noiseRMS))
        # All_RCR_Chi.append(getMaxChi(traces, 2*units.GHz, templates_RCR, 2*units.GHz))
        All_RCR_Chi.append(getMaxAllChi(traces, 2*units.GHz, template_series_RCR, 2*units.GHz))
        All_Times.append(stationtime)
#        All_Traces.append(traces)

        # zen = station[stnp.zenith]
        # azi = station[stnp.azimuth]
        # All_Zen.append(np.rad2deg(zen))
        # All_Azi.append(np.rad2deg(azi))

        if datetime.datetime.fromtimestamp(stationtime) > datetime.datetime(2019, 1, 1):
            continue
        for goodTime in allCutTimes:
            if datetime.datetime.fromtimestamp(stationtime) == goodTime:
                # correlationDirectionFitter.run(evt, station, det, n_index=1.35, ZenLim=[0*units.deg, 180*units.deg])
                try:
                    correlationDirectionFitter.run(evt, station, det, n_index=1.35)
                except LookupError:
                    print(f'Error for date {datetime.datetime.fromtimestamp(stationtime)}, skipping')
                    continue
                zen = station[stnp.zenith]
                azi = station[stnp.azimuth]


                print(f'found event on good day, plotting')
                PassingCut_SNRs.append(All_SNRs[-1])
                PassingCut_RCR_Chi.append(All_RCR_Chi[-1])
                PassingCut_Zen.append(np.rad2deg(zen))
                PassingCut_Azi.append(np.rad2deg(azi))
                PassingCut_Traces.append(traces)
                PassingCut_Times.append(stationtime)

                pT(traces, datetime.datetime.fromtimestamp(stationtime).strftime("%m-%d-%Y, %H:%M:%S") + f' Chi {PassingCut_RCR_Chi[-1]:.2f}, {np.rad2deg(zen):.1f}Deg Zen {np.rad2deg(azi):.1f}Deg Azi', 
                f'DeepLearning/plots/Station_{station_id}/GoldenDay/NurSearch_{i}_Chi{PassingCut_RCR_Chi[-1]:.2f}_SNR{PassingCut_SNRs[-1]:.2f}.png')
                # f'DeepLearning/plots/Station_{station_id}/GoldenDay/NurSearch_{i}_Chi{PassingCut_RCR_Chi[-1]:.2f}_SNR{PassingCut_SNRs[-1]:.2f}.png', average_fft_per_channel=average_fft_per_channel)
        if datetime.datetime(2017, 3, 29, 3, 25, 1) < datetime.datetime.fromtimestamp(stationtime) < datetime.datetime(2017, 3, 29, 3, 25, 5):
            chrisEvent = [All_SNRs[-1], All_RCR_Chi[-1]]
        if cut and (All_RCR_Chi[-1] > D04C_CutInBacklobeRCR.RCRChiSNRCut(All_SNRs[-1])):
            # Passing cut made, plot trace and save info
            try:
                correlationDirectionFitter.run(evt, station, det, n_index=1.35)
            except LookupError:
                print(f'Error for date {datetime.datetime.fromtimestamp(stationtime)}, skipping')
                continue
            # correlationDirectionFitter.run(evt, station, det, n_index=1.35, ZenLim=[0*units.deg, 180*units.deg])
            zen = station[stnp.zenith]
            azi = station[stnp.azimuth]

            PassingChiCut_SNRs.append(All_SNRs[-1])
            PassingChiCut_RCR_Chi.append(All_RCR_Chi[-1])
            PassingChiCut_Zen.append(np.rad2deg(zen))
            PassingChiCut_Azi.append(np.rad2deg(azi))
            PassingChiCut_Traces.append(traces)
            PassingChiCut_Times.append(stationtime)


            if False:
                pT(traces, datetime.datetime.fromtimestamp(stationtime).strftime("%m-%d-%Y, %H:%M:%S") + f' Chi {All_RCR_Chi[-1]:.2f}, {np.rad2deg(zen):.1f}Deg Zen {np.rad2deg(azi):.1f}Deg Azi', 
                f'DeepLearning/plots/Station_{station_id}/PassingSNRChiCut/NurSearch_{i}_Chi{All_RCR_Chi[-1]:.2f}_SNR{All_SNRs[-1]:.2f}.png')
            # f'DeepLearning/plots/Station_{station_id}/PassingSNRChiCut/NurSearch_{i}_Chi{All_RCR_Chi[-1]:.2f}_SNR{All_SNRs[-1]:.2f}.png', average_fft_per_channel=average_fft_per_channel)
            


    print(f'Saving the SNR, Chi, and reconstructed Zen/Azi angles')
    np.save(f'DeepLearning/data/{folder}/{savename}_SNR_Chi.npy', [All_SNRs, All_RCR_Chi,  
            PassingCut_SNRs, PassingCut_RCR_Chi, PassingCut_Azi, PassingCut_Zen])
    print(f'Saved to DeepLearning/data/{folder}/{savename}_SNR_Chi.npy')
    np.save(f'DeepLearning/data/{folder}/{savename}_Traces.npy', PassingCut_Traces)
    print(f'Saved traces to DeepLearning/data/{folder}/{savename}_Traces.npy')
    np.save(f'DeepLearning/data/{folder}/{savename}_SnrChiCut.npy', [PassingChiCut_SNRs, PassingChiCut_RCR_Chi, PassingChiCut_Azi, PassingChiCut_Zen, PassingChiCut_Traces])
    print(f'Saved to traces and data to DeepLearning/data/{folder}/{savename}_SnrChiCut.npy')
    np.save(f'DeepLearning/data/{folder}/{savename}_Times.npy', [All_Times, PassingCut_Times, PassingChiCut_Times])
    print(f'Saved to DeepLearning/data/{folder}/{savename}_Times.npy')

    return

    if len(PassingCut_SNRs) == 0:
        return

    print(f'Snrs {PassingCut_SNRs}')
    print(f'Chis {PassingCut_RCR_Chi}')

    if cut:
        snrchicutmask = D04C_CutInBacklobeRCR.RCRChiSNRCutMask(All_SNRs, All_RCR_Chi)



    SNRbins = np.logspace(0.477, 2, num=80)
    maxCorrBins = np.arange(0, 1.0001, 0.01)

    #Plot of all events in Chi-SNR space
    if cut:
        plt.hist2d(All_SNRs, All_RCR_Chi, bins=[SNRbins, maxCorrBins], label=f'{snrchicutmask.sum()} Passing SNR-Chi Cut',norm=matplotlib.colors.LogNorm())
        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(label=True)
        plt.legend()
    else:
        plt.hist2d(All_SNRs, All_RCR_Chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.xlim((3, 100))
    plt.ylim((0, 1))
    plt.xlabel('SNR')
    plt.ylabel('Avg Chi Highest Parallel Channels')
    # plt.legend()
    plt.xscale('log')
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.grid(visible=True, which='both', axis='both')
    plt.title(f'Station {station_id}')
    plt.savefig(f'DeepLearning/plots/Station_{station_id}/ChiSNR_All_Stnd{station_id}.png')


    #Plot of sim overlayed on top of all events
    # plotSimSNRChi(templates_RCR, noiseRMS)
    plotSimSNRChi(template_series_RCR, noiseRMS, cut=cut)
    if not cut:
        plt.scatter([], [], color='red', label='Simulated Air Showers')
    else:
        D04C_CutInBacklobeRCR.plotRCRChiSNRCut(label=False)
    plt.legend()
    plt.savefig(f'DeepLearning/plots/Station_{station_id}/ChiSNR_wSim_Stnd{station_id}.png')

    #Plot of station & sim, with events passing cuts circled
    plt.scatter(PassingCut_SNRs, PassingCut_RCR_Chi, label=f'{len(PassingCut_RCR_Chi)} Events Passing ML-Time Cuts', facecolor='none', edgecolor='black')
    if station_id == 19:
        plt.scatter(chrisEvent[0], chrisEvent[1], label='Persichilli Thesis Event', facecolor='none', edgecolor='red')
    plt.xlim((3, 100))
    plt.ylim((0, 1))
    plt.xlabel('SNR')
    plt.ylabel('Avg Chi Highest Parallel Channels')
    plt.legend()
    plt.xscale('log')
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.grid(visible=True, which='both', axis='both')
    plt.title(f'Station {station_id} RCR SNR-Chi')
    plt.savefig(f'DeepLearning/plots/Station_{station_id}/ChiSNR_PassedCuts_Stnd{station_id}.png')
    plt.clf()
    plt.close()

    # Redoing above but adding simulated backlobes on top of simulated air showers
    if True:
        if cut:
            plt.hist2d(All_SNRs, All_RCR_Chi, bins=[SNRbins, maxCorrBins], label=f'{snrchicutmask.sum()} Passing SNR-Chi Cut',norm=matplotlib.colors.LogNorm())
            plotSimSNRChi(template_series_RCR, noiseRMS, cut=True)
            plotSimSNRChi(template_series_RCR, noiseRMS, type='Backlobe', cut=True)    
            D04C_CutInBacklobeRCR.plotRCRChiSNRCut(label=True)
            plt.legend()
        else:
            plt.hist2d(All_SNRs, All_RCR_Chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
        # plotSimSNRChi(templates_RCR, noiseRMS)
        # plotSimSNRChi(templates_RCR, noiseRMS, type='Backlobe')    
            plotSimSNRChi(template_series_RCR, noiseRMS)
            plotSimSNRChi(template_series_RCR, noiseRMS, type='Backlobe')    
            plt.scatter([], [], color='red', label='Simulated Air Showers')
            plt.scatter([], [], color='green', label='Simulated Backlobe')
        plt.scatter(PassingCut_SNRs, PassingCut_RCR_Chi, label=f'{len(PassingCut_RCR_Chi)} Events Passing ML-Time Cuts', facecolor='none', edgecolor='black')

        plt.colorbar()
        plt.xlim((3, 100))
        plt.ylim((0, 1))
        plt.xlabel('SNR')
        plt.ylabel('Avg Chi Highest Parallel Channels')
        plt.xscale('log')
        plt.tick_params(axis='x', which='minor', bottom=True)
        plt.grid(visible=True, which='both', axis='both')
        plt.title(f'Station {station_id}')
        plt.legend()
        plt.savefig(f'DeepLearning/plots/Station_{station_id}/ChiSNR_wBacklobe_PassedCuts_Stnd{station_id}.png')
        plt.clf()
        plt.close()



    #Histogram of reconstructed directions
    #First of only the events passing cuts
    zenbins = np.arange(0, 180.1, 10)
    azibins = np.arange(0, 360.1, 20)

    plt.hist(PassingCut_Zen, bins=zenbins, density=True)
    plt.title(f'Station {station_id} Reconstructed Zen Passing Cuts')
    plt.savefig(f'DeepLearning/plots/Station_{station_id}/ReconstZen_PassedCuts_Stn{station_id}.png')
    plt.clf()

    plt.hist(PassingCut_Azi, bins=azibins, density=True)
    plt.title(f'Station {station_id} Reconstructed Azi Passing Cuts')
    plt.savefig(f'DeepLearning/plots/Station_{station_id}/ReconstAzi_PassedCuts_Stn{station_id}.png')
    plt.clf()

    #Plot with the density of all events as well behind them
    plt.hist(PassingCut_Zen, bins=zenbins, density=True, histtype='step', label='Passing Cuts')
    plt.hist(All_Zen, bins=zenbins, density=True, histtype='step', label='All Station Events')
    plt.legend()
    plt.title(f'Station {station_id} Zenith Reconstruction')
    plt.savefig(f'DeepLearning/plots/Station_{station_id}/ReconstZen_All_Stn{station_id}.png')
    plt.clf()

    plt.hist(PassingCut_Azi, bins=azibins, density=True, histtype='step', label='Passing Cuts')
    plt.hist(All_Zen, bins=azibins, density=True, histtype='step', label='All Station Events')
    plt.legend()
    plt.title(f'Station {station_id} Azi Reconstruction')
    plt.savefig(f'DeepLearning/plots/Station_{station_id}/ReconstAzi_All_Stn{station_id}.png')
    plt.clf()



    return
        

def plotSimSNRChi(templates_RCR, noiseRMS, amp='200s', type='RCR', ax=None, cut=False, plotOnlyAbove=False):
    if cut:
        import DeepLearning.D04C_CutInBacklobeRCR as D04C_CutInBacklobeRCR

    path = 'DeepLearning/data/3rdpass/'
    RCR_files = []
    if type == 'RCR':
        for filename in os.listdir(path):
            if filename.startswith(f'SimRCR_{amp}'):
                RCR_files.append(os.path.join(path, filename))
                RCR_weights_file = f'DeepLearning/data/3rdpass/SimWeights_{filename}'
    elif type == 'Backlobe':
        for filename in os.listdir(path):
            if filename.startswith(f'Backlobe_{amp}'):
                RCR_files.append(os.path.join(path, filename))
                RCR_weights_file = f'DeepLearning/data/3rdpass/SimWeights_{filename}'

    for file in RCR_files:
        RCR_sim = np.load(file)
    RCR_weights = np.load(RCR_weights_file)    

    sim_SNRs = []
    sim_Chi = []
    sim_weights = []
    for iR, RCR in enumerate(RCR_sim):


        traces = []
        for trace in RCR:
            traces.append(trace * units.V)
        sim_SNRs.append(getMaxSNR(traces, noiseRMS=noiseRMS))
        # sim_Chi.append(getMaxChi(traces, 2*units.GHz, templates_RCR, 2*units.GHz))
        sim_Chi.append(getMaxAllChi(traces, 2*units.GHz, templates_RCR, 2*units.GHz))

        sim_weights.append(RCR_weights[iR])
        if iR % 1000 == 0 and False:
            pT(traces, f'Backlobe SNR {sim_SNRs[-1]:.1f} Chi {sim_Chi[-1]:.2f}', f'DeepLearning/plots/Backlobe/SimBacklobe_SNR{sim_SNRs[-1]:.1f}_Chi{sim_Chi[-1]:.2f}_{iR}.png')

    if False:
        SNRbins = np.logspace(0.477, 2, num=80)
        maxCorrBins = np.arange(0, 1.0001, 0.01)

        plt.hist(sim_Chi, weights=sim_weights, bins=maxCorrBins, density=True)
        plt.xlabel('Chi')
        plt.savefig('DeepLearning/plots/200s_Chi_hist.png')
        plt.clf()

        plt.hist(sim_SNRs, weights=sim_weights, bins=SNRbins, density=True)
        plt.xlabel('SNR')
        plt.xscale('log')
        plt.savefig('DeepLearning/plots/200s_SNR_hist.png')
        plt.clf()

        quit()


    SNRbins = np.logspace(0.477, 2, num=80)
    maxCorrBins = np.arange(0, 1.0001, 0.01)

    print(f'len sim snr {len(sim_SNRs)} and chi {len(sim_Chi)}')
    # print(f'weights {sim_weights}')

    sim_weights = np.array(sim_weights)
    sim_SNRs = np.array(sim_SNRs)
    sim_Chi = np.array(sim_Chi)

    sort_order = sim_weights.argsort()
#    sim_SNRs = sim_SNRs[sort_order[::-1]]
#    sim_Chi = sim_Chi[sort_order[::-1]]
#    sim_weights = sim_weights[sort_order[::-1]]
    sim_SNRs = sim_SNRs[sort_order]
    sim_Chi = sim_Chi[sort_order]
    sim_weights = sim_weights[sort_order]

    if cut:
        cutMask = D04C_CutInBacklobeRCR.RCRChiSNRCutMask(sim_SNRs, sim_Chi)
        weight_cut = np.sum(sim_weights[cutMask]) / np.sum(sim_weights)
    if plotOnlyAbove:
        sim_SNRs = sim_SNRs[cutMask]
        sim_Chi = sim_Chi[cutMask]
        sim_weights = sim_weights[cutMask]

    if type == 'RCR':
        cmap = 'seismic'
    else:
        cmap = 'PiYG'
    # plt.scatter(sim_SNRs, sim_Chi, c=sim_weights, label=f'Simulated {type}', cmap=cmap, alpha=0.9, norm=matplotlib.colors.LogNorm())
    if ax is None:
        if cut:
            # weight_cut = np.sum(sim_weights[cutMask]) / np.sum(sim_weights)
            eff = weight_cut*100
            if type == 'Backlobe':
                eff = 1 - eff
            plt.scatter(sim_SNRs, sim_Chi, c=sim_weights, label=f'{type} {eff:.1f}% Efficiency',cmap=cmap, alpha=0.9, norm=matplotlib.colors.LogNorm())
        else:
            plt.scatter(sim_SNRs, sim_Chi, c=sim_weights, cmap=cmap, alpha=0.9, norm=matplotlib.colors.LogNorm())
    else:
        if cut:
            # weight_cut = np.sum(sim_weights[cutMask]) / np.sum(sim_weights)
            eff = weight_cut*100
            if type == 'Backlobe':
                eff = 1 - eff
            ax.scatter(sim_SNRs, sim_Chi, c=sim_weights, label=f'{type} {eff:.1f}% Efficiency', cmap=cmap, alpha=0.9, norm=matplotlib.colors.LogNorm())
        else:
            ax.scatter(sim_SNRs, sim_Chi, c=sim_weights, cmap=cmap, alpha=0.9, norm=matplotlib.colors.LogNorm())
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Analysis on particular station')
    parser.add_argument('station', type=int, default=19, help='Station to run on')
    parser.add_argument('--folder', type=str, default='5thpass', help='Folder to save data to')
    parser.add_argument('--single_file', type=str, default=None, help='Single file to run on')
    parser.add_argument('--first_ch', type=int, default=0, help='First channel of 4 to save data from')
    parser.add_argument('--amp', type=str, default='200s', help='Amp type used')
    args = parser.parse_args()
    station_id = args.station
    folder = args.folder
    single_file = args.single_file
    first_ch = args.first_ch
    amp = args.amp
    
    if amp == '100s':
        noiseRMS = 20.0 * units.mV
    elif amp == '200s':
        noiseRMS = 22.53 * units.mV
    elif amp == '300s':
        noiseRMS = 10 * units.mV # Placeholder, need to look up TODO


    #Existing data conversion

#    station_id = 17

    # detector = generic_detector.GenericDetector(json_filename=f'DeepLearning/station_configs/station{station_id}.json', assume_inf=False, antenna_by_depth=False, default_station=station_id)
    detector = detector.Detector('../NuRadioMC/NuRadioReco/detector/ARIANNA/arianna_detector_db.json', 'json')   #Relative path from running folder

    if single_file is None:
        if not station_id == 52:
            station_path = f"/dfs8/sbarwick_lab/ariannaproject/station_nur/station_{station_id}/"
        elif station_id == 52:
            station_path = f"/dfs8/sbarwick_lab/ariannaproject/leshanz_backup/arianna/station_{station_id}/data/"
        DataFiles = []
        for filename in os.listdir(station_path):
            if filename.endswith('_statDatPak.root.nur'):
                continue
            elif not filename.endswith('.nur'):
                continue
            else:
                if os.path.getsize(os.path.join(station_path, filename)) == 0:
                    print(f'File {filename} is empty, skipping')
                    continue
                DataFiles.append(os.path.join(station_path, filename))
        savename = f'FilteredStation{station_id}_Data'
    else:
        DataFiles = [single_file]
        savename = f'FilteredStation{station_id}_Data_{single_file.split("/")[-1].replace(".root.nur", "")}'

#    DataFiles = DataFiles[0:1]     #Just 1 file for testing purposes

    saveChannels = np.arange(first_ch, first_ch+4)
    #converter(DataFiles, folder, f'Station{station_id}_Data', saveChannels, station_id = station_id, filter=False, saveTimes=True, plot=False)
    converter(DataFiles, folder, savename, saveChannels, station_id = station_id, det=detector, filter=True, saveTimes=True, plot=False)
#    for file in DataFiles:
#        converter(file, folder, f'FilteredStation{station_id}_Data', saveChannels, station_id = station_id, filter=True, saveTimes=True, plot=False)
