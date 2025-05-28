import datetime
from NuRadioReco.utilities import units
from NuRadioReco.modules import channelResampler as channelResampler
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
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.modules.io import NuRadioRecoio
import numpy as np
import os
from NuRadioReco.detector import generic_detector
from NuRadioReco.detector import detector
import datetime
import json
from NuRadioReco.utilities import units, fft
from NuRadioReco.utilities.cr_flux import get_cr_event_rate
from icecream import ic
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

channelResampler = channelResampler.channelResampler()
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

def pT(traces, title, saveLoc, sampling_rate=2, show=False):
    #Sampling rate should be in GHz

    x = np.linspace(1, int(256 / sampling_rate), num=256)
    x_freq = np.fft.rfftfreq(len(x), d=(1 / sampling_rate*units.GHz)) / units.MHz


    #fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False)
    plt.plot(x, traces[0]*100, color='orange')
    plt.plot(x, traces[1]*100, color='blue')
    plt.plot(x, traces[2]*100, color='purple')
    plt.plot(x, traces[3]*100, color='green')
    plt.xlabel('time [ns]',fontsize=18)
    plt.ylabel('Amplitude (mV)')
    plt.xlim(-3,260 / sampling_rate)
    plt.suptitle(title)
    plt.savefig(f'DeepLearning/plots/Station_19/GoldenDay/NuSearchTraces_{title}.png', format='png')
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
    plt.suptitle(title)
    plt.savefig(f'DeepLearning/plots/Station_19/GoldenDay/NuSearchFreqs_{title}.png', format='png')
#    plt.savefig(saveLoc, format='png')
    plt.clf()
    plt.close()
    return

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
        plt.savefig(saveLoc, format='png')
    plt.clf()
    plt.close()
    return

def getVrms(nurFiles, save_chans, station_id, det, check_forced=False, max_check=1000, plot_avg_trace=False, saveLoc='plots/'):
    template = NuRadioRecoio.NuRadioRecoio(nurFiles)


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


        



def converter(nurFiles, folder, save_prefix, save_chans, station_id = 1, det=None, plot=False, blackout=True,
              filter=False, BW=[80*units.MHz, 500*units.MHz], normalize=False, saveTimes=False, timeAdjust=True, sim=False, reconstruct=False, forced=True):
    ###
    #Converts a nur file, or collection of nur files, into numpy arrays for data processing
    ###
    #Inputs:
    #
    # nurFiles      : list of strings
    #       -list of paths to each nur file, [path/to/file/1.nur, path/to/file/2.nur, ...]
    # folder        : string
    #       -name of folder to save events to
    # save_prefix   : string
    #       -prefix to append to saved file names
    # save_chans    : array of ints
    #       -list of channels to save traces of
    # station_id    : int
    #       -id of the station data is saved on in the files
    # det           : NuRadioReco.detector
    #       -detector object used in simulation. Needed for reconstruction
    # plot          : bool
    #       -if True, plots every 1000th trace to view a sub-selection with
    save_prefix = save_prefix + f'_forced{forced}'
    os.makedirs(f'DeepLearning/data/{folder}', exist_ok=True)

    count = 0
    part = 0
    max_events = 500000
    ary = np.zeros((max_events, 4, 256))
    if saveTimes:
        art = np.zeros(max_events)
    if sim:
        #Add weights based off of energy/zenith for sim events
        arw = np.zeros(max_events)
        #Also add the true energy, zenith and azimuth of the event
        arz = np.zeros((max_events,3))
    template = NuRadioRecoio.NuRadioRecoio(nurFiles)

#    station_id = 1

    #Normalizing will save traces with values of sigma, rather than voltage
    if normalize:
        Vrms = getVrms(nurFiles, save_chans, station_id, det)
        print(f'normalizing to {Vrms} vrms')

    if reconstruct:
        correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
        correlationDirectionFitter.begin(debug=False)
        arr = np.zeros(max_events)


#    i = 0  #Maybe I need to do iteration outside of enumerate because we skip events in blackoutTime?
    for i, evt in enumerate(template.get_events()):

        #If in a blackout region, skip event
        station = evt.get_station(station_id)
        stationtime = station.get_station_time().unix

        # Skip forced triggers in flagged
        if not forced and not station.has_triggered():
            continue


        if 0 and station.has_triggered():
            for ch in [0, 1, 2, 3]:
                print(f'times are {station.get_channel(ch).get_times()}')
            quit()

        #Checking if event on Chris' golden day
        checkGolden = False
        if i % 1000 == 0:
            print(f'{i} events processed...')
        if checkGolden and datetime.datetime(2017, 3, 29, 3, 25) < datetime.datetime.fromtimestamp(stationtime) < datetime.datetime(2017, 3, 29, 3, 26):
            print(f'found event on day, plotting')
            traces = []
            channelBandPassFilter.run(evt, station, det, passband=[1*units.MHz, 2000*units.MHz], filter_type='butter', order=10)
            for ChId, channel in enumerate(station.iter_channels(use_channels=save_chans)):
                y = channel.get_trace()
                traces.append(y)
            pT(traces, datetime.datetime.fromtimestamp(stationtime).strftime("%m-%d-%Y, %H:%M:%S"), f'DeepLearning/plots/Station_19/GoldenDay/NurSearch_{i}.png')
#        continue

        if blackout and inBlackoutTime(stationtime, blackoutTimes):
            continue


        if i % 1000 == 0:
            print(f'{i} events processed...')
#        if count % max_events == 0 and not count == 0:
#            print(f'just testing number of events...not saving')
#        continue
        count = i - max_events * part
        if count >= max_events:
            saveName = f'DeepLearning/data/{folder}/{save_prefix}_{max_events}events_part{part}.npy'
            print(f'Reached cap, saving events to {saveName}')
            np.save(saveName, ary)
            if saveTimes:
                saveTimes = f'DeepLearning/data/{folder}/DateTime_{save_prefix}_{max_events}events_part{part}.npy'
                print(f'Saving times to {saveTimes}')
                np.save(saveTimes, art)
                art = np.zeros(max_events)
            if sim:
                saveWeights = f'DeepLearning/data/{folder}/SimWeights_{save_prefix}_{max_events}events_part{part}.npy'
                saveParams = f'DeepLearning/data/{folder}/SimZeniths_{save_prefix}_{max_events}events_part{part}.npy'
                print(f'Saving times to {saveWeights}')
                np.save(saveWeights, arw)
                np.save(saveParams, arz)
                arw = np.zeros(max_events)
                arz = np.zeros(max_events)
            if reconstruct:
                saveReconstruct = f'DeepLearning/data/{folder}/SimReconZeniths_{save_prefix}_{max_events}events_part{part}.npy'
                np.save(saveParams, arr)
                arr = np.zeros(max_events)
            part += 1
            ary = np.zeros((max_events, 4, 256))
        station = evt.get_station(station_id)
        i = i - max_events * part

        if saveTimes:
            art[i] = stationtime
        if sim:
            sim_shower = evt.get_sim_shower(0)
            sim_energy = sim_shower[shp.energy]
            sim_zen = sim_shower[shp.zenith]
            sim_azi = sim_shower[shp.azimuth]
            eventRate = get_cr_event_rate(np.log10(sim_energy), np.rad2deg(sim_zen)*units.deg)
            arw[i] = eventRate
            arz[i] = [np.log10(sim_energy), np.rad2deg(sim_zen), np.rad2deg(sim_azi)]
#            print(f'energy {np.log10(sim_energy)} and zen {np.rad2deg(sim_zen)} got event rate {eventRate}')

        #Example of getting date from saved stationtime/testing it works
        if False:
            print(f'Datetime event is ' + datetime.utcfromtimestamp(art[i]).strftime('%Y-%m-%d %H:%M:%S'))

        if reconstruct:
            correlationDirectionFitter.run(evt, station, det, n_index=1.35, ZenLim=[0*units.deg, 180*units.deg])
            zen = station[stnp.zenith]
            arr[i] = np.rad2deg(zen)

        for ChId, channel in enumerate(station.iter_channels(use_channels=save_chans)):

            if filter:
                channelBandPassFilter.run(evt, station, det, passband=[BW[0], 1000*units.MHz], filter_type='butter', order=10)
                channelBandPassFilter.run(evt, station, det, passband=[1*units.MHz, BW[1]], filter_type='butter', order=5)
            channelStopFilter.run(evt, station, det, prepend=0*units.ns, append=0*units.ns)

            y = channel.get_trace()
            t = channel.get_times()
            if len(y) > 257:
                channelLengthAdjuster.run(evt, station, channel_ids=[channel.get_id()])
                y = channel.get_trace()
                t = channel.get_times()
#                plt.plot(t, y)
#                plt.title(f'ch {channel.get_id()}')
#                plt.show()
#                print(f'len y {len(y)}')
#                print(f'num of samples{channel.get_number_of_samples()}')
#            continue
            #Array with 3 dimensions
            #Dim 1 = event number
            #Dim 2 = Channel identifier (0-X)
            #Dim 3 = Samples, 256 long, voltage trace
            if normalize:
                y = y / Vrms
            ary[i, ChId] = y

        if plot and i % 1000 == 0:
            plotTrace(ary[i], f"Sim RCR {i}",f"DeepLearning/data/{folder}/Sim_RCR_{i}.png")


#        i = i + 1

    print(f'shape ary {ary.shape}')
    print(f'count - max events * part = {count} - {max_events} * {part} = {count - max_events * part}')
    print(f'meanwhile i is {i}')
#    ary = ary[0:(count - max_events * part)]
    ary = ary[0:i]
    print(ary.shape)


    saveName = f'DeepLearning/data/{folder}/{save_prefix}_{len(ary)}events_Filter{filter}_part{part}.npy'
    print(f'Saving to {saveName}')
    np.save(saveName, ary)

    if saveTimes:
#        art = art[0:(count - max_events * part)]
        art = art[0:i]
        saveTimes = f'DeepLearning/data/{folder}/DateTime_{save_prefix}_{len(ary)}events_part{part}.npy'
        np.save(saveTimes, art)
    if sim:
        arw = arw[0:i]
        arz = arz[0:i]
        saveWeights = f'DeepLearning/data/{folder}/SimWeights_{save_prefix}_{len(ary)}events_part{part}.npy'
        saveParams = f'DeepLearning/data/{folder}/SimParams_{save_prefix}_{len(ary)}events_part{part}.npy'
        np.save(saveWeights, arw)
        np.save(saveParams, arz)
    if reconstruct:
        arr = arr[0:i]
        saveReconstruct = f'DeepLearning/data/{folder}/SimReconZeniths_{save_prefix}_{len(ary)}events_part{part}.npy'
        np.save(saveParams, arr)


    return
        
import configparser


config = configparser.ConfigParser()
config.read('DeepLearning/config.ini')
amp_type = config['SIMPARAMETERS']['amp']
simdate = config['SIMPARAMETERS']['date']
folder = config['SIMPARAMETERS']['datapass']
noise = config['SIMPARAMETERS']['noise']
filter = config['SIMPARAMETERS']['filter']

amp_type = amp_type[:-1]

#Convert RCR simulated data
if True:
    det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_MB_old_{amp_type}s_footprint576m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
    # station_files_path = 'FootprintAnalysis/output/'
    station_files_path = f'SimpleFootprintSimulation/output/RCR/{simdate}/{amp_type}s/'
    SimRCRFiles = []
    for filename in os.listdir(station_files_path):
        # if filename.startswith(f'RCRs_MB_MB_old_{amp_type}s_refracted') and filename.endswith('.nur'):
        if filename.startswith(f'RCR_MB') and filename.endswith('.nur'):
            # Check if file has NoiseTrue or NoiseFalse in the name
            if noise == 'True' and 'NoiseFalse' in filename:
                continue
            if noise == 'False' and 'NoiseTrue' in filename:
                continue
            print(f'adding events from file {filename}')
            SimRCRFiles.append(os.path.join(station_files_path, filename))

    saveChannels = [4, 5, 6, 7]
    if filter == 'True':
        filter = True
    else:
        filter = False
    converter(SimRCRFiles, folder, f'SimRCR_{amp_type}s_Noise{noise}', saveChannels, station_id = 1, det=det, filter=filter, saveTimes=False, plot=False, sim=True, reconstruct=False, blackout=False, forced=False)
    print(f'saved RCRs!')

    quit()

#Convert simulated backlobe data
if False:
#    det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_MB_BacklobeTest_{amp_type}s_footprint576m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
    det = generic_detector.GenericDetector(json_filename=f'configurations/gen2_MB_old_{amp_type}s_footprint576m_infirn.json', assume_inf=False, antenna_by_depth=False, default_station=1)
#    station_files_path = 'FootprintAnalysis/output/'
    station_files_path = f'SimpleFootprintSimulation/output/Backlobe/10.25.24/{amp_type}s/'
    SimRCRFiles = []
    for filename in os.listdir(station_files_path):
#        if filename.startswith(f'Backlobes_BacklobeTest_{amp_type}s_Refl_CRs_10000Evts_Noise_False') and filename.endswith('.nur'):
#        if filename.startswith(f'NewBacklobes_MB_BacklobeTest_{amp_type}s_direct_CRs') and filename.endswith('.nur'):
        if filename.endswith('.nur'):
            print(f'Adding file {filename}')
            SimRCRFiles.append(os.path.join(station_files_path, filename))

    saveChannels = [0, 1, 2, 3]
    converter(SimRCRFiles, folder, f'Backlobe_{amp_type}s', saveChannels, station_id = 1, det=det, filter=True, saveTimes=False, plot=False, sim=True, reconstruct=False, blackout=False, forced=False)
    print(f'saved backlobes!!')

    quit()

#Existing data conversion

station_id = 30
station_path = f"../../../../../dfs8/sbarwick_lab/ariannaproject/station_nur/station_{station_id}/"


DataFiles = []
for filename in os.listdir(station_path):
    if filename.endswith('_statDatPak.root.nur'):
        continue
    else:
        DataFiles.append(os.path.join(station_path, filename))

#DataFiles = DataFiles[0:4]

saveChannels = [0, 1, 2, 3]
#converter(DataFiles, folder, f'Station{station_id}_Data', saveChannels, station_id = station_id, filter=False, saveTimes=True, plot=False)
converter(DataFiles, folder, f'FilteredStation{station_id}_Data', saveChannels, station_id = station_id, filter=True, saveTimes=True, plot=False, forced=False)
