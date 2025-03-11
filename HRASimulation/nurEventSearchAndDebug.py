import os
import configparser
import NuRadioReco.modules.io.eventReader
from icecream import ic
from NuRadioReco.framework.parameters import eventParameters as evtp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import units, fft

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

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('HRASimulation/config.ini')
    sim_folder = config['FOLDERS']['sim_folder']
    numpy_folder = config['FOLDERS']['numpy_folder']
    save_folder = config['FOLDERS']['save_folder']
    diameter = config['SIMPARAMETERS']['diameter']
    max_distance = float(diameter)/2*units.km

    if not os.exists(f'{save_folder}Traces'):
        os.makedirs(f'{save_folder}Traces')

    nur_files = []
    for file in os.listdir(sim_folder):
        if file.endswith('.nur'):
            nur_files.append(os.path.join(sim_folder, file))


    zenlim = [20, 40]
    azilim = [90, 180]
    englim = [10**17.5, 10**18.5]
    rdist = 5000
    station_check = [13, 14, 15, 17, 18, 19, 30]
    sigma = 4.5

    plot = 10
    plotted = 0

    eventReader = NuRadioReco.modules.io.eventReader.eventReader()
    for file in nur_files:
        eventReader.begin(file)
        for event in eventReader.run():
            triggered = False
            station_id = 0
            for station in event.get_stations():
                if station.get_id() in station_check:
                    if station.has_triggered() and station.has_triggered(trigger_name=f'primary_LPDA_2of4_{sigma}sigma'):
                        ic(f'{event.get_id()} {station.get_id()} {station.get_parameter(stnp.zenith)} {station.get_parameter(stnp.azimuth)}')
                        triggered = True
                        station_id = station.get_id()
                        break
            if not triggered:
                continue

            station = event.get_station(station_id)

            sim_shower = event.get_sim_shower(0)
            core_x = event.get_parameter(evtp.coreas_x)
            core_y = event.get_parameter(evtp.coreas_y)
            sim_energy = sim_shower[shp.energy]
            sim_zenith = sim_shower[shp.zenith]
            sim_azimuth = sim_shower[shp.azimuth]

            station_zenith = station.get_parameter(stnp.zenith)
            station_azimuth = station.get_parameter(stnp.azimuth)

            ic(core_x, core_y, sim_energy, sim_zenith/units.deg, sim_azimuth/units.deg, station_zenith/units.deg, station_azimuth/units.deg)

            pos_check = core_x**2 + core_y**2 >= rdist**2
            eng_check = englim[0] <= sim_energy <= englim[1]
            zen_check = zenlim[0] <= sim_zenith/units.deg <= zenlim[1]
            azi_check = azilim[0] <= sim_azimuth/units.deg <= azilim[1]

            ic(pos_check, eng_check, zen_check, azi_check)

            traces = []
            use_channels = [0, 1, 2, 3]
            for ChId, channel in enumerate(station.iter_channels(use_channels=use_channels)):
                y = channel.get_trace()
                traces.append(y)

            title = f'Stn {station_id}, Zen{sim_zenith/units.deg:.1f}deg Azi{sim_azimuth/units.deg:.1f}deg, Eng {np.log10(sim_energy/units.eV):.1f}log10eV, ({core_x/units.km:.1f}km, {core_y/units.km:.1f}km)'
            plotTrace(traces, title, f'{save_folder}Traces/{event.get_id()}_{station_id}_{core_x/units.km:.1f}-{core_y/units.km:.1f}.png')
            plotted += 1
            if plotted >= plot:
                quit()

