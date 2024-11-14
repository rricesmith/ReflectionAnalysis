import numpy as np
from glob import glob
import os
from icecream import ic
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def getTimestripAxs(yearStart=2014, yearEnd=2019):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=4)    #Originally 5 and 10
    plt.gca().xaxis.set_major_locator(locator)

    timeMin = datetime.datetime(yearStart, 9, 1)
    timeMax = datetime.datetime(yearEnd, 5, 1)
    delta_years = int(yearEnd-yearStart)
    delta_days = (timeMax - timeMin).days
    ic(timeMin, timeMax, delta_years, delta_days)

    fig, axs = plt.subplots(1, delta_years, sharey=True, facecolor='w')
    axs = np.atleast_1d(axs)
    return fig, axs

def getVerticalTimestripAxs(yearStart=2014, yearEnd=2019, n_stations=1):
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=4)    #Originally 5 and 10
    plt.gca().xaxis.set_major_locator(locator)

    timeMin = datetime.datetime(yearStart, 9, 1)
    timeMax = datetime.datetime(yearEnd, 5, 1)
    delta_years = int(yearEnd-yearStart)
    delta_days = (timeMax - timeMin).days
    ic(timeMin, timeMax, delta_years, delta_days)

    fig, axs = plt.subplots(n_stations, delta_years, sharey=True, sharex=True, facecolor='w', squeeze=False)
    ic(axs.shape, n_stations, delta_years)
    axs = np.atleast_2d(axs)
    return fig, axs



def timestripScatter(times, data, yearStart=2014, yearEnd=2019, legend=None, marker=None, color=None, markersize=2, fig=None, axs=None):
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    # locator = mdates.AutoDateLocator(minticks=3, maxticks=4)    #Originally 5 and 10
    # plt.gca().xaxis.set_major_locator(locator)

    # timeMin = datetime.datetime(yearStart, 10, 1)
    # timeMax = datetime.datetime(yearEnd, 6, 1)
    # delta_years = int(yearStart-yearStart)
    # delta_days = (timeMax - timeMin).days

    # fig, axs = plt.subplots(1, delta_years, sharey=True, facecolor='w')
    if np.all(axs) == None:
        fig, axs = getTimestripAxs(yearStart, yearEnd)

#    ic(fig, axs, len(axs))

    for iA, ax in enumerate(axs):
        ax.scatter(times, data, label=legend, marker=marker, color=color, s=markersize)
        # ax.set_xlim(left=timeMin, right=timeMin + datetime.timedelta(days=365) * (iA+1))
        # ax.set_title(f'{}')
    plt.gcf().autofmt_xdate()

    # Set spines and separate seasons if multiple years
    for iA, ax in enumerate(axs):
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlim(left=datetime.datetime(yearStart+iA, 9, 1), right=datetime.datetime(yearStart+1+iA, 5, 1))
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    axs[0].spines['left'].set_visible(True)
    axs[-1].spines['right'].set_visible(True)


    # Set diagonal slashes to separate time years. d may need to be changed to be dynamic based on number of axis
    d = 0.015
    kwargs = dict(transform=axs[0].transAxes, color='k', clip_on=False)
    for iE, ax in enumerate(axs):
        kwargs.update(transform=ax.transAxes)
        # Plot left hatch
        if not iE == 0:
            ax.plot((-d, +d), (-d, +d), **kwargs)
            ax.plot((-d, +d), (1-d, 1+d), **kwargs)
        # Plot right hatch
        if not (iE == len(axs)-1):
            ax.plot((1-d, 1+d), (-d, +d), **kwargs)
            ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)


    # if not legend == None:
    #     plt.legend()
    # fig.suptitle(title)
    # plt.savefig(saveLoc + f'HistStrip.png', format='png')
    # plt.clf()
    return fig, axs


def findClusterTimes(times, data, n_cluster=10, chi_cut=0.6):
    # Find days that correspond to high number of high chi events, which indicates high noise that day
    # ic(data)
    datamask = data > chi_cut

    ctimes = times[datamask]
    cdata = data[datamask]
    # ic(len(times), len(ctimes), len(data), len(cdata))


    cluster_days = []
    for iD, idate in enumerate(ctimes):
        if idate.day in cluster_days:
            continue
        n_day = 1
        for jD, jdate in enumerate(ctimes[iD:]):
            if (iD == jD) or (n_day > n_cluster) or (jdate.day in cluster_days):
                continue
            if idate.day == jdate.day:
                n_day += 1
            if n_day > n_cluster:
                cluster_days.append(idate)

    # ic(len(cluster_days), cluster_days[0:10])
    return cluster_days


def plotClusterTimes(times, data, fig, axs, n_cluster=10, chi_cut=0.6):
    ic('finding cluster times')
    cluster_days = findClusterTimes(times, data, n_cluster=n_cluster, chi_cut=chi_cut)

    for iA, ax in enumerate(axs):
        for cdate in cluster_days:
            ax.axvspan(cdate, cdate + datetime.timedelta(days=1), color='r', alpha=0.5, hatch='/', zorder=-1)
    plt.gcf().autofmt_xdate()
    return


if __name__ == "__main__":

    datapass = '7thpass'
    stations_100s = [13, 15, 18, 32]
    stations_200s = [14, 17, 19, 30]
    stations = {100: stations_100s, 200: stations_200s}


    for series in stations.keys():
        # continue
        for station_id in stations[series]:
            station_data_folder = f'DeepLearning/data/{datapass}/Station{station_id}'

            data = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_SNR_Chi.npy', allow_pickle=True)
            data_SnrChiCut = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_SnrChiCut.npy', allow_pickle=True)
            data_in2016 = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_In2016.npy', allow_pickle=True)
            times = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_Times.npy', allow_pickle=True)

            plotfolder = f'StationDataAnalysis/plots/Station_{station_id}/TimeStrips'
            if not os.path.exists(plotfolder):
                os.makedirs(plotfolder)

            # Load all the data
            All_SNRs, All_RCR_Chi, MLCut_SNRs, MLCut_RCR_Chi, MLCut_Azi, MLCut_Zen = data
            ChiCut_SNRS, ChiCut_RCR_Chi, ChiCut_Azi, ChiCut_Zen, ChiCut_Traces = data_SnrChiCut
            in2016_SNRs, in2016_RCR_Chi, in2016_Azi, in2016_Zen, in2016_Traces, in2016_datetimes = data_in2016
            All_datetimes, MLCut_datetimes, ChiCut_datetimes = times


            ic(len(All_datetimes), len(MLCut_datetimes), len(ChiCut_datetimes), len(in2016_datetimes))
            if len(All_datetimes) == 0:
                print(f'Skipping station {station_id} because no data')
                continue
            All_datetimes = np.vectorize(datetime.datetime.fromtimestamp)(All_datetimes)
            if not len(MLCut_datetimes) == 0:
                MLCut_datetimes = np.vectorize(datetime.datetime.fromtimestamp)(MLCut_datetimes)
            ChiCut_datetimes = np.vectorize(datetime.datetime.fromtimestamp)(ChiCut_datetimes)
            if not len(in2016_datetimes) == 0:
                in2016_datetimes = np.vectorize(datetime.datetime.fromtimestamp)(in2016_datetimes)

            All_RCR_Chi = np.array(All_RCR_Chi)

            # Iterate through each year, then also plot all years at once
            years = [2014, 2015, 2016, 2017, 2018, 2019]
            for iY in range(len(years)):
                if iY == len(years)-1:
                    yStart = years[0]
                    yEnd = years[-1]
                else:
                    yStart = years[iY]
                    yEnd = years[iY+1]
                # yStart = years[0]
                # yEnd = years[-1]

                # Plot the data
                fig, axs = getTimestripAxs(yStart, yEnd)
                plotClusterTimes(ChiCut_datetimes, ChiCut_RCR_Chi, fig, axs, n_cluster=10, chi_cut=0.6)
                timestripScatter(All_datetimes, All_RCR_Chi, yearStart=yStart, yearEnd=yEnd, legend='All data', marker='o', color='k', markersize=2, fig=fig, axs=axs)
                plt.legend()
                fig.suptitle(f'Station {station_id} {yStart}-{yEnd}')
                savename = f'{plotfolder}/Station{station_id}_Timestrip_AllData_{yStart}-{yEnd}.png'
                plt.savefig(savename, format='png')
                ic(f'Saved {savename}')

                timestripScatter(MLCut_datetimes, MLCut_RCR_Chi, yearStart=yStart, yearEnd=yEnd, legend='ML Cut', marker='o', color='b', markersize=2, fig=fig, axs=axs)
                plt.legend()
                savename = f'{plotfolder}/Station{station_id}_Timestrip_MLCut_{yStart}-{yEnd}.png'
                plt.savefig(savename, format='png')
                ic(f'Saved {savename}')

                timestripScatter(ChiCut_datetimes, ChiCut_RCR_Chi, yearStart=yStart, yearEnd=yEnd, legend='Chi Cut', marker='o', color='m', markersize=2, fig=fig, axs=axs)
                plt.legend()
                savename = f'{plotfolder}/Station{station_id}_Timestrip_ChiCut_{yStart}-{yEnd}.png'
                plt.savefig(savename, format='png')
                ic(f'Saved {savename}')

                timestripScatter(in2016_datetimes, in2016_RCR_Chi, yearStart=yStart, yearEnd=yEnd, legend='2016 Events', marker='*', color='g', markersize=8, fig=fig, axs=axs)
                plt.legend()
                savename = f'{plotfolder}/Station{station_id}_Timestrip_2016Events_{yStart}-{yEnd}.png'
                plt.savefig(savename, format='png')
                ic(f'Saved {savename}')


                plt.close(fig)
                # quit()


    # Plot all stations on a single timestrip
    # Iterate through each year, then also plot all years at once
    years = [2014, 2015, 2016, 2017, 2018, 2019]
    for iY in range(len(years)):
        if iY == len(years)-1:
            yStart = years[0]
            yEnd = years[-1]
        else:
            yStart = years[iY]
            yEnd = years[iY+1]

        plotfolder = f'StationDataAnalysis/plots/StackedTimeStrips'
        if not os.path.exists(plotfolder):
            os.makedirs(plotfolder)

        ic(yStart, yEnd)

        fig_all, axs_all = getVerticalTimestripAxs(yearStart=yStart, yearEnd=yEnd, n_stations=len(stations_100s)+len(stations_200s))
        axs_all = np.atleast_2d(axs_all)
        ic(axs_all.shape, axs_all[0].shape)
        i_station = 0
        for series in stations.keys():
            for station_id in stations[series]:
                station_data_folder = f'DeepLearning/data/{datapass}/Station{station_id}'

                data = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_SNR_Chi.npy', allow_pickle=True)
                data_SnrChiCut = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_SnrChiCut.npy', allow_pickle=True)
                data_in2016 = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_In2016.npy', allow_pickle=True)
                times = np.load(f'{station_data_folder}/FilteredStation{station_id}_Data_Times.npy', allow_pickle=True)

#                plotfolder = f'StationDataAnalysis/plots/Station_{station_id}/TimeStrips'
                if not os.path.exists(plotfolder):
                    os.makedirs(plotfolder)

                # Load all the data
                All_SNRs, All_RCR_Chi, MLCut_SNRs, MLCut_RCR_Chi, MLCut_Azi, MLCut_Zen = data
                ChiCut_SNRS, ChiCut_RCR_Chi, ChiCut_Azi, ChiCut_Zen, ChiCut_Traces = data_SnrChiCut
                in2016_SNRs, in2016_RCR_Chi, in2016_Azi, in2016_Zen, in2016_Traces, in2016_datetimes = data_in2016
                All_datetimes, MLCut_datetimes, ChiCut_datetimes = times



                ic(len(All_datetimes), len(MLCut_datetimes), len(ChiCut_datetimes), len(in2016_datetimes))
                if len(All_datetimes) == 0:
                    print(f'Skipping station {station_id} because no data')
                    continue
                All_datetimes = np.vectorize(datetime.datetime.fromtimestamp)(All_datetimes)
                if not len(MLCut_datetimes) == 0:
                    MLCut_datetimes = np.vectorize(datetime.datetime.fromtimestamp)(MLCut_datetimes)
                ChiCut_datetimes = np.vectorize(datetime.datetime.fromtimestamp)(ChiCut_datetimes)
                if not len(in2016_datetimes) == 0:
                    in2016_datetimes = np.vectorize(datetime.datetime.fromtimestamp)(in2016_datetimes)

                All_RCR_Chi = np.array(All_RCR_Chi)


                # Plot all data
                plotClusterTimes(ChiCut_datetimes, ChiCut_RCR_Chi, fig_all, axs_all[i_station], n_cluster=10, chi_cut=0.6)
                timestripScatter(All_datetimes, All_RCR_Chi, yearStart=yStart, yearEnd=yEnd, legend='All data', marker='o', color='k', markersize=2, fig=fig_all, axs=axs_all[i_station])
                timestripScatter(MLCut_datetimes, MLCut_RCR_Chi, yearStart=yStart, yearEnd=yEnd, legend='ML Cut', marker='o', color='b', markersize=2, fig=fig_all, axs=axs_all[i_station])
                timestripScatter(ChiCut_datetimes, ChiCut_RCR_Chi, yearStart=yStart, yearEnd=yEnd, legend='Chi Cut', marker='o', color='m', markersize=2, fig=fig_all, axs=axs_all[i_station])
                timestripScatter(in2016_datetimes, in2016_RCR_Chi, yearStart=yStart, yearEnd=yEnd, legend='2016 Events', marker='*', color='g', markersize=8, fig=fig_all, axs=axs_all[i_station])
                plt.legend()

                # Title on middle plot
                axs_all[i_station][int(len(axs_all[i_station])/2)].title.set_text(f'Station {station_id} {yStart}-{yEnd}')

                i_station += 1

        savename = f'{plotfolder}/Timestrip_AllData_{yStart}-{yEnd}.png'
        plt.savefig(savename, format='png')
        ic(f'Saved {savename}')
