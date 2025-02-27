from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import eventParameters as evtp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.modules.io.eventReader
from icecream import ic
import os
import numpy as np
import astrotools.auger as auger
import matplotlib.colors
import matplotlib.pyplot as plt
import configparser

class HRAevent:
    def __init__(self, event, DEBUG=False):
        # event should be the NuRadioReco event object
        # All parameters should have NuRadioReco units attached

        self.coreas_x = event.get_parameter(evtp.coreas_x) 
        self.coreas_y = event.get_parameter(evtp.coreas_y)
        self.event_id = event.get_id()
        self.weight = {} # Dictionary of list of weights for primary and secondary triggers for each station and sigma

        sim_shower = event.get_sim_shower(0)
        self.energy = sim_shower[shp.energy]
        self.zenith = sim_shower[shp.zenith]
        self.azimuth = sim_shower[shp.azimuth]

        self.recon_zenith = {}
        self.recon_azimuth = {}
        for station in event.get_stations():
            if station.has_triggered():
                self.recon_zenith[station.get_id()] = station.get_parameter(stnp.zenith)
                self.recon_azimuth[station.get_id()] = station.get_parameter(stnp.azimuth)

        if DEBUG:
            ic(self.event_id, self.energy, self.zenith, self.azimuth)

        self.trigger_sigmas = [3.5, 4, 4.5, 5, 5.5, 6]
        for sigma in self.trigger_sigmas:
            self.weight[sigma] = {}
        self.station_triggers = {}
        self.secondary_station_triggers = {}
        for sigma in self.trigger_sigmas:
            self.station_triggers[sigma] = []
            self.secondary_station_triggers[sigma] = []
        # Only doing the 3.5sigma triggers to begin with
        for station in event.get_stations():
            for sigma in self.trigger_sigmas:
                self.weight[sigma][station.get_id()] = [np.nan, np.nan]
            if station.has_triggered():

                for sigma in self.trigger_sigmas:
                    if station.has_triggered(trigger_name=f'primary_LPDA_2of4_{sigma}sigma'):
                        self.addTrigger(station.get_id(), sigma)

                if station.get_id() == 52:
                    for sigma in self.trigger_sigmas:
                        if station.has_triggered(trigger_name=f'secondary_LPDA_2of4_{sigma}sigma'):
                            self.addSecondaryTrigger(station.get_id(), sigma)


        self.direct_triggers = {}
        for sigma in self.trigger_sigmas:
            self.direct_triggers[sigma] = []
            for station_id in self.station_triggers[sigma]:
                if station_id < 100:
                    self.direct_triggers[sigma].append(station_id)

        self.reflected_triggers = {}
        for sigma in self.trigger_sigmas:
            self.reflected_triggers[sigma] = []
            for station_id in self.station_triggers[sigma]:
                if station_id > 100:
                    self.reflected_triggers[sigma].append(station_id)

    def getCoreasPosition(self):
        return self.coreas_x, self.coreas_y

    def getEnergy(self):
        return self.energy

    def getAngles(self):
        return self.zenith, self.azimuth

    def getEventID(self):
        return self.event_id

    def primaryTriggers(self):
        return self.station_triggers

    def secondaryTriggers(self):
        return self.secondary_station_triggers
    
    def directTriggers(self, sigma=5):
        return self.direct_triggers[sigma]
    
    def reflectedTriggers(self, sigma=5):
        return self.reflected_triggers[sigma]

    def addTrigger(self, station_id, sigma):
        if station_id not in self.station_triggers[sigma]:
            self.station_triggers[sigma].append(station_id)

    def addSecondaryTrigger(self, station_id, sigma):
        if station_id not in self.station_triggers[sigma]:
            self.station_triggers[sigma].append(station_id)

    def hasCoincidence(self, num=1, bad_stations=None, use_secondary=False, sigma=5):
        # Bad Stations should be a list of station IDs that are not to be included in the coincidence
        n_coinc = len(self.station_triggers[sigma])
        if use_secondary:
            n_coinc += len(self.secondary_station_triggers[sigma])
            n_coinc -= self.hasTriggered(station_id=52)
        if bad_stations is not None:
            for station_id in bad_stations:
                if station_id in self.station_triggers[sigma] and station_id != 52:
                    n_coinc -= 1
                elif station_id == 52 and use_secondary:
                    if station_id in self.secondary_station_triggers[sigma]:
                        n_coinc -= 1    
            return n_coinc > num
        return n_coinc > num

    def hasSecondaryCoincidence(self, sigma=5):
        return (len(self.station_triggers[sigma]) + len(self.secondary_station_triggers[sigma])) > 1

    def hasTriggered(self, station_id=None, sigma=5):
        if station_id is None:
            return len(self.station_triggers[sigma]) > 0
        return station_id in self.station_triggers[sigma]

    def inEnergyZenithBin(self, e_low, e_high, z_low, z_high):
        return e_low <= self.energy <= e_high and z_low <= self.zenith <= z_high
    
    def setWeight(self, weight, weight_name, primary=True, sigma=5):
        if weight_name not in self.weight:
            # Weights can be station ids, or can be a string such as 'all reflected', or '52 with direct only'
            self.weight[sigma][weight_name] = [np.nan, np.nan]
        if primary:
            self.weight[sigma][weight_name][0] = weight
        else:
            self.weight[sigma][weight_name][1] = weight


    def getWeight(self, weight_name, primary=True, sigma=5):
        if primary:
            return self.weight[sigma][weight_name][0]
        else:
            return self.weight[sigma][weight_name][1]

    def hasWeight(self, weight_name, sigma=5):
        return weight_name in self.weight[sigma]
    


def getHRAevents(nur_files):
    # Input a list of nur files to get a list of HRAevent objects

    eventReader = NuRadioReco.modules.io.eventReader.eventReader()
    HRAeventList = []
    for file in nur_files:
        eventReader.begin(file)
        for event in eventReader.run():
            HRAeventList.append(HRAevent(event))
        eventReader.end()

    return HRAeventList


def getHRAeventsFromDir(directory):
    # Input a directory to get a list of HRAevent objects from all nur files in that directory

    nur_files = []
    for file in os.listdir(directory):
        if file.endswith('.nur'):
            nur_files.append(os.path.join(directory, file))

    return getHRAevents(nur_files)

def getEnergyZenithBins():
    # Define the bins that are constant
    min_energy = 16.0
    max_energy = 20.1
    e_bins = 10**np.arange(min_energy, max_energy, 0.5) * units.eV
    z_bins = np.arange(0, 1.01, 0.2)
    z_bins = np.arccos(z_bins)
    z_bins[np.isnan(z_bins)] = 0
    z_bins = z_bins * units.rad
    z_bins = np.sort(z_bins)
    ic(e_bins, z_bins)

    return e_bins, z_bins

def getEnergyZenithArray():
    # Returns an array of the energy-zenith bins
    e_bins, z_bins = getEnergyZenithBins()

    return np.zeros((len(e_bins)-1, len(z_bins)-1))

def getnThrows(HRAeventList):
    # Returns the number of throws in each energy-zenith bin
    e_bins, z_bins = getEnergyZenithBins()

    n_throws = getEnergyZenithArray()

    for event in HRAeventList:
        energy_bin = np.digitize(event.getEnergy(), e_bins) - 1
        zenith_bin = np.digitize(event.getAngles()[0], z_bins) - 1
        if energy_bin < 0 or zenith_bin < 0 or energy_bin >= len(e_bins) or zenith_bin >= len(z_bins):
            ic(f'Outside of bins, {event.getEnergy()}, {event.getAngles()[0]} {energy_bin}, {zenith_bin}')
            continue
        n_throws[energy_bin][zenith_bin] += 1

    return n_throws    


def getBinnedTriggerRate(HRAeventList, num_coincidence=0, use_secondary=False, sigma=5):
    # Input a list of HRAevent objects to get the event rate in each energy-zenith bin

    e_bins, z_bins = getEnergyZenithBins()

    # Create a dictionary for the event rate per station
    direct_trigger_rate_dict = {}
    reflected_trigger_rate_dict = {}
    n_throws = getnThrows(HRAeventList)

    for event in HRAeventList:
        # n_throws += 1
        if not event.hasCoincidence(num_coincidence, use_secondary=use_secondary, sigma=sigma):
            # Event not triggered or meeting coincidence bar
            continue
        for station_id in event.directTriggers(sigma=sigma):
            energy_bin = np.digitize(event.getEnergy(), e_bins) - 1
            zenith_bin = np.digitize(event.getAngles()[0], z_bins) - 1
            if energy_bin < 0 or zenith_bin < 0 or energy_bin >= len(e_bins) or zenith_bin >= len(z_bins):
                ic(f'Outside of bins , {event.getEnergy()}, {event.getAngles()[0]} {energy_bin}, {zenith_bin}')
                continue
            if station_id not in direct_trigger_rate_dict:
                direct_trigger_rate_dict[station_id] = getEnergyZenithArray()
            direct_trigger_rate_dict[station_id][energy_bin][zenith_bin] += 1
        for station_id in event.reflectedTriggers(sigma=sigma):
            energy_bin = np.digitize(event.getEnergy(), e_bins) - 1
            zenith_bin = np.digitize(event.getAngles()[0], z_bins) - 1
            if energy_bin < 0 or zenith_bin < 0 or energy_bin >= len(e_bins) or zenith_bin >= len(z_bins):
                ic(f'Outside of bins , {event.getEnergy()}, {event.getAngles()[0]} {energy_bin}, {zenith_bin}')
                continue
            if station_id not in reflected_trigger_rate_dict:
                reflected_trigger_rate_dict[station_id] = getEnergyZenithArray()
            reflected_trigger_rate_dict[station_id][energy_bin][zenith_bin] += 1

    # Create an array of the combined rate of a single station using statistics of all stations, ignoring 52 and 32
    combined_trigger_rate = {}
    combined_trigger_rate['direct'] = getEnergyZenithArray()
    combined_trigger_rate['reflected'] = getEnergyZenithArray()
    # combined_throws = 0
    combined_throws = np.zeros_like(n_throws)
    for station_id in direct_trigger_rate_dict:
        if station_id in [32, 52]:
            continue
        combined_trigger_rate['direct'] += direct_trigger_rate_dict[station_id]
        combined_throws += n_throws
    combined_trigger_rate['direct'] /= combined_throws
    # combined_throws = 0
    combined_throws = np.zeros_like(n_throws)
    for station_id in reflected_trigger_rate_dict:
        if station_id in [132, 152]:
            continue
        combined_trigger_rate['reflected'] += reflected_trigger_rate_dict[station_id]
        combined_throws += n_throws
    combined_trigger_rate['reflected'] /= combined_throws

    # Normalise the event rate
    for station_id in direct_trigger_rate_dict:
        direct_trigger_rate_dict[station_id] /= n_throws
        direct_trigger_rate_dict[station_id][np.isnan(direct_trigger_rate_dict[station_id])] = 0
    for station_id in reflected_trigger_rate_dict:
        reflected_trigger_rate_dict[station_id] /= n_throws
        reflected_trigger_rate_dict[station_id][np.isnan(reflected_trigger_rate_dict[station_id])] = 0

    return direct_trigger_rate_dict, reflected_trigger_rate_dict, combined_trigger_rate, e_bins, z_bins

def getEventRateArray(e_bins, z_bins):
    # Returns an array of the event rate in each energy-zenith bin in evts/km^2/yr
    if e_bins[0] > 100:
        logE_bins = np.log10(e_bins/units.eV)
    else:
        logE_bins = e_bins
    eventRateArray = getEnergyZenithArray()
    for i in range(len(e_bins)-1):
        for j in range(len(z_bins)-1):
            # Old method did full geometric exposure, pi*(1+cos)(1-cos)
            # auger.event_rate has 2pi(1-cos), so correcting factor added
            # TODO find out argument why this needs to be included
            high_flux = auger.event_rate(logE_bins[i], logE_bins[i+1], zmax=z_bins[j+1]/units.deg, area=1*0.5*(1+np.cos(z_bins[j+1])))
            low_flux = auger.event_rate(logE_bins[i], logE_bins[i+1], zmax=z_bins[j]/units.deg, area=1*0.5*(1+np.cos(z_bins[j])))
            # high_flux = auger.event_rate(logE_bins[i], logE_bins[i+1], zmax=z_bins[j+1]/units.deg, area=1)
            # low_flux = auger.event_rate(logE_bins[i], logE_bins[i+1], zmax=z_bins[j]/units.deg, area=1)
            # ic(logE_bins[i], logE_bins[i+1], z_bins[j+1]/units.deg, z_bins[j]/units.deg, high_flux, low_flux)
            eventRateArray[i][j] = high_flux - low_flux

    return eventRateArray


def getEventRate(trigger_rate, e_bins, z_bins, max_distance=6.0*units.km):
    # Input a single trigger rate list to get the event rate in each energy-zenith bin

    logE_bins = np.log10(e_bins/units.eV)
    area = np.pi * max_distance**2
    
    eventRateArray = getEventRateArray(e_bins, z_bins)

    return eventRateArray * trigger_rate * area/units.km**2

def setHRAeventListRateWeight(HRAeventList, trigger_rate_array, weight_name, max_distance=6.0*units.km, sigma=5):
    # Set the event rate weight for each event in the HRAeventList

    e_bins, z_bins = getEnergyZenithBins()
    eventRateArray = getEventRateArray(e_bins, z_bins)
    n_throws = getnThrows(HRAeventList)

    area = np.pi * max_distance**2

    # Event weight = event_rate_bin / n_triggers_bin
    for event in HRAeventList:
        energy_bin = np.digitize(event.getEnergy(), e_bins) - 1
        zenith_bin = np.digitize(event.getAngles()[0], z_bins) - 1
        if energy_bin < 0 or zenith_bin < 0 or energy_bin >= len(e_bins) or zenith_bin >= len(z_bins):
            ic(f'Outside of bins , {event.getEnergy()}, {event.getAngles()[0]} {energy_bin}, {zenith_bin}')
            continue
        event_rate = eventRateArray[energy_bin][zenith_bin]

        n_trig = trigger_rate_array[energy_bin][zenith_bin] * n_throws[energy_bin][zenith_bin]
        if n_trig == 0:
            event.setWeight(0, weight_name, sigma=sigma)
        else:
            event.setWeight(event_rate * trigger_rate_array[energy_bin][zenith_bin] * (area/units.km**2) / n_trig, weight_name, sigma=sigma)

    return


def getCoincidencesTriggerRates(HRAeventList, bad_stations, use_secondary=False, force_station=None, sigma=5):
    # Return a list of coincidence events
    # As well as a dictionary of the trigger rate array for each number of coincidences
    e_bins, z_bins = getEnergyZenithBins()
    n_throws = getnThrows(HRAeventList)

    trigger_rate_coincidence =  {}
    for i in [2, 3, 4, 5, 6, 7]:
        trigger_rate_coincidence[i] = getEnergyZenithArray()
        for event in HRAeventList:
            if not event.hasCoincidence(i, bad_stations, use_secondary, sigma=sigma):
                # Event not triggered or meeting coincidence bar
                continue
            if force_station is not None and force_station not in event.station_triggers[sigma]:
                # Event not triggered by the station we want
                continue
            energy_bin = np.digitize(event.getEnergy(), e_bins) - 1
            zenith_bin = np.digitize(event.getAngles()[0], z_bins) - 1
            if energy_bin < 0 or zenith_bin < 0 or energy_bin >= len(e_bins) or zenith_bin >= len(z_bins):
                ic(f'Outside of bins , {event.getEnergy()}, {event.getAngles()[0]} {energy_bin}, {zenith_bin}')
                continue
            trigger_rate_coincidence[i][energy_bin][zenith_bin] += 1
        trigger_rate_coincidence[i] /= n_throws

    return trigger_rate_coincidence

def set_bad_imshow(array, value):
    ma = np.ma.masked_where(array == value, array)
    cmap = matplotlib.cm.viridis
    cmap.set_bad(color='white')
    return ma, cmap


def imshowRate(rate, title, savename, colorbar_label='Evts/yr'):

    e_bins, z_bins = getEnergyZenithBins()
    e_bins = np.log10(e_bins/units.eV)
    cos_bins = np.cos(z_bins)


    rate, cmap = set_bad_imshow(rate.T, 0)

    fig, ax = plt.subplots()

    # rate[1,:] = 100
    # rate[:, 2] = 200

    im = ax.imshow(rate, aspect='auto', origin='lower', extent=[min(e_bins), max(e_bins), min(cos_bins), max(cos_bins)], norm=matplotlib.colors.LogNorm(), cmap=cmap)
    # im = ax.imshow(rate, aspect='auto', origin='upper', extent=[min(e_bins), max(e_bins), min(cos_bins), max(cos_bins)], norm=matplotlib.colors.LogNorm(), cmap=cmap)

    # Since the y-axis is not evenly spaced in zenith, need to adjust axis labels
    ax_labels = []
    for z in z_bins:
        ax_labels.append('{:.0f}'.format(z/units.deg))
    ax_labels.reverse()
    # ax = plt.gca() # Removing to attempt to fix tick problems
    ic(cos_bins, ax_labels)
    ax.set_yticks(cos_bins)
    ax.set_yticklabels(ax_labels)
    ax.set_ylabel('Zenith Angle (deg)')

    ax.set_xlabel('log10(E/eV)')
    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(title)
    fig.savefig(savename)
    ic(f'Saved {savename}')
    plt.close(fig)
    return

def getXYWeights(HRAeventList, weight_name, use_primary=True):
    # Get a list of the events x/y with associated event rate as a weight

    x = []
    y = []
    weights = []


    for event in HRAeventList:
        cor_x, cor_y = event.getCoreasPosition()
        # Need to get the event rate corresponding to each event, previously calculated

        x.append(cor_x)
        y.append(cor_y)
        weights.append(event.getWeight(weight_name, use_primary))    # Append all events because non-triggering events have a weight of zero    

    return np.array(x), np.array(y), np.array(weights)

def getDirectReflTriggered(HRAeventList, use_primary=True):
    # Return two lists of the stations that were direct triggered and reflected triggered
    stations = [13, 14, 15, 17, 18, 19, 30, 32, 52]
    refl_stations = [113, 114, 115, 117, 118, 119, 130, 132, 152]

    direct_triggered = []
    reflected_triggered = []

    for event in HRAeventList:
        if len(stations) == len(direct_triggered) and len(refl_stations) == len(reflected_triggered):
            break
        if event.hasTriggered():
            for station_id in event.directTriggers():
                if station_id in stations and station_id not in direct_triggered:
                    direct_triggered.append(station_id)
            for station_id in event.reflectedTriggers():
                if station_id in refl_stations and station_id not in reflected_triggered:
                    reflected_triggered.append(station_id)

    return direct_triggered, reflected_triggered

def histAreaRate(x, y, weights, title, savename, dir_trig=[], refl_trig=[], exclude=[], colorbar_label='Evts/yr', max_distance=6.0*units.km):
    x_bins, y_bins = np.linspace(-max_distance/units.m, max_distance/units.m, 100), np.linspace(-max_distance/units.m, max_distance/units.m, 100)

    fig, ax = plt.subplots()

    # h, xedges, yedges = np.histogram2d(x, y, bins=(x_bins, y_bins), weights=weights)
    norm = matplotlib.colors.LogNorm(vmin=np.min(weights[np.nonzero(weights)]), vmax=np.max(weights)*5)
    h, xedges, yedges, im = ax.hist2d(x, y, bins=(x_bins, y_bins), weights=weights, cmap='viridis', norm=norm)
    # h, cmap = set_bad_imshow(h, 0)
    # extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
    # im = ax.imshow(h, extent=extent, norm=matplotlib.colors.LogNorm(), cmap=cmap)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax = plotStationLocations(ax, triggered=dir_trig, exclude=exclude, reflected_triggers=refl_trig)
    ax.legend()
    # plt.colorbar(label=colorbar_label)
    ax.set_title(title)
    fig.savefig(savename)
    ic(f'Saved {savename}')
    plt.close(fig)
    return


def getErrorEventRates(trigger_rate, HRAeventList, max_distance=6.0*units.km):
    # Return the event rate array as well as an array of the error in each bin
    # Error is capped at 100% of the event rate, for bins with 1 trigger going to 0 event rate

    e_bins, z_bins = getEnergyZenithBins()
    n_throws = getnThrows(HRAeventList)


    # Need to get n_trig per bin
    n_trig = trigger_rate * n_throws    
    
    trig_rate_error = np.sqrt(n_trig) / n_throws

    return getEventRate(trig_rate_error, e_bins, z_bins, max_distance=max_distance)

def plotRateWithError(eventRate, errorRate, savename, title):
    # Plot the rate summed in zenith with error bars
    e_bins, z_bins = getEnergyZenithBins()
    e_bins = np.log10(e_bins/units.eV)

    fig, ax = plt.subplots()

    eventRate[np.isnan(eventRate)] = 0
    errorRate[np.isnan(errorRate)] = 0
    ax.fill_between((e_bins[1:]+e_bins[:-1])/2, np.sum(eventRate - errorRate,axis=1), np.sum(eventRate + errorRate,axis=1), alpha=0.5, label=f'{np.sum(eventRate):.2f} +/- {np.sum(errorRate):.2f} Evts/Yr')

    for iZ in range(len(z_bins)-1):
        ax.fill_between((e_bins[1:]+e_bins[:-1])/2, eventRate[:,iZ] - errorRate[:,iZ], eventRate[:,iZ] + errorRate[:,iZ], alpha=0.5, label=f'{z_bins[iZ]/units.deg:.1f}-{z_bins[iZ+1]/units.deg:.1f}deg')

    ax.set_xlabel('log10(E/eV)')
    ax.set_ylabel('Evts/Yr')
    ax.legend()
    ax.set_title(title)
    fig.savefig(savename)
    ic(f'Saved {savename}')
    plt.close(fig)

    return


def getAnglesReconWeights(HRAeventList, weight_name, station_ids, use_primary=True):
    # Get a list of the events x/y with associated event rate as a weight

    zenith = []
    recon_zenith = []
    azimuth = []
    recon_azimuth = []
    weights = []

    if not isinstance(station_ids, list):
        station_ids = [station_ids]

    for event in HRAeventList:
        zenith.append(event.getAngles()[0])
        azimuth.append(event.getAngles()[1])
        weights.append(event.getWeight(weight_name, use_primary))    # Append all events because non-triggering events have a weight of zero    

        # if station_id in event.recon_zenith:
        if np.in1d(station_ids, list(event.recon_zenith.keys())).all():
            recon_zenith.append(event.recon_zenith[station_id])
            recon_azimuth.append(event.recon_azimuth[station_id])
        else:
            recon_zenith.append(np.nan)
            recon_azimuth.append(np.nan)

    return np.array(zenith), np.array(recon_zenith), np.array(azimuth), np.array(recon_azimuth), np.array(weights)


def histAngleRecon(zenith, azimuth, recon_zenith, recon_azimuth, weights, title, savename, colorbar_label='Evts/yr'):
    if max(zenith) < 10:
        zenith = np.rad2deg(zenith)
        recon_zenith = np.rad2deg(recon_zenith)
        azimuth = np.rad2deg(azimuth)
        recon_azimuth = np.rad2deg(recon_azimuth)
    
    zenith_bins, azimuth_bins = np.linspace(0, 90, 100), np.linspace(0, 360, 100)
    ax, fig = plt.subplots(nrows=1, ncols=2)

    norm = matplotlib.colors.LogNorm(vmin=np.min(weights[np.nonzero(weights)]), vmax=np.max(weights)*5)
    h, xedges, yedges, im = fig[0].hist2d(zenith, recon_zenith, bins=(zenith_bins, zenith_bins), weights=weights, cmap='viridis', norm=norm)
    fig[0].set_xlabel('True Zenith (deg)')
    fig[0].set_ylabel('Reconstructed Zenith (deg)')
    fig[0].set_title('Zenith')
    fig[0].set_aspect('equal')
    fig[0].set_xlim(0, 90)
    fig[0].set_ylim(0, 90)
    fig[0].grid()

    h, xedges, yedges, im = fig[1].hist2d(azimuth, recon_azimuth, bins=(azimuth_bins, azimuth_bins), weights=weights, cmap='viridis', norm=norm)
    fig[1].set_xlabel('True Azimuth (deg)')
    fig[1].set_ylabel('Reconstructed Azimuth (deg)')
    fig[1].set_title('Azimuth')
    fig[1].set_aspect('equal')
    fig[1].set_xlim(0, 360)
    fig[1].set_ylim(0, 360)
    fig[1].grid()

    fig[0].colorbar(im, ax=fig[0], label=colorbar_label)
    fig[1].colorbar(im, ax=fig[1], label=colorbar_label)
    plt.suptitle(title)

    plt.savefig(savename)
    ic(f'Saved {savename}')
    plt.close()

    # Also plot the difference in true and reconstructed angles
    zenith_diff = np.abs(zenith - recon_zenith)
    azimuth_diff = np.abs(azimuth - recon_azimuth)

    fig, ax = plt.subplots(nrows=1, ncols=2)

    diff_bins = np.linspace(-90, 90, 100)

    h, xedges, yedges, im = ax[0].hist(zenith_diff, bins=diff_bins, weights=weights)
    ax[0].set_xlabel('True - Reconstructed Zenith (deg)')
    ax[0].set_ylabel('Evts')
    ax[0].set_title('Zenith')

    h, xedges, yedges, im = ax[1].hist(azimuth_diff, bins=diff_bins, weights=weights)
    ax[1].set_xlabel('True - Reconstructed Azimuth (deg)')
    ax[1].set_ylabel('Evts')
    ax[1].set_title('Azimuth')

    plt.suptitle(f'{title} Difference in True and Reconstructed Angles')
    plt.savefig(savename.replace('.png', '_diff.png'))
    ic(f'Saved {savename.replace(".png", "_diff.png")}')
    plt.close()

    return

def plotStationLocations(ax, triggered=[], reflected_triggers=[], exclude=[]):
    station_locations = {13: [1044.4451, -91.337], 14: [610.4495, 867.118], 15: [-530.8272, -749.382], 17: [503.6394, -805.116], 
                         18: [0, 0], 19: [-371.9322, 950.705], 30: [-955.2426, 158.383], 32: [463.6215, -893.988], 52: [436.7442, 168.904]}

    for station_id in station_locations:
        if station_id in exclude:
            continue
        if station_id in reflected_triggers:
            ax.scatter(station_locations[station_id][0], station_locations[station_id][1], marker='s', edgecolors='red', s=12, facecolors='none')
            ax.text(station_locations[station_id][0], station_locations[station_id][1], f'{station_id}', fontsize=10, color='red')
        elif station_id in triggered:
            ax.scatter(station_locations[station_id][0], station_locations[station_id][1], marker='s', edgecolors='orange', s=12, facecolors='none')
            ax.text(station_locations[station_id][0], station_locations[station_id][1], f'{station_id}', fontsize=10, color='orange')
        else:
            ax.scatter(station_locations[station_id][0], station_locations[station_id][1], marker='s', edgecolors='black', s=12, facecolors='none')
            ax.text(station_locations[station_id][0], station_locations[station_id][1], f'{station_id}', fontsize=10, color='black')

    # ax.scatter([], [], 's', edgecolors='black', markersize=12, facecolors='none', label='')
    if len(triggered) > 0:
        ax.scatter([], [], marker='s', edgecolors='orange', s=12, facecolors='none', label='BL Triggered')
    if len(reflected_triggers) > 0:
        ax.scatter([], [], marker='s', edgecolors='red', s=12, facecolors='none', label='Refl Triggered')

    return ax

if __name__ == "__main__":

    # sim_folder = 'HRASimulation/output/HRA/1.27.25/'
    # numpy_folder = 'HRASimulation/output/HRA/1.27.25/numpy/' # For saving data to not have to reprocess
    # save_folder = f'HRASimulation/plots/2.3.25/'
    config = configparser.ConfigParser()
    config.read('HRASimulation/config.ini')
    sim_folder = config['FOLDERS']['sim_folder']
    numpy_folder = config['FOLDERS']['numpy_folder']
    save_folder = config['FOLDERS']['save_folder']
    plot_sigma = float(config['PLOTPARAMETERS']['trigger_sigma'])
    ic(plot_sigma)

    os.makedirs(save_folder, exist_ok=True)

    if not os.path.exists(numpy_folder):
        os.makedirs(numpy_folder)
    if os.path.exists(f'{numpy_folder}HRAeventList.npy'):        
        HRAeventList = np.load(f'{numpy_folder}HRAeventList.npy', allow_pickle=True)
        direct_trigger_rate_dict, reflected_trigger_rate_dict, combined_trigger_rate, e_bins, z_bins = np.load(f'{numpy_folder}trigger_rate_dict.npy', allow_pickle=True)
        direct_event_rate, reflected_event_rate, combined_event_rate = np.load(f'{numpy_folder}event_rate_dict.npy', allow_pickle=True)

    else:
        HRAeventList = getHRAeventsFromDir(sim_folder)
        direct_trigger_rate_dict, reflected_trigger_rate_dict, combined_trigger_rate, e_bins, z_bins = getBinnedTriggerRate(HRAeventList)
        direct_event_rate = {}
        for station_id in direct_trigger_rate_dict:
            ic(station_id)
            direct_event_rate[station_id] = getEventRate(direct_trigger_rate_dict[station_id], e_bins, z_bins)
            setHRAeventListRateWeight(HRAeventList, direct_trigger_rate_dict[station_id], weight_name=station_id)
        reflected_event_rate = {}
        for station_id in reflected_trigger_rate_dict:
            ic(station_id)
            reflected_event_rate[station_id] = getEventRate(reflected_trigger_rate_dict[station_id], e_bins, z_bins)
            setHRAeventListRateWeight(HRAeventList, reflected_trigger_rate_dict[station_id], weight_name=station_id)

        combined_event_rate = {}
        combined_event_rate['direct'] = getEventRate(combined_trigger_rate['direct'], e_bins, z_bins)
        combined_event_rate['reflected'] = getEventRate(combined_trigger_rate['reflected'], e_bins, z_bins)
        setHRAeventListRateWeight(HRAeventList, combined_trigger_rate['direct'], weight_name='combined_direct')
        setHRAeventListRateWeight(HRAeventList, combined_trigger_rate['reflected'], weight_name='combined_reflected')


        # np.save(f'{numpy_folder}HRAeventList.npy', HRAeventList)
        np.save(f'{numpy_folder}trigger_rate_dict.npy', [direct_trigger_rate_dict, reflected_trigger_rate_dict, combined_trigger_rate, e_bins, z_bins])
        np.save(f'{numpy_folder}event_rate_dict.npy', [direct_event_rate, reflected_event_rate, combined_event_rate])

    logE_bins = np.log10(e_bins/units.eV)
    cos_bins = np.cos(z_bins)

    if not os.path.exists(save_folder+'error_rate/'):
        os.makedirs(save_folder+'error_rate/')

    for station_id in direct_event_rate:
        imshowRate(direct_trigger_rate_dict[station_id], f'Direct Trigger Rate for Station {station_id}', f'{save_folder}direct_trigger_rate_{station_id}.png', colorbar_label='Trigger Rate')
        imshowRate(direct_event_rate[station_id], f'Direct Event Rate for Station {station_id}', f'{save_folder}direct_event_rate_{station_id}.png', colorbar_label=f'Evts/yr, Sum {np.nansum(direct_event_rate[station_id]):.3f}')
        event_rate_error = getErrorEventRates(direct_trigger_rate_dict[station_id], HRAeventList)
        plotRateWithError(direct_event_rate[station_id], event_rate_error, f'{save_folder}error_rate/direct_event_rate_error_{station_id}.png', f'Direct Event Rate Error for Station {station_id}')
    for station_id in reflected_event_rate:
        imshowRate(reflected_trigger_rate_dict[station_id], f'Reflected Trigger Rate for Station {station_id}', f'{save_folder}reflected_trigger_rate_{station_id}.png', colorbar_label='Trigger Rate')
        imshowRate(reflected_event_rate[station_id], f'Reflected Event Rate for Station {station_id}', f'{save_folder}reflected_event_rate_{station_id}.png', colorbar_label=f'Evts/yr, Sum {np.nansum(reflected_event_rate[station_id]):.3f}')
        event_rate_error = getErrorEventRates(reflected_trigger_rate_dict[station_id], HRAeventList)
        plotRateWithError(reflected_event_rate[station_id], event_rate_error, f'{save_folder}error_rate/reflected_event_rate_error_{station_id}.png', f'Reflected Event Rate Error for Station {station_id}')

    # Combined trigger rate and event rate plots
    imshowRate(combined_trigger_rate['direct'], 'Combined Direct Trigger Rate', f'{save_folder}combined_direct_trigger_rate.png', colorbar_label='Trigger Rate')
    imshowRate(combined_event_rate['direct'], 'Combined Direct Event Rate', f'{save_folder}combined_direct_event_rate.png', colorbar_label=f'Evts/yr, Sum {np.nansum(combined_event_rate["direct"]):.3f}')
    event_rate_error = getErrorEventRates(combined_trigger_rate['direct'], HRAeventList)
    plotRateWithError(combined_event_rate['direct'], event_rate_error, f'{save_folder}error_rate/combined_direct_event_rate_error.png', 'Combined Direct Event Rate Error')
    imshowRate(combined_trigger_rate['reflected'], 'Combined Reflected Trigger Rate', f'{save_folder}combined_reflected_trigger_rate.png', colorbar_label='Trigger Rate')
    imshowRate(combined_event_rate['reflected'], 'Combined Reflected Event Rate', f'{save_folder}combined_reflected_event_rate.png', colorbar_label=f'Evts/yr, Sum {np.nansum(combined_event_rate["reflected"]):.3f}')
    event_rate_error = getErrorEventRates(combined_trigger_rate['reflected'], HRAeventList)
    plotRateWithError(combined_event_rate['reflected'], event_rate_error, f'{save_folder}error_rate/combined_reflected_event_rate_error.png', 'Combined Reflected Event Rate Error')

    # Coincidence plots with reflections
    bad_stations = [32, 52, 132, 152]
    trigger_rate_coincidence = getCoincidencesTriggerRates(HRAeventList, bad_stations)
    event_rate_coincidence = {}
    for i in trigger_rate_coincidence:
        if not np.any(trigger_rate_coincidence[i] > 0):
            ic(f'No events for {i} coincidences')
            continue
        imshowRate(trigger_rate_coincidence[i], f'Trigger Rate for {i} Coincidences, w/Refl', f'{save_folder}trigger_rate_coincidence_{i}.png', colorbar_label='Trigger Rate')
        event_rate_coincidence[i] = getEventRate(trigger_rate_coincidence[i], e_bins, z_bins)
        setHRAeventListRateWeight(HRAeventList, trigger_rate_coincidence[i], weight_name=f'{i}_coincidence_wrefl')
        ic(event_rate_coincidence[i]), np.sum(event_rate_coincidence[i])
        imshowRate(event_rate_coincidence[i], f'Event Rate for {i} Coincidences, w/Refl', f'{save_folder}event_rate_coincidence_{i}.png', colorbar_label=f'Evts/yr, Sum {np.nansum(event_rate_coincidence[i]):.3f}')
        event_rate_error = getErrorEventRates(trigger_rate_coincidence[i], HRAeventList)
        plotRateWithError(event_rate_coincidence[i], event_rate_error, f'{save_folder}error_rate/event_rate_coincidence_error_{i}.png', f'Event Rate Error for {i} Coincidences')

    # Coincidence plots without reflections
    bad_stations = [32, 52, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    trigger_rate_coincidence = getCoincidencesTriggerRates(HRAeventList, bad_stations)
    event_rate_coincidence = {}
    for i in trigger_rate_coincidence:
        if not np.any(trigger_rate_coincidence[i] > 0):
            ic(f'No events for {i} coincidences')
            continue
        imshowRate(trigger_rate_coincidence[i], f'Trigger Rate for {i} Coincidences, w/o Refl', f'{save_folder}trigger_rate_coincidence_norefl_{i}.png', colorbar_label='Trigger Rate')
        event_rate_coincidence[i] = getEventRate(trigger_rate_coincidence[i], e_bins, z_bins)
        setHRAeventListRateWeight(HRAeventList, trigger_rate_coincidence[i], weight_name=f'{i}_coincidence_norefl')
        ic(event_rate_coincidence[i]), np.sum(event_rate_coincidence[i])
        imshowRate(event_rate_coincidence[i], f'Event Rate for {i} Coincidences w/o Refl', f'{save_folder}event_rate_coincidence_norefl_{i}.png', colorbar_label=f'Evts/yr, Sum {np.nansum(event_rate_coincidence[i]):.3f}')
        event_rate_error = getErrorEventRates(trigger_rate_coincidence[i], HRAeventList)
        plotRateWithError(event_rate_coincidence[i], event_rate_error, f'{save_folder}error_rate/event_rate_coincidence_norefl_error_{i}.png', f'Event Rate Error for {i} Coincidences w/o Refl')


    # Coincidence with reflection and station 52 upwards LPDA
    bad_stations = [32, 132, 152]
    trigger_rate_coincidence = getCoincidencesTriggerRates(HRAeventList, bad_stations, use_secondary=False, force_station=52)
    event_rate_coincidence = {}
    for i in trigger_rate_coincidence:
        if not np.any(trigger_rate_coincidence[i] > 0):
            ic(f'No events for {i} coincidences')
            continue
        imshowRate(trigger_rate_coincidence[i], f'Trigger Rate, {i} Coincidences, 52 upward forced w/Refl', f'{save_folder}trigger_rate_coincidence_52up_{i}.png', colorbar_label='Trigger Rate')
        event_rate_coincidence[i] = getEventRate(trigger_rate_coincidence[i], e_bins, z_bins)
        setHRAeventListRateWeight(HRAeventList, trigger_rate_coincidence[i], weight_name=f'{i}_coincidence_52up_wrefl')
        ic(event_rate_coincidence[i]), np.sum(event_rate_coincidence[i])
        imshowRate(event_rate_coincidence[i], f'Event Rate, {i} Coincidences, 52 upward forced w/Refl', f'{save_folder}event_rate_coincidence_52up_{i}.png', colorbar_label=f'Evts/yr, Sum {np.nansum(event_rate_coincidence[i]):.3f}')
        event_rate_error = getErrorEventRates(trigger_rate_coincidence[i], HRAeventList)
        plotRateWithError(event_rate_coincidence[i], event_rate_error, f'{save_folder}error_rate/event_rate_coincidence_52up_error_{i}.png', f'Event Rate Error for {i} Coincidences, 52 upward forced w/Refl')


    # Coincidence without reflection and station 52 upwards LPDA
    bad_stations = [32, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    trigger_rate_coincidence = getCoincidencesTriggerRates(HRAeventList, bad_stations, use_secondary=False, force_station=52)
    event_rate_coincidence = {}
    for i in trigger_rate_coincidence:
        if not np.any(trigger_rate_coincidence[i] > 0):
            ic(f'No events for {i} coincidences')
            continue
        imshowRate(trigger_rate_coincidence[i], f'Trigger Rate, {i} Coincidences, 52 upward forced w/o Refl', f'{save_folder}trigger_rate_coincidence_norefl_52up_{i}.png', colorbar_label='Trigger Rate')
        event_rate_coincidence[i] = getEventRate(trigger_rate_coincidence[i], e_bins, z_bins)
        setHRAeventListRateWeight(HRAeventList, trigger_rate_coincidence[i], weight_name=f'{i}_coincidence_52up_norefl')
        ic(event_rate_coincidence[i]), np.sum(event_rate_coincidence[i])
        imshowRate(event_rate_coincidence[i], f'Event Rate, {i} Coincidences, 52 upward forced w/o Refl', f'{save_folder}event_rate_coincidence_norefl_52up_{i}.png', colorbar_label=f'Evts/yr, Sum {np.nansum(event_rate_coincidence[i]):.3f}')
        event_rate_error = getErrorEventRates(trigger_rate_coincidence[i], HRAeventList)
        plotRateWithError(event_rate_coincidence[i], event_rate_error, f'{save_folder}error_rate/event_rate_coincidence_norefl_52up_error_{i}.png', f'Event Rate Error for {i} Coincidences, 52 upward forced w/o Refl')


    # Plot the zenith and azimuth reconstruction
    angle_save_folder = f'{save_folder}recon_angles/'
    if not os.path.exists(angle_save_folder):
        os.makedirs(angle_save_folder)
    for station_id in direct_event_rate:
        zenith, recon_zenith, azimuth, recon_azimuth, weights = getAnglesReconWeights(HRAeventList, f'{station_id}', station_id)
        histAngleRecon(zenith, azimuth, recon_zenith, recon_azimuth, weights, f'Reconstruction Angles for Station {station_id}', f'{angle_save_folder}recon_angles_{station_id}.png')
    
    # Also plot angles for combination of all reflected and direct stations together
    zenith, recon_zenith, azimuth, recon_azimuth, weights = getAnglesReconWeights(HRAeventList, 'combined_backlobe', [13, 14, 15, 17, 18, 19, 30])
    histAngleRecon(zenith, azimuth, recon_zenith, recon_azimuth, weights, f'Reconstruction Angles for Combined Backlobe Stations', f'{angle_save_folder}recon_angles_combined_backlobe.png')

    zenith, recon_zenith, azimuth, recon_azimuth, weights = getAnglesReconWeights(HRAeventList, 'combined_reflected', [113, 114, 115, 117, 118, 119, 130])
    histAngleRecon(zenith, azimuth, recon_zenith, recon_azimuth, weights, f'Reconstruction Angles for Combined Reflected Stations', f'{angle_save_folder}recon_angles_combined_reflected.png')





    # Resave to save the weights
    if not os.path.exists(f'{numpy_folder}HRAeventList.npy'):        
        np.save(f'{numpy_folder}HRAeventList.npy', HRAeventList)


