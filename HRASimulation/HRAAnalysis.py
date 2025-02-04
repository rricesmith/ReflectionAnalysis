from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import eventParameters as evtp
from NuRadioReco.framework.parameters import showerParameters as shp
import NuRadioReco.modules.io.eventReader
from icecream import ic
import os
import numpy as np
import astrotools.auger as auger
import matplotlib.colors
import matplotlib.pyplot as plt

class HRAevent:
    def __init__(self, event):
        # event should be the NuRadioReco event object
        # All parameters should have NuRadioReco units attached

        self.coreas_x = event.get_parameter(evtp.coreas_x) 
        self.coreas_y = event.get_parameter(evtp.coreas_y)
        self.event_id = event.get_id()
        self.station_triggers = []
        self.secondary_station_triggers = []

        sim_shower = event.get_sim_shower(0)
        self.energy = sim_shower[shp.energy]
        self.zenith = sim_shower[shp.zenith]
        self.azimuth = sim_shower[shp.azimuth]
        ic(self.event_id, self.energy, self.zenith, self.azimuth)

        # Only doing the 3.5sigma triggers to begin with
        for station in event.get_stations():
            if station.has_triggered():
                self.addTrigger(station.get_id())
                if station.get_id() == 52:
                    if station.has_triggered(trigger_name='secondary_LPDA_2of4_3.5sigma'):
                        self.addSecondaryTrigger(station.get_id())

        self.direct_triggers = []
        for station_id in self.station_triggers:
            if station_id < 100:
                self.direct_triggers.append(station_id)

        self.reflected_triggers = []
        for station_id in self.station_triggers:
            if station_id > 100:
                self.reflected_triggers.append(station_id)

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
    
    def directTriggers(self):
        return self.direct_triggers
    
    def reflectedTriggers(self):
        return self.reflected_triggers

    def addTrigger(self, station_id):
        if station_id not in self.station_triggers:
            self.station_triggers.append(station_id)

    def addSecondaryTrigger(self, station_id):
        if station_id not in self.secondary_station_triggers:
            self.secondary_station_triggers.append(station_id)

    def hasCoincidence(self, num=1, bad_stations=None):
        # Bad Stations should be a list of station IDs that are not to be included in the coincidence
        if bad_stations is not None:
            coinc = len(self.station_triggers)
            for station_id in bad_stations:
                if station_id in self.station_triggers:
                    coinc -= 1
            return coinc > num
        return len(self.station_triggers) > num

    def hasSecondaryCoincidence(self):
        return (len(self.station_triggers) + len(self.secondary_station_triggers)) > 1

    def hasTriggered(self, station_id=None):
        if station_id is None:
            return len(self.station_triggers) > 0
        return station_id in self.station_triggers

    def inEnergyZenithBin(self, e_low, e_high, z_low, z_high):
        return e_low <= self.energy <= e_high and z_low <= self.zenith <= z_high
    





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
    min_energy = 17
    max_energy = 20.1
    e_bins = 10**np.arange(min_energy, max_energy, 0.5) * units.eV
    z_bins = np.arange(0, 1.01, 0.2)
    z_bins = np.arccos(z_bins)
    z_bins[np.isnan(z_bins)] = 0
    z_bins = z_bins * units.rad
    z_bins = np.sort(z_bins)
    ic(e_bins, z_bins)

    return e_bins, z_bins

def getnThrows(HRAeventList):
    # Returns the number of throws in each energy-zenith bin
    e_bins, z_bins = getEnergyZenithBins()

    n_throws = np.zeros((len(e_bins), len(z_bins)))

    for event in HRAeventList:
        energy_bin = np.digitize(event.getEnergy(), e_bins) - 1
        zenith_bin = np.digitize(event.getAngles()[0], z_bins) - 1
        n_throws[energy_bin][zenith_bin] += 1

    return n_throws    

def getBinnedTriggerRate(HRAeventList, num_coincidence=0):
    # Input a list of HRAevent objects to get the event rate in each energy-zenith bin

    e_bins, z_bins = getEnergyZenithBins()

    # Create a dictionary for the event rate per station
    direct_trigger_rate_dict = {}
    reflected_trigger_rate_dict = {}
    n_throws = getnThrows(HRAeventList)

    for event in HRAeventList:
        n_throws += 1
        if not event.hasCoincidence(num_coincidence):
            # Event not triggered or meeting coincidence bar
            continue
        for station_id in event.directTriggers():
            energy_bin = np.digitize(event.getEnergy(), e_bins) - 1
            zenith_bin = np.digitize(event.getAngles()[0], z_bins) - 1
            if station_id not in direct_trigger_rate_dict:
                direct_trigger_rate_dict[station_id] = np.zeros((len(e_bins), len(z_bins)))
            direct_trigger_rate_dict[station_id][energy_bin][zenith_bin] += 1
        for station_id in event.reflectedTriggers():
            energy_bin = np.digitize(event.getEnergy(), e_bins) - 1
            zenith_bin = np.digitize(event.getAngles()[0], z_bins) - 1
            if station_id not in reflected_trigger_rate_dict:
                reflected_trigger_rate_dict[station_id] = np.zeros((len(e_bins), len(z_bins)))
            reflected_trigger_rate_dict[station_id][energy_bin][zenith_bin] += 1

    # Normalise the event rate
    for station_id in direct_trigger_rate_dict:
        direct_trigger_rate_dict[station_id] /= n_throws
    for station_id in reflected_trigger_rate_dict:
        reflected_trigger_rate_dict[station_id] /= n_throws

    return direct_trigger_rate_dict, reflected_trigger_rate_dict, e_bins, z_bins

def getEventRateArray(e_bins, z_bins):
    # Returns an array of the event rate in each energy-zenith bin in evts/km^2/yr
    if e_bins[0] > 100:
        logE_bins = np.log10(e_bins/units.eV)
    else:
        logE_bins = e_bins
    eventRateArray = np.zeros((len(e_bins), len(z_bins)))
    for i in range(len(e_bins)-1):
        for j in range(len(z_bins)-1):
            high_flux = auger.event_rate(logE_bins[i], logE_bins[i+1], zmax=z_bins[j+1]/units.deg, area=1)
            low_flux = auger.event_rate(logE_bins[i], logE_bins[i+1], zmax=z_bins[j]/units.deg, area=1)
            ic(logE_bins[i], logE_bins[i+1], z_bins[j+1]/units.deg, z_bins[j]/units.deg, high_flux, low_flux)
            eventRateArray[i][j] = high_flux - low_flux

    return eventRateArray


def getEventRate(trigger_rate, e_bins, z_bins, max_distance=2.5*units.km):
    # Input a single trigger rate list to get the event rate in each energy-zenith bin

    logE_bins = np.log10(e_bins/units.eV)
    area = np.pi * max_distance**2
    
    eventRateArray = getEventRateArray(e_bins, z_bins)

    return eventRateArray * trigger_rate * area/units.km**2

def getCoincidencesTriggerRates(HRAeventList, bad_stations):
    # Return a list of coincidence events
    # As well as a dictionary of the trigger rate array for each number of coincidences
    e_bins, z_bins = getEnergyZenithBins()
    n_throws = getnThrows(HRAeventList)

    trigger_rate_coincidence =  {}
    for i in [2, 3, 4, 5, 6, 7]:
        trigger_rate_coincidence[i] = np.zeros((len(e_bins), len(z_bins)))
        for event in HRAeventList:
            if not event.hasCoincidence(i, bad_stations):
                # Event not triggered or meeting coincidence bar
                continue
            energy_bin = np.digitize(event.getEnergy(), e_bins) - 1
            zenith_bin = np.digitize(event.getAngles()[0], z_bins) - 1
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
    cos_bins = np.cos(z_bins)

    rate, cmap = set_bad_imshow(rate, 0)

    fig, ax = plt.subplots()

    # rate[1,:] = 100
    # rate[:, 0] = 200

    im = ax.imshow(rate, aspect='auto', origin='lower', extent=[min(e_bins), max(e_bins), min(cos_bins), max(cos_bins)], norm=matplotlib.colors.LogNorm(), cmap=cmap)

    # Since the y-axis is not evenly spaced in zenith, need to adjust axis labels
    ax_labels = []
    for z in z_bins:
        ax_labels.append('{:.0f}'.format(z/units.deg))
    ax = plt.gca()
    ax.set_yticks(cos_bins)
    ax.set_yticklabels(ax_labels)

    ax.set_xlabel('log10(E/eV)')
    ax.set_ylabel('Zenith Angle (deg)')
    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(title)
    fig.savefig(savename)
    ic(f'Saved {savename}')
    plt.close(fig)
    return

if __name__ == "__main__":

    sim_folder = 'HRASimulation/output/HRA/1.27.25/'
    numpy_folder = 'HRASimulation/output/HRA/1.27.25/numpy/' # For saving data to not have to reprocess
    save_folder = f'HRASimulation/plots/2.3.25/'
    os.makedirs(save_folder, exist_ok=True)

    if not os.path.exists(numpy_folder):
        os.makedirs(numpy_folder)
    if os.path.exists(f'{numpy_folder}HRAeventList.npy'):        
        HRAeventList = np.load(f'{numpy_folder}HRAeventList.npy', allow_pickle=True)
        direct_trigger_rate_dict, reflected_trigger_rate_dict, e_bins, z_bins = np.load(f'{numpy_folder}trigger_rate_dict.npy', allow_pickle=True)
    

    else:
        HRAeventList = getHRAeventsFromDir(sim_folder)
        direct_trigger_rate_dict, reflected_trigger_rate_dict, e_bins, z_bins = getBinnedTriggerRate(HRAeventList)
        direct_event_rate = {}
        for station_id in direct_trigger_rate_dict:
            ic(station_id)
            direct_event_rate[station_id] = getEventRate(direct_trigger_rate_dict[station_id], e_bins, z_bins)
        reflected_event_rate = {}
        for station_id in reflected_trigger_rate_dict:
            ic(station_id)
            reflected_event_rate[station_id] = getEventRate(reflected_trigger_rate_dict[station_id], e_bins, z_bins)

        np.save(f'{numpy_folder}HRAeventList.npy', HRAeventList)
        np.save(f'{numpy_folder}trigger_rate_dict.npy', [direct_trigger_rate_dict, reflected_trigger_rate_dict, e_bins, z_bins])


    logE_bins = np.log10(e_bins/units.eV)
    cos_bins = np.cos(z_bins)


    for station_id in direct_event_rate:
        imshowRate(direct_trigger_rate_dict[station_id], f'Direct Trigger Rate for Station {station_id}', f'{save_folder}direct_trigger_rate_{station_id}.png', colorbar_label='Trigger Rate')
        imshowRate(direct_event_rate[station_id], f'Direct Event Rate for Station {station_id}', f'{save_folder}direct_event_rate_{station_id}.png', colorbar_label=f'Evts/yr, Sum {np.sum(direct_event_rate[station_id]):.3f}')
    for station_id in reflected_event_rate:
        imshowRate(reflected_trigger_rate_dict[station_id], f'Reflected Trigger Rate for Station {station_id}', f'{save_folder}reflected_trigger_rate_{station_id}.png', colorbar_label='Trigger Rate')
        imshowRate(reflected_event_rate[station_id], f'Reflected Event Rate for Station {station_id}', f'{save_folder}reflected_event_rate_{station_id}.png', colorbar_label=f'Evts/yr, Sum {np.sum(reflected_event_rate[station_id]):.3f}')


    # Coincidence plots with reflections
    bad_stations = [32, 52, 132, 152]
    trigger_rate_coincidence = getCoincidencesTriggerRates(HRAeventList, bad_stations)
    event_rate_coincidence = {}
    for i in trigger_rate_coincidence:
        imshowRate(trigger_rate_coincidence[i], f'Trigger Rate for {i} Coincidences', f'{save_folder}trigger_rate_coincidence_{i}.png', colorbar_label='Trigger Rate')
        event_rate_coincidence[i] = getEventRate(trigger_rate_coincidence[i], e_bins, z_bins)

    for i in event_rate_coincidence:
        imshowRate(event_rate_coincidence[i], f'Event Rate for {i} Coincidences', f'{save_folder}event_rate_coincidence_{i}.png', colorbar_label=f'Evts/yr, Sum {np.sum(event_rate_coincidence[i]):.3f}')

    # Coincidence plots without reflections
    bad_stations = [32, 52, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    trigger_rate_coincidence = getCoincidencesTriggerRates(HRAeventList, bad_stations)
    event_rate_coincidence = {}
    for i in trigger_rate_coincidence:
        imshowRate(trigger_rate_coincidence[i], f'Trigger Rate for {i} Coincidences, no refl', f'{save_folder}trigger_rate_coincidence_norefl_{i}.png', colorbar_label='Trigger Rate')
        event_rate_coincidence[i] = getEventRate(trigger_rate_coincidence[i], e_bins, z_bins)

    for i in event_rate_coincidence:
        ic(event_rate_coincidence[i]), np.sum(event_rate_coincidence[i])
        imshowRate(event_rate_coincidence[i], f'Event Rate for {i} Coincidences', f'{save_folder}event_rate_coincidence_norefl_{i}.png', colorbar_label=f'Evts/yr, Sum {np.sum(event_rate_coincidence[i]):.3f}')

