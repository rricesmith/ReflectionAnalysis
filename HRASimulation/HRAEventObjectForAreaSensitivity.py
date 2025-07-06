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

stns_100s = [13, 15, 18, 113, 115, 118]
stns_100s_noise = 20 * units.mV
stns_200s = [14, 17, 19, 30, 114, 117, 119, 130]
stns_200s_noise = 23 * units.mV
stns_300s_noise = 11 * units.mV

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

        self.SNR = {} # Dictionary of SNR values for each station

        self.recon_zenith = {}
        self.recon_azimuth = {}
        for station in event.get_stations():
            if station.has_triggered():
                try:
                    self.recon_zenith[station.get_id()] = station.get_parameter(stnp.zenith)
                    self.recon_azimuth[station.get_id()] = station.get_parameter(stnp.azimuth)
                except KeyError:
                    # If the station does not have a zenith or azimuth, set to None
                    self.recon_zenith[station.get_id()] = None
                    self.recon_azimuth[station.get_id()] = None

        if DEBUG:
            ic(self.event_id, self.energy, self.zenith, self.azimuth)

        self.trigger_sigmas = [50, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        for sigma in self.trigger_sigmas:
            self.weight[sigma] = {}
        self.station_triggers = {}
        self.secondary_station_triggers = {}
        for sigma in self.trigger_sigmas:
            self.station_triggers[sigma] = []
            self.secondary_station_triggers[sigma] = []
        # Only doing the 3.5sigma triggers to begin with

        from HRAStationDataAnalysis.HRADataConvertToNpy import calcSNR
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

            if station.get_id() == 52:
                LPDA_channels = [4, 5, 6, 7]
            else:
                LPDA_channels = [0, 1, 2, 3]
            if station.get_id() in stns_100s:
                Vrms = stns_100s_noise
            elif station.get_id() in stns_200s:
                Vrms = stns_200s_noise
            else:
                Vrms = stns_300s_noise
            traces = []
            for chId, channel in enumerate(station.iter_channels(use_channels=LPDA_channels)):
                traces.append(channel.get_trace())
            self.SNR[station.get_id()] = calcSNR(traces, Vrms)
            
            


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
    
    def getSNR(self, station_id):
        if station_id in self.SNR:
            return self.SNR[station_id]
        else:
            return None

    def primaryTriggers(self):
        return self.station_triggers

    def secondaryTriggers(self):
        return self.secondary_station_triggers
    
    def directTriggers(self, sigma=4.5, sigma_52=7):
        if sigma == sigma_52:
            return self.direct_triggers[sigma]
        else:
            # ic(self.direct_triggers)
            dt = self.direct_triggers[sigma]
            dt_52 = self.direct_triggers[sigma_52]
            if 52 in dt_52 and 52 not in dt:
                dt.append(52)
            elif 52 in dt and 52 not in dt_52:
                dt.remove(52)
            return dt

    def reflectedTriggers(self, sigma=4.5, sigma_52=7):
        if sigma == sigma_52:
            return self.reflected_triggers[sigma]
        else:
            rt = self.reflected_triggers[sigma]
            rt_52 = self.reflected_triggers[sigma_52]
            if 52 in rt_52 and 52 not in rt:
                rt.append(52)
            elif 52 in rt and 52 not in rt_52:
                rt.remove(52)
            return rt

    def addTrigger(self, trigger_name, sigma):
        if trigger_name not in self.station_triggers[sigma]:
            self.station_triggers[sigma].append(trigger_name)

    def addSecondaryTrigger(self, station_id, sigma):
        if station_id not in self.station_triggers[sigma]:
            self.station_triggers[sigma].append(station_id)

    def hasCoincidence(self, num=1, bad_stations=None, use_secondary=False, sigma=4.5, sigma_52=7):
        # Bad Stations should be a list of station IDs that are not to be included in the coincidence
        n_coinc = len(self.station_triggers[sigma])
        for key in self.station_triggers[sigma]:
            if isinstance(key, str):
                # Remove the keys that are summed from others
                n_coinc -= 1
        if use_secondary:
            n_coinc += len(self.secondary_station_triggers[sigma_52])
            n_coinc -= self.hasTriggered(station_id=52, sigma=sigma_52)
        if bad_stations is not None:
            for station_id in bad_stations:
                if station_id in self.station_triggers[sigma] and station_id != 52:
                    n_coinc -= 1
                elif station_id == 52 and use_secondary:
                    if station_id in self.secondary_station_triggers[sigma_52]:
                        n_coinc -= 1    
            return n_coinc > num
        return n_coinc > num

    def hasSecondaryCoincidence(self, sigma_52=7):
        return (len(self.station_triggers[sigma_52]) + len(self.secondary_station_triggers[sigma_52])) > 1

    def hasTriggered(self, trigger_name=None, sigma=4.5):
        if trigger_name is None:
            return len(self.station_triggers[sigma]) > 0
        return trigger_name in self.station_triggers[sigma]

    def inEnergyZenithBin(self, e_low, e_high, z_low, z_high):
        return e_low <= self.energy <= e_high and z_low <= self.zenith <= z_high
    
    def setWeight(self, weight, weight_name, primary=True, sigma=4.5):
        if weight_name not in self.weight:
            # Weights can be station ids, or can be a string such as 'all reflected', or '52 with direct only'
            self.weight[sigma][weight_name] = [np.nan, np.nan]
        if primary:
            self.weight[sigma][weight_name][0] = weight
        else:
            self.weight[sigma][weight_name][1] = weight


    def getWeight(self, weight_name, primary=True, sigma=4.5):
        if primary:
            return self.weight[sigma][weight_name][0]
        else:
            return self.weight[sigma][weight_name][1]

    def hasWeight(self, weight_name, sigma=4.5):
        return weight_name in self.weight[sigma]
    