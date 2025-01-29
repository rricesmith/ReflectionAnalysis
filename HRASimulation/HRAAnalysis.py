from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import eventParameters as evtp
from NuRadioReco.framework.parameters import showerParameters as shp


class HRAevent:
    def __init__(self, event):
        # event should be the NuRadioReco event object
        # All parameters should have NuRadioReco units attached

        self.coreas_x = event.get_parameter(evtp.coreas_x) 
        self.coreas_y = event.get_parameter(evtp.coreas_y)
        self.event_id = event.get_parameter(event.get_id())
        self.station_triggers = []
        self.secondary_station_triggers = []

        sim_shower = evt.get_sim_shower(0)
        self.energy = sim_shower[shp.energy]
        self.zenith = sim_shower[shp.zenith]
        self.azimuth = sim_shower[shp.azimuth]
        ic(self.event_id, self.energy, self.zenith, self.azimuth)


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

    def addTrigger(self, station_id):
        if station_id not in self.station_triggers:
            self.station_triggers.append(station_id)

    def addSecondaryTrigger(self, station_id):
        if station_id not in self.secondary_station_triggers:
            self.secondary_station_triggers.append(station_id)

    