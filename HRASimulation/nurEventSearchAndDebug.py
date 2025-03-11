import os
import configparser
import NuRadioReco.modules.io.eventReader
from icecream import ic
from NuRadioReco.framework.parameters import eventParameters as evtp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('HRASimulation/config.ini')
    sim_folder = config['FOLDERS']['sim_folder']


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

    eventReader = NuRadioReco.modules.io.eventReader.eventReader()
    for file in nur_files:
        eventReader.begin(file)
        for event in eventReader.run():
            triggered = False
            station_id = 0
            for station in event.get_stations():
                if station.get_id() in station_check:
                    if station.has_triggered(trigger_name=f'primary_LPDA_2of4_{sigma}sigma'):
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

            ic(core_x, core_y, sim_energy, sim_zenith, sim_azimuth, station_zenith, station_azimuth)

            pos_check = (core_x - station.get_parameter(stnp.x))**2 + (core_y - station.get_parameter(stnp.y))**2 >= rdist**2
            eng_check = englim[0] <= sim_energy <= englim[1]
            zen_check = zenlim[0] <= sim_zenith <= zenlim[1]
            azi_check = azilim[0] <= sim_azimuth <= azilim[1]

            ic(pos_check, eng_check, zen_check, azi_check)
            quit()