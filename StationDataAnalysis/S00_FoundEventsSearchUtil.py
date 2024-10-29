import json
from icecream import ic


def inStation2016(time_unix, station_id='all', buffer=3):
    #This checks if given event is listed in found list of events for that particular station
    # Requires the time passed in to be a unix number
    # Station_id can be 'all' or a number of a station in data - 13, 14, 15, 17, 18, and 30
    # Buffer checks for if events are within the passed seconds of an event previously found, ie within 3 seconds before/after

    file = f'StationDataAnalysis/2016FoundEvents.json'
    with open(file) as f:
        data = json.load(f)

        if station_id == 'all':
            station_string = 'FoundEvents'
        else:
            station_string = f'Station{station_id}Found'
        for event in data[station_string]:
            # ic(event, time_unix, event == time_unix)
            if (event - buffer) <= time_unix <= (event + buffer):
                return True

    return False




