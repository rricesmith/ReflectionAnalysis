import numpy as np
import datetime



def getTimeEventMasks(times_raw, event_ids):
    # Global function to get the masks for good data to process from the total dataset


    zerotime_mask = times_raw != 0
    min_datetime_threshold = datetime.datetime(2013, 1, 1).timestamp()
    pretime_mask = times_raw >= min_datetime_threshold
    initial_mask = zerotime_mask & pretime_mask

    # Apply zerotime and pretime masks
    times_raw = times_raw[initial_mask]
    event_ids = event_ids[initial_mask]


    # Also create mask for unique event IDs
    time_eventid_pairs = np.stack((times_raw, event_ids), axis=-1)
    _, unique_indices = np.unique(time_eventid_pairs, axis=0, return_index=True)
    unique_indices.sort()

    return initial_mask, unique_indices


## Global parameter for date checking
import datetime
time_unix = datetime.datetime(2017, 2, 16, 19, 9, 51).timestamp() # RCR-BL event found

def timeInTimes(times_list):
    for time in times_list:
        if abs(time - time_unix) < 1:  # within 1 second
            return True
    return False