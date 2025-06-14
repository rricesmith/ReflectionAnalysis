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


