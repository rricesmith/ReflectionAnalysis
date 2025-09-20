import numpy as np
import datetime
import os
import glob


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


def load_station_events(
    date_str,
    station_id,
    time_files_template,
    event_id_files_template,
    external_cuts_file_path=None,
    apply_external_cuts=True,
    parameter_name=None,
    parameter_files_template=None,
    filename_event_count_regex=None,
):
    """
    Unified loader for station events.

    - Loads and concatenates Times and EventIDs using provided templates.
    - Applies initial time filter (non-zero, >= 2013-01-01) and uniqueness by (time,event_id).
    - Optionally applies external cuts loaded from `external_cuts_file_path`.

    Args:
      date_str: Date string used to format templates.
      station_id: Station number (int or str) used to format templates.
      time_files_template: Glob template with placeholders `{date}` and `{station_id}` for Times files.
      event_id_files_template: Glob template with placeholders `{date}` and `{station_id}` for EventIDs files.
      external_cuts_file_path: Path to cuts .npy file (dict of boolean arrays or a single boolean array).
      apply_external_cuts: If True and cuts file exists, apply combined cuts mask.

        Returns:
      A dict with keys:
        - 'all_times': concatenated raw times array
        - 'all_event_ids': concatenated raw event IDs array
        - 'initial_mask': boolean mask over raw arrays (time validity)
        - 'unique_indices': indices selecting unique (time, event_id) among initial_masked entries
        - 'pre_external_grcis': raw indices after initial+unique filtering
        - 'cuts_mask': combined external cuts mask (or None)
        - 'final_grcis': raw indices after all cuts
        - 'final_times': times after all cuts (ordered)
        - 'final_event_ids': event IDs after all cuts (ordered)
                - 'final_idx_to_grci': dict mapping final_idx (0..N-1) to raw concatenated index (GRCI)
                - 'parameters': dict mapping parameter_name -> {
                            'by_grci': {grci: value},
                            'by_final_idx': list aligned with final indices
                    } if parameter_name/templates provided; else empty dict
    """

    # Resolve files
    time_glob = time_files_template.format(date=date_str, station_id=station_id)
    evt_glob = event_id_files_template.format(date=date_str, station_id=station_id)
    sorted_time_files = sorted(glob.glob(time_glob))
    sorted_eventid_files = sorted(glob.glob(evt_glob))

    if not sorted_time_files or not sorted_eventid_files or (len(sorted_time_files) != len(sorted_eventid_files)):
        return {
            'all_times': np.array([]),
            'all_event_ids': np.array([]),
            'initial_mask': np.array([], dtype=bool),
            'unique_indices': np.array([], dtype=int),
            'pre_external_grcis': np.array([], dtype=int),
            'cuts_mask': None,
            'final_grcis': np.array([], dtype=int),
            'final_times': np.array([]),
            'final_event_ids': np.array([]),
            'final_idx_to_grci': {},
            'parameters': {},
        }

    # Load and concatenate
    all_times = np.concatenate([np.load(f) for f in sorted_time_files]) if sorted_time_files else np.array([])
    all_event_ids = np.concatenate([np.load(f) for f in sorted_eventid_files]) if sorted_eventid_files else np.array([])

    if all_times.shape[0] != all_event_ids.shape[0]:
        # Inconsistent inputs; return empty to signal caller
        return {
            'all_times': all_times,
            'all_event_ids': all_event_ids,
            'initial_mask': np.array([], dtype=bool),
            'unique_indices': np.array([], dtype=int),
            'pre_external_grcis': np.array([], dtype=int),
            'cuts_mask': None,
            'final_grcis': np.array([], dtype=int),
            'final_times': np.array([]),
            'final_event_ids': np.array([]),
            'final_idx_to_grci': {},
            'parameters': {},
        }

    # Initial time + uniqueness masks
    initial_mask, unique_indices = getTimeEventMasks(all_times, all_event_ids)
    grcis_time_passed = np.where(initial_mask)[0]
    pre_external_grcis = grcis_time_passed[unique_indices]

    # External cuts mask
    cuts_mask = None
    if apply_external_cuts and external_cuts_file_path and os.path.exists(external_cuts_file_path):
        try:
            cuts_data = np.load(external_cuts_file_path, allow_pickle=True)
            if isinstance(cuts_data, np.ndarray) and cuts_data.ndim == 0 and isinstance(cuts_data.item(), dict):
                cuts_data = cuts_data.item()
            if isinstance(cuts_data, dict) and cuts_data:
                first_key = next(iter(cuts_data))
                n = len(cuts_data[first_key])
                mask = np.ones(n, dtype=bool)
                for arr in cuts_data.values():
                    if len(arr) == n:
                        mask &= arr
                # Align lengths if needed
                if n != len(pre_external_grcis):
                    min_len = min(n, len(pre_external_grcis))
                    tmp = np.zeros(len(pre_external_grcis), dtype=bool)
                    tmp[:min_len] = mask[:min_len]
                    mask = tmp
                cuts_mask = mask
            elif isinstance(cuts_data, np.ndarray) and cuts_data.dtype == bool:
                if len(cuts_data) != len(pre_external_grcis):
                    min_len = min(len(cuts_data), len(pre_external_grcis))
                    tmp = np.zeros(len(pre_external_grcis), dtype=bool)
                    tmp[:min_len] = cuts_data[:min_len]
                    cuts_mask = tmp
                else:
                    cuts_mask = cuts_data
        except Exception:
            cuts_mask = None

    if cuts_mask is None:
        final_grcis = pre_external_grcis
    else:
        final_grcis = pre_external_grcis[cuts_mask]

    final_times = all_times[final_grcis] if final_grcis.size else np.array([])
    final_event_ids = all_event_ids[final_grcis] if final_grcis.size else np.array([])
    final_idx_to_grci = {i: grci for i, grci in enumerate(final_grcis)}

    # Optional: parameter loading
    parameters = {}
    if parameter_name and parameter_files_template and filename_event_count_regex is not None:
        actual_param_glob = parameter_files_template.format(
            date=date_str, station_id=station_id, parameter_name=parameter_name
        )
        sorted_param_files = sorted(glob.glob(actual_param_glob))
        by_grci = {}
        if sorted_param_files:
            cumulative_grci_offset = 0
            # For each parameter file, extract event count from filename; values are stored in same order as events
            for param_file_path in sorted_param_files:
                match = filename_event_count_regex.search(os.path.basename(param_file_path))
                if not match:
                    continue
                try:
                    events_in_file = int(match.group(1))
                except Exception:
                    continue
                try:
                    arr = np.load(param_file_path, allow_pickle=True)
                    if parameter_name != 'Traces':
                        if arr.ndim > 1:
                            arr = arr.squeeze()
                        if arr.ndim == 0:
                            arr = np.array([arr.item()])
                except Exception:
                    arr = None

                # Map local array entries to GRCI range
                file_grci_start = cumulative_grci_offset
                file_grci_end = cumulative_grci_offset + events_in_file
                if arr is not None:
                    max_len = min(len(arr), events_in_file)
                    for local_idx in range(max_len):
                        grci = file_grci_start + local_idx
                        by_grci[grci] = arr[local_idx]
                cumulative_grci_offset += events_in_file

        # Build by_final_idx aligned values
        by_final_idx = []
        for i in range(len(final_idx_to_grci)):
            grci = final_idx_to_grci[i]
            by_final_idx.append(by_grci.get(grci, np.nan))
        parameters[parameter_name] = {
            'by_grci': by_grci,
            'by_final_idx': by_final_idx,
        }

    return {
        'all_times': all_times,
        'all_event_ids': all_event_ids,
        'initial_mask': initial_mask,
        'unique_indices': unique_indices,
        'pre_external_grcis': pre_external_grcis,
        'cuts_mask': cuts_mask,
        'final_grcis': final_grcis,
        'final_times': final_times,
        'final_event_ids': final_event_ids,
        'final_idx_to_grci': final_idx_to_grci,
        'parameters': parameters,
    }


def build_station_cuts_path(date_cuts, date_str, station_id, station_data_root='HRAStationDataAnalysis/StationData'):
    """
    Construct the standardized path to a station cuts file using date_cuts.

    Returns path: {station_data_root}/cuts/{date_cuts}/{date_str}_Station{station_id}_Cuts.npy
    """
    return os.path.join(
        station_data_root,
        'cuts',
        str(date_cuts),
        f'{date_str}_Station{station_id}_Cuts.npy'
    )


## Global parameter for date checking
import datetime
time_unix = [datetime.datetime(2017, 2, 16, 19, 9, 51).timestamp()]# RCR-BL event found

def timeInTimes(times_list):
    for time in times_list:
        if np.any(np.abs(time - time_unix) <= 1):  # within 1 second
            return True
    return False