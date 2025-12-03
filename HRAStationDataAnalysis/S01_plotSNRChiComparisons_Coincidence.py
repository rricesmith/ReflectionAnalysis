import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import os
import glob
import gc
import h5py
import pickle
from icecream import ic
import configparser
import json
from datetime import datetime
from HRAStationDataAnalysis.C_utils import getTimeEventMasks
from HRASimulation.HRAEventObject import HRAevent 
from HRAStationDataAnalysis.C03_coincidenceEventPlotting import plot_single_master_event

# --- Utility Functions ---

def load_cuts_for_station(date, station_id, cuts_data_folder):
    """
    Load the cuts from C00 for a specific station and date.
    Returns the final combined mask that should be applied to data.
    """
    cuts_file = os.path.join(cuts_data_folder, f'{date}_Station{station_id}_Cuts.npy')
    if not os.path.exists(cuts_file):
        ic(f"Warning: Cuts file not found for station {station_id} on date {date}. No cuts will be applied.")
        return None
    
    try:
        ic(f"Loading cuts file: {cuts_file}")
        cuts_data = np.load(cuts_file, allow_pickle=True)[()]
        
        # Combine all cuts (L1 + Storm + Burst) as done in C00 and C01
        final_cuts_mask = np.ones(len(cuts_data['L1_mask']), dtype=bool)
        for cut_key in cuts_data.keys():
            ic(f"Applying cut: {cut_key}")
            final_cuts_mask &= cuts_data[cut_key]
        
        ic(f"Final cuts mask for station {station_id}: {np.sum(final_cuts_mask)}/{len(final_cuts_mask)} events pass")
        return final_cuts_mask
    except Exception as e:
        ic(f"Error loading cuts file {cuts_file}: {e}")
        return None

def filter_unique_events_by_day(times, station_ids):
    """
    Returns a boolean mask of the same length as times/station_ids.
    True means keep the event (it's the first one seen for that station/day).
    False means it's a duplicate.
    """
    seen_combinations = set()
    keep_mask = np.zeros(len(times), dtype=bool)
    
    for i, (t, sid) in enumerate(zip(times, station_ids)):
        # Convert unix timestamp to date string (YYYY-MM-DD)
        date_str = datetime.utcfromtimestamp(t).strftime('%Y-%m-%d')
        combo = (sid, date_str)
        
        if combo not in seen_combinations:
            seen_combinations.add(combo)
            keep_mask[i] = True
        else:
            keep_mask[i] = False
            
    return keep_mask

def loadHRAfromH5(filename):
    """
    Loads a list of HRAevent objects from an HDF5 file.
    """
    ic(f"Loading HRA event list from: {filename}")
    eventList = []
    with h5py.File(filename, 'r') as hf:
        for key in hf.keys():
            obj_bytes = hf[key][0]
            event = pickle.loads(obj_bytes)
            eventList.append(event)
    ic(f"Successfully loaded {len(eventList)} events.")
    return eventList

def load_station_data(folder, date, station_id, data_name):
    """
    Loads and concatenates data files for a specific station and data type.
    """
    file_pattern = os.path.join(folder, f'{date}_Station{station_id}_{data_name}*')
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        ic(f"Warning: No files found for {data_name} with pattern: {file_pattern}")
        return np.array([])
    
    data_arrays = [np.load(f, allow_pickle=True) for f in file_list]
    data_arrays = [arr for arr in data_arrays if arr.size > 0]
    if not data_arrays:
        return np.array([])
        
    return np.concatenate(data_arrays, axis=0)


def load_coincidence_events(filepath, requested_event_ids):
    """Load coincidence events from a pickle file for the requested IDs."""
    if not os.path.exists(filepath):
        ic(f"Warning: Coincidence file not found: {filepath}")
        return {}

    try:
        with open(filepath, 'rb') as handle:
            events_dict = pickle.load(handle)
    except Exception as exc:  # pylint: disable=broad-except
        ic(f"Error loading coincidence file {filepath}: {exc}")
        return {}

    coincidence_events = {}
    for event_id in requested_event_ids:
        event_obj = events_dict.get(event_id)
        if event_obj is None:
            event_obj = events_dict.get(str(event_id))
        if event_obj is None:
            ic(f"Warning: Coincidence event {event_id} not found in pickle data.")
            continue
        coincidence_events[event_id] = event_obj

    ic(f"Loaded {len(coincidence_events)} coincidence events from alternate search.")
    return coincidence_events


def build_coincidence_station_overlays(coincidence_events, station_ids):
    """Prepare per-station coincidence overlays for plotting."""
    station_set = set(station_ids)
    
    special_ids = {11230, 11243}
    special_station_map = {
        11230: {13: "RCR", 17: "Backlobe"},
        11243: {30: "RCR", 17: "Backlobe"},
    }

    overlays = {}

    def _init_station_entry():
        return {
            'Backlobe': {'snr': [], 'Chi2016': [], 'ChiRCR': []},
            'Backlobe_event_ids': set(),
            'RCR': {'snr': [], 'Chi2016': [], 'ChiRCR': []},
            'RCR_event_ids': set(),
            'RCR_annotations': []
        }

    for station_id in station_set:
        overlays[station_id] = _init_station_entry()

    for event_id, event_details in coincidence_events.items():
        stations_info = event_details.get('stations', {}) if isinstance(event_details, dict) else {}
        
        # Determine default category for this event
        default_category = "RCR" if event_id in special_ids else "Backlobe"

        for station_key, station_payload in stations_info.items():
            try:
                station_int = int(station_key)
            except (TypeError, ValueError):
                continue

            if station_int not in overlays:
                if station_int in station_set:
                    overlays[station_int] = _init_station_entry()
                else:
                    continue

            # Determine category for this station
            category = default_category
            if event_id in special_station_map:
                if station_int in special_station_map[event_id]:
                    category = special_station_map[event_id][station_int]

            snr_vals = np.asarray(station_payload.get('SNR', []), dtype=float)
            chi2016_vals = np.asarray(station_payload.get('Chi2016', []), dtype=float)
            chircr_vals = np.asarray(station_payload.get('ChiRCR', []), dtype=float)

            min_len = min(snr_vals.size, chi2016_vals.size, chircr_vals.size)
            if min_len == 0:
                continue

            snr_vals = snr_vals[:min_len]
            chi2016_vals = chi2016_vals[:min_len]
            chircr_vals = chircr_vals[:min_len]

            valid_mask = np.isfinite(snr_vals) & np.isfinite(chi2016_vals) & np.isfinite(chircr_vals)
            if not np.any(valid_mask):
                continue

            snr_vals = snr_vals[valid_mask]
            chi2016_vals = chi2016_vals[valid_mask]
            chircr_vals = chircr_vals[valid_mask]

            entry = overlays[station_int]
            
            entry[category]['snr'].extend(snr_vals.tolist())
            entry[category]['Chi2016'].extend(chi2016_vals.tolist())
            entry[category]['ChiRCR'].extend(chircr_vals.tolist())

            if category == 'RCR':
                entry['RCR_event_ids'].add(event_id)
                if snr_vals.size > 0:
                    annotations = [f"St {station_int} (Evt {event_id})"] + [None] * (snr_vals.size - 1)
                    entry['RCR_annotations'].extend(annotations)
            else:
                entry['Backlobe_event_ids'].add(event_id)

    for entry in overlays.values():
        for bucket in ('Backlobe', 'RCR'):
            for key in ('snr', 'Chi2016', 'ChiRCR'):
                entry[bucket][key] = np.asarray(entry[bucket][key], dtype=float)

    return overlays


def combine_coincidence_overlays(station_ids, overlays_map):
    """Combine per-station overlays for summed-station plots."""
    combined = {
        'Backlobe': {'snr': [], 'Chi2016': [], 'ChiRCR': []},
        'Backlobe_event_ids': set(),
        'RCR': {'snr': [], 'Chi2016': [], 'ChiRCR': []},
        'RCR_event_ids': set(),
        'RCR_annotations': []
    }

    for station_id in station_ids:
        station_overlay = overlays_map.get(station_id)
        if not station_overlay:
            continue

        for bucket in ('Backlobe', 'RCR'):
            for key in ('snr', 'Chi2016', 'ChiRCR'):
                combined[bucket][key].extend(station_overlay[bucket][key])

        combined['Backlobe_event_ids'].update(station_overlay.get('Backlobe_event_ids', set()))
        combined['RCR_event_ids'].update(station_overlay.get('RCR_event_ids', set()))
        combined['RCR_annotations'].extend(station_overlay.get('RCR_annotations', []))

    for bucket in ('Backlobe', 'RCR'):
        for key in ('snr', 'Chi2016', 'ChiRCR'):
            combined[bucket][key] = np.asarray(combined[bucket][key], dtype=float)

    return combined

def get_sim_data(HRAeventList, direct_weight_name, reflected_weight_name, direct_stations, reflected_stations, sigma=4.5):
    """
    Extracts SNR, Chi, and weights from the HRAevent list, separating them
    into direct and reflected triggers using separate weight names.
    """
    direct_data = {'snr': [], 'Chi2016': [], 'ChiRCR': [], 'weights': []}
    reflected_data = {'snr': [], 'Chi2016': [], 'ChiRCR': [], 'weights': []}

    for event in HRAeventList:
        direct_weight = event.getWeight(direct_weight_name, primary=True, sigma=sigma)
        if not np.isnan(direct_weight) and direct_weight > 0:
            triggered_direct = [st_id for st_id in direct_stations if event.hasTriggered(st_id, sigma)]
            if triggered_direct:
                split_weight = direct_weight / len(triggered_direct)
                for st_id in triggered_direct:
                    snr, chi_dict = event.getSNR(st_id), event.getChi(st_id)
                    if snr is not None and chi_dict:
                        direct_data['snr'].append(snr)
                        direct_data['Chi2016'].append(chi_dict.get('Chi2016', np.nan))
                        direct_data['ChiRCR'].append(chi_dict.get('ChiRCR', np.nan))
                        direct_data['weights'].append(split_weight)

        reflected_weight = event.getWeight(reflected_weight_name, primary=True, sigma=sigma)
        if not np.isnan(reflected_weight) and reflected_weight > 0:
            triggered_reflected = [st_id for st_id in reflected_stations if event.hasTriggered(st_id, sigma)]
            if triggered_reflected:
                split_weight = reflected_weight / len(triggered_reflected)
                for st_id in triggered_reflected:
                    snr, chi_dict = event.getSNR(st_id), event.getChi(st_id)
                    if snr is not None and chi_dict:
                        reflected_data['snr'].append(snr)
                        reflected_data['Chi2016'].append(chi_dict.get('Chi2016', np.nan))
                        reflected_data['ChiRCR'].append(chi_dict.get('ChiRCR', np.nan))
                        reflected_data['weights'].append(split_weight)

    for data_dict in [direct_data, reflected_data]:
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key])

        # Filter out data where (RCRchi - BLchi) > 0.15
        if len(data_dict['ChiRCR']) > 0:
            mask = (data_dict['ChiRCR'] - data_dict['Chi2016']) <= 0.15
            ic(f"Filtering simulation data: keeping {np.sum(mask)}/{len(mask)} events where (ChiRCR - Chi2016) <= 0.15")
            for key in data_dict:
                data_dict[key] = data_dict[key][mask]
            
    return direct_data, reflected_data

def get_sim_data_coincidence(HRAeventList, direct_weight_name, reflected_weight_name, direct_stations, reflected_stations, sigma=4.5):
    """
    Extracts SNR, Chi, and weights from the HRAevent list, separating them
    into direct and reflected triggers using separate weight names.
    
    Applies coincidence logic:
    - Direct: Requires coincidence >= 2 among direct stations (excluding reflected).
    - Reflected: Requires coincidence >= 2 (among all valid stations) AND at least one reflected station trigger.
    """
    direct_data = {'snr': [], 'Chi2016': [], 'ChiRCR': [], 'weights': []}
    reflected_data = {'snr': [], 'Chi2016': [], 'ChiRCR': [], 'weights': []}

    # Define station sets for coincidence checks
    # Direct coincidence: ignore reflected stations and 32, 52
    bad_stations_direct = [32, 52, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    
    # Reflected coincidence: ignore 32, 52, 132, 152. Require at least one from reflected_stations
    bad_stations_reflected = [32, 52, 132, 152]
    force_stations_reflected = [113, 114, 115, 117, 118, 119, 130]

    for event in HRAeventList:
        # --- Direct Data Processing ---
        # Check for Direct Coincidence (>= 2 stations from direct set)
        if event.hasCoincidence(2, bad_stations_direct, sigma=sigma):
            direct_weight = event.getWeight(direct_weight_name, primary=True, sigma=sigma)
            if not np.isnan(direct_weight) and direct_weight > 0:
                triggered_direct = [st_id for st_id in direct_stations if event.hasTriggered(st_id, sigma)]
                if triggered_direct:
                    split_weight = direct_weight / len(triggered_direct)
                    for st_id in triggered_direct:
                        snr, chi_dict = event.getSNR(st_id), event.getChi(st_id)
                        if snr is not None and chi_dict:
                            direct_data['snr'].append(snr)
                            direct_data['Chi2016'].append(chi_dict.get('Chi2016', np.nan))
                            direct_data['ChiRCR'].append(chi_dict.get('ChiRCR', np.nan))
                            direct_data['weights'].append(split_weight)

        # --- Reflected Data Processing ---
        # Check for Reflected Coincidence (>= 2 stations total, at least one reflected)
        if event.hasCoincidence(2, bad_stations_reflected, sigma=sigma):
            # Check if at least one forced station triggered
            triggers = event.station_triggers.get(sigma, [])
            if not set(force_stations_reflected).isdisjoint(triggers):
                reflected_weight = event.getWeight(reflected_weight_name, primary=True, sigma=sigma)
                if not np.isnan(reflected_weight) and reflected_weight > 0:
                    triggered_reflected = [st_id for st_id in reflected_stations if event.hasTriggered(st_id, sigma)]
                    if triggered_reflected:
                        split_weight = reflected_weight / len(triggered_reflected)
                        for st_id in triggered_reflected:
                            snr, chi_dict = event.getSNR(st_id), event.getChi(st_id)
                            if snr is not None and chi_dict:
                                reflected_data['snr'].append(snr)
                                reflected_data['Chi2016'].append(chi_dict.get('Chi2016', np.nan))
                                reflected_data['ChiRCR'].append(chi_dict.get('ChiRCR', np.nan))
                                reflected_data['weights'].append(split_weight)

    for data_dict in [direct_data, reflected_data]:
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key])

        # Filter out data where (RCRchi - BLchi) > 0.15
        if len(data_dict['ChiRCR']) > 0:
            mask = (data_dict['ChiRCR'] - data_dict['Chi2016']) <= 0.15
            ic(f"Filtering simulation data: keeping {np.sum(mask)}/{len(mask)} events where (ChiRCR - Chi2016) <= 0.15")
            for key in data_dict:
                data_dict[key] = data_dict[key][mask]
            
    return direct_data, reflected_data

def get_all_cut_masks(data_dict, cuts, cut_type='rcr'):
    """
    Applies all cut combinations and returns a dictionary of boolean masks.
    cut_type: 'rcr' for RCR events, 'backlobe' for backlobe events
    """
    snr = data_dict['snr']
    chircr = data_dict['ChiRCR']
    chi2016 = data_dict['Chi2016']
    
    if cut_type == 'rcr':
        # Original RCR cuts
        chi_rcr_snr_cut_values = np.interp(snr, cuts['chi_rcr_line_snr'], cuts['chi_rcr_line_chi'])
        chi_diff = chircr - chi2016
        
        masks = {}
        masks['snr_cut'] = snr < cuts['snr_max']
        masks['snr_line_cut'] = chircr > chi_rcr_snr_cut_values
        masks['chi_diff_cut'] = (chi_diff > cuts['chi_diff_threshold']) & (chi_diff < cuts.get('chi_diff_max', 999))
        
        masks['snr_and_snr_line'] = masks['snr_cut'] & masks['snr_line_cut']
        masks['all_cuts'] = masks['snr_cut'] & masks['snr_line_cut'] & masks['chi_diff_cut']
        
    elif cut_type == 'backlobe':
        # Backlobe cuts (reversed logic)
        chi_2016_snr_cut_values = np.interp(snr, cuts['chi_2016_line_snr'], cuts['chi_2016_line_chi'])
        chi_diff = chircr - chi2016
        
        masks = {}
        masks['snr_cut'] = snr < cuts['snr_max']
        masks['snr_line_cut'] = chi2016 > chi_2016_snr_cut_values
        masks['chi_diff_cut'] = (chi_diff < -cuts['chi_diff_threshold']) & (chi_diff > -cuts.get('chi_diff_max', 999))
        
        masks['snr_and_snr_line'] = masks['snr_cut'] & masks['snr_line_cut']
        masks['all_cuts'] = masks['snr_cut'] & masks['snr_line_cut'] & masks['chi_diff_cut']
    
    return masks

def calculate_cut_stats_table(data_dict, cuts, is_sim, title, pre_mask_count=None, cut_type='rcr'):
    """Calculates the percentage of events/weight passing each cut individually and all together."""
    lines = [f"{title}:"]
    
    if pre_mask_count is not None:
        lines.append(f"- {'All Data (pre-mask)':<20}: {pre_mask_count}")

    if is_sim:
        total_val = np.sum(data_dict['weights'])
        if total_val == 0: return f"{title}\nNo weight in dataset."
    else:
        total_val = len(data_dict['snr'])
        if total_val == 0: return f"{title}\nNo events in dataset."
        lines.append(f"- {'All Data (post-mask)':<20}: {total_val}")
    
    weights = data_dict.get('weights', np.ones_like(data_dict['snr']))
    masks = get_all_cut_masks(data_dict, cuts, cut_type=cut_type)
    
    cut_masks_to_report = {
        f"SNR < {cuts['snr_max']}": masks['snr_cut'],
        r"RCR-$\chi$ > SNR Line": masks['snr_line_cut'],
        f"RCR-$\chi$ - BL-$\chi$ > {cuts['chi_diff_threshold']}": masks['chi_diff_cut'],
        "All Cuts": masks['all_cuts']
    };
    
    for name, mask in cut_masks_to_report.items():
        passing_val = np.sum(weights[mask])
        if is_sim:
            lines.append(f"- {name:<20}: {passing_val/total_val:>7.2%}")
        else:
            lines.append(f"- {name:<20}: {int(passing_val)} ({passing_val/total_val:.2%})")
            
    return "\n".join(lines)


# --- Plotting Functions ---

def set_plot_labels(ax, xlabel, ylabel, title, xlim, ylim, xscale='linear', yscale='linear'):
    """Set common labels and properties for a subplot."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(visible=True, which='both', axis='both')

def draw_cut_visuals(ax, plot_key, cuts_dict, cut_type='rcr'):
    """Draws lines and shaded regions for cuts on a given subplot."""
    snr_max = cuts_dict['snr_max']
    chi_diff_threshold = cuts_dict['chi_diff_threshold']

    if 'snr' in plot_key:
        ax.axvline(x=snr_max, color='m', linestyle='--', linewidth=1.5)
        # Only shade the BAD region (SNR > threshold) with dashed pattern
        ax.fill_betweenx(ax.get_ylim(), snr_max, ax.get_xlim()[1], color='m', alpha=0.1, hatch='///')

    if plot_key == 'snr_vs_chircr':
        # Always draw RCR cut line here
        snr_line_snr = cuts_dict['chi_rcr_line_snr']
        snr_line_chi = cuts_dict['chi_rcr_line_chi']
        ax.plot(snr_line_snr, snr_line_chi, color='purple', linestyle='--', linewidth=1.5, label='RCR Cut')
    
    elif plot_key == 'snr_vs_chi2016':
        # Always draw Backlobe cut line here
        snr_line_snr = cuts_dict['chi_2016_line_snr']
        snr_line_chi = cuts_dict['chi_2016_line_chi']
        ax.plot(snr_line_snr, snr_line_chi, color='orange', linestyle='--', linewidth=1.5, label='BL Cut')
    
    elif plot_key == 'chi_vs_chi':
        chi_diff_max = cuts_dict.get('chi_diff_max', 1.5)
        rcr_chi_cut_val = cuts_dict['chi_rcr_line_chi'][0]
        bl_chi_cut_val = cuts_dict['chi_2016_line_chi'][0]
        
        # RCR Region: ChiRCR > rcr_chi_cut_val, ChiRCR > Chi2016 + threshold, ChiRCR < Chi2016 + max_diff
        x = np.linspace(0, 1, 200)
        y_rcr_lower = x + chi_diff_threshold
        y_rcr_upper = x + chi_diff_max
        
        # Effective bounds for RCR region
        y_lower_eff = np.maximum(rcr_chi_cut_val, y_rcr_lower)
        y_upper_eff = np.minimum(1.0, y_rcr_upper)
        
        # Only fill where lower < upper
        fill_mask = y_lower_eff < y_upper_eff
        if np.any(fill_mask):
            ax.fill_between(x[fill_mask], y_lower_eff[fill_mask], y_upper_eff[fill_mask], color='green', alpha=0.1, label='Pass RCR Cuts')
        
        # Draw boundaries for RCR
        # Only show lines where ChiRCR > rcr_chi_cut_val
        mask_rcr_lines = y_rcr_lower > rcr_chi_cut_val
        if np.any(mask_rcr_lines):
             ax.plot(x[mask_rcr_lines], y_rcr_lower[mask_rcr_lines], color='darkgreen', linestyle='--', linewidth=1.5, label='RCR Diff Cut')
        
        mask_rcr_max_lines = y_rcr_upper > rcr_chi_cut_val
        if np.any(mask_rcr_max_lines):
             ax.plot(x[mask_rcr_max_lines], y_rcr_upper[mask_rcr_max_lines], color='darkgreen', linestyle=':', linewidth=1.5, label='RCR Max Diff')

        # RCR Chi Cut: Horizontal line from x=0 to x=rcr_chi_cut_val (diagonal intersection)
        ax.plot([0, rcr_chi_cut_val], [rcr_chi_cut_val, rcr_chi_cut_val], color='purple', linestyle='--', linewidth=1.5, label='RCR Chi Cut')
        
        # Backlobe Region: ChiBL > bl_chi_cut_val, ChiRCR < Chi2016 - threshold, ChiRCR > Chi2016 - max_diff
        # For x (ChiBL) > bl_chi_cut_val, y (ChiRCR) < x - threshold
        x_bl = np.linspace(0, 1, 200)
        y_bl_upper = x_bl - chi_diff_threshold
        y_bl_lower = x_bl - chi_diff_max
        
        # Effective bounds for BL region
        # We need x > bl_chi_cut_val
        # And y < y_bl_upper
        # And y > y_bl_lower
        # And y > 0 (implicit)
        
        # We are filling between y_bl_lower and y_bl_upper, but only where x > bl_chi_cut_val
        
        # Also need to respect x > bl_chi_cut_val for the fill
        fill_mask_bl = (x_bl > bl_chi_cut_val) & (y_bl_lower < y_bl_upper)
        
        # Also ensure y is within [0, 1]
        y_bl_upper_eff = np.minimum(1.0, y_bl_upper)
        y_bl_lower_eff = np.maximum(0.0, y_bl_lower)
        
        fill_mask_bl &= (y_bl_lower_eff < y_bl_upper_eff)

        if np.any(fill_mask_bl):
            ax.fill_between(x_bl[fill_mask_bl], y_bl_lower_eff[fill_mask_bl], y_bl_upper_eff[fill_mask_bl], color='orange', alpha=0.1, label='Pass BL Cuts')
        
        # Draw boundaries for BL
        # Only show lines where ChiBL > bl_chi_cut_val
        mask_bl_lines = x_bl > bl_chi_cut_val
        if np.any(mask_bl_lines):
            ax.plot(x_bl[mask_bl_lines], y_bl_upper[mask_bl_lines], color='darkorange', linestyle='--', linewidth=1.5, label='BL Diff Cut')
            ax.plot(x_bl[mask_bl_lines], y_bl_lower[mask_bl_lines], color='darkorange', linestyle=':', linewidth=1.5, label='BL Max Diff')

        # BL Chi Cut: Vertical line from y=0 to y=bl_chi_cut_val (diagonal intersection)
        ax.plot([bl_chi_cut_val, bl_chi_cut_val], [0, bl_chi_cut_val], color='orange', linestyle='--', linewidth=1.5, label='BL Chi Cut')

    elif plot_key == 'snr_vs_chidiff':
        chi_diff_max = cuts_dict.get('chi_diff_max', 1.5)
        # Draw horizontal line for Chi difference cut (RCR)
        ax.axhline(y=chi_diff_threshold, color='darkgreen', linestyle='--', linewidth=1.5, label='RCR Diff Cut')
        ax.axhline(y=chi_diff_max, color='darkgreen', linestyle=':', linewidth=1.5, label='RCR Max Diff')
        # Draw horizontal line for Chi difference cut (Backlobe)
        ax.axhline(y=-chi_diff_threshold, color='darkorange', linestyle='--', linewidth=1.5, label='BL Diff Cut')
        ax.axhline(y=-chi_diff_max, color='darkorange', linestyle=':', linewidth=1.5, label='BL Max Diff')


def plot_2x2_grid(fig, axs, base_data_config, cuts_dict, overlays=None, hist_bins_dict=None):
    """
    Master function to generate a 2x2 grid. Plots a base layer (scatter or hist)
    and then adds any number of overlay layers on top.
    """
    im = None
    base_data = base_data_config['data']
    base_plot_type = base_data_config['type']

    plot_configs = {
        'snr_vs_chi2016': {'xlabel': 'SNR', 'ylabel': r'BL-$\chi$', 'xlim': (3, 100), 'ylim': (0, 1), 'xscale': 'log'},
        'snr_vs_chircr': {'xlabel': 'SNR', 'ylabel': r'RCR-$\chi$', 'xlim': (3, 100), 'ylim': (0, 1), 'xscale': 'log'},
        'chi_vs_chi': {'xlabel': r'BL-$\chi$', 'ylabel': r'RCR-$\chi$', 'xlim': (0, 1), 'ylim': (0, 1)},
        'snr_vs_chidiff': {'xlabel': 'SNR', 'ylabel': r'RCR-$\chi$ - BL-$\chi$', 'xlim': (3, 100), 'ylim': (-0.4, 0.4), 'xscale': 'log'}
    }
    
    for i, (key, p) in enumerate(plot_configs.items()):
        ax = axs.flatten()[i]
        set_plot_labels(ax, p['xlabel'], p['ylabel'], f"{p['ylabel']} vs {p['xlabel']}", p.get('xlim'), p.get('ylim'), p.get('xscale', 'linear'))
        
        # Plot Base Layer
        base_map = {
            'snr_vs_chi2016': {'x': base_data['snr'], 'y': base_data['Chi2016']},
            'snr_vs_chircr': {'x': base_data['snr'], 'y': base_data['ChiRCR']},
            'chi_vs_chi': {'x': base_data['Chi2016'], 'y': base_data['ChiRCR']},
            'snr_vs_chidiff': {'x': base_data['snr'], 'y': base_data['ChiRCR'] - base_data['Chi2016']}
        }
        if base_plot_type == 'scatter':
            ax.scatter(base_map[key]['x'], base_map[key]['y'], **base_data_config['style'])
        elif base_plot_type == 'hist' and np.sum(base_data.get('weights', 0)) > 0:
            h, xedges, yedges, im_temp = ax.hist2d(base_map[key]['x'], base_map[key]['y'], bins=hist_bins_dict[key], weights=base_data['weights'], norm=colors.LogNorm())
            if im is None: im = im_temp

        # Plot Overlay Layers
        if overlays:
            for overlay in overlays:
                overlay_data = overlay['data']
                overlay_map = {
                    'snr_vs_chi2016': {'x': overlay_data['snr'], 'y': overlay_data['Chi2016']},
                    'snr_vs_chircr': {'x': overlay_data['snr'], 'y': overlay_data['ChiRCR']},
                    'chi_vs_chi': {'x': overlay_data['Chi2016'], 'y': overlay_data['ChiRCR']},
                    'snr_vs_chidiff': {'x': overlay_data['snr'], 'y': overlay_data['ChiRCR'] - overlay_data['Chi2016']}
                }
                x_data, y_data = overlay_map[key]['x'], overlay_map[key]['y']
                
                if overlay['style'].get('color_by_weight'):
                    weights = overlay_data['weights']
                    if weights.size == 0:
                        continue
                    valid_weight_mask = weights > 0
                    if not np.any(valid_weight_mask):
                        continue
                    weights = weights[valid_weight_mask]
                    x_data = x_data[valid_weight_mask]
                    y_data = y_data[valid_weight_mask]
                    sort_indices = np.argsort(weights)
                    x_data, y_data, weights = x_data[sort_indices], y_data[sort_indices], weights[sort_indices]

                    min_weight = weights.min()
                    max_weight = weights.max()
                    if min_weight == max_weight:
                        max_weight *= 1.0001

                    # Enhanced weight visualization with better color mapping and size scaling
                    norm = colors.LogNorm(vmin=min_weight, vmax=max_weight)
                    cmap = overlay['style'].get('cmap', 'viridis')
                    scatter = ax.scatter(x_data, y_data, c=weights, cmap=cmap, norm=norm,
                                       alpha=overlay['style']['alpha'], s=overlay['style']['s'])
                    
                    # Add colorbar for weight visualization if this is the first overlay
                    if overlay == overlays[0]:  # Only add colorbar for first overlay to avoid duplicates
                        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
                        cbar.set_label('Event Weight', fontsize=8)
                        cbar.ax.tick_params(labelsize=7)
                else:
                    scatter = ax.scatter(x_data, y_data, **overlay['style'])

                    annotations = overlay.get('annotations')
                    if annotations:
                        offsets = overlay.get('annotation_offsets')
                        default_offset = overlay.get('annotation_offset', (4, 4))
                        annotation_color = overlay.get('annotation_color', overlay['style'].get('c', 'k'))
                        annotation_fontsize = overlay.get('annotation_fontsize', 8)
                        ha = overlay.get('annotation_ha', 'left')
                        va = overlay.get('annotation_va', 'bottom')

                        for idx, (x_val, y_val, label_text) in enumerate(zip(x_data, y_data, annotations)):
                            if label_text is None:
                                continue
                            offset = default_offset
                            if offsets and idx < len(offsets):
                                offset = offsets[idx]
                            ax.annotate(
                                label_text,
                                xy=(x_val, y_val),
                                xytext=offset,
                                textcoords='offset points',
                                fontsize=annotation_fontsize,
                                color=annotation_color,
                                ha=ha,
                                va=va
                            )

        if cuts_dict:
            draw_cut_visuals(ax, key, cuts_dict, cut_type='rcr')  # For now, only show RCR cuts on main plots
        if key == 'chi_vs_chi':
            ax.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=1)

    # --- Figure-level Legend ---
    legend_elements = []
    if base_data_config['label']:
        if base_plot_type == 'hist':
            legend_elements.append(plt.Rectangle((0,0),1,1,fc="lightblue", label=base_data_config['label']))
        else:
            legend_elements.append(Line2D([0], [0], marker=base_data_config['style'].get('marker', 'o'), color='w', label=base_data_config['label'], markerfacecolor=base_data_config['style']['c'], markersize=8))

    if overlays:
        for overlay in overlays:
            if overlay['label']:
                if overlay['style'].get('color_by_weight'):
                    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=overlay['label'], markerfacecolor='blue', markersize=8, alpha=0.5))
                else:
                    marker_face = overlay['style'].get('c', overlay['style'].get('color', 'gray'))
                    marker_edge = overlay['style'].get('edgecolors', 'w')
                    legend_elements.append(Line2D([0], [0], marker=overlay['style'].get('marker', 'o'), color='w', label=overlay['label'], markerfacecolor=marker_face, markeredgecolor=marker_edge, markersize=8))
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.22), ncol=3, fontsize=10, frameon=True)

    return im


def subset_data_by_mask(data_dict, mask):
    """Return a shallow copy of data_dict filtered by boolean mask."""
    return {key: data_dict[key][mask] for key in data_dict}


def plot_sim_only_comparisons(sim_direct, sim_reflected, cuts, hist_bins, plot_folder, date, rcr_cut_string):
    """Create simulation-only comparison plots with and without RCR cuts."""
    ic("Generating simulation-only comparison plots (no cuts)...")

    base_config = {'data': sim_direct, 'type': 'hist', 'label': 'Backlobe Sim'}
    reflected_overlay = {'data': sim_reflected, 'label': 'RCR Sim', 'style': {'s': 12, 'alpha': 0.5, 'color_by_weight': True, 'cmap': 'cool'}}

    # --- Plot 1: No Cuts (With Lines) ---
    fig_raw, axs_raw = plt.subplots(2, 2, figsize=(12, 15))
    fig_raw.suptitle(f'Simulation Comparison: Backlobe vs RCR (No Cuts)\n{rcr_cut_string}', fontsize=14)
    im_raw = plot_2x2_grid(fig_raw, axs_raw, base_config, cuts, overlays=[reflected_overlay], hist_bins_dict=hist_bins)

    direct_stats_rcr = calculate_cut_stats_table(sim_direct, cuts, True, "Backlobe Sim (RCR Cuts)", cut_type='rcr')
    reflected_stats_rcr = calculate_cut_stats_table(sim_reflected, cuts, True, "RCR Sim (RCR Cuts)", cut_type='rcr')
    direct_stats_back = calculate_cut_stats_table(sim_direct, cuts, True, "Backlobe Sim (Backlobe Cuts)", cut_type='backlobe')
    reflected_stats_back = calculate_cut_stats_table(sim_reflected, cuts, True, "RCR Sim (Backlobe Cuts)", cut_type='backlobe')
    
    # Split text into columns
    fig_raw.text(0.25, 0.01, f"{direct_stats_rcr}\n\n{reflected_stats_rcr}", ha='center', va='bottom', fontsize=9, fontfamily='monospace')
    fig_raw.text(0.75, 0.01, f"{direct_stats_back}\n\n{reflected_stats_back}", ha='center', va='bottom', fontsize=9, fontfamily='monospace')

    if im_raw:
        fig_raw.tight_layout(rect=[0, 0.28, 0.9, 0.95])
        cbar_ax_raw = fig_raw.add_axes([0.91, 0.28, 0.02, 0.65])
        fig_raw.colorbar(im_raw, cax=cbar_ax_raw, label='Backlobe Weighted Counts (Evts/Yr)')
    else:
        fig_raw.tight_layout(rect=[0, 0.28, 1, 0.95])
    plt.savefig(f'{plot_folder}SimOnly_Backlobe_vs_RCR_NoCuts_{date}.png')
    plt.close(fig_raw)

    # --- Plot 1b: No Cuts (No Lines) ---
    ic("Generating simulation-only comparison plots (no cuts, no lines)...")
    fig_raw_nl, axs_raw_nl = plt.subplots(2, 2, figsize=(12, 15))
    fig_raw_nl.suptitle(f'Simulation Comparison: Backlobe vs RCR (No Cuts, No Lines)\n{rcr_cut_string}', fontsize=14)
    im_raw_nl = plot_2x2_grid(fig_raw_nl, axs_raw_nl, base_config, None, overlays=[reflected_overlay], hist_bins_dict=hist_bins)

    fig_raw_nl.text(0.25, 0.01, f"{direct_stats_rcr}\n\n{reflected_stats_rcr}", ha='center', va='bottom', fontsize=9, fontfamily='monospace')
    fig_raw_nl.text(0.75, 0.01, f"{direct_stats_back}\n\n{reflected_stats_back}", ha='center', va='bottom', fontsize=9, fontfamily='monospace')

    if im_raw_nl:
        fig_raw_nl.tight_layout(rect=[0, 0.28, 0.9, 0.95])
        cbar_ax_raw_nl = fig_raw_nl.add_axes([0.91, 0.28, 0.02, 0.65])
        fig_raw_nl.colorbar(im_raw_nl, cax=cbar_ax_raw_nl, label='Backlobe Weighted Counts (Evts/Yr)')
    else:
        fig_raw_nl.tight_layout(rect=[0, 0.28, 1, 0.95])
    plt.savefig(f'{plot_folder}SimOnly_Backlobe_vs_RCR_NoCuts_NoLines_{date}.png')
    plt.close(fig_raw_nl)

    # --- Plot 1c: Backlobe Only (No Cuts) ---
    ic("Generating simulation-only plots (Backlobe only, no cuts)...")
    fig_bl_only, axs_bl_only = plt.subplots(2, 2, figsize=(12, 15))
    fig_bl_only.suptitle(f'Simulation: Backlobe Only (No Cuts)\n{rcr_cut_string}', fontsize=14)
    im_bl_only = plot_2x2_grid(fig_bl_only, axs_bl_only, base_config, cuts, overlays=[], hist_bins_dict=hist_bins)

    fig_bl_only.text(0.5, 0.01, f"{direct_stats_rcr}\n\n{direct_stats_back}", ha='center', va='bottom', fontsize=9, fontfamily='monospace')

    if im_bl_only:
        fig_bl_only.tight_layout(rect=[0, 0.28, 0.9, 0.95])
        cbar_ax_bl_only = fig_bl_only.add_axes([0.91, 0.28, 0.02, 0.65])
        fig_bl_only.colorbar(im_bl_only, cax=cbar_ax_bl_only, label='Backlobe Weighted Counts (Evts/Yr)')
    else:
        fig_bl_only.tight_layout(rect=[0, 0.28, 1, 0.95])
    plt.savefig(f'{plot_folder}SimOnly_Backlobe_NoCuts_{date}.png')
    plt.close(fig_bl_only)

    # --- Plot 1d: Backlobe Only (No Cuts, No Lines) ---
    ic("Generating simulation-only plots (Backlobe only, no cuts, no lines)...")
    fig_bl_only_nl, axs_bl_only_nl = plt.subplots(2, 2, figsize=(12, 15))
    fig_bl_only_nl.suptitle(f'Simulation: Backlobe Only (No Cuts, No Lines)\n{rcr_cut_string}', fontsize=14)
    im_bl_only_nl = plot_2x2_grid(fig_bl_only_nl, axs_bl_only_nl, base_config, None, overlays=[], hist_bins_dict=hist_bins)

    fig_bl_only_nl.text(0.5, 0.01, f"{direct_stats_rcr}\n\n{direct_stats_back}", ha='center', va='bottom', fontsize=9, fontfamily='monospace')

    if im_bl_only_nl:
        fig_bl_only_nl.tight_layout(rect=[0, 0.28, 0.9, 0.95])
        cbar_ax_bl_only_nl = fig_bl_only_nl.add_axes([0.91, 0.28, 0.02, 0.65])
        fig_bl_only_nl.colorbar(im_bl_only_nl, cax=cbar_ax_bl_only_nl, label='Backlobe Weighted Counts (Evts/Yr)')
    else:
        fig_bl_only_nl.tight_layout(rect=[0, 0.28, 1, 0.95])
    plt.savefig(f'{plot_folder}SimOnly_Backlobe_NoCuts_NoLines_{date}.png')
    plt.close(fig_bl_only_nl)


def run_analysis_for_station(station_id, station_data, event_ids, unique_indices, pre_mask_count, sim_direct, sim_reflected, cuts, rcr_cut_string, hist_bins, plot_folder, date, coincidence_overlay=None, backlobe_2016_overlay=None, excluded_events=None):
    """
    Runs the full plotting and saving pipeline for a given station ID and its data.
    """
    ic(f"--- Running analysis for Station {station_id} ---")

    # Ensure overlay data are numpy arrays to prevent AttributeError: 'list' object has no attribute 'size'
    for overlay in [coincidence_overlay, backlobe_2016_overlay]:
        if overlay:
            for cat in ['Backlobe', 'RCR']:
                if cat in overlay:
                    for data_key in ['snr', 'Chi2016', 'ChiRCR']:
                        if data_key in overlay[cat] and isinstance(overlay[cat][data_key], list):
                            overlay[cat][data_key] = np.array(overlay[cat][data_key])

    # --- Get Masks and Save Passing Events ---
    masks_rcr = get_all_cut_masks(station_data, cuts, cut_type='rcr')
    masks_backlobe = get_all_cut_masks(station_data, cuts, cut_type='backlobe')
    
    # --- Handle Excluded Events ---
    excluded_mask = np.zeros(len(event_ids), dtype=bool)
    if excluded_events and 'StationID' in station_data:
        st_ids = station_data['StationID']
        # Create a set for faster lookup
        excluded_set = set(excluded_events)
        
        # Only mark as excluded if it passes the cuts we make (RCR cuts or Backlobe cuts)
        passing_rcr = masks_rcr['all_cuts']
        passing_bl = masks_backlobe['all_cuts']
        
        for idx, (evt_id, st_id) in enumerate(zip(event_ids, st_ids)):
             if (st_id, evt_id) in excluded_set:
                 if passing_rcr[idx] or passing_bl[idx]:
                     excluded_mask[idx] = True
                     ic(f"Excluding event {evt_id} from Station {st_id}")

    # --- Apply Day-Cut (Uniqueness) ---
    # For RCR
    rcr_passing_indices = np.where(masks_rcr['all_cuts'])[0]
    if len(rcr_passing_indices) > 0:
        times_rcr = station_data['Time'][rcr_passing_indices]
        sids_rcr = station_data['StationID'][rcr_passing_indices]
        unique_mask_rcr = filter_unique_events_by_day(times_rcr, sids_rcr)
        
        # Update masks
        # The indices that are False in unique_mask_rcr are the duplicates
        duplicate_indices_rcr = rcr_passing_indices[~unique_mask_rcr]
        masks_rcr['all_cuts'][duplicate_indices_rcr] = False
        masks_rcr['day_cut_fail'] = np.zeros(len(station_data['snr']), dtype=bool)
        masks_rcr['day_cut_fail'][duplicate_indices_rcr] = True
        
        ic(f"RCR Day Cut: Removed {len(duplicate_indices_rcr)} duplicates.")
    else:
        masks_rcr['day_cut_fail'] = np.zeros(len(station_data['snr']), dtype=bool)

    # For Backlobe
    bl_passing_indices = np.where(masks_backlobe['all_cuts'])[0]
    if len(bl_passing_indices) > 0:
        times_bl = station_data['Time'][bl_passing_indices]
        sids_bl = station_data['StationID'][bl_passing_indices]
        unique_mask_bl = filter_unique_events_by_day(times_bl, sids_bl)
        
        # Update masks
        duplicate_indices_bl = bl_passing_indices[~unique_mask_bl]
        masks_backlobe['all_cuts'][duplicate_indices_bl] = False
        masks_backlobe['day_cut_fail'] = np.zeros(len(station_data['snr']), dtype=bool)
        masks_backlobe['day_cut_fail'][duplicate_indices_bl] = True
        
        ic(f"Backlobe Day Cut: Removed {len(duplicate_indices_bl)} duplicates.")
    else:
        masks_backlobe['day_cut_fail'] = np.zeros(len(station_data['snr']), dtype=bool)

    # Update masks to exclude these events from passing
    masks_rcr['all_cuts'] &= ~excluded_mask
    masks_backlobe['all_cuts'] &= ~excluded_mask
    
    passing_events_to_save = {}
    
    # Note: The keys here must be valid Python identifiers for np.savez
    passing_events_to_save['snr_cut_only'] = np.zeros(np.sum(masks_rcr['snr_cut']), dtype=[('event_id', 'i8'), ('unique_index', 'i8')])
    passing_events_to_save['snr_and_snr_line'] = np.zeros(np.sum(masks_rcr['snr_and_snr_line']), dtype=[('event_id', 'i8'), ('unique_index', 'i8')])
    passing_events_to_save['all_cuts'] = np.zeros(np.sum(masks_rcr['all_cuts']), dtype=[('event_id', 'i8'), ('unique_index', 'i8')])
    passing_events_to_save['backlobe_cut_only'] = np.zeros(np.sum(masks_backlobe['snr_cut']), dtype=[('event_id', 'i8'), ('unique_index', 'i8')])
    passing_events_to_save['backlobe_and_snr_line'] = np.zeros(np.sum(masks_backlobe['snr_and_snr_line']), dtype=[('event_id', 'i8'), ('unique_index', 'i8')])
    passing_events_to_save['backlobe_all_cuts'] = np.zeros(np.sum(masks_backlobe['all_cuts']), dtype=[('event_id', 'i8'), ('unique_index', 'i8')])

    passing_events_to_save['snr_cut_only']['event_id'] = event_ids[masks_rcr['snr_cut']]
    passing_events_to_save['snr_cut_only']['unique_index'] = unique_indices[masks_rcr['snr_cut']]
    passing_events_to_save['snr_and_snr_line']['event_id'] = event_ids[masks_rcr['snr_and_snr_line']]
    passing_events_to_save['snr_and_snr_line']['unique_index'] = unique_indices[masks_rcr['snr_and_snr_line']]
    passing_events_to_save['all_cuts']['event_id'] = event_ids[masks_rcr['all_cuts']]
    passing_events_to_save['all_cuts']['unique_index'] = unique_indices[masks_rcr['all_cuts']]
    passing_events_to_save['backlobe_cut_only']['event_id'] = event_ids[masks_backlobe['snr_cut']]
    passing_events_to_save['backlobe_cut_only']['unique_index'] = unique_indices[masks_backlobe['snr_cut']]
    passing_events_to_save['backlobe_and_snr_line']['event_id'] = event_ids[masks_backlobe['snr_and_snr_line']]
    passing_events_to_save['backlobe_and_snr_line']['unique_index'] = unique_indices[masks_backlobe['snr_and_snr_line']]
    passing_events_to_save['backlobe_all_cuts']['event_id'] = event_ids[masks_backlobe['all_cuts']]
    passing_events_to_save['backlobe_all_cuts']['unique_index'] = unique_indices[masks_backlobe['all_cuts']]
    
    savename = f'{plot_folder}PassingEvents_Station{station_id}_{date}.npz'
    np.savez(savename, **passing_events_to_save)
    ic(f"Saved passing event combinations for Station {station_id} to {savename}")

    # --- Generate Master Plots for Passing Events ---
    # (Removed as per request)

    # --- Plot 3: Data Only with layered cuts
    # (Removed as per request)
    
    # --- Setup Configurations for Requested Plots ---
    
    # 1. Sim Data Configs (Updated Labels)
    sim_base_config = {'data': sim_direct, 'type': 'hist', 'label': 'Coinc BL Sim'}
    reflected_overlay_config = {'data': sim_reflected, 'label': 'Coinc RCR Sim', 'style': {'s': 12, 'alpha': 0.5, 'color_by_weight': True, 'cmap': 'cool'}}

    # 2. Overlay Configs
    bl_2016_overlay_config = None
    if backlobe_2016_overlay:
        bl_2016_data = backlobe_2016_overlay.get('Backlobe')
        if bl_2016_data is not None and bl_2016_data.get('snr', np.array([])).size > 0:
             bl_2016_overlay_config = {
                'data': bl_2016_data,
                'label': f"2016 Backlobe",
                'style': {'marker': 's', 's': 40, 'alpha': 0.8, 'c': 'cyan', 'edgecolors': 'black', 'linewidths': 0.5}
            }

    coinc_backlobe_overlay_config = None
    coinc_rcr_overlay_config = None
    if coincidence_overlay:
        backlobe_data = coincidence_overlay.get('Backlobe')
        if backlobe_data is not None and backlobe_data.get('snr', np.array([])).size > 0:
            coinc_backlobe_overlay_config = {
                'data': backlobe_data,
                'label': f"Coinc Backlobe",
                'style': {'marker': 'o', 's': 55, 'alpha': 0.9, 'c': 'gold', 'edgecolors': 'black', 'linewidths': 0.4}
            }
        
        rcr_data = coincidence_overlay.get('RCR')
        if rcr_data is not None and rcr_data.get('snr', np.array([])).size > 0:
             coinc_rcr_overlay_config = {
                'data': rcr_data,
                'label': f"Coinc RCR",
                'style': {'marker': '*', 's': 140, 'alpha': 0.95, 'c': 'darkorange', 'edgecolors': 'black', 'linewidths': 0.6}
            }

    # Calculate Stats Strings
    sim_direct_rcr_stats = calculate_cut_stats_table(sim_direct, cuts, True, "Coinc BL Sim (RCR Cuts)", cut_type='rcr')
    sim_reflected_rcr_stats = calculate_cut_stats_table(sim_reflected, cuts, True, "Coinc RCR Sim (RCR Cuts)", cut_type='rcr')


    # --- Requested Plot 1: Coinc BL Sim ---
    ic("Generating Plot 1: Coinc BL Sim...")
    fig1, axs1 = plt.subplots(2, 2, figsize=(12, 15))
    fig1.suptitle(f'Coinc BL Sim - Station {station_id}\n{rcr_cut_string}', fontsize=14)
    im1 = plot_2x2_grid(fig1, axs1, sim_base_config, cuts, overlays=[], hist_bins_dict=hist_bins)
    fig1.text(0.5, 0.01, sim_direct_rcr_stats, ha='center', va='bottom', fontsize=9, fontfamily='monospace')
    if im1:
        fig1.tight_layout(rect=[0, 0.28, 0.9, 0.95])
        cbar_ax1 = fig1.add_axes([0.91, 0.28, 0.02, 0.65])
        fig1.colorbar(im1, cax=cbar_ax1, label='Coinc BL Sim Weighted Counts (Evts/Yr)')
    else:
        fig1.tight_layout(rect=[0, 0.28, 1, 0.95])
    plt.savefig(f'{plot_folder}CoincBLSim_Station{station_id}_{date}.png')
    plt.close(fig1)

    # --- Requested Plot 1 (No Cuts): Coinc BL Sim ---
    ic("Generating Plot 1 (No Cuts): Coinc BL Sim...")
    fig1_nc, axs1_nc = plt.subplots(2, 2, figsize=(12, 15))
    fig1_nc.suptitle(f'Coinc BL Sim - Station {station_id} (No Cuts)\n{rcr_cut_string}', fontsize=14)
    im1_nc = plot_2x2_grid(fig1_nc, axs1_nc, sim_base_config, None, overlays=[], hist_bins_dict=hist_bins)
    fig1_nc.text(0.5, 0.01, sim_direct_rcr_stats, ha='center', va='bottom', fontsize=9, fontfamily='monospace')
    if im1_nc:
        fig1_nc.tight_layout(rect=[0, 0.28, 0.9, 0.95])
        cbar_ax1_nc = fig1_nc.add_axes([0.91, 0.28, 0.02, 0.65])
        fig1_nc.colorbar(im1_nc, cax=cbar_ax1_nc, label='Coinc BL Sim Weighted Counts (Evts/Yr)')
    else:
        fig1_nc.tight_layout(rect=[0, 0.28, 1, 0.95])
    plt.savefig(f'{plot_folder}CoincBLSim_NoCuts_Station{station_id}_{date}.png')
    plt.close(fig1_nc)


    # --- Requested Plot 2: Coinc BL Sim + 2016 Backlobe + Coinc Backlobe + Coinc RCR ---
    ic("Generating Plot 2: Coinc BL Sim + 2016 Backlobe + Coinc Backlobe + Coinc RCR...")
    overlays_2 = []
    if bl_2016_overlay_config: overlays_2.append(bl_2016_overlay_config)
    if coinc_backlobe_overlay_config: overlays_2.append(coinc_backlobe_overlay_config)
    if coinc_rcr_overlay_config: overlays_2.append(coinc_rcr_overlay_config)

    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 15))
    fig2.suptitle(f'Coinc BL Sim + 2016 BL + Coinc BL + Coinc RCR - Station {station_id}\n{rcr_cut_string}', fontsize=14)
    im2 = plot_2x2_grid(fig2, axs2, sim_base_config, cuts, overlays=overlays_2, hist_bins_dict=hist_bins)
    fig2.text(0.5, 0.01, sim_direct_rcr_stats, ha='center', va='bottom', fontsize=9, fontfamily='monospace')
    if im2:
        fig2.tight_layout(rect=[0, 0.28, 0.9, 0.95])
        cbar_ax2 = fig2.add_axes([0.91, 0.28, 0.02, 0.65])
        fig2.colorbar(im2, cax=cbar_ax2, label='Coinc BL Sim Weighted Counts (Evts/Yr)')
    else:
        fig2.tight_layout(rect=[0, 0.28, 1, 0.95])
    plt.savefig(f'{plot_folder}CoincBLSim_vs_2016_vs_Coinc_Station{station_id}_{date}.png')
    plt.close(fig2)

    # --- Requested Plot 2 (No Cuts): Coinc BL Sim + 2016 Backlobe + Coinc Backlobe + Coinc RCR ---
    ic("Generating Plot 2 (No Cuts): Coinc BL Sim + 2016 Backlobe + Coinc Backlobe + Coinc RCR...")
    fig2_nc, axs2_nc = plt.subplots(2, 2, figsize=(12, 15))
    fig2_nc.suptitle(f'Coinc BL Sim + 2016 BL + Coinc BL + Coinc RCR - Station {station_id} (No Cuts)\n{rcr_cut_string}', fontsize=14)
    im2_nc = plot_2x2_grid(fig2_nc, axs2_nc, sim_base_config, None, overlays=overlays_2, hist_bins_dict=hist_bins)
    fig2_nc.text(0.5, 0.01, sim_direct_rcr_stats, ha='center', va='bottom', fontsize=9, fontfamily='monospace')
    if im2_nc:
        fig2_nc.tight_layout(rect=[0, 0.28, 0.9, 0.95])
        cbar_ax2_nc = fig2_nc.add_axes([0.91, 0.28, 0.02, 0.65])
        fig2_nc.colorbar(im2_nc, cax=cbar_ax2_nc, label='Coinc BL Sim Weighted Counts (Evts/Yr)')
    else:
        fig2_nc.tight_layout(rect=[0, 0.28, 1, 0.95])
    plt.savefig(f'{plot_folder}CoincBLSim_vs_2016_vs_Coinc_NoCuts_Station{station_id}_{date}.png')
    plt.close(fig2_nc)


    # --- Requested Plot 3: Coinc BL Sim + Coinc RCR Sim + 2016 Backlobe + Coinc Backlobe + Coinc RCR ---
    ic("Generating Plot 3: Coinc BL Sim + Coinc RCR Sim + 2016 Backlobe + Coinc Backlobe + Coinc RCR...")
    overlays_3 = [reflected_overlay_config] + overlays_2

    fig3, axs3 = plt.subplots(2, 2, figsize=(12, 15))
    fig3.suptitle(f'Coinc BL Sim + Coinc RCR Sim + 2016 BL + Coinc BL + Coinc RCR - Station {station_id}\n{rcr_cut_string}', fontsize=14)
    im3 = plot_2x2_grid(fig3, axs3, sim_base_config, cuts, overlays=overlays_3, hist_bins_dict=hist_bins)
    fig3.text(0.5, 0.01, sim_direct_rcr_stats + "\n\n" + sim_reflected_rcr_stats, ha='center', va='bottom', fontsize=9, fontfamily='monospace')
    if im3:
        fig3.tight_layout(rect=[0, 0.28, 0.9, 0.95])
        cbar_ax3 = fig3.add_axes([0.91, 0.28, 0.02, 0.65])
        fig3.colorbar(im3, cax=cbar_ax3, label='Coinc BL Sim Weighted Counts (Evts/Yr)')
    else:
        fig3.tight_layout(rect=[0, 0.28, 1, 0.95])
    plt.savefig(f'{plot_folder}CoincBLSim_vs_CoincRCRSim_vs_2016_vs_Coinc_Station{station_id}_{date}.png')
    plt.close(fig3)

    # --- Requested Plot 3 (No Cuts): Coinc BL Sim + Coinc RCR Sim + 2016 Backlobe + Coinc Backlobe + Coinc RCR ---
    ic("Generating Plot 3 (No Cuts): Coinc BL Sim + Coinc RCR Sim + 2016 Backlobe + Coinc Backlobe + Coinc RCR...")
    fig3_nc, axs3_nc = plt.subplots(2, 2, figsize=(12, 15))
    fig3_nc.suptitle(f'Coinc BL Sim + Coinc RCR Sim + 2016 BL + Coinc BL + Coinc RCR - Station {station_id} (No Cuts)\n{rcr_cut_string}', fontsize=14)
    im3_nc = plot_2x2_grid(fig3_nc, axs3_nc, sim_base_config, None, overlays=overlays_3, hist_bins_dict=hist_bins)
    fig3_nc.text(0.5, 0.01, sim_direct_rcr_stats + "\n\n" + sim_reflected_rcr_stats, ha='center', va='bottom', fontsize=9, fontfamily='monospace')
    if im3_nc:
        fig3_nc.tight_layout(rect=[0, 0.28, 0.9, 0.95])
        cbar_ax3_nc = fig3_nc.add_axes([0.91, 0.28, 0.02, 0.65])
        fig3_nc.colorbar(im3_nc, cax=cbar_ax3_nc, label='Coinc BL Sim Weighted Counts (Evts/Yr)')
    else:
        fig3_nc.tight_layout(rect=[0, 0.28, 1, 0.95])
    plt.savefig(f'{plot_folder}CoincBLSim_vs_CoincRCRSim_vs_2016_vs_Coinc_NoCuts_Station{station_id}_{date}.png')
    plt.close(fig3_nc)


if __name__ == "__main__":
    # --- Configuration and Setup ---
    config = configparser.ConfigParser()
    config.read('HRAStationDataAnalysis/config.ini')
    date = config['PARAMETERS']['date']
    date_cuts = config['PARAMETERS']['date_cuts']
    date_processing = config['PARAMETERS']['date_processing']
    
    sim_file = config['SIMULATION']['sim_file']
    direct_weight_name = config['SIMULATION']['direct_weight_name']
    reflected_weight_name = config['SIMULATION']['reflected_weight_name']
    sim_sigma = float(config['SIMULATION']['sigma'])

    station_data_folder = f'HRAStationDataAnalysis/StationData/nurFiles/{date}/'
    cuts_data_folder = f'HRAStationDataAnalysis/StationData/cuts/{date_cuts}/'
    plot_folder = f'HRAStationDataAnalysis/plots/{date_processing}/'
    os.makedirs(plot_folder, exist_ok=True)
    
    ic.configureOutput(prefix='Chi-SNR Analysis | ')
    
    # --- Define Stations and Load Sim Data ---
    station_ids_to_process = [13, 14, 15, 17, 18, 19, 30]
    HRAeventList = loadHRAfromH5(sim_file)
    direct_stations = [13, 14, 15, 17, 18, 19, 30]
    reflected_stations = [113,  114, 115, 117, 118, 119, 130]
    sim_direct, sim_reflected = get_sim_data_coincidence(HRAeventList, direct_weight_name, reflected_weight_name, direct_stations, reflected_stations, sigma=sim_sigma)

    # --- Define Cuts & Bins ---
    cuts = {
        'snr_max': 50,
        'chi_rcr_line_snr': np.array([0, 7, 8.5, 15, 20, 30, 100]),
        'chi_rcr_line_chi': np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]),  # Flat cut at 0.75
        'chi_diff_threshold': 0.0,
        'chi_diff_max': 0.2,
        'chi_2016_line_snr': np.array([0, 7, 8.5, 15, 20, 30, 100]),
        'chi_2016_line_chi': np.array([0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]) # Flat cut at 0.75
    }
    rcr_cut_string = f"RCR Cuts: SNR < {cuts['snr_max']} & RCR-$\chi$ > 0.75 & 0 < RCR-$\chi$ - BL-$\chi$ < {cuts['chi_diff_max']}"
    backlobe_cut_string = f"Backlobe Cuts: SNR < {cuts['snr_max']} & BL-$\chi$ > 0.75 & -{cuts['chi_diff_max']} < RCR-$\chi$ - BL-$\chi$ < 0"
    
    log_bins = np.logspace(np.log10(3), np.log10(100), 31)
    linear_bins = np.linspace(0, 1, 31)
    diff_bins = np.linspace(-0.4, 0.4, 31)
    hist_bins = {
        'snr_vs_chi2016': [log_bins, linear_bins], 'snr_vs_chircr': [log_bins, linear_bins],
        'chi_vs_chi': [linear_bins, linear_bins], 'snr_vs_chidiff': [log_bins, diff_bins]
    }
    
    coincidence_pickle_path = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/9.24.25_CoincidenceDatetimes_passing_cuts_with_all_params_recalcZenAzi_calcPol.pkl"
    requested_coincidence_event_ids = [
        3047,
        3432,
        10195,
        10231,
        10273,
        10284,
        10444,
        10449,
        10466,
        10471,
        10554,
        11197,
        11220,
        11230,
        11236,
        11243,
    ]
    # highlight_coincidence_event_ids = [11230, 11243] # Now handled internally

    coincidence_events = load_coincidence_events(coincidence_pickle_path, requested_coincidence_event_ids)
    coincidence_station_overlays = build_coincidence_station_overlays(coincidence_events, station_ids_to_process)

    # --- Prepare "2016 Backlobe" Overlay ---
    json_path = 'StationDataAnalysis/2016FoundEvents.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            found_events_json = json.load(f)
        ic(f"Loaded 2016 Found Events from {json_path}")
    else:
        ic(f"Warning: JSON file not found at {json_path}")
        found_events_json = {}

    backlobe_2016_station_overlays = {}

    # plot_sim_only_comparisons(sim_direct, sim_reflected, cuts, hist_bins, plot_folder, date, rcr_cut_string)

    excluded_events = [
        (18, 82), (18, 520), (18, 681),
        (15, 1472768),
        (19, 3621320), (19, 4599318), (19, 4599919)
    ]

    # --- Main Loop for Individual and Summed Stations ---
    all_stations_data = {key: [] for key in ['snr', 'Chi2016', 'ChiRCR', 'StationID', 'Time']}
    all_stations_event_ids = []
    all_stations_unique_indices = []
    total_pre_mask_count = 0

    for station_id in station_ids_to_process:
        # Load Data
        snr_array = load_station_data(station_data_folder, date, station_id, 'SNR')
        Chi2016_array = load_station_data(station_data_folder, date, station_id, 'Chi2016')
        ChiRCR_array = load_station_data(station_data_folder, date, station_id, 'ChiRCR')
        times = load_station_data(station_data_folder, date, station_id, 'Time')
        event_ids_raw = load_station_data(station_data_folder, date, station_id, 'EventIDs')
        
        pre_mask_count = len(snr_array)
        total_pre_mask_count += pre_mask_count
        
        # Prepare Backlobe 2016 Overlay for this station
        bl_2016_entry = {
            'Backlobe': {'snr': [], 'Chi2016': [], 'ChiRCR': []},
            'Backlobe_event_ids': set(),
            'RCR': {'snr': [], 'Chi2016': [], 'ChiRCR': []},
            'RCR_event_ids': set(),
            'RCR_annotations': []
        }
        
        station_key = f"Station{station_id}Found"
        if station_key in found_events_json:
            target_times = found_events_json[station_key]
            # Find indices in current data that match these times (approximate match or exact?)
            # Assuming exact match for now, or use event IDs if available.
            # The JSON has times.
            
            # Efficient matching
            found_indices = []
            # Create a map of time -> index
            time_map = {t: i for i, t in enumerate(times)}
            
            for t in target_times:
                if t in time_map:
                    found_indices.append(time_map[t])
            
            found_indices = np.unique(found_indices)
            ic(f"Station {station_id}: Found {len(found_indices)} Backlobe 2016 events matching {len(target_times)} target times.")
            
            if len(found_indices) > 0:
                bl_2016_entry['Backlobe']['snr'] = snr_array[found_indices]
                bl_2016_entry['Backlobe']['Chi2016'] = Chi2016_array[found_indices]
                bl_2016_entry['Backlobe']['ChiRCR'] = ChiRCR_array[found_indices]
                if 'event_ids_raw' in locals():
                     bl_2016_entry['Backlobe_event_ids'].update(event_ids_raw[found_indices])
        
        backlobe_2016_station_overlays[station_id] = bl_2016_entry
        
        # Load additional data for master plots
        traces_array = load_station_data(station_data_folder, date, station_id, 'Traces')
        zen_array = load_station_data(station_data_folder, date, station_id, 'Zen')
        azi_array = load_station_data(station_data_folder, date, station_id, 'Azi')

        if Chi2016_array.size == 0 or ChiRCR_array.size == 0:
            ic(f"Skipping Station {station_id} due to missing Chi data.")
            continue

        # Apply initial time and uniqueness cuts
        initial_mask, unique_indices = getTimeEventMasks(times, event_ids_raw)
        
        # Apply cuts from C00 before processing
        cuts_mask = load_cuts_for_station(date, station_id, cuts_data_folder)
        if cuts_mask is not None:
            # The cuts mask should be applied after initial_mask and unique_indices
            # to match the same data flow as in C01
            temp_times = times[initial_mask][unique_indices]
            temp_event_ids = event_ids_raw[initial_mask][unique_indices]
            
            # Ensure cuts mask length matches the post-initial processing length
            if len(cuts_mask) != len(temp_times):
                ic(f"Warning: Cuts mask length ({len(cuts_mask)}) doesn't match data length ({len(temp_times)}). Truncating cuts mask.")
                cuts_mask = cuts_mask[:len(temp_times)]
            
            # Apply cuts mask
            final_indices = unique_indices[cuts_mask]
            ic(f"Station {station_id}: {len(temp_times)} events after initial cuts, {np.sum(cuts_mask)} events after C00 cuts")
        else:
            # No cuts available, use all data after initial processing
            final_indices = unique_indices
            ic(f"Station {station_id}: {len(times[initial_mask][unique_indices])} events after initial cuts, no C00 cuts applied")

            # For now, actually error out if no cuts are found
            ic(f"Error: No cuts found for Station {station_id}. Please ensure cuts are available.")
            quit()

        station_data = {
            'snr': snr_array[initial_mask][final_indices],
            'Chi2016': Chi2016_array[initial_mask][final_indices],
            'ChiRCR': ChiRCR_array[initial_mask][final_indices],
            'Time': times[initial_mask][final_indices]
        }
        station_data['StationID'] = np.full(len(station_data['snr']), station_id, dtype=int)
        
        # Add optional data if available and matching length
        if traces_array.size > 0:
            try:
                station_data['Traces'] = traces_array[initial_mask][final_indices]
            except IndexError:
                ic(f"Warning: Traces array length mismatch for Station {station_id}. Skipping Traces.")
        
        if zen_array.size > 0:
            try:
                station_data['Zen'] = zen_array[initial_mask][final_indices]
            except IndexError:
                ic(f"Warning: Zen array length mismatch for Station {station_id}. Skipping Zen.")

        if azi_array.size > 0:
            try:
                station_data['Azi'] = azi_array[initial_mask][final_indices]
            except IndexError:
                ic(f"Warning: Azi array length mismatch for Station {station_id}. Skipping Azi.")

        station_event_ids = event_ids_raw[initial_mask][final_indices]
        
        ic(f"Station {station_id} has {len(station_data['snr'])} events after masking and cuts.")
        
        for key in all_stations_data:
            all_stations_data[key].append(station_data[key])
        all_stations_event_ids.append(station_event_ids)
        all_stations_unique_indices.append(final_indices)
        
        coincidence_overlay_for_station = coincidence_station_overlays.get(station_id)
        backlobe_2016_overlay_for_station = backlobe_2016_station_overlays.get(station_id)
        run_analysis_for_station(station_id, station_data, station_event_ids, final_indices, pre_mask_count, sim_direct, sim_reflected, cuts, rcr_cut_string, hist_bins, plot_folder, date, coincidence_overlay=coincidence_overlay_for_station, backlobe_2016_overlay=backlobe_2016_overlay_for_station, excluded_events=excluded_events)

    # --- Run Analysis for Summed Stations ---
    if len(all_stations_data['snr']) > 1:
        summed_station_data = {key: np.concatenate(all_stations_data[key]) for key in all_stations_data}
        summed_event_ids = np.concatenate(all_stations_event_ids)
        summed_unique_indices = np.concatenate(all_stations_unique_indices)
        summed_station_id = '+'.join(map(str, station_ids_to_process))
        combined_coincidence_overlay = combine_coincidence_overlays(station_ids_to_process, coincidence_station_overlays)
        combined_backlobe_2016_overlay = combine_coincidence_overlays(station_ids_to_process, backlobe_2016_station_overlays)
        run_analysis_for_station(summed_station_id, summed_station_data, summed_event_ids, summed_unique_indices, total_pre_mask_count, sim_direct, sim_reflected, cuts, rcr_cut_string, hist_bins, plot_folder, date, coincidence_overlay=combined_coincidence_overlay, backlobe_2016_overlay=combined_backlobe_2016_overlay, excluded_events=excluded_events)
    else:
        ic("Not enough station data to perform a summed analysis.")
