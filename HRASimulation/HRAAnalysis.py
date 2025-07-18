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
from HRASimulation.HRAEventObject import HRAevent
from HRASimulation.HRANurToNpy import loadHRAfromH5
import itertools
import collections
from scipy.special import comb as nCr 

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
    # ic(e_bins, z_bins)

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

def setNewTrigger(HRAEventList, trigger_name, bad_stations, sigma=4.5, sigma_52=7):
    # Set a new trigger for each event in the HRAEventList if it triggered w/sigma and isn't in bad stations list
    for event in HRAEventList:
        for trigger in event.station_triggers[sigma]:
            if trigger == 52 and 52 not in bad_stations:
                if event.hasTriggered(trigger, sigma_52) and not event.hasTriggered(trigger_name, sigma):
                    event.addTrigger(trigger_name, sigma)
            elif trigger not in bad_stations and not event.hasTriggered(trigger_name, sigma):
                event.addTrigger(trigger_name, sigma)

    return


def getBinnedTriggerRate(HRAeventList, num_coincidence=0, use_secondary=False, sigma=4.5, sigma_52=7):
    # Input a list of HRAevent objects to get the event rate in each energy-zenith bin

    e_bins, z_bins = getEnergyZenithBins()

    # Create a dictionary for the event rate per station
    direct_trigger_rate_dict = {}
    reflected_trigger_rate_dict = {}
    n_throws = getnThrows(HRAeventList)

    for event in HRAeventList:
        # n_throws += 1
        if not event.hasCoincidence(num_coincidence, use_secondary=use_secondary, sigma=sigma, sigma_52=sigma_52):
            # Event not triggered or meeting coincidence bar
            continue
        for station_id in event.directTriggers(sigma=sigma, sigma_52=sigma_52):
            energy_bin = np.digitize(event.getEnergy(), e_bins) - 1
            zenith_bin = np.digitize(event.getAngles()[0], z_bins) - 1
            if energy_bin < 0 or zenith_bin < 0 or energy_bin >= len(e_bins) or zenith_bin >= len(z_bins):
                ic(f'Outside of bins , {event.getEnergy()}, {event.getAngles()[0]} {energy_bin}, {zenith_bin}')
                continue
            if station_id not in direct_trigger_rate_dict:
                direct_trigger_rate_dict[station_id] = getEnergyZenithArray()
            direct_trigger_rate_dict[station_id][energy_bin][zenith_bin] += 1
        for station_id in event.reflectedTriggers(sigma=sigma, sigma_52=sigma_52):
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
    combined_trigger_rate['direct'][np.isnan(combined_trigger_rate['direct'])] = 0
    # combined_throws = 0
    combined_throws = np.zeros_like(n_throws)
    for station_id in reflected_trigger_rate_dict:
        if station_id in [132, 152]:
            continue
        combined_trigger_rate['reflected'] += reflected_trigger_rate_dict[station_id]
        combined_throws += n_throws
    combined_trigger_rate['reflected'] /= combined_throws
    combined_trigger_rate['reflected'][np.isnan(combined_trigger_rate['reflected'])] = 0

    # Also get rates for 100s and 200s separately
    stn_100s_trigger_rate = {}
    stns_100 = [13, 15, 18]
    stn_100s_trigger_rate['direct'] = getEnergyZenithArray()
    stn_100s_trigger_rate['reflected'] = getEnergyZenithArray()
    combined_throws = np.zeros_like(n_throws)
    for station_id in stns_100:
        stn_100s_trigger_rate['direct'] += direct_trigger_rate_dict[station_id]
        stn_100s_trigger_rate['reflected'] += reflected_trigger_rate_dict[station_id+100]
        combined_throws += n_throws
    stn_100s_trigger_rate['direct'] /= combined_throws
    stn_100s_trigger_rate['reflected'] /= combined_throws
    stn_100s_trigger_rate['direct'][np.isnan(stn_100s_trigger_rate['direct'])] = 0
    stn_100s_trigger_rate['reflected'][np.isnan(stn_100s_trigger_rate['reflected'])] = 0

    stn_200s_trigger_rate = {}
    stns_200 = [14, 17, 19, 30]
    stn_200s_trigger_rate['direct'] = getEnergyZenithArray()
    stn_200s_trigger_rate['reflected'] = getEnergyZenithArray()
    combined_throws = np.zeros_like(n_throws)
    for station_id in stns_200:
        stn_200s_trigger_rate['direct'] += direct_trigger_rate_dict[station_id]
        stn_200s_trigger_rate['reflected'] += reflected_trigger_rate_dict[station_id+100]
        combined_throws += n_throws
    stn_200s_trigger_rate['direct'] /= combined_throws
    stn_200s_trigger_rate['reflected'] /= combined_throws
    stn_200s_trigger_rate['direct'][np.isnan(stn_200s_trigger_rate['direct'])] = 0
    stn_200s_trigger_rate['reflected'][np.isnan(stn_200s_trigger_rate['reflected'])] = 0



    # Normalise the event rate
    for station_id in direct_trigger_rate_dict:
        direct_trigger_rate_dict[station_id] /= n_throws
        direct_trigger_rate_dict[station_id][np.isnan(direct_trigger_rate_dict[station_id])] = 0
    for station_id in reflected_trigger_rate_dict:
        reflected_trigger_rate_dict[station_id] /= n_throws
        reflected_trigger_rate_dict[station_id][np.isnan(reflected_trigger_rate_dict[station_id])] = 0

    return direct_trigger_rate_dict, reflected_trigger_rate_dict, combined_trigger_rate, stn_100s_trigger_rate, stn_200s_trigger_rate, e_bins, z_bins

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

def setHRAeventListRateWeight(HRAeventList, trigger_rate_array, weight_name, max_distance=6.0*units.km, sigma=4.5):
    # Set the event rate weight for each event in the HRAeventList

    e_bins, z_bins = getEnergyZenithBins()
    eventRateArray = getEventRateArray(e_bins, z_bins)
    n_throws = getnThrows(HRAeventList)

    area = np.pi * max_distance**2

    # Event weight = event_rate_bin / n_triggers_bin
    for event in HRAeventList:
        if not event.hasTriggered(weight_name, sigma):
            # Event not triggered
            event.setWeight(0, weight_name, sigma=sigma)
            continue
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


def getCoincidencesTriggerRates(HRAeventList, bad_stations, use_secondary=False, force_stations=None, sigma=4.5, sigma_52=7):
    """
    Calculates coincidence trigger rates, with an option to require a trigger 
    from a specific list of stations.

    Args:
        HRAeventList (list): List of HRA event objects.
        bad_stations (list): List of station IDs to ignore.
        use_secondary (bool): Flag to use secondary trigger conditions.
        force_stations (int or list, optional): A station ID or a list of station IDs.
            If provided, only events triggered by at least one of these stations will be counted.
            Defaults to None.
        sigma (float): The significance threshold for a station trigger.
        sigma_52 (float): The significance threshold for stations 52 and 53.

    Returns:
        dict: A dictionary where keys are the number of coincidences (2-7) and values
              are 2D numpy arrays of trigger rates binned by energy and zenith.
    """
    e_bins, z_bins = getEnergyZenithBins()
    n_throws = getnThrows(HRAeventList)

    trigger_rate_coincidence =  {}
    for i in [2, 3, 4, 5, 6, 7]:
        trigger_rate_coincidence[i] = getEnergyZenithArray()
        for event in HRAeventList:
            if not event.hasCoincidence(i, bad_stations, use_secondary, sigma=sigma, sigma_52=sigma_52):
                # Event not triggered or meeting coincidence bar
                continue

            if force_stations is not None:
                # Ensure force_stations is a list for uniform processing
                stations_to_check = force_stations
                if not isinstance(stations_to_check, list):
                    stations_to_check = [stations_to_check]
                
                # Use sets for an efficient check of whether any required station triggered.
                # `set.isdisjoint()` returns True if the sets have no common elements.
                # We `continue` if there is NO overlap between required stations and triggered stations.
                if set(stations_to_check).isdisjoint(event.station_triggers[sigma]):
                    continue

            energy_bin = np.digitize(event.getEnergy(), e_bins) - 1
            zenith_bin = np.digitize(event.getAngles()[0], z_bins) - 1
            if energy_bin < 0 or zenith_bin < 0 or energy_bin >= len(e_bins) or zenith_bin >= len(z_bins):
                ic(f'Outside of bins , {event.getEnergy()}, {event.getAngles()[0]} {energy_bin}, {zenith_bin}')
                continue
            trigger_rate_coincidence[i][energy_bin][zenith_bin] += 1
        trigger_rate_coincidence[i] /= n_throws

    return trigger_rate_coincidence

def categorize_events_by_coincidence(HRAeventList, stations_of_interest, sigma=4.5, snr_threshold=None):
    """
    Analyzes and categorizes each event by its effective n-coincidence for each reflected_mode,
    considering only a specific list of stations. Now includes n=1 categorization.
    If an snr_threshold is provided, only triggers meeting that threshold are considered.
    """
    stations_of_interest_set = set(stations_of_interest)
    reflected_stations_of_interest = {s + 100 for s in stations_of_interest}
    
    categorized_events = {
        'included': collections.defaultdict(lambda: collections.defaultdict(list)),
        'required': collections.defaultdict(lambda: collections.defaultdict(list)),
        'excluded': collections.defaultdict(lambda: collections.defaultdict(list)),
        'only': collections.defaultdict(lambda: collections.defaultdict(list))
    }

    for event in HRAeventList:
        triggered_stations_all = set(event.station_triggers.get(sigma, []))

        # --- New: Filter stations by SNR threshold if provided ---
        if snr_threshold is not None:
            snr_passed_stations = set()
            for station_id in triggered_stations_all:
                if event.getSNR(station_id) is None:
                    ic(f"Warning: Event {event.getEventID()} does not have SNR data for station {station_id}. Skipping SNR check.")
                    continue
                # Assumes event object has a getSNR method returning a float.
                if event.getSNR(station_id) >= snr_threshold:
                    snr_passed_stations.add(station_id)
            triggered_stations_all = snr_passed_stations
        # --- End of new code ---

        triggered_direct = {s for s in triggered_stations_all if s in stations_of_interest_set}
        triggered_reflected = {s for s in triggered_stations_all if s in reflected_stations_of_interest}
        
        reflected_bases = {s - 100 for s in triggered_reflected}
        union_bases = triggered_direct.union(reflected_bases)
        
        n_effective = len(union_bases)

        # --- n > 1 Coincidence Categorization (Unchanged) ---
        if n_effective > 1:
            combo = tuple(sorted(list(union_bases)))
            # 'included'
            categorized_events['included'][n_effective][combo].append(event)
            # 'required'
            if len(triggered_direct) > 0 and len(triggered_reflected) > 0:
                categorized_events['required'][n_effective][combo].append(event)
            # 'excluded'
            if len(triggered_direct) > 0 and len(triggered_reflected) == 0:
                # For n>1, the combo is the union_bases, which equals triggered_direct here
                categorized_events['excluded'][n_effective][combo].append(event)
            # 'only'
            if len(triggered_reflected) > 0 and len(triggered_direct) == 0:
                # For n>1, the combo is the union_bases, which equals reflected_bases here
                categorized_events['only'][n_effective][combo].append(event)

        # --- New n=1 (Single Station) Categorization (Unchanged) ---
        elif n_effective == 1:
            station_id = list(union_bases)[0]
            combo = (station_id,)
            has_direct = station_id in triggered_direct
            has_reflected = (station_id + 100) in triggered_reflected
            
            # 'included': direct=True or reflected=True or both
            if has_direct or has_reflected:
                categorized_events['included'][1][combo].append(event)
            # 'required': reflected=True, direct can be any
            if has_reflected:
                categorized_events['required'][1][combo].append(event)
            # 'excluded': direct=True and reflected=False
            if has_direct and not has_reflected:
                categorized_events['excluded'][1][combo].append(event)
            # 'only': direct=False and reflected=True
            if not has_direct and has_reflected:
                categorized_events['only'][1][combo].append(event)

    return categorized_events


def calculate_all_station_combination_rates(HRAeventList, output_file, max_distance, sigma=4.5, reflected_mode='required', snr_threshold=None):
    """
    Calculates initial "true" rates for n=1 to n=max, ensuring each event is counted
    only once for its specific n-fold coincidence. This version outputs all possible 
    combinations, even those with zero rate, to serve as input for rate propagation.
    """
    base_stations = [13, 14, 15, 17, 18, 19, 30]
    e_bins, z_bins = getEnergyZenithBins()
    n_throws = getnThrows(HRAeventList)

    ic("Pre-categorizing all events by effective n-coincidence...")
    # --- Modified call to pass the new parameter ---
    categorized_events = categorize_events_by_coincidence(
        HRAeventList, base_stations, sigma=sigma, snr_threshold=snr_threshold
    )
    ic("Categorization complete.")

    def calculate_rate_for_event_list(events, n_throws_binned):
        # ... (This helper function is unchanged)
        trigger_rate_array = getEnergyZenithArray()
        for event in events:
            energy = event.getEnergy()
            zenith = event.getAngles()[0]
            energy_bin = np.digitize(energy, e_bins) - 1
            zenith_bin = np.digitize(zenith, z_bins) - 1
            if 0 <= energy_bin < len(e_bins) - 1 and 0 <= zenith_bin < len(z_bins) - 1:
                trigger_rate_array[energy_bin][zenith_bin] += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            trigger_rate_array /= n_throws_binned
        trigger_rate_array[np.isnan(trigger_rate_array)] = 0
        return trigger_rate_array

    initial_rates = {}  # Store results in a dictionary for the next phase

    with open(output_file, 'w') as f:
        f.write(f"# Initial analysis for Mode: {reflected_mode}\n")
        f.write("Station Combination, Total_Event_Rate, Total_Error (both in Evts/Yr)\n")

        # Modified loop to run from n=max down to n=1
        for n_coincidence in range(len(base_stations), 0, -1):
            f.write(f"\n# {n_coincidence}-Fold Coincidences:\n")
            
            all_possible_combos = itertools.combinations(base_stations, n_coincidence)
            
            for combo_tuple in all_possible_combos:
                event_list = categorized_events[reflected_mode].get(n_coincidence, {}).get(combo_tuple, [])
                
                trigger_rate = calculate_rate_for_event_list(event_list, n_throws)
                event_rate_array = getEventRate(trigger_rate, e_bins, z_bins, max_distance)
                error_rate_array = getErrorEventRates(trigger_rate, HRAeventList, max_distance=max_distance)
                
                total_event_rate = np.nansum(event_rate_array)
                total_error = np.nansum(error_rate_array)
                
                initial_rates[combo_tuple] = (total_event_rate, total_error)

                combo_str = "-".join(map(str, combo_tuple))
                f.write(f"{combo_str} : {total_event_rate:.5e}, {total_error:.5e}\n")

    ic(f"Initial rate calculation complete. Data saved to {output_file}")
    return initial_rates


def propagate_downtime_rates(initial_rates, base_stations, downtime_prob=0.25):
    """
    Takes a dictionary of initial "true" rates (from n=1 to n=max) and propagates them 
    down to account for station downtime.
    """
    online_prob = 1 - downtime_prob
    
    # Initialize adjusted_rates for n=1 through n=max combinations
    adjusted_rates = {combo: [0, []] for combo in itertools.chain.from_iterable(
        itertools.combinations(base_stations, i) for i in range(1, len(base_stations) + 1)
    )}

    sorted_initial_combos = sorted(initial_rates.keys(), key=len, reverse=True)

    for n_combo in sorted_initial_combos:
        initial_rate, initial_error = initial_rates[n_combo]
        if initial_rate == 0:
            continue

        n = len(n_combo)
        
        # Iterate from k=n (observed size) down to k=1
        for k in range(n, 0, -1):
            prob_n_to_k = nCr(n, k) * (online_prob**k) * (downtime_prob**(n - k))
            
            rate_contribution = initial_rate * prob_n_to_k
            # Propagate the squared error, scaled by the probability squared
            error_sq_contribution = (initial_error * prob_n_to_k)**2

            sub_combos = itertools.combinations(n_combo, k)
            num_sub_combos = nCr(n, k)
            rate_per_sub = rate_contribution / num_sub_combos
            
            # The squared error is for the total rate contribution, so the error^2 for each
            # sub-combo's share is divided by num_sub_combos^2
            error_sq_per_sub = error_sq_contribution / (num_sub_combos**2)
            
            for sub_combo_tuple in sub_combos:
                adjusted_rates[sub_combo_tuple][0] += rate_per_sub
                adjusted_rates[sub_combo_tuple][1].append(error_sq_per_sub)

    # Finalize the error calculation
    final_adjusted_rates = {}
    for combo, (rate, errors_sq_list) in adjusted_rates.items():
        final_error = np.sqrt(np.sum(errors_sq_list))
        final_adjusted_rates[combo] = (rate, final_error)
        
    ic("Rate propagation complete.")
    return final_adjusted_rates

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
    # ic(cos_bins, ax_labels)
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

def getXYWeights(HRAeventList, weight_name, use_primary=True, in_array=None):
    # Get a list of the events x/y with associated event rate as a weight

    x = np.zeros(len(HRAeventList))
    y = np.zeros(len(HRAeventList))
    weights = getWeights(HRAeventList, weight_name, use_primary, in_array)

    for iE, event in enumerate(HRAeventList):
        cor_x, cor_y = event.getCoreasPosition()
        # Need to get the event rate corresponding to each event, previously calculated

        x[iE] = cor_x
        y[iE] = cor_y

    return x, y, weights

def getTrigMask(HRAeventList, station_ids, sigma=4.5):

    mask = np.zeros(len(HRAeventList), dtype=bool)
    for iE, event in enumerate(HRAeventList):
        if event.hasTriggered():
            for station_id in event.directTriggers(sigma):
                if station_id in station_ids:
                    mask[iE] = True
                    break

    return mask

def getWeights(HRAeventList, weight_name, use_primary=True, in_array=None):
    if in_array is None:
        in_array = np.zeros(len(HRAeventList))
    for iE, event in enumerate(HRAeventList):
        in_array[iE] = event.getWeight(weight_name, use_primary)
    return in_array

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
                if station_id in stations and not station_id in direct_triggered:
                    direct_triggered.append(station_id)
            for station_id in event.reflectedTriggers():
                if station_id in refl_stations and not station_id in reflected_triggered:
                    reflected_triggered.append(station_id)

    return direct_triggered, reflected_triggered

def histAreaRate(x, y, weights, title, savename, dir_trig=[], refl_trig=[], exclude=[], colorbar_label='Evts/yr', max_distance=6.0*units.km):

    x_bins, y_bins = getXYbins(max_distance)

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
    plt.close('all')
    return


def getErrorEventRates(trigger_rate, HRAeventList, max_distance=6.0*units.km, combined=False):
    # Return the event rate array as well as an array of the error in each bin
    # Error is capped at 100% of the event rate, for bins with 1 trigger going to 0 event rate

    e_bins, z_bins = getEnergyZenithBins()
    n_throws = getnThrows(HRAeventList)

    if combined == True:
        n_throws *= 7       # Assuming no station 52/32 or 152/132 used, 7 direct or 7 reflected stations in sum

    # Need to get n_trig per bin
    n_trig = trigger_rate * n_throws    
    
    trig_rate_error = np.sqrt(n_trig) / n_throws

    return getEventRate(trig_rate_error, e_bins, z_bins, max_distance=max_distance)

def plotRateWithError(eventRate, errorRate, savename, title):
    # Plot the rate summed in zenith with error bars
    e_bins, z_bins = getEnergyZenithBins()
    e_bins = np.log10(e_bins/units.eV)

    fig, ax = plt.subplots()
    color = plt.cm.rainbow(np.linspace(0, 1, len(z_bins)-1))

    eventRate[np.isnan(eventRate)] = 0
    errorRate[np.isnan(errorRate)] = 0
    ax.fill_between((e_bins[1:]+e_bins[:-1])/2, np.nansum(eventRate - errorRate,axis=1), np.nansum(eventRate + errorRate,axis=1), alpha=0.5, label=f'{np.nansum(eventRate):.2f} +/- {np.nansum(errorRate):.2f} Evts/Yr', color='black')
    ax.plot((e_bins[1:]+e_bins[:-1])/2, np.nansum(eventRate,axis=1), color='black', linestyle='--')

    for iZ in range(len(z_bins)-1):
        ax.fill_between((e_bins[1:]+e_bins[:-1])/2, eventRate[:,iZ] - errorRate[:,iZ], eventRate[:,iZ] + errorRate[:,iZ], alpha=0.5, label=f'{z_bins[iZ]/units.deg:.1f}-{z_bins[iZ+1]/units.deg:.1f}deg', color=color[iZ])
        ax.plot((e_bins[1:]+e_bins[:-1])/2, eventRate[:,iZ], color=color[iZ], linestyle='--')

    ax.set_xlabel('log10(E/eV)')
    ax.set_ylabel('Evts/Yr')
    ax.set_yscale('log')
    ax.set_ylim(bottom=10**-3)
    ax.legend(loc='lower left')
    ax.set_title(title)
    fig.savefig(savename)
    ic(f'Saved {savename}')
    plt.close(fig)

    return


def getAnglesReconWeights(HRAeventList, weight_name, station_ids, use_primary=True, sigma=4.5):
    # Get a list of the events x/y with associated event rate as a weight
    # station_ids can be a single station or a list of stations

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
        weights.append(event.getWeight(weight_name, use_primary, sigma))    # Append all events because non-triggering events have a weight of zero    

        # Add reconstructed angles, and average if detected in multiple stations
        recon_zenith.append(0)
        recon_azimuth.append(0)
        i = 0
        for station_id in station_ids:
            if station_id in list(event.recon_zenith.keys()):
                i += 1
                recon_zenith[-1] += event.recon_zenith[station_id]
                recon_azimuth[-1] += event.recon_azimuth[station_id]
        if i == 0:
            i = 1
        recon_zenith[-1] /= i
        recon_azimuth[-1] /= i

    return np.array(zenith), np.array(recon_zenith), np.array(azimuth), np.array(recon_azimuth), np.array(weights)


def histAngleRecon(zenith, azimuth, recon_zenith, recon_azimuth, weights, title, savename, colorbar_label='Evts/yr'):
    if max(zenith) < np.pi/2 or max(azimuth) < np.pi*2:
        zenith = np.rad2deg(zenith)
        recon_zenith = np.rad2deg(recon_zenith)
        azimuth = np.rad2deg(azimuth)
        recon_azimuth = np.rad2deg(recon_azimuth)
    
    zenith_bins, azimuth_bins = np.linspace(0, 90, 100), np.linspace(0, 360, 100)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    norm = matplotlib.colors.LogNorm(vmin=np.min(weights[np.nonzero(weights)]), vmax=np.max(weights)*5)


    ax[0].hist(zenith, bins=zenith_bins, weights=weights)
    ax[0].set_xlabel('True Zenith (deg)')
    ax[0].set_ylabel('Evts/Yr')
    ax[0].set_title('Zenith')

    ax[1].hist(azimuth, bins=azimuth_bins, weights=weights)
    ax[1].set_xlabel('True Azimuth (deg)')
    ax[1].set_ylabel('Evts/Yr')
    ax[1].set_title('Azimuth')

    plt.suptitle(title)
    plt.savefig(savename.replace('.png', '_1D.png'))
    ic(f'Saved {savename}')
    plt.close()

    savename.replace('_1D.png', '_2D.png')

    fig, ax = plt.subplots(nrows=1, ncols=2)
    h, xedges, yedges, im = ax[0].hist2d(zenith, recon_zenith, bins=(zenith_bins, zenith_bins), weights=weights, cmap='viridis', norm=norm)
    ax[0].plot([0, 90], [0, 90], color='black', linestyle='--')
    ax[0].plot(np.linspace(0, 90.1, 1), np.rad2deg(np.arcsin(np.sin(np.deg2rad(np.linspace(0, 90.1, 1)))/1.78)), color='red', linestyle='--')
    ax[0].set_xlabel('True Zenith (deg)')
    ax[0].set_ylabel('Reconstructed Zenith (deg)')
    ax[0].set_title('Zenith')
    ax[0].set_aspect('equal')
    ax[0].set_xlim(0, 90)
    ax[0].set_ylim(0, 90)
    ax[0].grid()
    # fig.colorbar(im, ax=ax[0], label=colorbar_label)

    h, xedges, yedges, im = ax[1].hist2d(azimuth, recon_azimuth, bins=(azimuth_bins, azimuth_bins), weights=weights, cmap='viridis', norm=norm)
    ax[1].plot([0, 360], [0, 360], color='black', linestyle='--')
    ax[1].set_xlabel('True Azimuth (deg)')
    ax[1].set_ylabel('Reconstructed Azimuth (deg)')
    ax[1].set_title('Azimuth')
    ax[1].set_aspect('equal')
    ax[1].set_xlim(0, 360)
    ax[1].set_ylim(0, 360)
    ax[1].grid()
    fig.colorbar(im, ax=ax[1], label=colorbar_label)

    # fig[0].colorbar(im, ax=fig[0], label=colorbar_label)
    # fig[1].colorbar(im, ax=fig[1], label=colorbar_label)
    plt.suptitle(title)

    plt.savefig(savename)
    ic(f'Saved {savename}')
    plt.close()

    # Also plot the difference in true and reconstructed angles
    zenith_diff = zenith - recon_zenith
    azimuth_diff = azimuth - recon_azimuth

    fig, ax = plt.subplots(nrows=1, ncols=2)

    diff_bins = np.linspace(-90, 90, 100)

    h, xedges, im = ax[0].hist(zenith_diff, bins=diff_bins, weights=weights)
    ax[0].set_xlabel('True - Reconstructed Zenith (deg)')
    ax[0].set_ylabel('Evts/Yr')
    ax[0].set_title('Zenith')

    diff_bins = np.linspace(-360, 360, 100)
    h, xedges, im = ax[1].hist(azimuth_diff, bins=diff_bins, weights=weights)
    ax[1].set_xlabel('True - Reconstructed Azimuth (deg)')
    ax[1].set_ylabel('Evts/Yr')
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

def getXYbins(max_distance=6.0*units.km, num=50):
    return np.linspace(-max_distance/units.m, max_distance/units.m, 50), np.linspace(-max_distance/units.m, max_distance/units.m, num)

def plotAreaAziZenArrows(x, y, zenith, azimuth, weights, title, savename, dir_trig=[], refl_trig=[], exclude=[], max_distance=6.0*units.km):
    # Plot 2d histogram area of stations, but at every bin have an arrow pointing in the direction of the weighted average azimuth and zenith
    # Arrow length is proportional to the weighted average zenith

    if max(zenith) < np.pi/2 or max(azimuth) < np.pi*2:
        zenith = np.rad2deg(zenith)
        azimuth = np.rad2deg(azimuth)
    

    x_bins, y_bins = getXYbins(max_distance, num=25)

    x_center = (x_bins[1:] + x_bins[:-1]) / 2
    y_center = (y_bins[1:] + y_bins[:-1]) / 2
    length = np.sqrt((x_bins[1] - x_bins[0])**2 + (y_bins[1] - y_bins[0])**2)


    avg_zen = np.zeros((len(x_center), len(y_center)))
    # avg_azi = np.zeros((len(x_center), len(y_center)))
    # In order to get proper azimuth weighted, need to sum vectors
    # Otherwise 1deg and 359deg would average to 180deg, rather than 0deg
    avg_azi_x = np.zeros((len(x_center), len(y_center)))
    avg_azi_y = np.zeros((len(x_center), len(y_center)))
    weighted_throws = np.zeros((len(x_center), len(y_center)))

    x_dig = np.digitize(x, x_bins) - 1
    y_dig = np.digitize(y, y_bins) - 1

    for iE in range(len(x)):
        if x_dig[iE] < 0 or y_dig[iE] < 0 or x_dig[iE] >= len(x_center) or y_dig[iE] >= len(y_center):
            ic("Outside of bins, shouldn't happen so quitting")
            quit()
        avg_zen[x_dig[iE]][y_dig[iE]] += zenith[iE] * weights[iE]
        # avg_azi[x_dig[iE]][y_dig[iE]] += azimuth[iE] * weights[iE]
        avg_azi_x[x_dig[iE]][y_dig[iE]] += np.cos(np.deg2rad(azimuth[iE])) * weights[iE]
        avg_azi_y[x_dig[iE]][y_dig[iE]] += np.sin(np.deg2rad(azimuth[iE])) * weights[iE]
        weighted_throws[x_dig[iE]][y_dig[iE]] += weights[iE]

    avg_azi_x /= weighted_throws
    avg_azi_y /= weighted_throws
    avg_azi = np.arctan2(avg_azi_y, avg_azi_x)
    avg_azi = np.rad2deg(avg_azi)

    avg_zen /= weighted_throws
    # avg_azi /= weighted_throws

    fig, ax = plt.subplots()
    plotStationLocations(ax, triggered=dir_trig, exclude=exclude, reflected_triggers=refl_trig)

    # Plot an arrow at each x_center and y_center pointing in the direction of the average azimuth with it's length proportional to the average zenith
    cmap = matplotlib.cm.viridis

    for iX, x in enumerate(x_center):
        for iY, y in enumerate(y_center):
            if weighted_throws[iX][iY] == 0:
                ax.scatter(x, y, marker='x', color='black', s=10)
                continue
            # color = colors[np.digitize(avg_zen[iX][iY], zen_bins)-1]
            # ic(avg_zen[iX][iY], avg_azi[iX][iY], color, zen_bins, np.digitize(avg_zen[iX][iY], zen_bins))
            # ic(x, y, 0.1*avg_zen[iX][iY]*np.cos(np.deg2rad(avg_azi[iX][iY])), 0.1*avg_zen[iX][iY]*-np.sin(np.deg2rad(avg_azi[iX][iY])))
            # ic(avg_azi[iX][iY], np.cos(np.deg2rad(avg_azi[iX][iY])), -np.sin(np.deg2rad(avg_azi[iX][iY])))
            ax.arrow(x, y, 4*avg_zen[iX][iY]*np.cos(np.deg2rad(avg_azi[iX][iY])), 4*avg_zen[iX][iY]*np.sin(np.deg2rad(avg_azi[iX][iY])), head_width=100, head_length=50, color=cmap(avg_zen[iX][iY]/90))


    ax.set_xlim(-max_distance/units.m, max_distance/units.m)
    ax.set_ylim(-max_distance/units.m, max_distance/units.m)
    fig.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=90), cmap=cmap), ax=ax, label='Zenith Angle (deg)')
    ax.legend(loc='upper right')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    fig.savefig(savename)
    ic(f'Saved {savename}')
    plt.close(fig)

    return



if __name__ == "__main__":

    # sim_folder = 'HRASimulation/output/HRA/1.27.25/'
    # numpy_folder = 'HRASimulation/output/HRA/1.27.25/numpy/' # For saving data to not have to reprocess
    # save_folder = f'HRASimulation/plots/2.3.25/'
    config = configparser.ConfigParser()
    config.read('HRASimulation/config.ini')
    sim_folder = config['FOLDERS']['sim_folder']
    numpy_folder = config['FOLDERS']['numpy_folder']
    save_folder = config['FOLDERS']['save_folder']
    downtime_prob = float(config['SIMPARAMETERS']['downtime_prob'])
    diameter = config['SIMPARAMETERS']['diameter']
    max_distance = float(diameter)/2*units.km
    plot_sigma = float(config['PLOTPARAMETERS']['trigger_sigma'])
    ic(plot_sigma)

    os.makedirs(save_folder, exist_ok=True)

    if not os.path.exists(numpy_folder):
        ic('No files to load, error')
        quit()
    # HRAeventList = np.load(f'{numpy_folder}HRAeventList.npy', allow_pickle=True)
    HRAeventList = loadHRAfromH5(f'{numpy_folder}HRAeventList.h5')
    direct_trigger_rate_dict, reflected_trigger_rate_dict, combined_trigger_rate, stn_100s_trigger_rate, stn_200s_trigger_rate, e_bins, z_bins = np.load(f'{numpy_folder}trigger_rate_dict.npy', allow_pickle=True)
    direct_event_rate, reflected_event_rate, combined_event_rate = np.load(f'{numpy_folder}event_rate_dict.npy', allow_pickle=True)

    logE_bins = np.log10(e_bins/units.eV)
    cos_bins = np.cos(z_bins)

    if not os.path.exists(save_folder+'error_rate/'):
        os.makedirs(save_folder+'error_rate/')


    # Calcing station combos first
    # Define all modes to be processed
    base_stations = [13, 14, 15, 17, 18, 19, 30]
    analysis_modes = ['required', 'included', 'excluded', 'only']
    snr_threshold = 7.0

    for mode in analysis_modes:
        ic(f"\n{'='*20} Starting Analysis for Mode: '{mode}' {'='*20}")

        # --- New: Add a suffix to filenames if SNR is used ---
        snr_suffix = f'_snr{snr_threshold}' if snr_threshold is not None else ''
        initial_rates_file = os.path.join(save_folder, f'initial_station_combination_rates_{mode}{snr_suffix}.txt')
        
        # --- Modified call to pass the SNR threshold ---
        initial_rates = calculate_all_station_combination_rates(
            HRAeventList,
            initial_rates_file,
            max_distance=max_distance,
            sigma=plot_sigma,
            reflected_mode=mode,
            snr_threshold=snr_threshold  # Pass the value here
        )

        # --- Step 2: Propagate Rates (Unchanged Logic) ---
        ic(f"Propagating rates for '{mode}' mode with {downtime_prob*100:.0f}% downtime...")
        
        final_adjusted_rates = propagate_downtime_rates(
            initial_rates,
            base_stations=base_stations,
            downtime_prob=downtime_prob
        )

        # --- New: Use the same suffix for the propagated rates file ---
        propagated_rates_file = os.path.join(save_folder, f'propagated_station_combination_rates_{mode}{snr_suffix}.txt')
        ic(f"Saving propagated rates to: {propagated_rates_file}")

        with open(propagated_rates_file, 'w') as f:
            # ... (writing to file is unchanged) ...
            f.write(f"# Propagated rates for Analysis Mode: {mode}\n")
            f.write(f"# Accounts for a {downtime_prob*100:.0f}% single-station downtime probability.\n")
            if snr_threshold is not None:
                f.write(f"# Event triggers are filtered with SNR >= {snr_threshold}\n")
            f.write("Station Combination, Adjusted_Event_Rate, Adjusted_Error (both in Evts/Yr)\n")

            for n_coincidence in range(len(base_stations), 0, -1):
                if n_coincidence > 1:
                    f.write(f"\n# {n_coincidence}-Fold Coincidences (Adjusted):\n")
                else:
                    f.write(f"\n# 1-Fold (Single Station) Rates (Adjusted):\n")
                
                all_possible_combos = sorted(list(itertools.combinations(base_stations, n_coincidence)))
                
                for combo_tuple in all_possible_combos:
                    rate, error = final_adjusted_rates.get(combo_tuple, (0, 0))
                    combo_str = "-".join(map(str, combo_tuple))
                    f.write(f"{combo_str} : {rate:.5e}, {error:.5e}\n")

    ic(f"\n{'='*20} All Analyses Complete {'='*20}")
    quit()

    for station_id in direct_event_rate:
        imshowRate(direct_trigger_rate_dict[station_id], f'Direct Trigger Rate for Station {station_id}', f'{save_folder}direct_trigger_rate_{station_id}.png', colorbar_label='Trigger Rate')
        imshowRate(direct_event_rate[station_id], f'Direct Event Rate for Station {station_id}', f'{save_folder}direct_event_rate_{station_id}.png', colorbar_label=f'Evts/yr, Sum {np.nansum(direct_event_rate[station_id]):.3f}')
        event_rate_error = getErrorEventRates(direct_trigger_rate_dict[station_id], HRAeventList, max_distance=max_distance)
        plotRateWithError(direct_event_rate[station_id], event_rate_error, f'{save_folder}error_rate/direct_event_rate_error_{station_id}.png', f'Direct Event Rate Error for Station {station_id}')
    for station_id in reflected_event_rate:
        imshowRate(reflected_trigger_rate_dict[station_id], f'Reflected Trigger Rate for Station {station_id}', f'{save_folder}reflected_trigger_rate_{station_id}.png', colorbar_label='Trigger Rate')
        imshowRate(reflected_event_rate[station_id], f'Reflected Event Rate for Station {station_id}', f'{save_folder}reflected_event_rate_{station_id}.png', colorbar_label=f'Evts/yr, Sum {np.nansum(reflected_event_rate[station_id]):.3f}')
        event_rate_error = getErrorEventRates(reflected_trigger_rate_dict[station_id], HRAeventList, max_distance=max_distance)
        plotRateWithError(reflected_event_rate[station_id], event_rate_error, f'{save_folder}error_rate/reflected_event_rate_error_{station_id}.png', f'Reflected Event Rate Error for Station {station_id}')

    # Combined trigger rate and event rate plots

    imshowRate(stn_100s_trigger_rate['direct'], '100s Direct Trigger Rate', f'{save_folder}100s_direct_trigger_rate.png', colorbar_label='Trigger Rate')
    imshowRate(combined_event_rate['100s_direct'], '100s Direct Event Rate', f'{save_folder}100s_direct_event_rate.png', colorbar_label=f'Evts/yr, Sum {np.nansum(combined_event_rate["100s_direct"]):.3f}')
    event_rate_error = getErrorEventRates(stn_100s_trigger_rate['direct'], HRAeventList, max_distance=max_distance, combined=True)
    plotRateWithError(combined_event_rate['100s_direct'], event_rate_error, f'{save_folder}error_rate/100s_direct_event_rate_error.png', '100s Direct Event Rate Error')

    imshowRate(stn_100s_trigger_rate['reflected'], '100s Reflected Trigger Rate', f'{save_folder}100s_reflected_trigger_rate.png', colorbar_label='Trigger Rate')
    imshowRate(combined_event_rate['100s_reflected'], '100s Reflected Event Rate', f'{save_folder}100s_reflected_event_rate.png', colorbar_label=f'Evts/yr, Sum {np.nansum(combined_event_rate["100s_reflected"]):.3f}')
    event_rate_error = getErrorEventRates(stn_100s_trigger_rate['reflected'], HRAeventList, max_distance=max_distance, combined=True)
    plotRateWithError(combined_event_rate['100s_reflected'], event_rate_error, f'{save_folder}error_rate/100s_reflected_event_rate_error.png', '100s Reflected Event Rate Error')

    imshowRate(stn_200s_trigger_rate['direct'], '200s Direct Trigger Rate', f'{save_folder}200s_direct_trigger_rate.png', colorbar_label='Trigger Rate')
    imshowRate(combined_event_rate['200s_direct'], '200s Direct Event Rate', f'{save_folder}200s_direct_event_rate.png', colorbar_label=f'Evts/yr, Sum {np.nansum(combined_event_rate["200s_direct"]):.3f}')
    event_rate_error = getErrorEventRates(stn_200s_trigger_rate['direct'], HRAeventList, max_distance=max_distance, combined=True)
    plotRateWithError(combined_event_rate['200s_direct'], event_rate_error, f'{save_folder}error_rate/200s_direct_event_rate_error.png', '200s Direct Event Rate Error')

    imshowRate(stn_200s_trigger_rate['reflected'], '200s Reflected Trigger Rate', f'{save_folder}200s_reflected_trigger_rate.png', colorbar_label='Trigger Rate')
    imshowRate(combined_event_rate['200s_reflected'], '200s Reflected Event Rate', f'{save_folder}200s_reflected_event_rate.png', colorbar_label=f'Evts/yr, Sum {np.nansum(combined_event_rate["200s_reflected"]):.3f}')
    event_rate_error = getErrorEventRates(stn_200s_trigger_rate['reflected'], HRAeventList, max_distance=max_distance, combined=True)
    plotRateWithError(combined_event_rate['200s_reflected'], event_rate_error, f'{save_folder}error_rate/200s_reflected_event_rate_error.png', '200s Reflected Event Rate Error')

    imshowRate(combined_trigger_rate['direct'], 'Combined Direct Trigger Rate', f'{save_folder}combined_direct_trigger_rate.png', colorbar_label='Trigger Rate')
    imshowRate(combined_event_rate['direct'], 'Combined Direct Event Rate', f'{save_folder}combined_direct_event_rate.png', colorbar_label=f'Evts/yr, Sum {np.nansum(combined_event_rate["direct"]):.3f}')
    event_rate_error = getErrorEventRates(combined_trigger_rate['direct'], HRAeventList, max_distance=max_distance, combined=True)
    plotRateWithError(combined_event_rate['direct'], event_rate_error, f'{save_folder}error_rate/combined_direct_event_rate_error.png', 'Combined Direct Event Rate Error')

    imshowRate(combined_trigger_rate['reflected'], 'Combined Reflected Trigger Rate', f'{save_folder}combined_reflected_trigger_rate.png', colorbar_label='Trigger Rate')
    imshowRate(combined_event_rate['reflected'], 'Combined Reflected Event Rate', f'{save_folder}combined_reflected_event_rate.png', colorbar_label=f'Evts/yr, Sum {np.nansum(combined_event_rate["reflected"]):.3f}')
    event_rate_error = getErrorEventRates(combined_trigger_rate['reflected'], HRAeventList, max_distance=max_distance, combined=True)
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
        event_rate_coincidence[i] = getEventRate(trigger_rate_coincidence[i], e_bins, z_bins, max_distance=max_distance)
        # setHRAeventListRateWeight(HRAeventList, trigger_rate_coincidence[i], weight_name=f'{i}_coincidence_wrefl', max_distance=max_distance)
        ic(event_rate_coincidence[i], np.nansum(event_rate_coincidence[i]))
        imshowRate(event_rate_coincidence[i], f'Event Rate for {i} Coincidences, w/Refl', f'{save_folder}event_rate_coincidence_{i}.png', colorbar_label=f'Evts/yr, Sum {np.nansum(event_rate_coincidence[i]):.3f}')
        event_rate_error = getErrorEventRates(trigger_rate_coincidence[i], HRAeventList, max_distance=max_distance)
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
        event_rate_coincidence[i] = getEventRate(trigger_rate_coincidence[i], e_bins, z_bins, max_distance=max_distance)
        # setHRAeventListRateWeight(HRAeventList, trigger_rate_coincidence[i], weight_name=f'{i}_coincidence_norefl', max_distance=max_distance)
        ic(event_rate_coincidence[i], np.nansum(event_rate_coincidence[i]))
        imshowRate(event_rate_coincidence[i], f'Event Rate for {i} Coincidences w/o Refl', f'{save_folder}event_rate_coincidence_norefl_{i}.png', colorbar_label=f'Evts/yr, Sum {np.nansum(event_rate_coincidence[i]):.3f}')
        event_rate_error = getErrorEventRates(trigger_rate_coincidence[i], HRAeventList, max_distance=max_distance)
        plotRateWithError(event_rate_coincidence[i], event_rate_error, f'{save_folder}error_rate/event_rate_coincidence_norefl_error_{i}.png', f'Event Rate Error for {i} Coincidences w/o Refl')


    # Coincidence with reflection and station 52 upwards LPDA
    bad_stations = [32, 132, 152]
    trigger_rate_coincidence = getCoincidencesTriggerRates(HRAeventList, bad_stations, use_secondary=False, force_stations=52)
    event_rate_coincidence = {}
    for i in trigger_rate_coincidence:
        if not np.any(trigger_rate_coincidence[i] > 0):
            ic(f'No events for {i} coincidences')
            continue
        imshowRate(trigger_rate_coincidence[i], f'Trigger Rate, {i} Coincidences, 52 upward forced w/Refl', f'{save_folder}trigger_rate_coincidence_52up_{i}.png', colorbar_label='Trigger Rate')
        event_rate_coincidence[i] = getEventRate(trigger_rate_coincidence[i], e_bins, z_bins, max_distance=max_distance)
        # setHRAeventListRateWeight(HRAeventList, trigger_rate_coincidence[i], weight_name=f'{i}_coincidence_52up_wrefl', max_distance=max_distance)
        ic(event_rate_coincidence[i], np.nansum(event_rate_coincidence[i]))
        imshowRate(event_rate_coincidence[i], f'Event Rate, {i} Coincidences, 52 upward forced w/Refl', f'{save_folder}event_rate_coincidence_52up_{i}.png', colorbar_label=f'Evts/yr, Sum {np.nansum(event_rate_coincidence[i]):.3f}')
        event_rate_error = getErrorEventRates(trigger_rate_coincidence[i], HRAeventList, max_distance=max_distance)
        plotRateWithError(event_rate_coincidence[i], event_rate_error, f'{save_folder}error_rate/event_rate_coincidence_52up_error_{i}.png', f'Event Rate Error for {i} Coincidences, 52 upward forced w/Refl')


    # Coincidence without reflection and station 52 upwards LPDA
    bad_stations = [32, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    trigger_rate_coincidence = getCoincidencesTriggerRates(HRAeventList, bad_stations, use_secondary=False, force_stations=52)
    event_rate_coincidence = {}
    for i in trigger_rate_coincidence:
        if not np.any(trigger_rate_coincidence[i] > 0):
            ic(f'No events for {i} coincidences')
            continue
        imshowRate(trigger_rate_coincidence[i], f'Trigger Rate, {i} Coincidences, 52 upward forced w/o Refl', f'{save_folder}trigger_rate_coincidence_norefl_52up_{i}.png', colorbar_label='Trigger Rate')
        event_rate_coincidence[i] = getEventRate(trigger_rate_coincidence[i], e_bins, z_bins, max_distance=max_distance)
        # setHRAeventListRateWeight(HRAeventList, trigger_rate_coincidence[i], weight_name=f'{i}_coincidence_52up_norefl', max_distance=max_distance)
        ic(event_rate_coincidence[i], np.nansum(event_rate_coincidence[i]))
        imshowRate(event_rate_coincidence[i], f'Event Rate, {i} Coincidences, 52 upward forced w/o Refl', f'{save_folder}event_rate_coincidence_norefl_52up_{i}.png', colorbar_label=f'Evts/yr, Sum {np.nansum(event_rate_coincidence[i]):.3f}')
        event_rate_error = getErrorEventRates(trigger_rate_coincidence[i], HRAeventList, max_distance=max_distance)
        plotRateWithError(event_rate_coincidence[i], event_rate_error, f'{save_folder}error_rate/event_rate_coincidence_norefl_52up_error_{i}.png', f'Event Rate Error for {i} Coincidences, 52 upward forced w/o Refl')


    # Coincidence with reflection required, no station 52
    bad_stations = [32, 52, 132, 152]
    force_stations = [113, 114, 115, 117, 118, 119, 130]   # Use all reflected stations except 52
    trigger_rate_coincidence = getCoincidencesTriggerRates(HRAeventList, bad_stations, use_secondary=False, force_stations=force_stations)
    event_rate_coincidence = {}
    for i in trigger_rate_coincidence:
        if not np.any(trigger_rate_coincidence[i] > 0):
            ic(f'No events for {i} coincidences')
            continue
        imshowRate(trigger_rate_coincidence[i], f'Trigger Rate, {i} Coincidences, Refl Required', f'{save_folder}trigger_rate_coincidence_reflReq_{i}.png', colorbar_label='Trigger Rate')
        event_rate_coincidence[i] = getEventRate(trigger_rate_coincidence[i], e_bins, z_bins, max_distance=max_distance)
        # setHRAeventListRateWeight(HRAeventList, trigger_rate_coincidence[i], weight_name=f'{i}_coincidence_refl', max_distance=max_distance)
        ic(event_rate_coincidence[i], np.nansum(event_rate_coincidence[i]))
        imshowRate(event_rate_coincidence[i], f'Event Rate, {i} Coincidences, Refl Required', f'{save_folder}event_rate_coincidence_reflReq_{i}.png', colorbar_label=f'Evts/yr, Sum {np.nansum(event_rate_coincidence[i]):.3f}')
        event_rate_error = getErrorEventRates(trigger_rate_coincidence[i], HRAeventList, max_distance=max_distance)
        plotRateWithError(event_rate_coincidence[i], event_rate_error, f'{save_folder}error_rate/event_rate_coincidence_reflReq_error_{i}.png', f'Event Rate Error for {i} Coincidences, Refl Required')


    # Coincidence with reflections only
    bad_stations = [13, 14, 15, 17, 18, 19, 30, 32, 52, 132, 152]   # Use all direct stations
    trigger_rate_coincidence = getCoincidencesTriggerRates(HRAeventList, bad_stations, use_secondary=False)
    event_rate_coincidence = {}
    for i in trigger_rate_coincidence:
        if not np.any(trigger_rate_coincidence[i] > 0):
            ic(f'No events for {i} coincidences')
            continue
        imshowRate(trigger_rate_coincidence[i], f'Trigger Rate, {i} Coincidences, Reflections Only', f'{save_folder}trigger_rate_coincidence_reflectionsOnly_{i}.png', colorbar_label='Trigger Rate')
        event_rate_coincidence[i] = getEventRate(trigger_rate_coincidence[i], e_bins, z_bins, max_distance=max_distance)
        # setHRAeventListRateWeight(HRAeventList, trigger_rate_coincidence[i], weight_name=f'{i}_coincidence_reflections', max_distance=max_distance)
        ic(event_rate_coincidence[i], np.nansum(event_rate_coincidence[i]))
        imshowRate(event_rate_coincidence[i], f'Event Rate, {i} Coincidences, Reflections Only', f'{save_folder}event_rate_coincidence_reflectionsOnly_{i}.png', colorbar_label=f'Evts/yr, Sum {np.nansum(event_rate_coincidence[i]):.3f}')
        event_rate_error = getErrorEventRates(trigger_rate_coincidence[i], HRAeventList, max_distance=max_distance)
        plotRateWithError(event_rate_coincidence[i], event_rate_error, f'{save_folder}error_rate/event_rate_coincidence_reflectionsOnly_error_{i}.png', f'Event Rate Error for {i} Coincidences, Reflections Only')

    # Plot the zenith and azimuth reconstruction
    angle_save_folder = f'{save_folder}recon_angles/'
    if not os.path.exists(angle_save_folder):
        os.makedirs(angle_save_folder)
    for station_id in direct_event_rate:
        zenith, recon_zenith, azimuth, recon_azimuth, weights = getAnglesReconWeights(HRAeventList, station_id, station_id)
        histAngleRecon(zenith, azimuth, recon_zenith, recon_azimuth, weights, f'Reconstruction Angles for Station {station_id}', f'{angle_save_folder}recon_angles_{station_id}.png')
    
    # Also plot angles for combination of all reflected and direct stations together
    zenith, recon_zenith, azimuth, recon_azimuth, weights = getAnglesReconWeights(HRAeventList, 'combined_direct', [13, 14, 15, 17, 18, 19, 30])
    histAngleRecon(zenith, azimuth, recon_zenith, recon_azimuth, weights, f'Reconstruction Angles for Combined Backlobe Stations', f'{angle_save_folder}recon_angles_combined_backlobe.png')

    zenith, recon_zenith, azimuth, recon_azimuth, weights = getAnglesReconWeights(HRAeventList, 'combined_reflected', [113, 114, 115, 117, 118, 119, 130])
    histAngleRecon(zenith, azimuth, recon_zenith, recon_azimuth, weights, f'Reconstruction Angles for Combined Reflected Stations', f'{angle_save_folder}recon_angles_combined_reflected.png')





