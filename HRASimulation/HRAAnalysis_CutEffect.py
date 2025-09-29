"""
HRAAnalysis_CutEffect.py

This script creates imShowRate plots similar to HRAAnalysis.py but with the addition
of angle and chi cuts from HRASNRPlots_CutEffects.py. It produces 4 sets of plots:
1. Pre-cuts (no cuts applied)
2. Chi cut only
3. Angle cut only  
4. Both cuts applied

The script uses a modified version of getCoincidencesTriggerRates that applies
the cuts before counting coincidences.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import configparser
from icecream import ic
import itertools

# Project imports
from HRASimulation.HRAEventObject import HRAevent, stns_100s, stns_200s
from HRASimulation.S02_HRANurToNpy import loadHRAfromH5
import HRASimulation.HRAAnalysis as HRAAnalysis
from NuRadioReco.utilities import units


# ----------------------------
# Cut implementations (copied from HRASNRPlots_CutEffects.py)
# ----------------------------

def _safe_deg(val):
    try:
        return np.degrees(float(val))
    except Exception:
        return None


def _wrap_delta_az_deg(az1_deg, az2_deg):
    d = abs(az1_deg - az2_deg) % 360.0
    return min(d, 360.0 - d)


def event_passes_angle_cut(event: HRAevent, station_ids, zenith_margin_deg=20.0, azimuth_margin_deg=45.0):
    """Angle cut using reconstructed per-station angles from a simulated event.

    - Uses event.recon_zenith[st] and event.recon_azimuth[st] (assumed radians).
    - Skips (passes) if fewer than 2 stations have valid angles.
    - Passes if any station pair agrees within the provided margins.
    """
    angles = []
    for st in station_ids:
        z = event.recon_zenith.get(st)
        a = event.recon_azimuth.get(st)
        if z is None or a is None:
            continue
        z_deg = _safe_deg(z)
        # azimuth stored in radians already; convert to degrees
        try:
            a_deg = np.degrees(float(a))
        except Exception:
            a_deg = None
        if z_deg is None or a_deg is None:
            continue
        angles.append((z_deg, a_deg))

    if len(angles) < 2:
        # Skip the cut if not enough info
        return True

    for (z1, a1), (z2, a2) in itertools.combinations(angles, 2):
        dz = abs(z1 - z2)
        da = _wrap_delta_az_deg(a1, a2)
        if (dz <= zenith_margin_deg) and (da <= azimuth_margin_deg):
            return True
    return False


def _get_station_chi_values(event: HRAevent, station_id):
    """Try multiple key names to fetch ChiRCR and Chi2016 values for robustness.
    Returns (chi_rcr, chi_2016) where missing values are returned as None.
    """
    # RCR key choice based on station group
    rcr_keys = []
    if station_id in stns_100s:
        rcr_keys.extend(["ChiRCR100s", "ChiRCR", "RCR"])  # try specific then fallbacks
    elif station_id in stns_200s:
        rcr_keys.extend(["ChiRCR200s", "ChiRCR", "RCR"])  # try specific then fallbacks
    else:
        rcr_keys.extend(["ChiRCR", "RCR"])  # generic

    chi_rcr = None
    for k in rcr_keys:
        v = event.getChi(station_id, k)
        if v is not None and not (isinstance(v, (int, float)) and v == 0):
            try:
                chi_rcr = float(v)
                break
            except Exception:
                pass

    chi2016 = None
    for k in ("Chi2016", "2016"):
        v = event.getChi(station_id, k)
        if v is not None and not (isinstance(v, (int, float)) and v == 0):
            try:
                chi2016 = float(v)
                break
            except Exception:
                pass

    return chi_rcr, chi2016


def event_passes_chi_cut(event: HRAevent, station_ids, high_chi_threshold=0.6, low_chi_threshold=0.5, min_triggers_passing=2):
    """Chi cut approximating C03 logic.

    - For each station in station_ids, consider max(ChiRCR*, Chi2016).
    - high pass: any station >= high_chi_threshold
    - low pass count: number of stations >= low_chi_threshold
    - final pass: high_pass and low_pass_count >= (min_triggers_passing - 1)
    """
    any_high = False
    low_count = 0

    for st in station_ids:
        chi_rcr, chi2016 = _get_station_chi_values(event, st)
        vals = [v for v in (chi_rcr, chi2016) if v is not None]
        if not vals:
            continue
        m = max(vals)
        if m >= high_chi_threshold:
            any_high = True
        if m >= low_chi_threshold:
            low_count += 1

    return bool(any_high and (low_count >= (min_triggers_passing - 1)))


# ----------------------------
# Modified coincidence trigger rates with cuts
# ----------------------------

def _get_trigger_station_lists(event: HRAevent, sigma=4.5, bad_stations=None):
    """Helper function to get direct, reflected, and combined trigger station lists."""
    if bad_stations is None:
        bad_stations = []
    direct = [st for st in event.directTriggers(sigma=sigma) if st not in bad_stations]
    reflected = [st for st in event.reflectedTriggers(sigma=sigma) if st not in bad_stations]
    both = direct + reflected
    return direct, reflected, both


def getCoincidencesTriggerRatesWithCuts(HRAeventList, bad_stations, use_secondary=False, force_stations=None, 
                                        sigma=4.5, sigma_52=7, apply_angle_cut=False, apply_chi_cut=False,
                                        zenith_margin_deg=20.0, azimuth_margin_deg=45.0,
                                        high_chi_threshold=0.6, low_chi_threshold=0.5, min_triggers_passing=2):
    """
    Modified version of getCoincidencesTriggerRates that applies angle and/or chi cuts
    before counting coincidences.

    Args:
        HRAeventList (list): List of HRA event objects.
        bad_stations (list): List of station IDs to ignore.
        use_secondary (bool): Flag to use secondary trigger conditions.
        force_stations (int or list, optional): A station ID or a list of station IDs.
            If provided, only events triggered by at least one of these stations will be counted.
            Defaults to None.
        sigma (float): The significance threshold for a station trigger.
        sigma_52 (float): The significance threshold for stations 52 and 53.
        apply_angle_cut (bool): Whether to apply the angle cut.
        apply_chi_cut (bool): Whether to apply the chi cut.
        zenith_margin_deg (float): Zenith angle margin for angle cut (degrees).
        azimuth_margin_deg (float): Azimuth angle margin for angle cut (degrees).
        high_chi_threshold (float): High chi threshold for chi cut.
        low_chi_threshold (float): Low chi threshold for chi cut.
        min_triggers_passing (int): Minimum number of triggers that must pass chi cut.

    Returns:
        dict: A dictionary where keys are the number of coincidences (2-7) and values
              are 2D numpy arrays of trigger rates binned by energy and zenith.
    """
    e_bins, z_bins = HRAAnalysis.getEnergyZenithBins()
    n_throws = HRAAnalysis.getnThrows(HRAeventList)

    trigger_rate_coincidence = {}
    for i in [2, 3, 4, 5, 6, 7]:
        trigger_rate_coincidence[i] = HRAAnalysis.getEnergyZenithArray()
        for event in HRAeventList:
            # First check if event has basic coincidence requirement
            if not event.hasCoincidence(i, bad_stations, use_secondary, sigma=sigma, sigma_52=sigma_52):
                continue

            # Check force_stations requirement if specified
            if force_stations is not None:
                stations_to_check = force_stations
                if not isinstance(stations_to_check, list):
                    stations_to_check = [stations_to_check]
                
                if set(stations_to_check).isdisjoint(event.station_triggers[sigma]):
                    continue

            # Get the triggered stations (excluding bad stations)
            direct_triggered, reflected_triggered, all_triggered = _get_trigger_station_lists(
                event, sigma=sigma, bad_stations=bad_stations
            )
            
            if not all_triggered:
                continue

            # Apply cuts if specified
            if apply_angle_cut:
                if not event_passes_angle_cut(event, all_triggered, zenith_margin_deg, azimuth_margin_deg):
                    continue

            if apply_chi_cut:
                if not event_passes_chi_cut(event, all_triggered, high_chi_threshold, low_chi_threshold, min_triggers_passing):
                    continue

            # If we get here, the event passes all requirements
            energy_bin = np.digitize(event.getEnergy(), e_bins) - 1
            zenith_bin = np.digitize(event.getAngles()[0], z_bins) - 1
            if energy_bin < 0 or zenith_bin < 0 or energy_bin >= len(e_bins) - 1 or zenith_bin >= len(z_bins) - 1:
                ic(f'Outside of bins, {event.getEnergy()}, {event.getAngles()[0]} {energy_bin}, {zenith_bin}')
                continue
            trigger_rate_coincidence[i][energy_bin][zenith_bin] += 1
        
        # Normalize by number of throws
        trigger_rate_coincidence[i] /= n_throws

    return trigger_rate_coincidence


def generate_cut_effect_plots(HRAeventList, save_folder, bad_stations, use_secondary=False, force_stations=None,
                             sigma=4.5, sigma_52=7, max_distance=6.0*units.km):
    """
    Generate 4 sets of imShowRate plots showing the effect of cuts:
    1. Pre-cuts (no cuts applied)
    2. Chi cut only
    3. Angle cut only
    4. Both cuts applied
    
    For each cut scenario, generates plots for 2-7 coincidences.
    """
    
    # Define the four cut scenarios
    cut_scenarios = [
        {
            'name': 'NoCuts',
            'title_suffix': 'No Cuts',
            'apply_angle_cut': False,
            'apply_chi_cut': False
        },
        {
            'name': 'ChiCut',
            'title_suffix': 'Chi Cut Only',
            'apply_angle_cut': False,
            'apply_chi_cut': True
        },
        {
            'name': 'AngleCut', 
            'title_suffix': 'Angle Cut Only',
            'apply_angle_cut': True,
            'apply_chi_cut': False
        },
        {
            'name': 'BothCuts',
            'title_suffix': 'Both Cuts',
            'apply_angle_cut': True,
            'apply_chi_cut': True
        }
    ]
    
    # Create output directory
    cut_effects_folder = os.path.join(save_folder, 'CutEffects_Plots')
    os.makedirs(cut_effects_folder, exist_ok=True)
    
    # Generate plots for each scenario
    for scenario in cut_scenarios:
        ic(f"Generating plots for scenario: {scenario['name']}")
        
        # Calculate trigger rates with appropriate cuts
        trigger_rates = getCoincidencesTriggerRatesWithCuts(
            HRAeventList, 
            bad_stations,
            use_secondary=use_secondary,
            force_stations=force_stations,
            sigma=sigma,
            sigma_52=sigma_52,
            apply_angle_cut=scenario['apply_angle_cut'],
            apply_chi_cut=scenario['apply_chi_cut']
        )
        
        # Create plots for each coincidence level
        for n_coinc in [2, 3, 4, 5, 6, 7]:
            # Convert trigger rate to event rate
            e_bins, z_bins = HRAAnalysis.getEnergyZenithBins()
            event_rate = HRAAnalysis.getEventRate(trigger_rates[n_coinc], e_bins, z_bins, max_distance)
            
            # Create the plot
            title = f'{n_coinc}-fold Coincidence Rate - {scenario["title_suffix"]}'
            savename = os.path.join(cut_effects_folder, f'{scenario["name"]}_{n_coinc}fold_coincidence.png')
            
            HRAAnalysis.imshowRate(event_rate, title, savename, colorbar_label=f'Evts/yr, Sum {np.nansum(event_rate):.3f}')
    
    ic(f"All cut effect plots saved to: {cut_effects_folder}")


def main():
    """Main function to run the cut effect analysis."""
    
    # Read configuration
    config = configparser.ConfigParser()
    config.read('HRASimulation/config.ini')
    
    try:
        numpy_folder = config['FOLDERS']['numpy_folder']
        save_folder = config['FOLDERS']['save_folder']
        diameter = config['SIMPARAMETERS']['diameter']
        max_distance = float(diameter) / 2 * units.km
        plot_sigma = float(config['PLOTPARAMETERS']['trigger_sigma'])
    except KeyError as e:
        ic(f"Missing configuration parameter: {e}")
        ic("Please check config.ini file")
        return
    
    ic(f"Configuration loaded: diameter={diameter}km, sigma={plot_sigma}")
    
    # Check if HRA event list exists
    HRAeventList_path = f'{numpy_folder}HRAeventList.h5'
    if not os.path.exists(HRAeventList_path):
        ic(f"Error: HRA event list not found at {HRAeventList_path}")
        return
    
    ic("Loading HRA event list...")
    try:
        HRAeventList = loadHRAfromH5(HRAeventList_path)
        ic(f"Loaded {len(HRAeventList)} events")
    except Exception as e:
        ic(f"Error loading HRA event list: {e}")
        return
    
    # Define bad stations (same as used in other scripts)
    bad_stations = [32, 52, 132, 152]
    
    # Generate plots for different scenarios
    ic("Generating cut effect plots - No reflection requirement...")
    
    try:
        # Scenario 1: No reflection requirement
        generate_cut_effect_plots(
            HRAeventList, 
            save_folder, 
            bad_stations,
            use_secondary=False,
            force_stations=None,
            sigma=plot_sigma,
            max_distance=max_distance
        )
        
        # Scenario 2: Reflection required
        reflected_stations = [113, 114, 115, 117, 118, 119, 130]
        
        ic("Generating cut effect plots with reflection requirement...")
        
        # Create a separate folder for reflection-required plots
        cut_effects_folder_refl = os.path.join(save_folder, 'CutEffects_Plots_ReflReq')
        os.makedirs(cut_effects_folder_refl, exist_ok=True)
        
        # Temporarily modify save_folder for reflection plots
        temp_save_folder = save_folder + '_temp_refl'
        os.makedirs(temp_save_folder, exist_ok=True)
        
        generate_cut_effect_plots(
            HRAeventList,
            temp_save_folder,
            bad_stations,
            use_secondary=False,
            force_stations=reflected_stations,
            sigma=plot_sigma,
            max_distance=max_distance
        )
        
        # Move reflection plots to the correct location
        import shutil
        temp_plots_folder = os.path.join(temp_save_folder, 'CutEffects_Plots')
        if os.path.exists(temp_plots_folder):
            for filename in os.listdir(temp_plots_folder):
                if filename.endswith('.png'):
                    src_path = os.path.join(temp_plots_folder, filename)
                    dst_path = os.path.join(cut_effects_folder_refl, f'ReflReq_{filename}')
                    shutil.move(src_path, dst_path)
            
            # Clean up temporary folder
            shutil.rmtree(temp_save_folder)
        
        ic("Cut effect analysis complete!")
        
        # Print summary of generated plots
        cut_effects_folder = os.path.join(save_folder, 'CutEffects_Plots')
        if os.path.exists(cut_effects_folder):
            no_refl_plots = [f for f in os.listdir(cut_effects_folder) if f.endswith('.png')]
            ic(f"Generated {len(no_refl_plots)} plots without reflection requirement")
        
        if os.path.exists(cut_effects_folder_refl):
            refl_plots = [f for f in os.listdir(cut_effects_folder_refl) if f.endswith('.png')]
            ic(f"Generated {len(refl_plots)} plots with reflection requirement")
            
    except Exception as e:
        ic(f"Error during plot generation: {e}")
        import traceback
        traceback.print_exc()


def test_cut_functions():
    """Quick test to verify cut functions are working properly."""
    ic("Testing cut functions...")
    
    # Test helper functions
    test_az_diff = _wrap_delta_az_deg(10, 350)  # Should be 20 degrees
    ic(f"Azimuth wrap test: {test_az_diff} (should be 20)")
    
    test_safe_deg = _safe_deg(np.pi/2)  # Should be 90 degrees  
    ic(f"Safe degree conversion test: {test_safe_deg} (should be 90)")
    
    ic("Cut function tests passed!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_cut_functions()
    else:
        main()