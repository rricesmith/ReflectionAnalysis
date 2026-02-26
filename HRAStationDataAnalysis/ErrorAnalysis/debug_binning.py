"""
Debug script: diagnose why BL (direct) sim events show zero weighted rate
after bin-by-bin weight assignment from S04 event rates.

Run from ReflectionAnalysis/ on the HPC:
    python HRAStationDataAnalysis/ErrorAnalysis/debug_binning.py

Checks (in order):
  1. Unit values (units.eV, units.rad) — are they 1 or conversion factors?
  2. getEnergyZenithBins() actual bin edges
  3. HRA event raw energy/zenith values + which bins they digitize into
  4. S04 RCR event energies for comparison
  5. S04 direct trigger rate + event rate per bin
  6. Overlap: bins with S04 rate > 0 vs bins with HRA events > 0
  7. Simulated weight assignment result
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import configparser
import h5py
import pickle
from icecream import ic

from NuRadioReco.utilities import units
from HRASimulation.HRAEventObject import HRAevent
from RCRSimulation.RCRAnalysis import (
    getEnergyZenithBins, getEventRate, getBinnedTriggerRate,
    get_all_r_triggers, getnThrows,
)
from RCRSimulation.S04_RCRChapter4Plots import load_combined_events, MB_REFLECTED_SIMS


def print_2d(array, log_e_centers, zen_centers_deg, fmt="{:10.4f}"):
    """Print a 2D (n_e, n_z) array as a labeled grid."""
    header = f"  {'':>12} " + "".join(f"zen{z:5.1f}d " for z in zen_centers_deg)
    print(header)
    for ie in range(array.shape[0]):
        row = f"  E={log_e_centers[ie]:5.2f}  "
        for jz in range(array.shape[1]):
            row += f"  {fmt.format(array[ie, jz])} "
        print(row)


# =========================================================================
# SECTION 0: Unit check
# =========================================================================
print("=" * 80)
print("  SECTION 0: Unit Values")
print("=" * 80)
print(f"  units.eV  = {units.eV}")
print(f"  units.rad = {units.rad}")
print(f"  units.deg = {units.deg}")
print(f"  units.km  = {units.km}")
print()

# =========================================================================
# SECTION 1: Bin edges from getEnergyZenithBins()
# =========================================================================
print("=" * 80)
print("  SECTION 1: Energy-Zenith Bin Edges")
print("=" * 80)

e_bins, z_bins = getEnergyZenithBins()
n_e = len(e_bins) - 1
n_z = len(z_bins) - 1

print(f"\n  Energy bin edges (raw): {e_bins}")
print(f"  Energy bin edges / units.eV: {e_bins / units.eV}")
print(f"  log10(E/eV) edges: {np.log10(e_bins / units.eV)}")
print(f"  -> {n_e} energy bins")

print(f"\n  Zenith bin edges (raw): {z_bins}")
print(f"  Zenith bin edges / units.rad: {z_bins / units.rad}")
print(f"  Zenith bin edges in degrees: {np.rad2deg(z_bins / units.rad)}")
print(f"  -> {n_z} zenith bins")

log_e_centers = (np.log10(e_bins[:-1] / units.eV) + np.log10(e_bins[1:] / units.eV)) / 2
zen_centers_deg = np.rad2deg((z_bins[:-1] + z_bins[1:]) / 2 / units.rad)

# =========================================================================
# SECTION 2: HRA Simulation Events
# =========================================================================
print("\n" + "=" * 80)
print("  SECTION 2: HRA Simulation Events (from HRAeventList.h5)")
print("=" * 80)

config = configparser.ConfigParser()
config.read('HRAStationDataAnalysis/config.ini')
sim_file = config['SIMULATION']['sim_file']
sigma = float(config['SIMULATION']['sigma'])
direct_weight_name = config['SIMULATION']['direct_weight_name']
reflected_weight_name = config['SIMULATION']['reflected_weight_name']

print(f"  Loading: {sim_file}")
HRAeventList = []
with h5py.File(sim_file, 'r') as hf:
    for key in hf.keys():
        obj_bytes = hf[key][0]
        event = pickle.loads(obj_bytes)
        HRAeventList.append(event)
print(f"  Loaded {len(HRAeventList)} HRA events")

print(f"\n  First 10 HRA events (raw energy and zenith):")
for i, evt in enumerate(HRAeventList[:10]):
    e = evt.energy
    z = evt.zenith
    # Try digitizing this one event to show what happens
    e_idx = np.digitize(e, e_bins) - 1
    z_idx = np.digitize(z, z_bins) - 1
    print(f"    [{i:3d}] energy={e:.4e}  log10={np.log10(e) if e > 0 else 'N/A':>8}  "
          f"e_bin_idx={e_idx}  |  "
          f"zenith={z:.6f} rad ({np.rad2deg(z):.2f} deg)  z_bin_idx={z_idx}")

# Also try with units.eV multiplier to see if that changes things
print(f"\n  Same events digitized with energy * units.eV (in case HRA stores raw floats):")
for i, evt in enumerate(HRAeventList[:5]):
    e_raw = evt.energy
    e_scaled = e_raw * units.eV
    e_idx_raw = np.digitize(e_raw, e_bins) - 1
    e_idx_scaled = np.digitize(e_scaled, e_bins) - 1
    print(f"    [{i}] raw={e_raw:.4e} -> bin {e_idx_raw}  |  "
          f"*units.eV={e_scaled:.4e} -> bin {e_idx_scaled}")

# =========================================================================
# SECTION 3: Extract HRA direct/reflected triggers with energy/zenith
# =========================================================================
print("\n" + "=" * 80)
print("  SECTION 3: HRA Direct & Reflected Trigger Extraction")
print("=" * 80)

direct_stations = [13, 14, 15, 17, 18, 19, 30]
reflected_stations = [113, 114, 115, 117, 118, 119, 130]

direct_energies, direct_zeniths, direct_weights_raw = [], [], []
reflected_energies, reflected_zeniths, reflected_weights_raw = [], [], []

for event in HRAeventList:
    dw = event.getWeight(direct_weight_name, primary=True, sigma=sigma)
    if not np.isnan(dw) and dw > 0:
        triggered = [s for s in direct_stations if event.hasTriggered(s, sigma)]
        if triggered:
            sw = dw / len(triggered)
            for _ in triggered:
                direct_energies.append(event.energy)
                direct_zeniths.append(event.zenith)
                direct_weights_raw.append(sw)

    rw = event.getWeight(reflected_weight_name, primary=True, sigma=sigma)
    if not np.isnan(rw) and rw > 0:
        triggered = [s for s in reflected_stations if event.hasTriggered(s, sigma)]
        if triggered:
            sw = rw / len(triggered)
            for _ in triggered:
                reflected_energies.append(event.energy)
                reflected_zeniths.append(event.zenith)
                reflected_weights_raw.append(sw)

direct_energies = np.array(direct_energies)
direct_zeniths = np.array(direct_zeniths)
direct_weights_raw = np.array(direct_weights_raw)
reflected_energies = np.array(reflected_energies)
reflected_zeniths = np.array(reflected_zeniths)
reflected_weights_raw = np.array(reflected_weights_raw)

print(f"  Direct triggers:    {len(direct_energies)} entries, raw weight sum={np.sum(direct_weights_raw):.4f}")
print(f"  Reflected triggers: {len(reflected_energies)} entries, raw weight sum={np.sum(reflected_weights_raw):.4f}")

for label, energies, zeniths in [("Direct", direct_energies, direct_zeniths),
                                  ("Reflected", reflected_energies, reflected_zeniths)]:
    if len(energies) == 0:
        print(f"\n  {label}: NO EVENTS — this alone explains zero rate!")
        continue

    print(f"\n  {label} energy range: {energies.min():.4e} to {energies.max():.4e} eV "
          f"(log10: {np.log10(energies.min()):.2f} to {np.log10(energies.max()):.2f})")
    print(f"  {label} zenith range: {zeniths.min():.4f} to {zeniths.max():.4f} rad "
          f"({np.rad2deg(zeniths.min()):.1f} to {np.rad2deg(zeniths.max()):.1f} deg)")

    e_idx = np.digitize(energies, e_bins) - 1
    z_idx = np.digitize(zeniths, z_bins) - 1
    in_range = (e_idx >= 0) & (e_idx < n_e) & (z_idx >= 0) & (z_idx < n_z)
    print(f"  {label} events in valid bin range: {np.sum(in_range)} / {len(energies)}")
    if not np.all(in_range):
        out_e = (e_idx < 0) | (e_idx >= n_e)
        out_z = (z_idx < 0) | (z_idx >= n_z)
        print(f"    Out of energy range: {np.sum(out_e)} (idx range: {e_idx.min()} to {e_idx.max()})")
        print(f"    Out of zenith range: {np.sum(out_z)} (idx range: {z_idx.min()} to {z_idx.max()})")

    counts = np.zeros((n_e, n_z), dtype=int)
    for k in range(len(energies)):
        ei, zi = int(e_idx[k]), int(z_idx[k])
        if 0 <= ei < n_e and 0 <= zi < n_z:
            counts[ei, zi] += 1

    print(f"\n  {label} HRA event counts per (E, zen) bin:")
    print_2d(counts, log_e_centers, zen_centers_deg, fmt="{:6.0f}")
    print(f"  Total in bins: {counts.sum()}")


# =========================================================================
# SECTION 4: S04 RCR Simulation — Event Rates
# =========================================================================
print("\n" + "=" * 80)
print("  SECTION 4: S04 RCR Simulation (HRA_MB_576m) Event Rates")
print("=" * 80)

rcr_config = configparser.ConfigParser()
rcr_config.read('RCRSimulation/config.ini')
numpy_folder = rcr_config.get("FOLDERS", "numpy_folder",
                               fallback="RCRSimulation/output/numpy")
max_distance = float(rcr_config.get("SIMULATION", "distance_km",
                                     fallback="5")) / 2 * units.km

sim_name = MB_REFLECTED_SIMS["HRA"]
print(f"  Loading: {sim_name} from {numpy_folder}")
rcr_events = load_combined_events(numpy_folder, sim_name)

if rcr_events is None:
    print("  ERROR: Could not load RCR events! Check numpy_folder path.")
    sys.exit(1)

rcr_list = list(rcr_events)
print(f"  Loaded {len(rcr_list)} RCR events")

print(f"\n  First 5 RCR events:")
for i, evt in enumerate(rcr_list[:5]):
    print(f"    [{i}] energy={evt.get_energy():.4e}  zenith={evt.zenith:.6f} rad")

# Triggers
r_triggers = get_all_r_triggers(rcr_list)
print(f"\n  R triggers found: {sorted(r_triggers.keys())}")
for r_val, trig_name in sorted(r_triggers.items()):
    print(f"    R={r_val}: {trig_name}")

# Use first R trigger
first_r = sorted(r_triggers.keys())[0]
trig = r_triggers[first_r]
print(f"\n  Computing rates with trigger: {trig}")

dir_rate, ref_rate, n_throws = getBinnedTriggerRate(rcr_list, trig)

print(f"\n  n_throws per bin:")
print_2d(n_throws, log_e_centers, zen_centers_deg, fmt="{:6.0f}")

print(f"\n  Direct trigger rate per bin:")
print_2d(dir_rate, log_e_centers, zen_centers_deg, fmt="{:8.5f}")
print(f"  Any non-zero direct trigger rate bins? {np.any(dir_rate > 0)}")

print(f"\n  Reflected trigger rate per bin:")
print_2d(ref_rate, log_e_centers, zen_centers_deg, fmt="{:8.5f}")

# Convert to event rates
direct_event_rate = getEventRate(dir_rate, e_bins, z_bins, max_distance)
reflected_event_rate = getEventRate(ref_rate, e_bins, z_bins, max_distance)

print(f"\n  Direct event rate (evts/station/yr) per bin:")
print_2d(direct_event_rate, log_e_centers, zen_centers_deg, fmt="{:10.4f}")
print(f"  TOTAL direct: {np.sum(direct_event_rate):.4f} evts/station/yr")

print(f"\n  Reflected event rate (evts/station/yr) per bin:")
print_2d(reflected_event_rate, log_e_centers, zen_centers_deg, fmt="{:10.4f}")
print(f"  TOTAL reflected: {np.sum(reflected_event_rate):.4f} evts/station/yr")


# =========================================================================
# SECTION 5: Overlap Diagnostic — the critical check
# =========================================================================
print("\n" + "=" * 80)
print("  SECTION 5: Overlap Diagnostic (S04 rate bins vs HRA event bins)")
print("=" * 80)

livetime = 11.7

for label, energies, zeniths in [("DIRECT (BL)", direct_energies, direct_zeniths),
                                  ("REFLECTED (RCR)", reflected_energies, reflected_zeniths)]:
    if label.startswith("DIRECT"):
        rate = direct_event_rate
    else:
        rate = reflected_event_rate

    events_per_bin = rate * livetime

    if len(energies) == 0:
        print(f"\n  {label}: No HRA events to assign weights to!")
        continue

    # Build HRA counts in bins
    e_idx = np.digitize(energies, e_bins) - 1
    z_idx = np.digitize(zeniths, z_bins) - 1
    hra_counts = np.zeros((n_e, n_z), dtype=int)
    for k in range(len(energies)):
        ei, zi = int(e_idx[k]), int(z_idx[k])
        if 0 <= ei < n_e and 0 <= zi < n_z:
            hra_counts[ei, zi] += 1

    print(f"\n  --- {label} ---")
    print(f"\n  S04 expected events per bin (rate * {livetime} yr):")
    print_2d(events_per_bin, log_e_centers, zen_centers_deg, fmt="{:10.4f}")
    print(f"  Total expected: {np.sum(events_per_bin):.4f}")

    # Mismatch analysis
    n_rate_gt0_hra_eq0 = 0
    n_hra_gt0_rate_eq0 = 0
    missed_events = 0
    orphan_hra = 0

    print(f"\n  Bins with S04 rate > 0 but NO HRA events (lost expected events):")
    for ie in range(n_e):
        for jz in range(n_z):
            if events_per_bin[ie, jz] > 0 and hra_counts[ie, jz] == 0:
                print(f"    E={log_e_centers[ie]:5.2f}, zen={zen_centers_deg[jz]:5.1f}d: "
                      f"expected={events_per_bin[ie, jz]:.4f}, HRA=0")
                n_rate_gt0_hra_eq0 += 1
                missed_events += events_per_bin[ie, jz]

    print(f"\n  Bins with HRA events > 0 but S04 rate = 0 (HRA events get weight=0):")
    for ie in range(n_e):
        for jz in range(n_z):
            if hra_counts[ie, jz] > 0 and events_per_bin[ie, jz] == 0:
                print(f"    E={log_e_centers[ie]:5.2f}, zen={zen_centers_deg[jz]:5.1f}d: "
                      f"HRA={hra_counts[ie, jz]}, expected=0 -> weight=0!")
                n_hra_gt0_rate_eq0 += 1
                orphan_hra += hra_counts[ie, jz]

    print(f"\n  Bins with BOTH S04 rate > 0 AND HRA events > 0 (good overlap):")
    good_bins = 0
    for ie in range(n_e):
        for jz in range(n_z):
            if hra_counts[ie, jz] > 0 and events_per_bin[ie, jz] > 0:
                w = events_per_bin[ie, jz] / hra_counts[ie, jz]
                print(f"    E={log_e_centers[ie]:5.2f}, zen={zen_centers_deg[jz]:5.1f}d: "
                      f"expected={events_per_bin[ie, jz]:.4f}, HRA={hra_counts[ie, jz]}, "
                      f"weight/event={w:.6f}")
                good_bins += 1

    print(f"\n  SUMMARY for {label}:")
    print(f"    Rate>0 but no HRA events: {n_rate_gt0_hra_eq0} bins, {missed_events:.4f} lost events")
    print(f"    HRA events but rate=0:    {n_hra_gt0_rate_eq0} bins, {orphan_hra} orphan HRA events")
    print(f"    Good overlap bins:        {good_bins}")

    # Simulate weight assignment
    weights = np.zeros(len(energies))
    for k in range(len(energies)):
        ei = int(np.digitize(energies[k], e_bins) - 1)
        zi = int(np.digitize(zeniths[k], z_bins) - 1)
        if 0 <= ei < n_e and 0 <= zi < n_z and hra_counts[ei, zi] > 0:
            weights[k] = events_per_bin[ei, zi] / hra_counts[ei, zi]

    n_nonzero = np.sum(weights > 0)
    print(f"    Resulting weights: {n_nonzero}/{len(weights)} non-zero, sum={np.sum(weights):.4f}")
    if n_nonzero > 0:
        print(f"    Weight range (non-zero): {weights[weights>0].min():.6f} to {weights[weights>0].max():.6f}")
    else:
        print(f"    *** ALL WEIGHTS ARE ZERO — this is the bug ***")


print("\n\n" + "=" * 80)
print("  DEBUG COMPLETE")
print("=" * 80)
