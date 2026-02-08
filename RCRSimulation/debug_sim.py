#!/usr/bin/env python3
"""Debug script to diagnose RCR simulation file loading and trigger issues.

Run from the ReflectiveAnalysis directory (same as the main simulation):
    python RCRSimulation/debug_sim.py [--file-range 600 610] [--site MB]

This script checks:
1. Whether CoREAS files exist at the expected paths for a given range
2. What energies/zeniths those files contain
3. Whether the phased array trigger fires on a sample event (with verbose output)
"""
from __future__ import annotations

import argparse
import os
import sys
import h5py
import numpy as np

# Add paths for NuRadioReco
sys.path.insert(0, os.getcwd())

from NuRadioReco.utilities import units


def check_files(site: str, min_file: int, max_file: int) -> list:
    """Check which files exist and print their energies."""
    print(f"\n{'='*60}")
    print(f"FILE CHECK: site={site}, range=[{min_file}, {max_file})")
    print(f"{'='*60}")

    if site.upper() == "MB":
        base = "../MBFootprints"
        fmt = "00{:04d}.hdf5"
    elif site.upper() in ("SP", "GL"):
        base = f"../{site}Footprints"
        fmt = "00{:04d}.hdf5"
    else:
        print(f"Unknown site: {site}")
        return []

    print(f"Base directory: {os.path.abspath(base)}")
    print(f"Directory exists: {os.path.isdir(base)}")

    if os.path.isdir(base):
        all_files = sorted(os.listdir(base))
        print(f"Total files in directory: {len(all_files)}")
        if all_files:
            print(f"First 5 files: {all_files[:5]}")
            print(f"Last 5 files:  {all_files[-5:]}")

    found = []
    missing = []
    energies = []
    zeniths = []

    for i in range(min_file, max_file):
        filepath = os.path.join(base, fmt.format(i))
        if os.path.exists(filepath):
            found.append(filepath)
            try:
                with h5py.File(filepath, "r") as f:
                    erange = f['inputs'].attrs["ERANGE"]
                    thetap = f['inputs'].attrs["THETAP"]
                    energy_eV = erange[0] * 1e9  # GeV -> eV
                    energies.append(energy_eV)
                    zeniths.append(thetap[0])
            except Exception as e:
                print(f"  ERROR reading {filepath}: {e}")
                energies.append(None)
                zeniths.append(None)
        else:
            missing.append(filepath)

    print(f"\nFiles found: {len(found)} / {max_file - min_file}")
    print(f"Files missing: {len(missing)}")

    if missing and len(missing) <= 20:
        for m in missing:
            print(f"  MISSING: {m}")
    elif missing:
        print(f"  First missing: {missing[0]}")
        print(f"  Last missing:  {missing[-1]}")

    valid_energies = [e for e in energies if e is not None]
    valid_zeniths = [z for z in zeniths if z is not None]

    if valid_energies:
        print(f"\nEnergy range: {min(valid_energies):.2e} - {max(valid_energies):.2e} eV")
        print(f"  log10(E/eV): {np.log10(min(valid_energies)):.2f} - {np.log10(max(valid_energies)):.2f}")
        # Print per-file energies for small ranges
        if len(found) <= 20:
            print("\nPer-file details:")
            for i, (fp, e, z) in enumerate(zip(found, energies, zeniths)):
                if e is not None:
                    print(f"  [{min_file + i:4d}] E={e:.2e} eV (log={np.log10(e):.2f}), theta={z:.1f} deg  -- {os.path.basename(fp)}")
    else:
        print("\nNo valid energies found!")

    if valid_zeniths:
        print(f"\nZenith range: {min(valid_zeniths):.1f} - {max(valid_zeniths):.1f} deg")

    return found


def check_energy_distribution(site: str, max_file: int, sample_every: int = 50):
    """Sample files across the full range to see energy distribution."""
    print(f"\n{'='*60}")
    print(f"ENERGY DISTRIBUTION (sampling every {sample_every} files)")
    print(f"{'='*60}")

    if site.upper() == "MB":
        base = "../MBFootprints"
        fmt = "00{:04d}.hdf5"
    else:
        print("Only MB supported for distribution check")
        return

    print(f"{'Index':>6} | {'Energy (eV)':>14} | {'log10(E)':>9} | {'Zenith':>8}")
    print("-" * 50)

    for i in range(0, max_file, sample_every):
        filepath = os.path.join(base, fmt.format(i))
        if os.path.exists(filepath):
            try:
                with h5py.File(filepath, "r") as f:
                    erange = f['inputs'].attrs["ERANGE"]
                    thetap = f['inputs'].attrs["THETAP"]
                    energy_eV = erange[0] * 1e9
                    print(f"{i:6d} | {energy_eV:14.2e} | {np.log10(energy_eV):9.2f} | {thetap[0]:8.1f}")
            except Exception as e:
                print(f"{i:6d} | ERROR: {e}")
        else:
            print(f"{i:6d} | FILE NOT FOUND")


def check_trigger_params():
    """Print the current trigger parameter values that would be used."""
    print(f"\n{'='*60}")
    print("TRIGGER PARAMETERS (from S01_RCRSim.py constants)")
    print(f"{'='*60}")

    # Import from S01 to check actual values
    try:
        # These are the constants defined in S01_RCRSim.py
        from RCRSimulation.S01_RCRSim import (
            PA_N_BEAMS_8CH, PA_N_BEAMS_4CH,
            PA_UPSAMPLING_8CH, PA_UPSAMPLING_4CH,
            PA_WINDOW_NS, PA_STEP_NS,
            PA_THRESHOLDS_8CH, PA_THRESHOLDS_4CH,
            PA_MAIN_LOW_ANGLE, PA_MAIN_HIGH_ANGLE,
            PA_REF_INDEX,
        )
        print(f"PA_N_BEAMS_8CH = {PA_N_BEAMS_8CH}")
        print(f"PA_N_BEAMS_4CH = {PA_N_BEAMS_4CH}")
        print(f"PA_UPSAMPLING_8CH = {PA_UPSAMPLING_8CH}")
        print(f"PA_UPSAMPLING_4CH = {PA_UPSAMPLING_4CH}")
        print(f"PA_WINDOW_NS = {PA_WINDOW_NS}")
        print(f"PA_STEP_NS = {PA_STEP_NS}")
        print(f"PA_THRESHOLDS_8CH = {PA_THRESHOLDS_8CH}")
        print(f"PA_THRESHOLDS_4CH = {PA_THRESHOLDS_4CH}")
        print(f"PA_MAIN_LOW_ANGLE = {PA_MAIN_LOW_ANGLE / units.deg:.2f} deg")
        print(f"PA_MAIN_HIGH_ANGLE = {PA_MAIN_HIGH_ANGLE / units.deg:.2f} deg")
        print(f"PA_REF_INDEX = {PA_REF_INDEX}")

        # Calculate what the actual trigger threshold would be for 100 Hz rate
        # threshold = PA_THRESHOLDS_8CH[100] * pa_vrms^2
        print(f"\nFor 100 Hz noise rate:")
        print(f"  8ch threshold factor: {PA_THRESHOLDS_8CH[100]}")
        print(f"  4ch threshold factor: {PA_THRESHOLDS_4CH[100]}")

        # Show what window/step would be for typical sampling rates
        for sr_ghz in [0.5, 1.0, 2.0, 3.2]:
            sr = sr_ghz * 1e9  # Hz
            window_8 = int(PA_WINDOW_NS * 1e-9 * sr * PA_UPSAMPLING_8CH)
            step_8 = int(PA_STEP_NS * 1e-9 * sr * PA_UPSAMPLING_8CH)
            window_4 = int(PA_WINDOW_NS * 1e-9 * sr * PA_UPSAMPLING_4CH)
            step_4 = int(PA_STEP_NS * 1e-9 * sr * PA_UPSAMPLING_4CH)
            print(f"\n  ADC rate = {sr_ghz} GHz:")
            print(f"    8ch: window={window_8} samples, step={step_8} samples")
            print(f"    4ch: window={window_4} samples, step={step_4} samples")

    except ImportError as e:
        print(f"Could not import S01_RCRSim constants: {e}")
        print("Make sure you're running from the ReflectiveAnalysis directory")


def check_detector_config(config_path: str):
    """Read detector config and show trigger ADC sampling frequency."""
    print(f"\n{'='*60}")
    print(f"DETECTOR CONFIG: {config_path}")
    print(f"{'='*60}")

    import json
    try:
        with open(config_path) as f:
            config = json.load(f)

        # Look for trigger_adc_sampling_frequency in channels
        if "channels" in config:
            for ch_id, ch_data in config["channels"].items():
                if "trigger_adc_sampling_frequency" in ch_data:
                    print(f"  Channel {ch_id}: trigger_adc_sampling_frequency = {ch_data['trigger_adc_sampling_frequency']}")
        elif isinstance(config, list):
            for station in config:
                if "channels" in station:
                    print(f"  Station {station.get('station_id', '?')}:")
                    for ch_id, ch_data in station["channels"].items():
                        if "trigger_adc_sampling_frequency" in ch_data:
                            print(f"    Channel {ch_id}: trigger_adc_sampling_frequency = {ch_data['trigger_adc_sampling_frequency']}")
    except Exception as e:
        print(f"  Error reading config: {e}")

    # Also try loading via NuRadioReco detector
    try:
        import datetime
        from NuRadioReco.detector import detector
        det = detector.Detector(config_path, "json", antenna_by_depth=False)
        det.update(datetime.datetime(2018, 10, 1))
        for sid in sorted(det.get_station_ids()):
            print(f"\n  Station {sid}:")
            for ch_id in sorted(det.get_channel_ids(sid)):
                ch = det.get_channel(sid, ch_id)
                adc_freq = ch.get("trigger_adc_sampling_frequency", "NOT SET")
                print(f"    Channel {ch_id}: trigger_adc_sampling_frequency = {adc_freq}")
                # Only show first few channels
                if ch_id >= 5:
                    remaining = len(det.get_channel_ids(sid)) - ch_id - 1
                    if remaining > 0:
                        print(f"    ... ({remaining} more channels)")
                    break
    except Exception as e:
        print(f"  Could not load detector via NuRadioReco: {e}")


def main():
    parser = argparse.ArgumentParser(description="Debug RCR simulation file loading and trigger")
    parser.add_argument("--site", default="MB", help="Site name (MB, SP)")
    parser.add_argument("--file-range", nargs=2, type=int, default=[0, 20],
                        metavar=("MIN", "MAX"), help="File range to check (default: 0 20)")
    parser.add_argument("--max-file", type=int, default=1000,
                        help="Max file index for energy distribution check")
    parser.add_argument("--sample-every", type=int, default=50,
                        help="Sample interval for energy distribution (default: 50)")
    parser.add_argument("--detector-config", type=str, default=None,
                        help="Path to detector JSON config to inspect")
    parser.add_argument("--skip-trigger", action="store_true",
                        help="Skip trigger parameter check (avoids importing S01)")
    args = parser.parse_args()

    # 1. Check specific file range
    check_files(args.site, args.file_range[0], args.file_range[1])

    # 2. Sample energy distribution across full range
    check_energy_distribution(args.site, args.max_file, args.sample_every)

    # 3. Check trigger parameters
    if not args.skip_trigger:
        check_trigger_params()

    # 4. Check detector config if provided
    if args.detector_config:
        check_detector_config(args.detector_config)
    else:
        # Check the standard configs
        configs = [
            "RCRSimulation/configurations/MB/Gen2_deep_576m_combined.json",
            "RCRSimulation/configurations/MB/Gen2_shallow_576m_combined.json",
            "RCRSimulation/configurations/MB/HRA_shallow_576m_combined.json",
        ]
        for cfg in configs:
            if os.path.exists(cfg):
                check_detector_config(cfg)


if __name__ == "__main__":
    main()
