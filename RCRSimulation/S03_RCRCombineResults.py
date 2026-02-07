#!/usr/bin/env python3
"""Combine per-task RCR simulation results into a single file per simulation.

After running a SLURM array job, each task produces its own .npy file.
This script globs for matching files, concatenates the RCREvent arrays,
saves a combined file, and prints summary statistics.

Usage:
    python RCRSimulation/S03_RCRCombineResults.py <sim_name> [--numpy-dir DIR]

Examples:
    python RCRSimulation/S03_RCRCombineResults.py Gen2_deep_MB_576m
    python RCRSimulation/S03_RCRCombineResults.py Gen2_deep_MB_576m --numpy-dir RCRSimulation/output/02.07.26/numpy/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def find_part_files(sim_name: str, numpy_dir: Path) -> list[Path]:
    """Find all per-task numpy files for a simulation."""
    pattern = f"{sim_name}_part*_RCReventList.npy"
    files = sorted(numpy_dir.glob(pattern))

    if not files:
        # Also try without _part suffix (from init_all_simulations.sh or single runs)
        pattern_alt = f"{sim_name}*_RCReventList.npy"
        files = sorted(numpy_dir.glob(pattern_alt))

    return files


def combine_and_save(files: list[Path], output_path: Path) -> np.ndarray:
    """Load all files, concatenate events, save combined file."""
    all_events = []
    for f in files:
        events = np.load(f, allow_pickle=True)
        all_events.extend(events)
        print(f"  Loaded {len(events)} events from {f.name}")

    combined = np.array(all_events, dtype=object)
    np.save(output_path, combined, allow_pickle=True)
    return combined


def print_summary(events: np.ndarray, sim_name: str) -> None:
    """Print summary statistics for combined results."""
    print(f"\n{'='*60}")
    print(f"Combined Results: {sim_name}")
    print(f"{'='*60}")
    print(f"Total events: {len(events)}")

    if len(events) == 0:
        print("No events to summarize.")
        return

    # Trigger statistics
    all_trigger_names = set()
    direct_count = 0
    reflected_count = 0
    both_count = 0
    neither_count = 0
    energies = []
    zeniths = []

    for evt in events:
        energies.append(evt.energy)
        zeniths.append(np.rad2deg(evt.zenith))

        for trig_name in evt.all_trigger_names():
            all_trigger_names.add(trig_name)

        has_direct = any(evt.has_direct_trigger(t) for t in evt.all_trigger_names())
        has_reflected = any(evt.has_reflected_trigger(t) for t in evt.all_trigger_names())

        if has_direct and has_reflected:
            both_count += 1
        elif has_direct:
            direct_count += 1
        elif has_reflected:
            reflected_count += 1
        else:
            neither_count += 1

    total_direct = direct_count + both_count
    total_reflected = reflected_count + both_count
    total_any = direct_count + reflected_count + both_count

    print(f"\nTrigger names: {sorted(all_trigger_names)}")
    print(f"Direct only:   {direct_count}")
    print(f"Reflected only: {reflected_count}")
    print(f"Both:          {both_count}")
    print(f"Neither:       {neither_count}")
    print(f"\nDirect rate:    {total_direct}/{len(events)} = {100*total_direct/len(events):.2f}%")
    print(f"Reflected rate: {total_reflected}/{len(events)} = {100*total_reflected/len(events):.2f}%")
    print(f"Any trigger:    {total_any}/{len(events)} = {100*total_any/len(events):.2f}%")
    print(f"\nEnergy range: {min(energies):.2e} - {max(energies):.2e} eV")
    print(f"Zenith range: {min(zeniths):.1f} - {max(zeniths):.1f} deg")


def main():
    parser = argparse.ArgumentParser(description="Combine per-task RCR simulation results.")
    parser.add_argument("sim_name", help="Simulation name (e.g., Gen2_deep_MB_576m)")
    parser.add_argument(
        "--numpy-dir",
        default=None,
        help="Directory containing numpy files. If not specified, searches RCRSimulation/output/*/numpy/",
    )
    args = parser.parse_args()

    # Find the numpy directory
    if args.numpy_dir:
        numpy_dir = Path(args.numpy_dir)
    else:
        # Search for most recent output directory
        output_base = Path("RCRSimulation/output/")
        if not output_base.exists():
            print(f"Error: {output_base} does not exist")
            sys.exit(1)

        # Find directories with numpy subdirs, sorted by name (date-based)
        candidates = sorted(
            [d / "numpy" for d in output_base.iterdir() if d.is_dir() and (d / "numpy").exists()],
            reverse=True,
        )
        if not candidates:
            print(f"Error: No numpy output directories found under {output_base}")
            sys.exit(1)

        numpy_dir = candidates[0]
        print(f"Using most recent numpy dir: {numpy_dir}")

    if not numpy_dir.exists():
        print(f"Error: Directory not found: {numpy_dir}")
        sys.exit(1)

    # Find part files
    files = find_part_files(args.sim_name, numpy_dir)
    if not files:
        print(f"Error: No files matching '{args.sim_name}' found in {numpy_dir}")
        sys.exit(1)

    print(f"Found {len(files)} part files for {args.sim_name}")

    # Combine and save
    output_path = numpy_dir / f"{args.sim_name}_combined_RCReventList.npy"
    events = combine_and_save(files, output_path)

    print(f"\nSaved combined file: {output_path}")

    # Print summary
    print_summary(events, args.sim_name)


if __name__ == "__main__":
    main()
