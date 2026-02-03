#!/usr/bin/env python3
"""Quick analysis script to verify RCR simulation test results.

Usage:
    python RCRSimulation/check_test_results.py [numpy_file]

If no file is specified, looks for the most recent test output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np


def find_latest_test_file() -> Path:
    """Find the most recent test numpy file."""
    test_dir = Path("RCRSimulation/output/test/numpy/")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test output directory not found: {test_dir}")

    npy_files = list(test_dir.glob("*_RCReventList.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No numpy files found in {test_dir}")

    return max(npy_files, key=lambda p: p.stat().st_mtime)


def analyze_rcr_events(npy_path: Path) -> None:
    """Analyze RCREvent list from numpy file."""
    print(f"Loading: {npy_path}")
    events = np.load(npy_path, allow_pickle=True)

    print(f"\n{'='*60}")
    print(f"RCR Simulation Results Summary")
    print(f"{'='*60}")
    print(f"Total events: {len(events)}")

    if len(events) == 0:
        print("No events to analyze!")
        return

    # Get first event to understand structure
    first_event = events[0]
    print(f"Event type: {type(first_event).__name__}")

    # Collect statistics
    all_trigger_names = set()
    direct_triggered_count = 0
    reflected_triggered_count = 0
    both_triggered_count = 0
    neither_triggered_count = 0

    energies = []
    zeniths = []

    for evt in events:
        energies.append(evt.energy)
        zeniths.append(np.rad2deg(evt.zenith))

        # Get all trigger names from this event
        for trig_name in evt.all_trigger_names():
            all_trigger_names.add(trig_name)

        # Count trigger types for first trigger name found
        has_direct = False
        has_reflected = False

        for trig_name in evt.all_trigger_names():
            if evt.has_direct_trigger(trig_name):
                has_direct = True
            if evt.has_reflected_trigger(trig_name):
                has_reflected = True

        if has_direct and has_reflected:
            both_triggered_count += 1
        elif has_direct:
            direct_triggered_count += 1
        elif has_reflected:
            reflected_triggered_count += 1
        else:
            neither_triggered_count += 1

    print(f"\n--- Trigger Summary ---")
    print(f"Trigger names found: {sorted(all_trigger_names)}")
    print(f"Direct only triggers: {direct_triggered_count}")
    print(f"Reflected only triggers: {reflected_triggered_count}")
    print(f"Both triggered: {both_triggered_count}")
    print(f"Neither triggered: {neither_triggered_count}")

    total_direct = direct_triggered_count + both_triggered_count
    total_reflected = reflected_triggered_count + both_triggered_count

    print(f"\n--- Rates ---")
    print(f"Direct trigger rate: {total_direct}/{len(events)} = {100*total_direct/len(events):.2f}%")
    print(f"Reflected trigger rate: {total_reflected}/{len(events)} = {100*total_reflected/len(events):.2f}%")
    print(f"Combined trigger rate: {(direct_triggered_count + reflected_triggered_count + both_triggered_count)}/{len(events)} = {100*(direct_triggered_count + reflected_triggered_count + both_triggered_count)/len(events):.2f}%")

    print(f"\n--- Event Properties ---")
    print(f"Energy range: {min(energies):.2e} - {max(energies):.2e} eV")
    print(f"Zenith range: {min(zeniths):.1f} - {max(zeniths):.1f} deg")

    # Show first few events as examples
    print(f"\n--- Sample Events ---")
    for i, evt in enumerate(events[:5]):
        print(f"  {evt}")


def main():
    if len(sys.argv) > 1:
        npy_path = Path(sys.argv[1])
    else:
        try:
            npy_path = find_latest_test_file()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Usage: python check_test_results.py <numpy_file>")
            sys.exit(1)

    if not npy_path.exists():
        print(f"Error: File not found: {npy_path}")
        sys.exit(1)

    analyze_rcr_events(npy_path)


if __name__ == "__main__":
    main()
