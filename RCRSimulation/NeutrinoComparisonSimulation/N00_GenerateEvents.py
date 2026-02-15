"""Generate neutrino event lists for the comparison simulation.

Creates HDF5 input files for NuRadioMC, split by energy bin and site.
Each energy bin uses a single energy value (monochromatic) to allow
fine-grained control over statistics.

Usage:
    python N00_GenerateEvents.py [--site MB|SP|both] [--test]

    --site: Which site to generate for (default: both)
    --test: Generate small test files (100 events per bin)
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import numpy as np
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder


# Ice depths per site (km)
SITE_DEPTHS = {
    "MB": -0.575,
    "SP": -2.7,
}

# Simulation volume radius (km)
MAX_RADIUS = 4.0

# Energy bins: 10^17.0 to 10^19.5 in 0.5 log steps
ENERGY_BINS = np.logspace(17.0, 19.5, num=6)

# Events per energy bin (monochromatic — Emin = Emax for each bin)
# Lower energies need more events because trigger rate is lower
def get_n_events(energy):
    if energy < 10**18.5:
        return int(1e5)
    else:
        return int(1e4)

# Max events per HDF5 file partition (auto-splits into .partNNNN)
N_EVENTS_PER_FILE = 50000


def generate_for_site(site, output_dir, test_mode=False):
    """Generate event lists for a single site.

    Args:
        site: 'MB' or 'SP'
        output_dir: Base output directory
        test_mode: If True, generate only 100 events per bin
    """
    depth_km = SITE_DEPTHS[site]
    site_dir = os.path.join(output_dir, site)
    os.makedirs(site_dir, exist_ok=True)

    volume = {
        "fiducial_zmin": depth_km * units.km,
        "fiducial_zmax": 0 * units.km,
        "fiducial_rmin": 0 * units.km,
        "fiducial_rmax": MAX_RADIUS * units.km,
    }

    for energy in ENERGY_BINS:
        n_events = 100 if test_mode else get_n_events(energy)
        n_per_file = 100 if test_mode else N_EVENTS_PER_FILE

        filename = os.path.join(site_dir, f"nu_{site}_{energy:.4e}.hdf5")
        print(f"  Generating {site} E={energy:.2e} eV: {n_events} events -> {filename}")

        generate_eventlist_cylinder(
            filename,
            n_events,
            energy * units.eV,
            energy * units.eV,  # monochromatic: Emin = Emax
            volume,
            n_events_per_file=n_per_file,
        )

    print(f"  Done: {site} ({len(ENERGY_BINS)} energy bins)")


def main():
    parser = argparse.ArgumentParser(description="Generate neutrino event lists")
    parser.add_argument(
        "--site",
        type=str,
        default="both",
        choices=["MB", "SP", "both"],
        help="Which site to generate for (default: both)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate small test files (100 events per bin)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="RCRSimulation/NeutrinoComparisonSimulation/GeneratedEvents",
        help="Base output directory",
    )
    args = parser.parse_args()

    sites = ["MB", "SP"] if args.site == "both" else [args.site]

    print(f"Generating neutrino event lists")
    print(f"  Sites: {sites}")
    print(f"  Energy bins: {len(ENERGY_BINS)} ({ENERGY_BINS[0]:.1e} - {ENERGY_BINS[-1]:.1e} eV)")
    print(f"  Test mode: {args.test}")
    print()

    for site in sites:
        generate_for_site(site, args.output_dir, test_mode=args.test)

    print("\nAll event lists generated.")


if __name__ == "__main__":
    main()
