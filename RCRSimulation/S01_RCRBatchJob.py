"""RCR Simulation Batch Job Submitter.

Submits all 9 simulation configurations needed for Chapter 4, with each split
into multiple subjobs for parallel execution. Each subjob processes a subset
of input files with its own seed.

Usage:
    python RCRSimulation/S01_RCRBatchJob.py [--test] [--dry-run] [--simulations SIM1 SIM2 ...]

Options:
    --test          Run small test simulations (50 cores, 10 files, 5 subjobs)
    --dry-run       Print commands without submitting
    --simulations   Only run specified simulations (by name)

Simulations:
    1. HRA_MB_shallow       - HRA baseline comparison
    2. Gen2_deep_MB_576m    - Gen2 deep at Moore's Bay
    3. Gen2_shallow_MB_576m - Gen2 shallow at Moore's Bay
    4. Gen2_deep_SP_300m    - Gen2 deep at South Pole (300m layer)
    5. Gen2_deep_SP_500m    - Gen2 deep at South Pole (500m layer)
    6. Gen2_deep_SP_830m    - Gen2 deep at South Pole (830m layer)
    7. Gen2_shallow_SP_300m - Gen2 shallow at South Pole (300m layer)
    8. Gen2_shallow_SP_500m - Gen2 shallow at South Pole (500m layer)
    9. Gen2_shallow_SP_830m - Gen2 shallow at South Pole (830m layer)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for A00_SlurmUtil import
sys.path.insert(0, str(Path(__file__).parent.parent))

import A00_SlurmUtil
import numpy as np
import configparser
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Optional


CONFIG_PATH = Path("RCRSimulation/config.ini")
COMMAND_PATH = "RCRSimulation/S01_RCRSim.py"
DEFAULT_PARTITION = "standard"


@dataclass
class SimulationConfig:
    """Configuration for a single simulation type."""
    name: str
    station_type: str
    station_depth: str
    site: str
    layer_depth: str
    layer_dB: str
    atten_model: str
    detector_config: str
    max_file: int


# All 9 simulation configurations for Chapter 4
SIMULATIONS: list[SimulationConfig] = [
    # 1. HRA MB (shallow) - baseline comparison
    SimulationConfig(
        name="HRA_MB_shallow",
        station_type="HRA", station_depth="shallow", site="MB",
        layer_depth="-576", layer_dB="1.7", atten_model="MB_freq",
        detector_config="RCRSimulation/configurations/MB/HRA_shallow_576m_combined.json",
        max_file=1000,
    ),
    # 2. Gen2 deep MB (576m layer)
    SimulationConfig(
        name="Gen2_deep_MB_576m",
        station_type="Gen2", station_depth="deep", site="MB",
        layer_depth="-576", layer_dB="1.7", atten_model="MB_freq",
        detector_config="RCRSimulation/configurations/MB/Gen2_deep_576m_combined.json",
        max_file=1000,
    ),
    # 3. Gen2 shallow MB (576m layer)
    SimulationConfig(
        name="Gen2_shallow_MB_576m",
        station_type="Gen2", station_depth="shallow", site="MB",
        layer_depth="-576", layer_dB="1.7", atten_model="MB_freq",
        detector_config="RCRSimulation/configurations/MB/Gen2_shallow_576m_combined.json",
        max_file=1000,
    ),
    # 4. Gen2 deep SP (300m layer)
    SimulationConfig(
        name="Gen2_deep_SP_300m",
        station_type="Gen2", station_depth="deep", site="SP",
        layer_depth="-300", layer_dB="0", atten_model="None",
        detector_config="RCRSimulation/configurations/SP/Gen2_deep_300m_combined.json",
        max_file=2100,
    ),
    # 5. Gen2 deep SP (500m layer)
    SimulationConfig(
        name="Gen2_deep_SP_500m",
        station_type="Gen2", station_depth="deep", site="SP",
        layer_depth="-500", layer_dB="0", atten_model="None",
        detector_config="RCRSimulation/configurations/SP/Gen2_deep_500m_combined.json",
        max_file=2100,
    ),
    # 6. Gen2 deep SP (830m layer)
    SimulationConfig(
        name="Gen2_deep_SP_830m",
        station_type="Gen2", station_depth="deep", site="SP",
        layer_depth="-830", layer_dB="0", atten_model="None",
        detector_config="RCRSimulation/configurations/SP/Gen2_deep_830m_combined.json",
        max_file=2100,
    ),
    # 7. Gen2 shallow SP (300m layer)
    SimulationConfig(
        name="Gen2_shallow_SP_300m",
        station_type="Gen2", station_depth="shallow", site="SP",
        layer_depth="-300", layer_dB="0", atten_model="None",
        detector_config="RCRSimulation/configurations/SP/Gen2_shallow_300m_combined.json",
        max_file=2100,
    ),
    # 8. Gen2 shallow SP (500m layer)
    SimulationConfig(
        name="Gen2_shallow_SP_500m",
        station_type="Gen2", station_depth="shallow", site="SP",
        layer_depth="-500", layer_dB="0", atten_model="None",
        detector_config="RCRSimulation/configurations/SP/Gen2_shallow_500m_combined.json",
        max_file=2100,
    ),
    # 9. Gen2 shallow SP (830m layer)
    SimulationConfig(
        name="Gen2_shallow_SP_830m",
        station_type="Gen2", station_depth="shallow", site="SP",
        layer_depth="-830", layer_dB="0", atten_model="None",
        detector_config="RCRSimulation/configurations/SP/Gen2_shallow_830m_combined.json",
        max_file=2100,
    ),
]


def chunk_file_range(min_file: int, max_file: int, num_subjobs: int) -> list[tuple[int, int]]:
    """Split file range [min_file, max_file] into num_subjobs chunks.

    Uses linspace to ensure even distribution, with lower file number as seed
    for each subjob.
    """
    if max_file <= min_file:
        return [(min_file, max_file)]

    # Use linspace like HRA does for even distribution
    file_range = np.linspace(min_file, max_file, num_subjobs + 1, dtype=int)

    ranges: list[tuple[int, int]] = []
    for i in range(len(file_range) - 1):
        ranges.append((int(file_range[i]), int(file_range[i + 1])))
    return ranges


def submit_simulation_jobs(
    sim: SimulationConfig,
    n_cores: int,
    num_subjobs: int,
    output_folder: str,
    numpy_folder: str,
    distance_km: float = 5.0,
    dry_run: bool = False,
    max_file_override: Optional[int] = None,
) -> int:
    """Submit subjobs for a simulation configuration.

    Splits the file range [0, max_file] into num_subjobs subjobs,
    each processing a subset of files.

    Returns:
        Number of jobs submitted.
    """
    min_file = 0
    max_file = max_file_override if max_file_override is not None else sim.max_file
    file_ranges = chunk_file_range(min_file, max_file, num_subjobs)

    print(f"\n{'='*60}")
    print(f"Simulation: {sim.name}")
    print(f"  Type: {sim.station_type}, Depth: {sim.station_depth}, Site: {sim.site}")
    print(f"  Layer: {sim.layer_depth}m, dB: {sim.layer_dB}")
    print(f"  Files: {min_file} to {max_file}, split into {len(file_ranges)} subjobs")
    print(f"  Cores per subjob: {n_cores}")
    print(f"{'='*60}")

    jobs_submitted = 0
    for lower_file, upper_file in file_ranges:
        # Use lower file number as seed for variation (like HRA does)
        seed = lower_file

        # Output name includes file range
        output_name = f"{sim.name}_files{lower_file}-{upper_file}_{n_cores}cores"
        job_name = f"RCR_{sim.name}_{lower_file}-{upper_file}"

        # Build command
        cmd = (
            f"python {COMMAND_PATH} {output_name} "
            f"--station-type {sim.station_type} "
            f"--station-depth {sim.station_depth} "
            f"--site {sim.site} "
            f"--propagation by_depth "
            f"--detector-config {sim.detector_config} "
            f"--n-cores {n_cores} "
            f"--distance-km {distance_km} "
            f"--min-file {lower_file} "
            f"--max-file {upper_file} "
            f"--seed {seed} "
            f"--layer-depth {sim.layer_depth} "
            f"--layer-db {sim.layer_dB} "
            f"--attenuation-model {sim.atten_model} "
            f"--add-noise "
            f"--output-folder {output_folder} "
            f"--numpy-folder {numpy_folder}"
        )

        if dry_run:
            print(f"  [DRY-RUN] Would submit: {job_name}")
            print(f"    Files {lower_file}-{upper_file}, seed={seed}")
        else:
            print(f"  Submitting: {job_name} (files {lower_file}-{upper_file})")
            A00_SlurmUtil.makeAndRunJob(
                cmd,
                jobName=job_name,
                runDirectory='run/RCRSimulation',
                partition=DEFAULT_PARTITION,
            )

        jobs_submitted += 1

    return jobs_submitted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit RCR simulation batch jobs for Chapter 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available simulations:
  HRA_MB_shallow       - HRA baseline comparison
  Gen2_deep_MB_576m    - Gen2 deep at Moore's Bay
  Gen2_shallow_MB_576m - Gen2 shallow at Moore's Bay
  Gen2_deep_SP_300m    - Gen2 deep at South Pole (300m layer)
  Gen2_deep_SP_500m    - Gen2 deep at South Pole (500m layer)
  Gen2_deep_SP_830m    - Gen2 deep at South Pole (830m layer)
  Gen2_shallow_SP_300m - Gen2 shallow at South Pole (300m layer)
  Gen2_shallow_SP_500m - Gen2 shallow at South Pole (500m layer)
  Gen2_shallow_SP_830m - Gen2 shallow at South Pole (830m layer)
        """,
    )
    parser.add_argument("--test", action="store_true", help="Run small test simulations")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting")
    parser.add_argument(
        "--simulations", nargs="+",
        help="Only run specified simulations (by name, e.g., Gen2_deep_MB_576m)",
    )
    parser.add_argument(
        "--num-subjobs", type=int, default=None,
        help="Number of subjobs per simulation (default: 100 for production, 5 for test)",
    )
    parser.add_argument(
        "--n-cores", type=int, default=None,
        help="Cores per subjob (default: 1000 for production, 50 for test)",
    )
    args = parser.parse_args()

    # Date tag for output directories
    date_tag = datetime.now().strftime("%m.%d.%y")

    # Try to get folders from config, with fallbacks
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    try:
        base_output = config.get('FOLDERS', 'sim_folder')
    except (configparser.NoSectionError, configparser.NoOptionError):
        base_output = f"/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/{date_tag}/"

    numpy_folder = f"RCRSimulation/output/{date_tag}/numpy/"

    # Test vs Production settings
    if args.test:
        n_cores = args.n_cores if args.n_cores else 50
        num_subjobs = args.num_subjobs if args.num_subjobs else 5
        max_file_override = 10
        print("=== TEST MODE: Small simulations ===")
    else:
        n_cores = args.n_cores if args.n_cores else 1000
        num_subjobs = args.num_subjobs if args.num_subjobs else 100
        max_file_override = None
        print("=== PRODUCTION MODE: Full simulations ===")

    print(f"Output folder: {base_output}")
    print(f"Numpy folder: {numpy_folder}")
    print(f"Subjobs per simulation: {num_subjobs}")
    print(f"Cores per subjob: {n_cores}")

    # Create output directories
    Path(numpy_folder).mkdir(parents=True, exist_ok=True)
    Path("run/RCRSimulation/logs").mkdir(parents=True, exist_ok=True)

    # Filter simulations if specified
    simulations_to_run = SIMULATIONS
    if args.simulations:
        sim_names = {s.name for s in SIMULATIONS}
        unknown = set(args.simulations) - sim_names
        if unknown:
            print(f"Warning: Unknown simulations ignored: {unknown}")
            print(f"Available: {sorted(sim_names)}")

        simulations_to_run = [
            sim for sim in SIMULATIONS
            if sim.name in args.simulations
        ]
        if not simulations_to_run:
            print("No matching simulations found.")
            return

    print(f"\nSimulations to run: {[s.name for s in simulations_to_run]}")

    # Submit all simulation jobs
    total_jobs = 0
    for sim in simulations_to_run:
        jobs = submit_simulation_jobs(
            sim=sim,
            n_cores=n_cores,
            num_subjobs=num_subjobs,
            output_folder=base_output,
            numpy_folder=numpy_folder,
            dry_run=args.dry_run,
            max_file_override=max_file_override,
        )
        total_jobs += jobs

    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"[DRY-RUN] Would submit {total_jobs} total jobs")
    else:
        print(f"Total jobs submitted: {total_jobs}")
    print(f"  ({len(simulations_to_run)} simulations x ~{num_subjobs} subjobs each)")
    print(f"Monitor with: squeue -u $USER")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
