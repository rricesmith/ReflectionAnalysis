from __future__ import annotations

import math
import configparser
from pathlib import Path

import numpy as np

import A00_SlurmUtil

CONFIG_PATH = Path("RCRSimulation/config.ini")
COMMAND_PATH = "RCRSimulation/S01_RCRSim.py"
DEFAULT_PARTITION = "standard"


def _chunk_file_range(min_file: int, max_file: int, job_count: int) -> list[tuple[int, int]]:
    if max_file == -1 or max_file <= min_file:
        return [(min_file, max_file)]

    span = max_file - min_file
    files_per_job = max(1, math.ceil(span / job_count))

    ranges: list[tuple[int, int]] = []
    start = min_file
    while start < max_file:
        end = min(max_file, start + files_per_job)
        ranges.append((start, end))
        start = end
    return ranges


def main() -> None:
    config = configparser.ConfigParser()
    if not config.read(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

    sim_cfg = config["SIMULATION"]
    folder_cfg = config["FOLDERS"]
    batch_cfg = config["BATCH"] if config.has_section("BATCH") else {}

    station_type = sim_cfg.get("station_type", "HRA")
    site = sim_cfg.get("site", "MB")
    station_depth = sim_cfg.get("station_depth", "shallow").strip().lower()
    if station_depth not in {"deep", "shallow"}:
        raise ValueError(f"Unsupported station depth variant '{station_depth}'.")
    propagation = sim_cfg.get("propagation_mode", "direct")
    if propagation == "reflected":
        station_id = config.getint("SIMULATION", "station_id_reflected", fallback=1)
    else:
        station_id = config.getint("SIMULATION", "station_id_direct", fallback=1)

    n_cores = config.getint("SIMULATION", "n_cores", fallback=500)
    distance_km = config.getfloat("SIMULATION", "distance_km", fallback=12.0)
    add_noise = config.getboolean("SIMULATION", "add_noise", fallback=True)
    min_file = config.getint("SIMULATION", "min_file", fallback=0)
    max_file = config.getint("SIMULATION", "max_file", fallback=-1)
    seed_base = config.getint("SIMULATION", "seed", fallback=0)

    job_count = int(batch_cfg.get("job_count", max(1, n_cores // 20)))
    partition = batch_cfg.get("partition", DEFAULT_PARTITION)

    output_folder = Path(folder_cfg.get("sim_output_folder", "RCRSimulation/output")).expanduser()
    run_directory = Path(folder_cfg.get("run_directory", "run/RCRSimulation")).expanduser()

    sim_config_dir = config.get("SIMULATION", "config_dir", fallback=None)
    folder_config_dir = config.get("FOLDERS", "config_dir", fallback=None)
    detector_config_root = folder_config_dir or sim_config_dir or folder_cfg.get(
        "detector_config_dir", "RCRSimulation/configurations"
    )
    detector_config_path = Path(detector_config_root).expanduser()

    output_folder.mkdir(parents=True, exist_ok=True)
    run_directory.mkdir(parents=True, exist_ok=True)

    site_lower = site.strip().lower()

    if site_lower == "icetop":
        energy_min = config.getfloat("SIMULATION", "energy_min", fallback=16.0)
        energy_max = config.getfloat("SIMULATION", "energy_max", fallback=18.6)
        energy_step = config.getfloat("SIMULATION", "energy_step", fallback=0.1)
        sin2_min = config.getfloat("SIMULATION", "sin2_min", fallback=0.0)
        sin2_max = config.getfloat("SIMULATION", "sin2_max", fallback=1.0)
        sin2_step = config.getfloat("SIMULATION", "sin2_step", fallback=0.1)
        num_icetop = config.getint("SIMULATION", "num_icetop", fallback=10)

        energy_values = np.arange(energy_min, energy_max, energy_step)
        if not energy_values.size:
            energy_values = np.array([energy_min])
        sin2_values = np.arange(sin2_min, sin2_max + 0.5 * sin2_step, sin2_step)
        sin2_values = np.clip(sin2_values, 0.0, 1.0)
        if not sin2_values.size:
            sin2_values = np.array([0.0])

        jobs: list[tuple[float, float, float]] = []
        for energy in energy_values:
            energy_low = float(np.round(energy, 5))
            energy_high = float(np.round(min(energy + energy_step, energy_max), 5))
            for sin2 in sin2_values:
                sin2_val = float(np.round(sin2, 5))
                jobs.append((energy_low, energy_high, sin2_val))

        print(
            f"Submitting {len(jobs)} IceTop jobs for station {station_id} "
            f"({station_type}, {site}, {station_depth}, {propagation})."
        )

        def format_bin(value: float) -> str:
            text = f"{value:.3f}"
            text = text.rstrip("0").rstrip(".")
            return text if text else "0"

        for idx, (energy_low, energy_high, sin2_val) in enumerate(jobs):
            seed_value = seed_base + idx
            sin2_label = format_bin(sin2_val)
            output_label = (
                f"{station_type}_{site}_{propagation}_stn{station_id}_"
                f"E{energy_low:.1f}-{energy_high:.1f}_sin2_{sin2_label}_{n_cores}cores"
            )

            cmd_parts = [
                "python",
                COMMAND_PATH,
                output_label,
                "--config",
                str(CONFIG_PATH),
                "--station-type",
                station_type,
                "--site",
                site,
                "--propagation",
                propagation,
                "--station-depth",
                station_depth,
                "--station-id",
                str(station_id),
                "--n-cores",
                str(n_cores),
                "--distance-km",
                f"{distance_km}",
                "--energy-min",
                f"{energy_low}",
                "--energy-max",
                f"{energy_high}",
                "--sin2",
                f"{sin2_val}",
                "--num-icetop",
                str(num_icetop),
                "--seed",
                str(seed_value),
            ]

            if add_noise:
                cmd_parts.append("--add-noise")
            else:
                cmd_parts.append("--no-noise")

            cmd_parts.extend(["--detector-config", str(detector_config_path)])

            command = " ".join(cmd_parts)
            job_name = f"RCR_{station_id}_E{energy_low:.1f}-{energy_high:.1f}_sin2_{sin2_label}"

            A00_SlurmUtil.makeAndRunJob(
                command,
                job_name,
                runDirectory=str(run_directory),
                partition=partition,
            )

        return

    file_ranges = _chunk_file_range(min_file, max_file, job_count)

    print(
        f"Submitting {len(file_ranges)} jobs for station {station_id} "
        f"({station_type}, {site}, {station_depth}, {propagation})."
    )

    for idx, (lower_file, upper_file) in enumerate(file_ranges):
        output_label = f"{station_type}_{site}_{propagation}_stn{station_id}_files{lower_file}-{upper_file if upper_file != -1 else 'all'}_{n_cores}cores"
        seed_value = seed_base + idx

        cmd_parts = [
            "python",
            COMMAND_PATH,
            output_label,
            "--config",
            str(CONFIG_PATH),
            "--station-type",
            station_type,
            "--site",
            site,
            "--propagation",
            propagation,
            "--station-depth",
            station_depth,
            "--station-id",
            str(station_id),
            "--n-cores",
            str(n_cores),
            "--distance-km",
            f"{distance_km}",
            "--min-file",
            str(lower_file),
        ]

        if upper_file != -1:
            cmd_parts.extend(["--max-file", str(upper_file)])

        cmd_parts.extend(["--seed", str(seed_value)])

        if add_noise:
            cmd_parts.append("--add-noise")
        else:
            cmd_parts.append("--no-noise")

        cmd_parts.extend(["--detector-config", str(detector_config_path)])

        command = " ".join(cmd_parts)
        job_name = f"RCR_{station_id}_{lower_file}-{upper_file if upper_file != -1 else 'all'}"

        A00_SlurmUtil.makeAndRunJob(
            command,
            job_name,
            runDirectory=str(run_directory),
            partition=partition,
        )


if __name__ == "__main__":
    main()
