from __future__ import annotations

import math
import configparser
from pathlib import Path

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
