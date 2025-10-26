#!/usr/bin/env python3
"""Single-station reflected cosmic-ray simulation runner.

This script mirrors the structure of ``HRASimulation/S01A_HRASim.py`` but is
specialised for simulations that only target one station at a time. Station
behaviour (direct vs reflected) and site-specific settings are controlled
through ``RCRSimulation/config.ini`` with optional command-line overrides.
"""

from __future__ import annotations

import argparse
import configparser
import datetime
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from NuRadioReco.utilities import units
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelStopFilter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.trigger.highLowThreshold
import NuRadioReco.modules.triggerTimeAdjuster
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.io.coreas.simulationSelector
import NuRadioReco.modules.io.eventWriter

from NuRadioReco.framework.parameters import eventParameters as evtp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.detector import detector

import readCoREASStationGrid

from SimpleFootprintSimulation.SimHelpers import pullFilesForSimulation, calculateNoisePerChannel


LOGGER = logging.getLogger("RCRSimulation")


DEFAULT_CONFIG_PATH = Path("RCRSimulation/config.ini")
COINCIDENCE_WINDOW = 40 * units.ns
NOISE_TRIGGER_SIGMA = 2.0
PRIMARY_CHANNELS = [0, 1, 2, 3]


@dataclass
class RCRSimEvent:
    """Light-weight container summarising a simulated event."""

    event_id: int
    station_id: int
    coreas_x_m: float
    coreas_y_m: float
    energy_eV: float
    zenith_deg: float
    azimuth_deg: float
    stn_zenith: float | None = None
    stn_azimuth: float | None = None
    trigger_sigma: float = 0.0
    triggered: bool = False

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single-station cosmic-ray simulation using CoREAS footprints."
    )
    parser.add_argument(
        "output_name",
        type=str,
        help="Base filename for the .nur output (without extension)."
             " If no directory is supplied, sim_output_folder from the config is used.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to the configuration file (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--station-type", choices=["HRA", "SP", "Gen2"], help="Override station type.")
    parser.add_argument("--site", choices=["MB", "SP"], help="Override site selection.")
    parser.add_argument("--propagation", choices=["direct", "reflected"], help="Override propagation mode.")
    parser.add_argument("--station-id", type=int, help="Explicit station id to simulate.")
    parser.add_argument("--detector-config", help="Path to detector JSON to load.")
    parser.add_argument("--n-cores", type=int, help="Number of cores to throw per run.")
    parser.add_argument("--distance-km", type=float, help="Throw radius in kilometres.")
    parser.add_argument("--min-file", type=int, help="Minimum CoREAS file index to include.")
    parser.add_argument("--max-file", type=int, help="Maximum CoREAS file index (inclusive).")
    parser.add_argument("--seed", type=int, help="Random seed supplied to readCoREAS.")
    parser.add_argument("--attenuation-model", help="Override attenuation model string.")
    parser.add_argument("--layer-depth", type=float, help="Layer depth in metres (negative for below surface).")
    parser.add_argument(
        "--trigger-sigma",
        type=float,
        help="Override the trigger sigma used for the primary station trigger.",
    )
    parser.add_argument(
        "--add-noise",
        dest="force_noise_on",
        action="store_true",
        help="Force enabling noise injection regardless of config value.",
    )
    parser.add_argument(
        "--no-noise",
        dest="force_noise_off",
        action="store_true",
        help="Force disabling noise injection regardless of config value.",
    )
    parser.add_argument(
        "--output-folder",
        help="Optional folder to place the .nur output (overrides config sim_output_folder).",
    )
    parser.add_argument(
        "--numpy-folder",
        help="Optional folder to place the numpy event summaries (overrides config numpy_folder).",
    )
    return parser.parse_args()


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def format_sigma_value(sigma: float) -> str:
    return f"{sigma:g}"


def primary_trigger_name(sigma: float) -> str:
    return f"primary_LPDA_2of4_{format_sigma_value(sigma)}sigma"


def _select_detector_config(candidates: Sequence[Path], station_type: str, site: str) -> Path | None:
    if not candidates:
        return None

    search_terms = [f"{station_type}_{site}", f"{station_type}{site}", f"{station_type}-{site}", station_type]
    for term in search_terms:
        term_lower = term.lower()
        for candidate in candidates:
            if term_lower in candidate.name.lower():
                return candidate

    return candidates[0]


def detector_config_from_path(path: Path, station_type: str, site: str) -> Path:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Detector configuration path not found: {path}")

    if path.is_file():
        return path

    search_roots: List[Path] = [path]
    for variant in {station_type, station_type.lower(), station_type.upper()}:
        search_roots.append(path / variant)
        search_roots.append(path / variant / site)
        search_roots.append(path / site / variant)
    for variant in {site, site.lower(), site.upper()}:
        search_roots.append(path / variant)

    seen_roots: set[Path] = set()
    candidates: List[Path] = []
    seen_files: set[Path] = set()
    patterns = ("*.json", "*.yaml", "*.yml", "*.ini")

    def add_candidates(root: Path) -> None:
        if root in seen_roots:
            return
        seen_roots.add(root)
        if root.is_file() and root.suffix.lower() in {".json", ".yaml", ".yml", ".ini"}:
            if root not in seen_files:
                seen_files.add(root)
                candidates.append(root)
        elif root.is_dir():
            for pattern in patterns:
                for candidate in sorted(root.glob(pattern)):
                    if candidate not in seen_files:
                        seen_files.add(candidate)
                        candidates.append(candidate)

    for root in search_roots:
        if root.exists():
            add_candidates(root)

    if not candidates:
        for pattern in patterns:
            for candidate in sorted(path.rglob(pattern)):
                if candidate not in seen_files:
                    seen_files.add(candidate)
                    candidates.append(candidate)

    match = _select_detector_config(candidates, station_type, site)
    if match is None:
        raise FileNotFoundError(
            f"Could not locate detector configuration under {path} for station type '{station_type}'"
        )
    return match


def merge_settings(args: argparse.Namespace, config: configparser.ConfigParser) -> Dict[str, object]:
    cfg_sim = config["SIMULATION"] if "SIMULATION" in config else {}
    cfg_paths = config["FOLDERS"] if "FOLDERS" in config else {}

    propagation = args.propagation or cfg_sim.get("propagation_mode", "direct")
    if propagation not in {"direct", "reflected"}:
        raise ValueError(f"Unsupported propagation mode '{propagation}'.")

    station_type = args.station_type or cfg_sim.get("station_type", "HRA")
    site = args.site or cfg_sim.get("site", "MB")

    station_id = args.station_id
    if station_id is None:
        key = "station_id_direct" if propagation == "direct" else "station_id_reflected"
        if key in cfg_sim:
            station_id = config.getint("SIMULATION", key)
        elif "station_id" in cfg_sim:
            station_id = config.getint("SIMULATION", "station_id")
    if station_id is None:
        raise ValueError("Station id must be provided via config or --station-id argument.")

    n_cores = args.n_cores or config.getint("SIMULATION", "n_cores", fallback=500)
    distance_km = args.distance_km or config.getfloat("SIMULATION", "distance_km", fallback=12.0)
    min_file = args.min_file if args.min_file is not None else config.getint("SIMULATION", "min_file", fallback=0)
    max_file = args.max_file if args.max_file is not None else config.getint("SIMULATION", "max_file", fallback=1000)
    seed = args.seed if args.seed is not None else config.getint("SIMULATION", "seed", fallback=0)

    attenuation_model = args.attenuation_model or cfg_sim.get("attenuation_model", "MB_freq")
    if attenuation_model and attenuation_model.lower() in {"none", "null"}:
        attenuation_model = None

    layer_depth = args.layer_depth if args.layer_depth is not None else config.getfloat("SIMULATION", "layer_depth_m", fallback=-576.0)

    trigger_sigma_key = f"trigger_sigma_{station_type}".lower()
    if args.trigger_sigma is not None:
        trigger_sigma = float(args.trigger_sigma)
    else:
        if not config.has_option("SIMULATION", trigger_sigma_key):
            raise KeyError(f"Missing configuration entry '{trigger_sigma_key}' in [SIMULATION] section.")
        trigger_sigma = config.getfloat("SIMULATION", trigger_sigma_key)

    if args.force_noise_on:
        add_noise = True
    elif args.force_noise_off:
        add_noise = False
    else:
        add_noise = parse_bool(cfg_sim.get("add_noise", "false"))

    sim_config_dir = config.get("SIMULATION", "config_dir", fallback=None)
    folder_config_dir = config.get("FOLDERS", "config_dir", fallback=None)
    detector_config_root = folder_config_dir or sim_config_dir or cfg_paths.get(
        "detector_config_dir", "RCRSimulation/configurations"
    )

    if args.detector_config:
        detector_config = detector_config_from_path(Path(args.detector_config), station_type, site)
    else:
        detector_config = detector_config_from_path(Path(detector_config_root), station_type, site)

    output_folder = (
        Path(args.output_folder)
        if args.output_folder
        else Path(cfg_paths.get("sim_output_folder", "RCRSimulation/output"))
    ).expanduser()
    numpy_folder = (
        Path(args.numpy_folder)
        if args.numpy_folder
        else Path(cfg_paths.get("numpy_folder", "RCRSimulation/output/numpy"))
    ).expanduser()
    save_folder = Path(cfg_paths.get("save_folder", "RCRSimulation/plots")).expanduser()

    run_directory = Path(cfg_paths.get("run_directory", "run/RCRSimulation")).expanduser()

    return {
        "station_type": station_type,
        "site": site,
        "propagation": propagation,
        "station_id": station_id,
        "n_cores": n_cores,
        "distance_km": distance_km,
        "min_file": min_file,
        "max_file": max_file,
        "seed": seed,
        "attenuation_model": attenuation_model,
        "layer_depth_m": layer_depth,
    "trigger_sigma": trigger_sigma,
    "noise_sigma": NOISE_TRIGGER_SIGMA,
        "add_noise": add_noise,
        "detector_config": detector_config,
        "output_folder": output_folder,
        "numpy_folder": numpy_folder,
        "save_folder": save_folder,
        "run_directory": run_directory,
    }


def resolve_output_paths(output_name: str, folders: Dict[str, Path]) -> Dict[str, Path]:
    base_path = Path(output_name)
    if base_path.parent == Path("."):
        base_path = folders["output_folder"] / base_path.name
    nur_path = base_path.with_suffix(".nur")

    numpy_folder = folders["numpy_folder"]
    numpy_filename = f"{base_path.stem}_RCReventList.npy"
    numpy_path = numpy_folder / numpy_filename

    return {
        "base": base_path,
        "nur": nur_path,
        "numpy": numpy_path,
    }


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def build_thresholds(post_amp_vrms: Dict[int, float], sigmas: List[float]) -> Dict[str, Dict[str, Dict[int, float]]]:
    thresholds: Dict[str, Dict[str, Dict[int, float]]] = {}
    for sigma in sigmas:
        key = format_sigma_value(sigma)
        thresholds[key] = {
            "high": {ch: val * sigma for ch, val in post_amp_vrms.items()},
            "low": {ch: -val * sigma for ch, val in post_amp_vrms.items()},
        }
    return thresholds


def run_simulation(settings: Dict[str, object], output_paths: Dict[str, Path]) -> None:
    station_id = settings["station_id"]
    trigger_sigma: float = settings["trigger_sigma"]
    noise_sigma: float = settings["noise_sigma"]
    add_noise: bool = settings["add_noise"]
    site: str = settings["site"]
    propagation: str = settings["propagation"]
    distance_km: float = settings["distance_km"]

    LOGGER.info(
        "Starting simulation for station %s (%s, %s, propagation=%s, sigma=%.2f)",
        station_id,
        settings["station_type"],
        site,
        propagation,
        trigger_sigma,
    )

    input_files = pullFilesForSimulation(site, settings["min_file"], settings["max_file"])
    if not input_files:
        raise RuntimeError(f"No CoREAS input files found for site '{site}' with the provided range.")

    det = detector.Detector(str(settings["detector_config"]), "json")
    det.update(datetime.datetime(2018, 10, 1))

    simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
    simulationSelector.begin()

    efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
    efieldToVoltageConverter.begin(debug=False)

    hardwareResponseIncorporator = NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()

    channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
    channelGenericNoiseAdder.begin()

    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
    channelBandPassFilter.begin()

    eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
    channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()

    channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
    channelResampler.begin()

    triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
    triggerTimeAdjuster.begin(trigger_name=primary_trigger_name(trigger_sigma))

    highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()

    correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
    correlationDirectionFitter.begin(debug=False)

    readCoREAS = readCoREASStationGrid.readCoREAS()
    distance_m = distance_km * 1000
    readCoREAS.begin(
        input_files,
        -distance_m / 2,
        distance_m / 2,
        -distance_m / 2,
        distance_m / 2,
        n_cores=settings["n_cores"],
        shape="radial",
        seed=settings["seed"],
        log_level=logging.INFO,
    )

    eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
    eventWriter.begin(str(output_paths["nur"]))

    events: List[RCRSimEvent] = []
    thresholds: Dict[str, Dict[str, Dict[int, float]]] | None = None

    for evt, event_idx, coreas_x, coreas_y in readCoREAS.run(
        detector=det,
        ray_type=propagation,
        layer_depth=settings["layer_depth_m"] * units.m,
        layer_dB=0,
        attenuation_model=settings["attenuation_model"],
        output_mode=2,
    ):
        LOGGER.debug("Processing event %s (index %s)", evt.get_id(), event_idx)
        evt.set_parameter(evtp.coreas_x, coreas_x)
        evt.set_parameter(evtp.coreas_y, coreas_y)

        station = evt.get_station(station_id)
        if station is None:
            LOGGER.warning("Station %s not present in event %s, skipping.", station_id, evt.get_id())
            continue

        eventTypeIdentifier.run(evt, station, mode="forced", forced_event_type="cosmic_ray")
        efieldToVoltageConverter.run(evt, station, det)
        channelResampler.run(evt, station, det, 2 * units.GHz)

        if thresholds is None:
            pre_amp_vrms, post_amp_vrms = calculateNoisePerChannel(
                det,
                station=station,
                amp=True,
                hardwareResponseIncorporator=hardwareResponseIncorporator,
                channelBandPassFilter=channelBandPassFilter,
            )
            thresholds = build_thresholds(post_amp_vrms, [trigger_sigma, noise_sigma])
            LOGGER.info("Computed noise thresholds for station %s", station_id)
            LOGGER.debug("Thresholds: %s", thresholds)

        hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)

        base_channels = PRIMARY_CHANNELS

        noise_key = format_sigma_value(noise_sigma)
        final_key = format_sigma_value(trigger_sigma)
        noise_trigger = primary_trigger_name(noise_sigma)
        final_trigger = primary_trigger_name(trigger_sigma)

        highLowThreshold.run(
            evt,
            station,
            det,
            threshold_high=thresholds[noise_key]["high"],
            threshold_low=thresholds[noise_key]["low"],
            coinc_window=COINCIDENCE_WINDOW,
            triggered_channels=base_channels,
            number_concidences=2,
            trigger_name=noise_trigger,
        )

        triggered = False
        stn_zenith = None
        stn_azimuth = None

        if station.has_trigger(noise_trigger):
            if add_noise:
                hardwareResponseIncorporator.run(evt, station, det, sim_to_data=False)
                channelGenericNoiseAdder.run(evt, station, det, type="rayleigh", amplitude=pre_amp_vrms)
                hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)

            highLowThreshold.run(
                evt,
                station,
                det,
                threshold_high=thresholds[final_key]["high"],
                threshold_low=thresholds[final_key]["low"],
                coinc_window=COINCIDENCE_WINDOW,
                triggered_channels=base_channels,
                number_concidences=2,
                trigger_name=final_trigger,
            )

            triggerTimeAdjuster.run(evt, station, det)
            channelStopFilter.run(evt, station, det, prepend=0 * units.ns, append=0 * units.ns)
            correlationDirectionFitter.run(
                evt,
                station,
                det,
                n_index=1.35,
                ZenLim=[0 * units.deg, 180 * units.deg],
                channel_pairs=((PRIMARY_CHANNELS[0], PRIMARY_CHANNELS[2]), (PRIMARY_CHANNELS[1], PRIMARY_CHANNELS[3])),
            )
            stn_zenith = station.get_parameter(stnp.zenith)
            stn_azimuth = station.get_parameter(stnp.azimuth)

            triggered = station.has_trigger(final_trigger)

        sim_shower = evt.get_sim_shower(0)
        events.append(
            RCRSimEvent(
                event_id=evt.get_id(),
                station_id=station_id,
                coreas_x_m=float(evt.get_parameter(evtp.coreas_x) / units.m),
                coreas_y_m=float(evt.get_parameter(evtp.coreas_y) / units.m),
                energy_eV=float(sim_shower[shp.energy] / units.eV),
                zenith_deg=float(sim_shower[shp.zenith] / units.deg),
                azimuth_deg=float(sim_shower[shp.azimuth] / units.deg),
                stn_zenith=stn_zenith / units.deg if stn_zenith is not None else None,
                stn_azimuth=stn_azimuth / units.deg if stn_azimuth is not None else None,
                trigger_sigma=trigger_sigma,
                triggered=triggered,
            )
        )

        eventWriter.run(evt)

    nevents = eventWriter.end()
    run_time_s = readCoREAS.end()
    LOGGER.info("Processed %s events. readCoREAS runtime: %.2fs", nevents, run_time_s)

    npy_array = np.array(events, dtype=object)
    np.save(output_paths["numpy"], npy_array, allow_pickle=True)
    LOGGER.info("Saved %d event summaries to %s", len(events), output_paths["numpy"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)

    settings = merge_settings(args, config)

    ensure_directories([
        settings["output_folder"],
        settings["numpy_folder"],
        settings["save_folder"],
    ])

    output_paths = resolve_output_paths(args.output_name, settings)

    ensure_directories([output_paths["nur"].parent, output_paths["numpy"].parent])

    run_simulation(settings, output_paths)


if __name__ == "__main__":
    main()
