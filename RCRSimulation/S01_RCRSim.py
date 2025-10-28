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
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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
GEN2_FILTER_RIPPLE = 0.1


def layer_depth_to_label(layer_depth: float | None) -> str:
    """Translate a numeric layer depth to the config filename label."""
    if layer_depth is None:
        return "surface"
    if isinstance(layer_depth, float) and not math.isfinite(layer_depth):
        return "surface"
    if abs(layer_depth) < 1e-3:
        return "surface"
    depth_m = abs(layer_depth)
    if math.isclose(depth_m, round(depth_m)):
        depth_m = int(round(depth_m))
        return f"{depth_m}m"
    return f"{depth_m:g}m"


def parse_layer_depth_value(raw_value: str | float | None, default: float) -> float:
    """Convert layer depth config entries to a float, supporting 'surface'."""
    if raw_value is None:
        return default
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    text = str(raw_value).strip()
    if not text:
        return default
    lowered = text.lower()
    if lowered in {"surface", "surf"}:
        return 0.0
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported layer depth value '{raw_value}'.") from exc


def build_gen2_filter_config(
    det_obj: detector.Detector,
    station_id: int,
    station_depth: str,
) -> Dict[str, Dict[int, object]]:
    """Construct per-channel band-pass settings for Gen2 stations."""
    channel_ids = sorted(det_obj.get_channel_ids(station_id))
    if not channel_ids:
        return {"passband_low": {}, "passband_high": {}, "filter_type": {}, "order_low": {}, "order_high": {}}

    passband_low: Dict[int, Tuple[float, float]] = {}
    passband_high: Dict[int, Tuple[float, float]] = {}
    filter_type: Dict[int, str] = {}
    order_low: Dict[int, int] = {}
    order_high: Dict[int, int] = {}

    shallow_candidates = [ch for ch in channel_ids if ch <= 3]
    pa_candidates = [ch for ch in channel_ids if ch not in shallow_candidates]

    def apply_shallow_filters(channels: Iterable[int]) -> None:
        for ch in channels:
            passband_low[ch] = (1 * units.MHz, 1000 * units.MHz)
            passband_high[ch] = (0.08 * units.GHz, 800 * units.GHz)
            filter_type[ch] = "butter"
            order_low[ch] = 10
            order_high[ch] = 5

    def apply_deep_filters(channels: Iterable[int]) -> None:
        for ch in channels:
            passband_low[ch] = (0 * units.MHz, 1000 * units.MHz)
            passband_high[ch] = (96 * units.MHz, 100 * units.GHz)
            filter_type[ch] = "cheby1"
            order_low[ch] = 7
            order_high[ch] = 4

    if station_depth.lower() == "deep":
        if shallow_candidates:
            apply_shallow_filters(shallow_candidates)
        if pa_candidates:
            apply_deep_filters(pa_candidates)
    else:
        target_channels = shallow_candidates if shallow_candidates else channel_ids
        apply_shallow_filters(target_channels)

    return {
        "passband_low": passband_low,
        "passband_high": passband_high,
        "filter_type": filter_type,
        "order_low": order_low,
        "order_high": order_high,
    }


def apply_gen2_filters(
    bandpass_module: NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter,
    evt,
    station,
    det,
    config: Dict[str, Dict[int, object]],
) -> None:
    if not config["passband_low"]:
        return
    bandpass_module.run(
        evt,
        station,
        det,
        passband=config["passband_low"],
        filter_type=config["filter_type"],
        order=config["order_low"],
        rp=GEN2_FILTER_RIPPLE,
    )
    bandpass_module.run(
        evt,
        station,
        det,
        passband=config["passband_high"],
        filter_type=config["filter_type"],
        order=config["order_high"],
        rp=GEN2_FILTER_RIPPLE,
    )


def resolve_trigger_channels(
    det_obj: detector.Detector,
    station_id: int,
    station_type: str,
    station_depth: str,
) -> List[int]:
    """Determine which channels participate in the primary trigger."""
    if station_type.lower() != "gen2":
        return list(PRIMARY_CHANNELS)

    channel_ids = sorted(det_obj.get_channel_ids(station_id))
    if not channel_ids:
        raise ValueError(f"No channels found for station {station_id} in detector configuration.")

    depth_value = station_depth.lower()

    if depth_value == "deep":
        pa_candidates = [ch for ch in channel_ids if ch >= 4]
        if len(pa_candidates) >= 4:
            return pa_candidates[:4]
        if len(channel_ids) >= 4:
            return channel_ids[-4:]
        return channel_ids

    # Shallow configuration: prioritise the first LPDA-style channels
    shallow_candidates = [ch for ch in channel_ids if ch < 4]
    if len(shallow_candidates) >= 4:
        return shallow_candidates[:4]
    return channel_ids[:4]


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
    parser.add_argument(
        "--station-depth",
        choices=["deep", "shallow"],
        help="Select the station configuration depth variant (deep or shallow).",
    )
    parser.add_argument("--site", choices=["MB", "SP"], help="Override site selection.")
    parser.add_argument(
        "--propagation",
        choices=["direct", "reflected", "by_depth"],
        help="Override propagation mode.",
    )
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


def detector_config_from_path(
    path: Path,
    station_type: str,
    site: str,
    depth_variant: str | None = None,
    layer_label: str | None = None,
) -> Path:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Detector configuration path not found: {path}")

    if path.is_file():
        return path

    station_norm = station_type.lower()
    depth_norm = depth_variant.lower() if depth_variant else None
    layer_norm = layer_label.lower() if layer_label else None
    site_norm = site.lower()

    search_roots: List[Path] = [path]
    for variant in {station_type, station_type.lower(), station_type.upper()}:
        search_roots.append(path / variant)
        search_roots.append(path / variant / site)
        search_roots.append(path / site / variant)
    for variant in {site, site.lower(), site.upper()}:
        search_roots.append(path / variant)
    if depth_variant:
        for variant in {depth_variant, depth_variant.lower(), depth_variant.upper()}:
            search_roots.append(path / variant)
            search_roots.append(path / station_type / variant)
    if layer_label:
        for variant in {layer_label, layer_label.lower(), layer_label.upper()}:
            search_roots.append(path / variant)
            search_roots.append(path / station_type / variant)
            if depth_variant:
                search_roots.append(path / variant / depth_variant)

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

    priority_patterns: List[str] = []

    def add_pattern(pattern: str | None) -> None:
        if pattern and pattern not in priority_patterns:
            priority_patterns.append(pattern)

    if depth_norm and layer_norm:
        add_pattern(f"{station_norm}_{depth_norm}_{layer_norm}")
        add_pattern(f"{station_norm}-{depth_norm}-{layer_norm}")
        add_pattern(f"{station_norm}{depth_norm}{layer_norm}")
    if depth_norm:
        add_pattern(f"{station_norm}_{depth_norm}")
        add_pattern(f"{station_norm}-{depth_norm}")
        add_pattern(f"{station_norm}{depth_norm}")
    if layer_norm:
        add_pattern(f"{station_norm}_{layer_norm}")
    add_pattern(f"{station_norm}_{site_norm}")
    add_pattern(f"{station_norm}{site_norm}")
    if layer_norm:
        add_pattern(layer_norm)
    add_pattern(site_norm)
    add_pattern(station_norm)

    for pattern in priority_patterns:
        for candidate in candidates:
            candidate_name = candidate.stem.lower()
            if pattern and pattern in candidate_name:
                return candidate

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
    if propagation not in {"direct", "reflected", "by_depth"}:
        raise ValueError(f"Unsupported propagation mode '{propagation}'.")

    station_type = args.station_type or cfg_sim.get("station_type", "HRA")
    site = args.site or cfg_sim.get("site", "MB")

    station_depth = args.station_depth or cfg_sim.get("station_depth", "shallow")
    if station_depth is None:
        station_depth = "shallow"
    station_depth = str(station_depth).strip().lower()
    if station_depth not in {"deep", "shallow"}:
        raise ValueError(f"Unsupported station depth variant '{station_depth}'.")

    station_id = args.station_id
    if station_id is None:
        propagation_key_map = {
            "direct": ["station_id_direct"],
            "reflected": ["station_id_reflected"],
            "by_depth": ["station_id_by_depth", "station_id_reflected", "station_id_direct"],
        }
        candidate_keys = propagation_key_map.get(propagation, []) + ["station_id"]
        for key in candidate_keys:
            if key and config.has_option("SIMULATION", key):
                station_id = config.getint("SIMULATION", key)
                break
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

    if args.layer_depth is not None:
        layer_depth = float(args.layer_depth)
    else:
        raw_layer_depth = cfg_sim.get("layer_depth_m", None)
        layer_depth = parse_layer_depth_value(raw_layer_depth, default=-576.0)
    layer_label = layer_depth_to_label(layer_depth)

    trigger_sigma = None
    trigger_sigma_key_used = None
    if args.trigger_sigma is not None:
        trigger_sigma = float(args.trigger_sigma)
        trigger_sigma_key_used = "cli"
    else:
        candidate_trigger_keys: List[str] = []
        if station_depth:
            candidate_trigger_keys.append(f"trigger_sigma_{station_type}_{station_depth}")
        candidate_trigger_keys.append(f"trigger_sigma_{station_type}")

        for key in candidate_trigger_keys:
            if config.has_option("SIMULATION", key):
                trigger_sigma = config.getfloat("SIMULATION", key)
                trigger_sigma_key_used = key
                break

        if trigger_sigma is None:
            raise KeyError(
                "Missing configuration entry for trigger sigma. Tried keys: "
                + ", ".join(candidate_trigger_keys)
            )

    if args.force_noise_on:
        add_noise = True
    elif args.force_noise_off:
        add_noise = False
    else:
        add_noise = parse_bool(cfg_sim.get("add_noise", "false"))

    debug_enabled = parse_bool(cfg_sim.get("debug", "false"))

    sim_config_dir = config.get("SIMULATION", "config_dir", fallback=None)
    folder_config_dir = config.get("FOLDERS", "config_dir", fallback=None)
    detector_config_root = folder_config_dir or sim_config_dir or cfg_paths.get(
        "detector_config_dir", "RCRSimulation/configurations"
    )

    if args.detector_config:
        detector_config = detector_config_from_path(
            Path(args.detector_config), station_type, site, station_depth, layer_label
        )
    else:
        detector_config = detector_config_from_path(
            Path(detector_config_root), station_type, site, station_depth, layer_label
        )

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
    log_folder = Path(
        cfg_paths.get("log_folder", "RCRSimulation/logs")
    ).expanduser()

    return {
        "station_type": station_type,
        "site": site,
        "station_depth": station_depth,
        "propagation": propagation,
        "station_id": station_id,
        "n_cores": n_cores,
        "distance_km": distance_km,
        "min_file": min_file,
        "max_file": max_file,
        "seed": seed,
        "attenuation_model": attenuation_model,
        "layer_depth_m": layer_depth,
        "layer_descriptor": layer_label,
        "trigger_sigma": trigger_sigma,
    "trigger_sigma_key": trigger_sigma_key_used,
        "noise_sigma": NOISE_TRIGGER_SIGMA,
        "add_noise": add_noise,
        "detector_config": detector_config,
        "output_folder": output_folder,
        "numpy_folder": numpy_folder,
        "save_folder": save_folder,
        "run_directory": run_directory,
        "debug": debug_enabled,
        "log_folder": log_folder,
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


def summarize_events(events: Sequence[RCRSimEvent]) -> Dict[Tuple[str, str, str], Dict[str, int]]:
    summary: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    for event in events:
        energy_label = f"{event.energy_eV:.6e}" if event.energy_eV is not None else "n/a"
        zenith_label = f"{event.zenith_deg:.3f}" if event.zenith_deg is not None else "n/a"
        azimuth_label = f"{event.azimuth_deg:.3f}" if event.azimuth_deg is not None else "n/a"
        key = (energy_label, zenith_label, azimuth_label)
        entry = summary.setdefault(key, {"count": 0, "triggered": 0})
        entry["count"] += 1
        if event.triggered:
            entry["triggered"] += 1
    return summary


def write_debug_log(
    settings: Dict[str, Any],
    output_paths: Dict[str, Path],
    thresholds: Dict[str, Dict[str, Dict[int, float]]] | None,
    noise_stats: Dict[str, Dict[int, float]],
    events: Sequence[RCRSimEvent],
    run_time_s: float,
) -> None:
    log_folder: Path = settings["log_folder"]
    log_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_folder / f"{output_paths['base'].stem}_{timestamp}_debug.txt"

    total_events = len(events)
    triggered_total = sum(1 for event in events if event.triggered)
    global_rate = triggered_total / total_events if total_events else 0.0

    summary_map = summarize_events(events)

    lines: List[str] = []
    lines.append("RCR Simulation Debug Log")
    lines.append("========================")
    lines.append(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Output base: {output_paths['base']}")
    lines.append(f"readCoREAS runtime (s): {run_time_s:.2f}")
    lines.append("")

    lines.append("[Configuration]")
    config_keys = [
        "station_type",
        "site",
        "station_depth",
        "propagation",
        "station_id",
        "n_cores",
        "distance_km",
        "min_file",
        "max_file",
        "seed",
        "attenuation_model",
        "layer_depth_m",
        "layer_descriptor",
        "trigger_sigma",
    "trigger_sigma_key",
        "noise_sigma",
        "add_noise",
        "detector_config",
        "output_folder",
        "numpy_folder",
        "save_folder",
        "run_directory",
        "debug",
        "log_folder",
    ]
    for key in config_keys:
        if key not in settings:
            continue
        value = settings[key]
        value_repr = str(value)
        lines.append(f"{key}: {value_repr}")
    lines.append("")

    lines.append("[Simulation Values]")
    trigger_channels = settings.get("trigger_channels", PRIMARY_CHANNELS)
    lines.append(f"Primary channels: {', '.join(str(ch) for ch in trigger_channels)}")
    lines.append(f"Primary trigger sigma: {settings['trigger_sigma']}")
    lines.append(f"Noise trigger sigma: {settings['noise_sigma']}")
    lines.append(f"Coincidence window (ns): {COINCIDENCE_WINDOW / units.ns:.3f}")

    pre_amp_vrms = noise_stats.get("pre_amp_vrms", {}) or {}
    post_amp_vrms = noise_stats.get("post_amp_vrms", {}) or {}
    if pre_amp_vrms:
        lines.append("Pre-amplifier Vrms per channel (V):")
        for ch in sorted(pre_amp_vrms):
            lines.append(f"  ch{ch}: {pre_amp_vrms[ch]:.6g}")
    if post_amp_vrms:
        lines.append("Post-amplifier Vrms per channel (V):")
        for ch in sorted(post_amp_vrms):
            lines.append(f"  ch{ch}: {post_amp_vrms[ch]:.6g}")

    if thresholds:
        for sigma_key in sorted(thresholds.keys(), key=lambda s: float(s)):
            sigma_data = thresholds[sigma_key]
            lines.append(f"Thresholds for {sigma_key} sigma (V):")
            lines.append("  High:")
            for ch in sorted(sigma_data["high"]):
                lines.append(f"    ch{ch}: {sigma_data['high'][ch]:.6g}")
            lines.append("  Low:")
            for ch in sorted(sigma_data["low"]):
                lines.append(f"    ch{ch}: {sigma_data['low'][ch]:.6g}")
    else:
        lines.append("Thresholds were not computed (no events processed).")

    lines.append("")
    lines.append("[Event Summary]")
    lines.append(f"Total events processed: {total_events}")
    lines.append(f"Total events triggered: {triggered_total}")
    lines.append(f"Global trigger rate: {global_rate:.2%}")

    if summary_map:
        lines.append("Grouped by energy/zenith/azimuth:")
        for key in sorted(summary_map.keys()):
            energy_label, zenith_label, azimuth_label = key
            entry = summary_map[key]
            rate = entry["triggered"] / entry["count"] if entry["count"] else 0.0
            lines.append(
                "  Energy="
                f"{energy_label} eV; Zenith={zenith_label} deg; Azimuth={azimuth_label} deg -> "
                f"{entry['count']} events, {entry['triggered']} triggered ({rate:.2%})"
            )
    else:
        lines.append("No events were processed in this run.")

    lines.append("")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote debug log to %s", log_path)


def run_simulation(settings: Dict[str, object], output_paths: Dict[str, Path]) -> None:
    station_id = settings["station_id"]
    trigger_sigma: float = settings["trigger_sigma"]
    noise_sigma: float = settings["noise_sigma"]
    add_noise: bool = settings["add_noise"]
    site: str = settings["site"]
    propagation: str = settings["propagation"]
    distance_km: float = settings["distance_km"]
    debug_enabled: bool = bool(settings.get("debug"))
    station_type: str = settings["station_type"]
    station_depth: str = settings["station_depth"]
    is_gen2 = station_type.lower() == "gen2"

    LOGGER.info(
        "Starting simulation for station %s (%s, %s, propagation=%s, sigma=%.2f)",
        station_id,
        station_type,
        site,
        propagation,
        trigger_sigma,
    )

    input_files = pullFilesForSimulation(site, settings["min_file"], settings["max_file"])
    if not input_files:
        # Some selections will have no files, return with no error
        LOGGER.warning("Quitting, No CoREAS input files found for site '%s' with the provided range.", site)
        quit()


    det = detector.Detector(str(settings["detector_config"]), "json")
    det.update(datetime.datetime(2018, 10, 1))

    configured_trigger_channels = resolve_trigger_channels(det, station_id, station_type, station_depth)
    settings["trigger_channels"] = configured_trigger_channels


    efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
    efieldToVoltageConverter.begin(debug=False)

    hardwareResponseIncorporator = None
    if not is_gen2:
        hardwareResponseIncorporator = (
            NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator.hardwareResponseIncorporator()
        )

    channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
    channelGenericNoiseAdder.begin()

    channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
    channelBandPassFilter.begin()

    gen2_filter_config: Dict[str, Dict[int, object]] | None = None
    if is_gen2:
        gen2_filter_config = build_gen2_filter_config(det, station_id, station_depth)

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
    pre_amp_vrms: Dict[int, float] | None = None
    post_amp_vrms: Dict[int, float] | None = None
    active_trigger_channels: List[int] | None = None

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

        eventTypeIdentifier.run(evt, station, mode="forced", forced_event_type="cosmic ray")
        efieldToVoltageConverter.run(evt, station, det)
        channelResampler.run(evt, station, det, 2 * units.GHz)

        if gen2_filter_config is not None:
            apply_gen2_filters(channelBandPassFilter, evt, station, det, gen2_filter_config)

        if thresholds is None:
            if is_gen2:
                pre_amp_vrms, post_amp_vrms = calculateNoisePerChannel(
                    det,
                    station=station,
                    amp=False,
                    channelBandPassFilter=channelBandPassFilter,
                )
            else:
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

            configured_channels = settings.get("trigger_channels", list(PRIMARY_CHANNELS))
            available_channels = [ch for ch in configured_channels if ch in post_amp_vrms]
            if not available_channels:
                available_channels = sorted(post_amp_vrms.keys())
                LOGGER.warning(
                    "No overlap between configured trigger channels %s and available noise channels; using %s instead.",
                    configured_channels,
                    available_channels,
                )
            active_trigger_channels = available_channels
            settings["trigger_channels"] = available_channels

        if active_trigger_channels is None:
            active_trigger_channels = settings.get("trigger_channels", list(PRIMARY_CHANNELS))
        if not active_trigger_channels:
            raise RuntimeError("No trigger channels available for high/low threshold evaluation.")

        if hardwareResponseIncorporator is not None:
            hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)
        
        base_channels = active_trigger_channels

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

        if station.has_triggered(trigger_name=noise_trigger):
            if add_noise:
                if hardwareResponseIncorporator is not None:
                    hardwareResponseIncorporator.run(evt, station, det, sim_to_data=False)
                channelGenericNoiseAdder.run(evt, station, det, type="rayleigh", amplitude=pre_amp_vrms)
                if hardwareResponseIncorporator is not None:
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
            triggered = station.has_triggered(trigger_name=final_trigger)
            # if triggered:
                # correlationDirectionFitter.run(
                #     evt,
                #     station,
                #     det,
                #     n_index=1.35,
                #     ZenLim=[0 * units.deg, 180 * units.deg],
                #     channel_pairs=((PRIMARY_CHANNELS[0], PRIMARY_CHANNELS[2]), (PRIMARY_CHANNELS[1], PRIMARY_CHANNELS[3])),
                # )
                # stn_zenith = station.get_parameter(stnp.zenith)
                # stn_azimuth = station.get_parameter(stnp.azimuth)


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
    run_time = readCoREAS.end()
    if isinstance(run_time, datetime.timedelta):
        run_time_s = run_time.total_seconds()
    elif run_time is None:
        run_time_s = 0.0
    else:
        run_time_s = float(run_time)
    LOGGER.info("Processed %s events. readCoREAS runtime: %.2fs", nevents, run_time_s)

    npy_array = np.array(events, dtype=object)
    np.save(output_paths["numpy"], npy_array, allow_pickle=True)
    LOGGER.info("Saved %d event summaries to %s", len(events), output_paths["numpy"])

    if debug_enabled:
        noise_stats = {
            "pre_amp_vrms": pre_amp_vrms or {},
            "post_amp_vrms": post_amp_vrms or {},
        }
        write_debug_log(
            settings,
            output_paths,
            thresholds,
            noise_stats,
            events,
            run_time_s,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)

    settings = merge_settings(args, config)

    required_dirs = [
        settings["output_folder"],
        settings["numpy_folder"],
        settings["save_folder"],
    ]
    if settings.get("debug"):
        required_dirs.append(settings["log_folder"])
    ensure_directories(required_dirs)

    output_paths = resolve_output_paths(args.output_name, settings)

    ensure_directories([output_paths["nur"].parent, output_paths["numpy"].parent])

    run_simulation(settings, output_paths)


if __name__ == "__main__":
    main()
