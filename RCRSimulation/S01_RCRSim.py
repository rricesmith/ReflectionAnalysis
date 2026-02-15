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
import NuRadioReco.modules.phasedarray.triggerSimulator
import NuRadioReco.modules.triggerTimeAdjuster
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.io.eventWriter

from NuRadioReco.framework.parameters import eventParameters as evtp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.detector import detector

import readCoREASStationGrid
from readCoREASStationGrid import compute_MB_freq_attenuation

from SimpleFootprintSimulation.SimHelpers import (
    pullFilesForSimulation,
    calculateNoisePerChannel
)
from RCRSimulation.RCREventObject import RCREvent, REFLECTED_STATION_OFFSET


LOGGER = logging.getLogger("RCRSimulation")


DEFAULT_CONFIG_PATH = Path("RCRSimulation/config.ini")
COINCIDENCE_WINDOW = 40 * units.ns
NOISE_TRIGGER_SIGMA = 3.5
PRIMARY_CHANNELS = [0, 1, 2, 3]
GEN2_FILTER_RIPPLE = 0.1

# Gen2 shallow LPDA channel definitions (for direct configs with 8 LPDAs)
GEN2_SHALLOW_DOWN_CHANNELS = [0, 1, 2, 3]  # Down-pointing LPDAs (theta=0)
GEN2_SHALLOW_UP_CHANNELS = [4, 5, 6, 7]    # Up-pointing LPDAs (theta=180)

# Phased array configuration (from Gen2 TDR)
PA_CHANNELS_8CH = [4, 5, 6, 7, 8, 9, 10, 11]
PA_CHANNELS_4CH = [8, 9, 10, 11]
PA_MAIN_LOW_ANGLE = -59.55 * units.deg
PA_MAIN_HIGH_ANGLE = 59.55 * units.deg
PA_REF_INDEX = 1.75
PA_N_BEAMS_8CH = 21
PA_N_BEAMS_4CH = 11
PA_UPSAMPLING_8CH = 4
PA_UPSAMPLING_4CH = 2
# Window/step are computed at runtime from detector trigger_adc_sampling_frequency
PA_WINDOW_NS = 16  # ns, converted to samples in event loop
PA_STEP_NS = 8     # ns, converted to samples in event loop
# Thresholds calibrated for different noise rates (Vrms^2 units)
# From reference: 10kHz->46.98, 100Hz->61.90, 1Hz->76.83, 1mHz->99.22 (8ch)
# From reference: 100Hz->30.68, 1Hz->38.62, 1mHz->50.53 (4ch)
PA_THRESHOLDS_8CH = {100: 61.90, 1: 76.83, 0.001: 99.22}  # Hz -> threshold
PA_THRESHOLDS_4CH = {100: 30.68, 1: 38.62, 0.001: 50.53}  # Hz -> threshold


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




def parse_layer_db_value(raw_value: str | float | None, default: float) -> float:
    if raw_value is None:
        return default
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    text = str(raw_value).strip()
    if not text:
        return default
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return default
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid layer_dB value: {raw_value}") from exc


def format_layer_db_metadata(layer_db: float | None) -> tuple[str, str]:
    try:
        numeric = float(layer_db)
    except (TypeError, ValueError):
        return "unknown", ""
    if not math.isfinite(numeric):
        return "unknown", ""
    text = f"{numeric:g}"
    return text, f"{text}dB".lower()


def calcSNR(traces, Vrms, n_avg=2):
    """Calculate SNR from the average of top n_avg channel p2p amplitudes.

    For each channel trace, computes peak-to-peak / (2 * Vrms).
    Returns the mean of the top n_avg values.

    Args:
        traces: List of voltage trace arrays (one per channel)
        Vrms: Noise RMS voltage (single value, same units as traces)
        n_avg: Number of top channels to average (2 for shallow, 4 for deep)
    """
    SNRs = []
    for trace in traces:
        p2p = np.max(trace) - np.min(trace)
        SNRs.append(p2p / (2 * Vrms))
    SNRs.sort(reverse=True)
    return np.mean(SNRs[:n_avg])


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
            passband_low[ch] = (1 * units.MHz, 150 * units.MHz)
            passband_high[ch] = (0.08 * units.GHz, 800 * units.GHz)
            filter_type[ch] = "butter"
            order_low[ch] = 10
            order_high[ch] = 5

    def apply_deep_filters(channels: Iterable[int]) -> None:
        for ch in channels:
            passband_low[ch] = (0 * units.MHz, 220 * units.MHz)
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
        # For phased array, return all available PA channels (4-11)
        pa_candidates = [ch for ch in channel_ids if ch >= 4 and ch <= 11]
        if len(pa_candidates) >= 4:
            return pa_candidates  # Return all PA channels for phased array
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
    layer_dB: float | None = None
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
    parser.add_argument("--site", choices=["MB", "SP", "IceTop"], help="Override site selection.")
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
    parser.add_argument("--energy-min", type=float, help="Lower log10(E/eV) edge for IceTop selections.")
    parser.add_argument("--energy-max", type=float, help="Upper log10(E/eV) edge for IceTop selections.")
    parser.add_argument("--sin2", type=float, help="sin^2(zenith) bin value for IceTop selections.")
    parser.add_argument("--num-icetop", type=int, help="Number of IceTop footprints to include per bin.")
    parser.add_argument("--attenuation-model", help="Override attenuation model string.")
    parser.add_argument("--layer-depth", type=float, help="Layer depth in metres (negative for below surface).")
    parser.add_argument(
        "--layer-db",
        type=float,
        help="Relative power loss in the reflective layer in dB.",
    )
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
    parser.add_argument(
        "--layer-db-list",
        type=str,
        default=None,
        help="Comma-separated dB loss values for multi-reflectivity sweep "
             "(e.g., '40,45,50,55' for SP, '0,1.7,3.0' for MB). "
             "When provided, reflected stations are tested at each dB value "
             "with separate triggers. Overrides --layer-db for reflected stations.",
    )
    parser.add_argument(
        "--min-energy-log10",
        type=float,
        default=None,
        help="Skip events with log10(E/eV) below this value "
             "(e.g., 18.0 to only simulate showers >= 10^18 eV). "
             "Skipped events are not recorded, reducing output size and runtime.",
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


def primary_trigger_name(sigma: float, layer_db: float | None = None) -> str:
    name = f"primary_LPDA_2of4_{format_sigma_value(sigma)}sigma"
    if layer_db is not None:
        name += f"_{layer_db:.1f}dB"
    return name


def pa_trigger_name(n_channels: int, noise_rate_hz: float, layer_db: float | None = None) -> str:
    """Generate trigger name for phased array trigger."""
    name = f"PA_{n_channels}ch_{noise_rate_hz:g}Hz"
    if layer_db is not None:
        name += f"_{layer_db:.1f}dB"
    return name


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


def _resolve_detector_config_root(
    base_path: Path,
    station_type: str,
    site: str,
    depth_variant: str | None,
    layer_label: str | None,
) -> Path:
    base_path = base_path.expanduser()

    if base_path.is_file():
        return detector_config_from_path(base_path, station_type, site, depth_variant, layer_label)

    site_key = (site or "").strip().lower()
    if site_key and base_path.name.lower() == site_key:
        return detector_config_from_path(base_path, station_type, site, depth_variant, layer_label)

    candidate_roots: List[Path] = []
    if site_key:
        variants = [site, site.lower(), site.upper()]
        seen: set[str] = set()
        for variant in variants:
            if not variant:
                continue
            norm = variant.lower()
            if norm in seen:
                continue
            seen.add(norm)
            candidate_roots.append(base_path / variant)
    candidate_roots.append(base_path)

    last_error: FileNotFoundError | None = None
    for candidate in candidate_roots:
        try:
            return detector_config_from_path(candidate, station_type, site, depth_variant, layer_label)
        except FileNotFoundError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise FileNotFoundError(
        f"Could not locate detector configuration under {base_path} for station type '{station_type}'"
    )


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

    search_roots: List[Path] = []
    path_is_site = path.name.lower() == site_norm
    if not path_is_site:
        for variant in {site, site.lower(), site.upper()}:
            if variant:
                candidate = path / variant
                if candidate not in search_roots:
                    search_roots.append(candidate)
    if path not in search_roots:
        search_roots.append(path)
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

    site_lower = site.lower()
    # IceTop uses energy/sin2 binned simulations; SP and MB use sequential file loading
    if site_lower == "icetop":
        energy_min = args.energy_min if args.energy_min is not None else config.getfloat(
            "SIMULATION", "energy_min", fallback=16.0
        )
        energy_max = args.energy_max if args.energy_max is not None else config.getfloat(
            "SIMULATION", "energy_max", fallback=18.6
        )
        sin2_value = args.sin2 if args.sin2 is not None else config.getfloat("SIMULATION", "sin2", fallback=0.0)
        num_icetop = args.num_icetop if args.num_icetop is not None else config.getint(
            "SIMULATION", "num_icetop", fallback=20
        )

        if energy_max <= energy_min:
            raise ValueError("IceTop energy_max must be greater than energy_min.")
        if not 0.0 <= sin2_value <= 1.0:
            raise ValueError("IceTop sin2 must be within [0, 1].")
        energy_settings = (energy_min, energy_max, sin2_value, num_icetop)
    else:
        energy_settings = (None, None, None, None)
    energy_min, energy_max, sin2_value, num_icetop = energy_settings

    attenuation_model = args.attenuation_model or cfg_sim.get("attenuation_model", "MB_freq")
    if attenuation_model and attenuation_model.lower() in {"none", "null"}:
        attenuation_model = None

    if args.layer_depth is not None:
        layer_depth = float(args.layer_depth)
    else:
        raw_layer_depth = cfg_sim.get("layer_depth_m", None)
        layer_depth = parse_layer_depth_value(raw_layer_depth, default=-576.0)
    layer_label = layer_depth_to_label(layer_depth)

    if args.layer_db is not None:
        layer_db = float(args.layer_db)
    else:
        raw_layer_db = cfg_sim.get("layer_dB", None)
        layer_db = parse_layer_db_value(raw_layer_db, default=0.0)
    layer_db_text, layer_db_tag = format_layer_db_metadata(layer_db)

    # Multi-dB reflectivity sweep (optional)
    layer_db_list: list[float] | None = None
    if args.layer_db_list is not None:
        layer_db_list = sorted(float(v.strip()) for v in args.layer_db_list.split(","))
        LOGGER.info("Multi-dB sweep enabled: %s", layer_db_list)

    if math.isfinite(layer_depth):
        if math.isclose(layer_depth, 0.0, abs_tol=1e-6):
            layer_depth_text = "surface"
            layer_depth_tag = "layer_surface"
        else:
            depth_value = int(round(layer_depth)) if math.isclose(layer_depth, round(layer_depth)) else layer_depth
            depth_text = f"{depth_value:g}"
            layer_depth_text = f"{depth_text} m"
            layer_depth_tag = f"layer_{depth_text}m".lower()
    else:
        layer_depth_text = "unknown"
        layer_depth_tag = "layer_unknown"

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

    noise_tag = "noise_on" if add_noise else "noise_off"
    noise_text = "on" if add_noise else "off"

    debug_enabled = parse_bool(cfg_sim.get("debug", "false"))

    sim_config_dir = config.get("SIMULATION", "config_dir", fallback=None)
    folder_config_dir = config.get("FOLDERS", "config_dir", fallback=None)
    detector_config_root_raw = folder_config_dir or sim_config_dir or cfg_paths.get(
        "detector_config_dir", "RCRSimulation/configurations"
    )
    detector_config_root_path = Path(detector_config_root_raw).expanduser()

    if args.detector_config:
        detector_config_arg = Path(args.detector_config).expanduser()
        if detector_config_arg.suffix.lower() in {".json", ".yaml", ".yml", ".ini"}:
            detector_config = detector_config_from_path(
                detector_config_arg, station_type, site, station_depth, layer_label
            )
        else:
            detector_config = _resolve_detector_config_root(
                detector_config_arg, station_type, site, station_depth, layer_label
            )
    else:
        detector_config = _resolve_detector_config_root(
            detector_config_root_path, station_type, site, station_depth, layer_label
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

    settings = {
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
        "layer_depth_text": layer_depth_text,
        "layer_depth_tag": layer_depth_tag,
        "layer_dB": layer_db,
        "layer_dB_text": layer_db_text,
        "layer_dB_tag": layer_db_tag,
        "layer_dB_list": layer_db_list,
        "trigger_sigma": trigger_sigma,
        "trigger_sigma_key": trigger_sigma_key_used,
        "noise_sigma": NOISE_TRIGGER_SIGMA,
        "add_noise": add_noise,
        "noise_tag": noise_tag,
        "noise_text": noise_text,
        "detector_config": detector_config,
        "energy_min": energy_min,
        "energy_max": energy_max,
        "sin2": sin2_value,
        "num_icetop": num_icetop,
        "output_folder": output_folder,
        "numpy_folder": numpy_folder,
        "save_folder": save_folder,
        "run_directory": run_directory,
        "debug": debug_enabled,
        "log_folder": log_folder,
    }

    settings["min_energy_log10"] = args.min_energy_log10

    settings["filename_tags"] = [
        str(value).lower()
        for value in (
            station_type,
            site,
            station_depth,
            propagation,
            layer_depth_tag,
            settings.get("layer_dB_tag"),
            noise_tag,
        )
        if value
    ]
    return settings


def resolve_output_paths(output_name: str, folders: Dict[str, Any]) -> Dict[str, Path]:
    raw_path = Path(output_name)
    if raw_path.is_absolute() or raw_path.parent != Path("."):
        resolved_path = raw_path
    else:
        resolved_path = folders["output_folder"] / raw_path.name
    resolved_path = resolved_path.expanduser()

    if resolved_path.suffix.lower() == ".nur":
        base_dir = resolved_path.parent
        base_name = resolved_path.stem
    else:
        base_dir = resolved_path.parent
        base_name = resolved_path.name

    tags = [str(tag).lower() for tag in folders.get("filename_tags", []) if tag]

    if not isinstance(base_name, str):
        base_name = str(base_name)

    if tags:
        tag_suffix = "_".join(tags)
        base_name = f"{base_name}_{tag_suffix}" if base_name else tag_suffix

    base_path = base_dir / base_name
    nur_path = base_dir / f"{base_name}.nur"

    numpy_folder = folders["numpy_folder"]
    numpy_filename = f"{base_name}_RCReventList.npy"
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


def summarize_events(events: Sequence[RCREvent]) -> Dict[Tuple[str, str, str], Dict[str, int]]:
    summary: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    for event in events:
        energy_label = f"{event.energy:.6e}" if event.energy is not None else "n/a"
        zenith_label = f"{np.rad2deg(event.zenith):.3f}" if event.zenith is not None else "n/a"
        azimuth_label = f"{np.rad2deg(event.azimuth):.3f}" if event.azimuth is not None else "n/a"
        key = (energy_label, zenith_label, azimuth_label)
        entry = summary.setdefault(key, {"count": 0, "triggered": 0})
        entry["count"] += 1
        # Check if any trigger fired
        if any(event.has_triggered(t) for t in event.all_trigger_names()):
            entry["triggered"] += 1
    return summary


def write_debug_log(
    settings: Dict[str, Any],
    output_paths: Dict[str, Path],
    thresholds: Dict[int, Dict[str, Dict[str, Dict[int, float]]]] | None,
    noise_stats: Dict[str, Dict[int, float]],
    events: Sequence[RCREvent],
    run_time_s: float,
) -> None:
    log_folder: Path = settings["log_folder"]
    log_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_label = output_paths["base"].name
    log_path = log_folder / f"{base_label}_{timestamp}_debug.txt"

    total_events = len(events)
    triggered_total = sum(
        1 for event in events
        if any(event.has_triggered(t) for t in event.all_trigger_names())
    )
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
        "layer_depth_text",
        "layer_depth_tag",
        "layer_dB",
        "layer_dB_text",
        "layer_dB_tag",
        "trigger_sigma",
        "trigger_sigma_key",
        "energy_min",
        "energy_max",
        "sin2",
        "num_icetop",
        "min_energy_log10",
        "noise_sigma",
        "add_noise",
        "noise_text",
        "noise_tag",
        "detector_config",
        "output_folder",
        "numpy_folder",
        "save_folder",
        "run_directory",
        "debug",
        "log_folder",
        "filename_tags",
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
        # thresholds is Dict[station_id, Dict[sigma_key, Dict["high"/"low", Dict[ch, float]]]]
        for stn_id in sorted(thresholds.keys()):
            stn_thresholds = thresholds[stn_id]
            lines.append(f"Station {stn_id} Thresholds (V):")
            for sigma_key in sorted(stn_thresholds.keys(), key=lambda s: float(s)):
                sigma_data = stn_thresholds[sigma_key]
                lines.append(f"  {sigma_key} sigma:")
                lines.append("    High:")
                for ch in sorted(sigma_data["high"]):
                    lines.append(f"      ch{ch}: {sigma_data['high'][ch]:.6g}")
                lines.append("    Low:")
                for ch in sorted(sigma_data["low"]):
                    lines.append(f"      ch{ch}: {sigma_data['low'][ch]:.6g}")
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
    base_station_id = settings["station_id"]  # Base station ID (direct)
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
    layer_db_text = settings.get("layer_dB_text", "unknown")

    LOGGER.info(
        "Starting simulation for station %s (%s, %s, propagation=%s, sigma=%.2f, layer_dB=%s dB)",
        base_station_id,
        station_type,
        site,
        propagation,
        trigger_sigma,
        layer_db_text,
    )

    site_lower = site.lower()
    if site_lower == "icetop":
        energy_range = (settings["energy_min"], settings["energy_max"])
        sin2_value = settings["sin2"]
        num_icetop = settings["num_icetop"] or 20
        input_files = pullFilesForSimulation(
            "IceTop",
            settings["min_file"],
            settings["max_file"],
            energy_range=energy_range,
            sin2_value=sin2_value,
            num_icetop=num_icetop,
        )
    else:
        # SP uses sequential file loading (000XXX.hdf5 / SIM00XXX.hdf5 up to 2100)
        # MB uses sequential file loading (00XXXX.hdf5 up to ~3999)
        input_files = pullFilesForSimulation(site, settings["min_file"], settings["max_file"])
    if not input_files:
        # Some selections will have no files, return with no error
        LOGGER.warning("Quitting, No CoREAS input files found for site '%s' with the provided range.", site)
        quit()


    det = detector.Detector(str(settings["detector_config"]), "json", antenna_by_depth=False)
    det.update(datetime.datetime(2018, 10, 1))

    # Get all station IDs from detector (includes both direct and reflected)
    all_station_ids = sorted(det.get_station_ids())
    LOGGER.info("Processing stations: %s", all_station_ids)

    # Determine trigger channels for the base station type
    configured_trigger_channels = resolve_trigger_channels(det, base_station_id, station_type, station_depth)
    settings["trigger_channels"] = configured_trigger_channels

    # Per-station data structures
    thresholds_per_station: Dict[int, Dict[str, Dict[str, Dict[int, float]]]] = {}
    pre_amp_vrms_per_station: Dict[int, Dict[int, float]] = {}
    post_amp_vrms_per_station: Dict[int, Dict[int, float]] = {}
    trigger_channels_per_station: Dict[int, List[int]] = {}
    gen2_filter_config_per_station: Dict[int, Dict[str, Dict[int, object]]] = {}


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

    # Build filter configs for all stations
    if is_gen2:
        for stn_id in all_station_ids:
            gen2_filter_config_per_station[stn_id] = build_gen2_filter_config(det, stn_id, station_depth)

    eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
    channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()

    channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
    channelResampler.begin()

    triggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()
    triggerTimeAdjuster.begin(trigger_name=primary_trigger_name(trigger_sigma))

    highLowThreshold = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()

    # Initialize phased array trigger for Gen2 deep
    phasedArrayTrigger = None
    paTriggerTimeAdjuster = None
    if is_gen2 and station_depth == "deep":
        phasedArrayTrigger = NuRadioReco.modules.phasedarray.triggerSimulator.triggerSimulator()
        # Create PA-specific trigger time adjuster (will set trigger_name in the event loop)
        paTriggerTimeAdjuster = NuRadioReco.modules.triggerTimeAdjuster.triggerTimeAdjuster()

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

    rcr_events: List[RCREvent] = []

    # Pre-calculated noise values for Gen2
    GEN2_DEEP_NOISE = 5.627 * units.micro * units.V
    GEN2_SHALLOW_NOISE = 3.789 * units.micro * units.V

    _trigger_params_logged = False  # Log PA trigger params once
    _n_events_processed = 0

    layer_db_list = settings.get("layer_dB_list")
    # When multi-dB sweep is active, run readCoREAS with layer_dB=0 so the
    # reflected E-field has base attenuation but no reflection dB loss applied.
    # The dB loss (and attenuation R correction) are applied per-value in the loop below.
    run_layer_dB = 0.0 if layer_db_list else settings["layer_dB"]

    # R_base used by readCoREAS for the MB_freq attenuation model (hardcoded default)
    R_BASE = 0.82

    for evt, event_idx, coreas_x, coreas_y in readCoREAS.run(
        detector=det,
        ray_type=propagation,
        layer_depth=settings["layer_depth_m"] * units.m,
        layer_dB=run_layer_dB,
        attenuation_model=settings["attenuation_model"],
        output_mode=2,
    ):
        LOGGER.debug("Processing event %s (index %s)", evt.get_id(), event_idx)

        # Energy cutoff: skip events below threshold to save compute time
        min_energy_log10 = settings.get("min_energy_log10")
        if min_energy_log10 is not None:
            sim_shower = evt.get_sim_shower(0)
            energy_eV = sim_shower[shp.energy] / units.eV
            if np.log10(energy_eV) < min_energy_log10:
                LOGGER.debug("Skipping event %s: energy %.2e eV below cutoff 10^%.1f eV",
                             evt.get_id(), energy_eV, min_energy_log10)
                continue

        evt.set_parameter(evtp.coreas_x, coreas_x)
        evt.set_parameter(evtp.coreas_y, coreas_y)
        evt_snr_cache: Dict[int, float] = {}  # station_id -> SNR for this event
        _n_events_processed += 1

        # Process each station (both direct and reflected)
        for station_id in all_station_ids:
            station = evt.get_station(station_id)
            if station is None:
                LOGGER.debug("Station %s not present in event %s, skipping.", station_id, evt.get_id())
                continue

            eventTypeIdentifier.run(evt, station, mode="forced", forced_event_type="cosmic ray")
            efieldToVoltageConverter.run(evt, station, det)

            if is_gen2:
                channelResampler.run(evt, station, det, 2.4 * units.GHz)
            else:
                channelResampler.run(evt, station, det, 2 * units.GHz)

            # Apply Gen2 filters if configured
            if station_id in gen2_filter_config_per_station:
                apply_gen2_filters(channelBandPassFilter, evt, station, det, gen2_filter_config_per_station[station_id])

            # Initialize per-station thresholds on first encounter
            if station_id not in thresholds_per_station:
                if is_gen2:
                    if station_depth == "deep":
                        pa_channels = [ch for ch in det.get_channel_ids(station_id) if ch >= 4 and ch <= 11]
                        pre_amp_vrms = {ch: GEN2_DEEP_NOISE for ch in pa_channels}
                        post_amp_vrms = {ch: GEN2_DEEP_NOISE for ch in pa_channels}
                    else:
                        pre_amp_vrms = {ch: GEN2_SHALLOW_NOISE for ch in PRIMARY_CHANNELS}
                        post_amp_vrms = {ch: GEN2_SHALLOW_NOISE for ch in PRIMARY_CHANNELS}
                else:
                    pre_amp_vrms, post_amp_vrms = calculateNoisePerChannel(
                        det,
                        station=station,
                        amp=True,
                        hardwareResponseIncorporator=hardwareResponseIncorporator,
                        channelBandPassFilter=channelBandPassFilter,
                    )
                pre_amp_vrms_per_station[station_id] = pre_amp_vrms
                post_amp_vrms_per_station[station_id] = post_amp_vrms
                thresholds_per_station[station_id] = build_thresholds(post_amp_vrms, [trigger_sigma, noise_sigma])

                # Determine trigger channels for this station
                configured_channels = settings.get("trigger_channels", list(PRIMARY_CHANNELS))
                available_channels = [ch for ch in configured_channels if ch in post_amp_vrms]
                if not available_channels:
                    available_channels = sorted(post_amp_vrms.keys())
                trigger_channels_per_station[station_id] = available_channels
                LOGGER.info("Initialized thresholds for station %s, channels: %s", station_id, available_channels)

            # Get per-station values
            pre_amp_vrms = pre_amp_vrms_per_station[station_id]
            post_amp_vrms = post_amp_vrms_per_station[station_id]
            thresholds = thresholds_per_station[station_id]
            base_channels = trigger_channels_per_station[station_id]

            if hardwareResponseIncorporator is not None:
                hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)

            # ---- Multi-dB sweep setup ----
            # For reflected stations with layer_db_list, test multiple reflectivity
            # values by applying spectral corrections to the post-amplifier traces.
            is_reflected_station = station_id >= REFLECTED_STATION_OFFSET
            if is_reflected_station and layer_db_list:
                db_sweep = sorted(layer_db_list)  # lowest dB (strongest signal) first
                # Save clean post-amplifier frequency spectra before any dB modification
                _clean_spectra = {}
                for ch_id in base_channels:
                    ch = station.get_channel(ch_id)
                    _clean_spectra[ch_id] = (ch.get_frequency_spectrum().copy(), ch.get_sampling_rate())
            else:
                db_sweep = [None]  # None = original single-dB behavior, no dB tag
                _clean_spectra = None

            _db_precheck_failed = False  # Early exit: if weakest dB fails pre-check, skip stronger

            for db_val in db_sweep:
                # Apply dB correction to reflected station spectra
                if _clean_spectra is not None and db_val is not None:
                    if _db_precheck_failed:
                        break  # Higher dB = weaker signal, won't pass either
                    R_new = 10 ** (-db_val / 20)
                    atten_model = settings["attenuation_model"]
                    dist_per_ch = readCoREAS.channel_dist_traveled.get(station_id, {})

                    for ch_id in base_channels:
                        orig_spectrum, sr = _clean_spectra[ch_id]
                        modified = orig_spectrum.copy()

                        # Frequency-dependent attenuation correction for MB_freq model
                        if atten_model == 'MB_freq' and ch_id in dist_per_ch:
                            trace_len = len(station.get_channel(ch_id).get_trace())
                            freqs = np.fft.rfftfreq(trace_len, d=1.0 / sr)
                            base_atten = compute_MB_freq_attenuation(freqs, dist_per_ch[ch_id], R=R_BASE)
                            new_atten = compute_MB_freq_attenuation(freqs, dist_per_ch[ch_id], R=R_new)
                            correction = new_atten / np.where(base_atten > 0, base_atten, 1.0)
                            modified = modified * correction

                        # Apply reflection dB loss (amplitude factor)
                        modified = modified * R_new
                        station.get_channel(ch_id).set_frequency_spectrum(modified, sr)

                # ---- Trigger logic (runs once per db_val) ----

                if phasedArrayTrigger is not None:
                    # Use phased array trigger for Gen2 deep configuration
                    n_pa_channels = len(base_channels)
                    pa_noise_rate = 100  # Hz - use 100 Hz threshold

                    if n_pa_channels >= 8:
                        pa_threshold = PA_THRESHOLDS_8CH.get(pa_noise_rate, PA_THRESHOLDS_8CH[100])
                        pa_n_beams = PA_N_BEAMS_8CH
                        pa_upsampling = PA_UPSAMPLING_8CH
                    else:
                        pa_threshold = PA_THRESHOLDS_4CH.get(pa_noise_rate, PA_THRESHOLDS_4CH[100])
                        pa_n_beams = PA_N_BEAMS_4CH
                        pa_upsampling = PA_UPSAMPLING_4CH

                    # Calculate window/step in samples from detector ADC rate (reference approach)
                    det_channel = det.get_channel(station_id, base_channels[0])
                    sampling_rate_pa = det_channel["trigger_adc_sampling_frequency"]
                    pa_window = int(PA_WINDOW_NS * units.ns * sampling_rate_pa * pa_upsampling)
                    pa_step = int(PA_STEP_NS * units.ns * sampling_rate_pa * pa_upsampling)

                    pa_vrms = post_amp_vrms[base_channels[0]]
                    final_trigger = pa_trigger_name(n_pa_channels, pa_noise_rate, layer_db=db_val)

                    phasing_angles = np.arcsin(
                        np.linspace(
                            np.sin(PA_MAIN_LOW_ANGLE),
                            np.sin(PA_MAIN_HIGH_ANGLE),
                            pa_n_beams
                        )
                    )

                    # One-time log of PA trigger parameters
                    if not _trigger_params_logged:
                        LOGGER.info("=== PA TRIGGER PARAMS (station %s) ===", station_id)
                        LOGGER.info("  channels: %s (%d ch)", base_channels, n_pa_channels)
                        LOGGER.info("  Vrms: %.6e", pa_vrms)
                        LOGGER.info("  threshold_factor: %.2f", pa_threshold)
                        LOGGER.info("  threshold_abs: %.6e (factor * Vrms^2)", pa_threshold * pa_vrms ** 2)
                        LOGGER.info("  n_beams: %d", pa_n_beams)
                        LOGGER.info("  upsampling: %d", pa_upsampling)
                        LOGGER.info("  ADC sampling rate: %s", sampling_rate_pa)
                        LOGGER.info("  window: %d samples, step: %d samples", pa_window, pa_step)
                        LOGGER.info("  trigger_name: %s", final_trigger)
                        LOGGER.info("  phasing_angles (deg): %s", np.rad2deg(phasing_angles).round(2))
                        _trigger_params_logged = True

                    if add_noise:
                        channelGenericNoiseAdder.run(evt, station, det, type="rayleigh", amplitude=pre_amp_vrms)

                    phasedArrayTrigger.run(
                        evt,
                        station,
                        det,
                        Vrms=pa_vrms,
                        threshold=pa_threshold * pa_vrms ** 2,
                        triggered_channels=base_channels,
                        phasing_angles=phasing_angles,
                        ref_index=PA_REF_INDEX,
                        trigger_name=final_trigger,
                        trigger_adc=True,
                        adc_output="voltage",
                        window=pa_window,
                        step=pa_step,
                        upsampling_factor=pa_upsampling,
                    )

                    # Per-event trigger diagnostics (first 20 events per station)
                    triggered = station.has_triggered(trigger_name=final_trigger)
                    if _n_events_processed <= 20 or triggered:
                        sim_shower = evt.get_sim_shower(0)
                        energy_eV = sim_shower[shp.energy] / units.eV
                        zenith_deg = sim_shower[shp.zenith] / units.deg
                        # Get max trace voltage across PA channels for signal strength context
                        max_voltage = 0
                        for ch_id in base_channels:
                            ch_trace = station.get_channel(ch_id).get_trace()
                            max_voltage = max(max_voltage, np.max(np.abs(ch_trace)))
                        LOGGER.info(
                            "TRIGGER_DIAG event=%s stn=%d E=%.2e eV zen=%.1f° triggered=%s "
                            "max_ch_V=%.3e Vrms=%.3e ratio=%.1f",
                            evt.get_id(), station_id, energy_eV, zenith_deg, triggered,
                            max_voltage, pa_vrms, max_voltage / pa_vrms if pa_vrms > 0 else 0,
                        )

                    if station.has_triggered(trigger_name=final_trigger):
                        # Note: Skipping triggerTimeAdjuster - simulation traces are shorter than
                        # hardware expectations. Trigger decision is valid; traces just won't be
                        # time-aligned to the trigger for downstream analysis.
                        channelStopFilter.run(evt, station, det, prepend=0 * units.ns, append=0 * units.ns)
                else:
                    # Use standard high/low threshold trigger for shallow/HRA configurations
                    noise_key = format_sigma_value(noise_sigma)
                    final_key = format_sigma_value(trigger_sigma)

                    # Check if this is Gen2 shallow direct config with both down and up channels
                    has_down_channels = all(ch in base_channels for ch in GEN2_SHALLOW_DOWN_CHANNELS)
                    has_up_channels = all(ch in base_channels for ch in GEN2_SHALLOW_UP_CHANNELS)
                    is_gen2_shallow_direct = has_down_channels and has_up_channels

                    if is_gen2_shallow_direct:
                        # Gen2 shallow direct: Run separate triggers for down and up pointing LPDAs
                        # Add noise once at start so both directions see same noise
                        if add_noise:
                            if hardwareResponseIncorporator is not None:
                                hardwareResponseIncorporator.run(evt, station, det, sim_to_data=False)
                            channelGenericNoiseAdder.run(evt, station, det, type="rayleigh", amplitude=pre_amp_vrms)
                            if hardwareResponseIncorporator is not None:
                                hardwareResponseIncorporator.run(evt, station, det, sim_to_data=True)

                        channel_sets = [
                            (GEN2_SHALLOW_DOWN_CHANNELS, "down"),
                            (GEN2_SHALLOW_UP_CHANNELS, "up"),
                        ]

                        for trigger_channels, direction_suffix in channel_sets:
                            final_trigger = f"primary_{direction_suffix}_{final_key}sigma"

                            highLowThreshold.run(
                                evt,
                                station,
                                det,
                                threshold_high=thresholds[final_key]["high"],
                                threshold_low=thresholds[final_key]["low"],
                                coinc_window=COINCIDENCE_WINDOW,
                                triggered_channels=trigger_channels,
                                number_concidences=2,
                                trigger_name=final_trigger,
                            )

                        # Finalize if any trigger fired
                        if station.has_triggered():
                            # Re-begin adjuster with whichever directional trigger fired
                            for _, dsuffix in channel_sets:
                                adj_name = f"primary_{dsuffix}_{final_key}sigma"
                                if station.has_triggered(trigger_name=adj_name):
                                    triggerTimeAdjuster.begin(trigger_name=adj_name)
                                    triggerTimeAdjuster.run(evt, station, det)
                                    break
                            channelStopFilter.run(evt, station, det, prepend=0 * units.ns, append=0 * units.ns)
                    else:
                        # Standard single trigger for HRA and Gen2 shallow reflected (4 channels)
                        noise_trigger = primary_trigger_name(noise_sigma, layer_db=db_val)
                        final_trigger = primary_trigger_name(trigger_sigma, layer_db=db_val)

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

                            triggerTimeAdjuster.begin(trigger_name=final_trigger)
                            triggerTimeAdjuster.run(evt, station, det)
                            channelStopFilter.run(evt, station, det, prepend=0 * units.ns, append=0 * units.ns)
                        else:
                            # Noise pre-check failed at this dB — higher dB values will also fail
                            if _clean_spectra is not None:
                                _db_precheck_failed = True

                # ---- End of trigger logic for this db_val ----
                # Restore clean spectra for next dB iteration
                if _clean_spectra is not None:
                    for ch_id, (spec, sr) in _clean_spectra.items():
                        station.get_channel(ch_id).set_frequency_spectrum(spec.copy(), sr)

            # ---- End of dB sweep loop ----

            # Calculate SNR on post-noise traces for stations that triggered
            if station.has_triggered():
                snr_channels = base_channels
                snr_traces = [station.get_channel(ch).get_trace() for ch in snr_channels]
                if snr_traces:
                    snr_vrms = post_amp_vrms[snr_channels[0]]
                    # Deep: average top half of channels; Shallow: average top 2
                    n_avg = max(len(snr_traces) // 2, 1) if (phasedArrayTrigger is not None) else min(2, len(snr_traces))
                    station_snr = calcSNR(snr_traces, snr_vrms, n_avg=n_avg)
                    evt_snr_cache[station_id] = station_snr

            LOGGER.debug("Station %s triggered: %s", station_id, station.has_triggered())
        # End of station loop - now save event with all station triggers
        # When multi-dB is active, dB info is encoded in trigger names, not a single value
        event_layer_dB = None if layer_db_list else settings.get("layer_dB")
        rcr_event = RCREvent.from_nuradio_event(evt, layer_dB=event_layer_dB)
        for snr_sid, snr_val in evt_snr_cache.items():
            rcr_event.set_snr(snr_sid, snr_val)

        # Compute per-station average polarization angle from readCoREAS
        for stn_id in all_station_ids:
            pol_angles = readCoREAS.channel_polarization_angle.get(stn_id, {})
            if pol_angles:
                avg_pol = float(np.mean(list(pol_angles.values())))
                rcr_event.set_polarization_angle(stn_id, avg_pol)

        rcr_events.append(rcr_event)

        # Log summary for this event
        n_direct = sum(len(rcr_event.direct_triggers(t)) for t in rcr_event.all_trigger_names())
        n_reflected = sum(len(rcr_event.reflected_triggers(t)) for t in rcr_event.all_trigger_names())
        LOGGER.debug("Event %s: %d direct triggers, %d reflected triggers", evt.get_id(), n_direct, n_reflected)

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

    # Trigger summary
    n_triggered = sum(1 for e in rcr_events if e.has_any_trigger())
    LOGGER.info("=== TRIGGER SUMMARY ===")
    LOGGER.info("  Total events: %d, Triggered: %d (%.1f%%)",
                len(rcr_events), n_triggered,
                100 * n_triggered / max(len(rcr_events), 1))
    if rcr_events:
        energies = [e.energy for e in rcr_events]
        LOGGER.info("  Energy range: %.2e - %.2e eV", min(energies), max(energies))

    # Save RCREvent list
    npy_array = np.array(rcr_events, dtype=object)
    np.save(output_paths["numpy"], npy_array, allow_pickle=True)
    LOGGER.info("Saved %d event summaries to %s", len(rcr_events), output_paths["numpy"])

    if debug_enabled:
        # Collect noise stats from all stations
        noise_stats = {
            "pre_amp_vrms_per_station": pre_amp_vrms_per_station,
            "post_amp_vrms_per_station": post_amp_vrms_per_station,
        }
        write_debug_log(
            settings,
            output_paths,
            thresholds_per_station,
            noise_stats,
            rcr_events,
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
