"""Evaluate coincidence events against template collections."""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


# Ensure repository root is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

try:
    load_templates_module = importlib.import_module("TemplateTesting.loadTemplates")
except ModuleNotFoundError:  # pragma: no cover - fallback when run inside package context
    load_templates_module = importlib.import_module("loadTemplates")

TemplateRecord = load_templates_module.TemplateRecord
load_cr_templates = load_templates_module.load_cr_templates
load_data_bl_templates = load_templates_module.load_data_bl_templates
load_rcr_templates = load_templates_module.load_rcr_templates
load_sim_bl_templates = load_templates_module.load_sim_bl_templates

from templateCrossCorr import (
    MATCH_PLOT_ROOT,
    DEFAULT_TRACE_SAMPLING_HZ,
    evaluate_events_against_templates,
    plot_snr_chi_summary,
)

DEFAULT_GOOD_EVENT_PICKLE = "9.24.25_CoincidenceDatetimes_passing_cuts_with_all_params_recalcZenAzi_calcPol.pkl"
DEFAULT_GOOD_EVENT_IDS: Tuple[int, ...] = (
    3047,
    3432,
    10195,
    10231,
    10273,
    10284,
    10444,
    10449,
    10466,
    10471,
    10554,
    11197,
    11220,
    11230,
    11236,
    11243,
)
SPECIAL_EVENT_IDS: Tuple[int, ...] = (
    11230,
    11243,
)
SPECIAL_EVENT_STATION_CATEGORIES: Dict[int, Dict[str, str]] = {
    11230: {
        "13": "RCR",
        "17": "Backlobe",
    },
    11243: {
        "30": "RCR",
        "17": "Backlobe",
    },
}
COINCIDENCE_SEARCH_ROOTS: Tuple[Path, ...] = (
    Path.cwd(),
    Path.cwd() / "HRAStationDataAnalysis",
    Path.cwd() / "HRAStationDataAnalysis" / "StationData",
    Path.cwd() / "HRAStationDataAnalysis" / "StationData" / "processedNumpyData",
)
STN51_EVENTS_DIR = Path("TemplateTesting/Stn51Events")
STN51_SAMPLING_RATE_HZ = 1e9
COINCIDENCE_PLOT_ROOT = MATCH_PLOT_ROOT / "Coincidence"
STN51_PLOT_ROOT = MATCH_PLOT_ROOT / "Stn51"


def _find_file_recursive(filename: str, search_roots: Iterable[Path]) -> Optional[Path]:
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        candidate = root_path / filename
        if candidate.is_file():
            return candidate
        for path in root_path.rglob(filename):
            if path.is_file():
                return path
    return None


def load_good_coincidence_events(
    event_ids: Optional[Iterable[int]] = None,
    pickle_name: str = DEFAULT_GOOD_EVENT_PICKLE,
    search_roots: Iterable[Path] = COINCIDENCE_SEARCH_ROOTS,
) -> Dict[int, Dict[str, object]]:
    """Locate and load coincidence events, optionally filtering by ID."""
    search_root_tuple = tuple(Path(root) for root in search_roots)
    pickle_path = _find_file_recursive(pickle_name, search_root_tuple)
    if pickle_path is None:
        raise FileNotFoundError(f"Could not find '{pickle_name}' under {[str(r) for r in search_root_tuple]}")

    with open(pickle_path, "rb") as fin:
        payload = pickle.load(fin)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {pickle_path}, got {type(payload).__name__}")

    data = payload

    if event_ids is None:
        target_ids = None
    else:
        target_ids = {int(eid) for eid in event_ids}

    filtered: Dict[int, Dict[str, object]] = {}
    missing: List[int] = []

    for key, value in data.items():
        try:
            event_id = int(key)
        except (TypeError, ValueError):
            continue
        if target_ids is not None and event_id not in target_ids:
            continue
        filtered[event_id] = value

    if target_ids is not None:
        missing = sorted(target_ids - set(filtered.keys()))
    if missing:
        print(f"Warning: missing {len(missing)} events in {pickle_path}: {missing}")

    print(f"Loaded {len(filtered)} backlobe events from {pickle_path}")
    return filtered


def _collect_trace_candidates(obj: object) -> List[np.ndarray]:
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            candidates: List[np.ndarray] = []
            for item in obj.flat:
                candidates.extend(_collect_trace_candidates(item))
            return candidates
        if obj.ndim == 1:
            return [obj]
        if obj.ndim >= 2 and obj.shape[0] <= 8:
            return [obj[idx] for idx in range(obj.shape[0])]
        return [obj.reshape(-1)]
    if isinstance(obj, (list, tuple)):
        candidates = []
        for item in obj:
            candidates.extend(_collect_trace_candidates(item))
        return candidates
    return [np.array(obj)]


def _extract_traces_from_npz(path: Path) -> List[np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        payload_candidates: List[np.ndarray] = []
        for key in ("traces", "Traces", "signals", "Signals", "data", "Data"):
            if key in archive.files:
                payload_candidates = _collect_trace_candidates(archive[key])
                break
        else:
            for key in sorted(archive.files):
                payload_candidates.extend(_collect_trace_candidates(archive[key]))

    traces: List[np.ndarray] = []
    for candidate in payload_candidates:
        arr = np.array(candidate, dtype=float, copy=True)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        if arr.size == 0:
            continue
        traces.append(arr)
    return traces


def _calc_trace_snr(traces: Iterable[object], vrms: float) -> Optional[float]:
    if vrms <= 0:
        return None
    snr_values: List[float] = []
    for trace in traces or []:
        arr = np.asarray(trace, dtype=float).reshape(-1)
        if arr.size == 0:
            continue
        peak_to_peak = float(np.max(arr) - np.min(arr))
        snr_values.append(peak_to_peak / (2.0 * vrms))
    if not snr_values:
        return None
    snr_values.sort(reverse=True)
    if len(snr_values) == 1:
        return snr_values[0]
    return 0.5 * (snr_values[0] + snr_values[1])


def _compute_event_snr(
    event_payload: Dict[str, object],
    vrms: float,
) -> Tuple[Optional[float], Dict[str, float]]:
    stations = event_payload.get("stations", {}) if isinstance(event_payload, dict) else {}
    if not isinstance(stations, dict):
        return None, {}
    best_snr: Optional[float] = None
    station_snrs: Dict[str, float] = {}
    for station_key, station_data in stations.items():
        if not isinstance(station_data, dict):
            continue
        traces_collection = station_data.get("Traces", [])
        try:
            triggers = list(traces_collection)
        except TypeError:
            continue
        station_best: Optional[float] = None
        for trigger_traces in triggers:
            if trigger_traces is None:
                continue
            try:
                traces_iter = list(trigger_traces)
            except TypeError:
                traces_iter = [trigger_traces]
            snr_value = _calc_trace_snr(traces_iter, vrms)
            if snr_value is None:
                continue
            if station_best is None or snr_value > station_best:
                station_best = snr_value
        if station_best is None:
            continue
        station_label = str(station_key)
        station_snrs[station_label] = station_best
        if best_snr is None or station_best > best_snr:
            best_snr = station_best
    return best_snr, station_snrs


def load_station51_events(directory: Path = STN51_EVENTS_DIR) -> Dict[str, Dict[str, object]]:
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Stn51 directory not found: {dir_path}")
        return {}

    events: Dict[str, Dict[str, object]] = {}
    for idx, npz_path in enumerate(sorted(dir_path.glob("*.npz"))):
        try:
            traces = _extract_traces_from_npz(npz_path)
        except Exception as exc:
            print(f"Failed to load {npz_path.name}: {exc}")
            continue
        if not traces:
            print(f"No traces extracted from {npz_path.name}")
            continue

        event_id = f"stn51_{npz_path.stem}"
        event_payload = {
            "stations": {
                "Stn51": {
                    "Traces": [traces],
                    "sampling_rate_hz": STN51_SAMPLING_RATE_HZ,
                }
            },
            "plot_root": STN51_PLOT_ROOT,
            "category": "Station 51",
        }
        if event_id in events:
            suffix = 1
            while f"{event_id}_{suffix}" in events:
                suffix += 1
            event_id = f"{event_id}_{suffix}"
        events[event_id] = event_payload

    print(f"Loaded {len(events)} Stn51 events from {dir_path}")
    return events


def load_template_groups() -> Dict[str, List[TemplateRecord]]:
    groups = {
        "RCR": load_rcr_templates(),
        "SimBL": load_sim_bl_templates(),
        "DataBL": load_data_bl_templates(),
        "CR": load_cr_templates(),
    }
    return {key: value for key, value in groups.items() if value}


def run_evaluation(
    event_ids: Optional[Iterable[int]] = DEFAULT_GOOD_EVENT_IDS,
    pickle_name: str = DEFAULT_GOOD_EVENT_PICKLE,
    search_roots: Iterable[Path] = COINCIDENCE_SEARCH_ROOTS,
    output_root: Path = MATCH_PLOT_ROOT,
    trace_sampling_rate_hz: float = DEFAULT_TRACE_SAMPLING_HZ,
    station51_dir: Optional[Path] = STN51_EVENTS_DIR,
) -> Dict[Union[int, str], Dict[str, Dict[str, object]]]:
    template_groups = load_template_groups()
    if not template_groups:
        raise RuntimeError("No templates available for matching.")

    base_events = load_good_coincidence_events(event_ids, pickle_name=pickle_name, search_roots=search_roots)
    events: Dict[Union[int, str], Dict[str, object]] = dict(base_events)
    if station51_dir is not None:
        station_events = load_station51_events(station51_dir)
        if station_events:
            events.update(station_events)
    special_ids = {int(eid) for eid in SPECIAL_EVENT_IDS}
    for event_id, event_payload in events.items():
        if not isinstance(event_payload, dict):
            continue
        stations = event_payload.get("stations", {})
        if isinstance(stations, dict):
            for station_data in stations.values():
                if isinstance(station_data, dict) and "sampling_rate_hz" not in station_data:
                    station_data["sampling_rate_hz"] = trace_sampling_rate_hz
        if "plot_root" not in event_payload:
            event_payload["plot_root"] = COINCIDENCE_PLOT_ROOT
        category = event_payload.get("category")
        if category is None:
            if isinstance(event_id, int) and event_id in special_ids:
                category = "RCR"
            else:
                category = "Backlobe"
            event_payload["category"] = category
        vrms = 0.01 if category == "Station 51" else 0.02
        event_payload["vrms"] = vrms
        event_snr, station_snrs = _compute_event_snr(event_payload, vrms)
        event_payload["event_snr"] = event_snr
        event_payload["station_snrs"] = station_snrs
        if isinstance(stations, dict):
            station_categories: Dict[str, str] = {}
            special_station_map: Dict[str, str] = {}
            if isinstance(event_id, int) and event_id in SPECIAL_EVENT_STATION_CATEGORIES:
                special_station_map = {
                    str(station_key): str(station_category)
                    for station_key, station_category in SPECIAL_EVENT_STATION_CATEGORIES[event_id].items()
                }
            for station_key in stations.keys():
                station_label = str(station_key)
                station_categories[station_label] = special_station_map.get(station_label, category)
            event_payload["station_categories"] = station_categories
    results = evaluate_events_against_templates(
        events,
        template_groups,
        output_root=output_root,
        trace_sampling_rate_hz=trace_sampling_rate_hz,
        prefer_secondary={"Backlobe": {"DataBL"}},
    )
    output_root_path = Path(output_root)
    summary_outputs: List[Tuple[str, Optional[Path]]] = []
    summary_outputs.append(
        (
            "Combined",
            plot_snr_chi_summary(
                results,
                output_root_path / "snr_chi_summary.png",
            ),
        )
    )
    for category in ("Backlobe", "RCR", "Station 51"):
        filename = f"snr_chi_summary_{category.lower().replace(' ', '_')}.png"
        summary_outputs.append(
            (
                category,
                plot_snr_chi_summary(
                    results,
                    output_root_path / filename,
                    category_filter=category,
                ),
            )
        )
    for label, path in summary_outputs:
        if path is not None:
            print(f"Saved {label} SNR-chi plot to {path}")
    print(f"Generated matches for {len(results)} events")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate coincidence events against template families")
    parser.add_argument(
        "--events",
        nargs="*",
        type=int,
        default=list(DEFAULT_GOOD_EVENT_IDS),
        help="Specific coincidence event IDs to evaluate (defaults to curated list)",
    )
    parser.add_argument(
        "--pickle",
        default=DEFAULT_GOOD_EVENT_PICKLE,
        help="Filename of the coincidence pickle/npz archive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MATCH_PLOT_ROOT,
        help="Directory root for match plots",
    )
    parser.add_argument(
        "--sampling",
        type=float,
        default=DEFAULT_TRACE_SAMPLING_HZ,
        help="Sampling rate for station traces in Hz",
    )
    parser.add_argument(
        "--stn51-dir",
        type=Path,
        default=STN51_EVENTS_DIR,
        help="Directory containing Stn51 .npz events",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_evaluation(
        event_ids=args.events,
        pickle_name=args.pickle,
        search_roots=COINCIDENCE_SEARCH_ROOTS,
        output_root=args.output,
        trace_sampling_rate_hz=args.sampling,
        station51_dir=args.stn51_dir,
    )


if __name__ == "__main__":
    main()
