"""Evaluate coincidence events against template collections."""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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
COINCIDENCE_SEARCH_ROOTS: Tuple[Path, ...] = (
    Path.cwd(),
    Path.cwd() / "HRAStationDataAnalysis",
    Path.cwd() / "HRAStationDataAnalysis" / "StationData",
    Path.cwd() / "HRAStationDataAnalysis" / "StationData" / "processedNumpyData",
)


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

    print(f"Loaded {len(filtered)} coincidence events from {pickle_path}")
    return filtered


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
) -> Dict[int, Dict[str, Dict[str, object]]]:
    template_groups = load_template_groups()
    if not template_groups:
        raise RuntimeError("No templates available for matching.")

    events = load_good_coincidence_events(event_ids, pickle_name=pickle_name, search_roots=search_roots)
    results = evaluate_events_against_templates(
        events,
        template_groups,
        output_root=output_root,
        trace_sampling_rate_hz=trace_sampling_rate_hz,
    )
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_evaluation(
        event_ids=args.events,
        pickle_name=args.pickle,
        search_roots=COINCIDENCE_SEARCH_ROOTS,
        output_root=args.output,
        trace_sampling_rate_hz=args.sampling,
    )


if __name__ == "__main__":
    main()
