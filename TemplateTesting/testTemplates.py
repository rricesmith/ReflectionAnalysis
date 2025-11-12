"""Load template collections and generate diagnostic plots."""

from __future__ import annotations

import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Ensure repository root is importable for sibling packages.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))


from DeepLearning.D00_helperFunctions import loadMultipleTemplates


RCR_SERIES: Tuple[str, ...] = ("100s", "200s")
SIM_BL_PICKLE_ROOT = Path("TemplateTesting/BL/pickles")
CR_PICKLE_ROOT = Path(
	"/dfs8/sbarwick_lab/ariannaproject/Tingwei_liu/TemplateTesting/Nobacklobe/pickles/"
)
PLOT_ROOT = Path("TemplateTesting/plots/templates")


@dataclass
class TemplateRecord:
	template_type: str
	identifier: str
	trace: np.ndarray
	time: Optional[np.ndarray] = None
	source: Optional[Path] = None


def _ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def _ensure_1d(trace: np.ndarray) -> np.ndarray:
	arr = np.array(trace, copy=True)
	if arr.ndim == 1:
		return arr
	reshaped = arr.reshape(arr.shape[0], -1)
	channel_index = int(np.argmax(np.max(np.abs(reshaped), axis=1)))
	return np.array(arr[channel_index], copy=True)


def _select_trace_with_times(
	traces: Dict[int, np.ndarray],
	times: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
	best_trace: Optional[np.ndarray] = None
	best_times: Optional[np.ndarray] = None
	best_amp = -np.inf
	for channel_id, trace in traces.items():
		arr = np.array(trace, copy=True)
		amp = float(np.max(np.abs(arr)))
		if amp > best_amp:
			best_amp = amp
			best_trace = arr
			best_times = np.array(times.get(channel_id), copy=True) if channel_id in times else None
	if best_trace is None:
		raise ValueError("No traces available in payload")
	return best_trace, best_times


def _sanitize_identifier(identifier: str) -> str:
	return re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier).strip("_") or "template"


def _plot_template(record: TemplateRecord, output_dir: Path) -> None:
	x_axis = record.time if record.time is not None else np.arange(record.trace.size)
	plt.figure(figsize=(10, 4))
	plt.plot(x_axis, record.trace)
	plt.title(f"{record.template_type} | {record.identifier}")
	plt.xlabel("time [ns]" if record.time is not None else "sample")
	plt.ylabel("amplitude")
	plt.tight_layout()
	filename = _sanitize_identifier(f"{record.template_type}_{record.identifier}") + ".png"
	plt.savefig(output_dir / filename, dpi=200)
	plt.close()


def load_rcr_templates() -> List[TemplateRecord]:
	records: List[TemplateRecord] = []
	for series in RCR_SERIES:
		templates = loadMultipleTemplates(series, addSingle=True, bad=False)
		for key in sorted(templates):
			trace = _ensure_1d(np.array(templates[key], copy=True))
			identifier = f"{series}_{key}"
			records.append(
				TemplateRecord(
					template_type="RCR",
					identifier=identifier,
					trace=trace,
				)
			)
	return records


def load_sim_bl_templates() -> List[TemplateRecord]:
	if not SIM_BL_PICKLE_ROOT.is_dir():
		raise FileNotFoundError(f"Sim BL pickle directory not found: {SIM_BL_PICKLE_ROOT}")

	sigma_pattern = re.compile(r"_sigma(\d+)", re.IGNORECASE)
	records: List[TemplateRecord] = []

	for group_dir in sorted(p for p in SIM_BL_PICKLE_ROOT.iterdir() if p.is_dir()):
		pickle_paths = list(group_dir.glob("*.pkl"))
		if not pickle_paths:
			continue

		def sigma_value(path: Path) -> int:
			match = sigma_pattern.search(path.stem)
			if match:
				return int(match.group(1))
			raise ValueError(f"Could not parse sigma from filename {path.name}")

		try:
			selected_path = max(pickle_paths, key=sigma_value)
		except ValueError as exc:
			print(f"Skipping directory {group_dir.name}: {exc}")
			continue

		with selected_path.open("rb") as fin:
			payload = pickle.load(fin)

		stations: Dict[int, Dict[str, Dict[int, np.ndarray]]] = payload.get("stations", {})
		if not stations:
			print(f"No station data in {selected_path}")
			continue

		best_trace: Optional[np.ndarray] = None
		best_time: Optional[np.ndarray] = None
		best_amp = -np.inf
		for station_info in stations.values():
			traces = station_info.get("traces", {})
			times = station_info.get("times", {})
			try:
				trace, time = _select_trace_with_times(traces, times)
			except ValueError:
				continue
			amp = float(np.max(np.abs(trace)))
			if amp > best_amp:
				best_amp = amp
				best_trace = trace
				best_time = time

		if best_trace is None:
			print(f"No usable traces found in {selected_path}")
			continue

		records.append(
			TemplateRecord(
				template_type="SimBL",
				identifier=selected_path.relative_to(SIM_BL_PICKLE_ROOT).as_posix(),
				trace=best_trace,
				time=best_time,
				source=selected_path,
			)
		)

	return records


def load_data_bl_templates() -> List[TemplateRecord]:
	records: List[TemplateRecord] = []
	seen_keys: Set[str] = set()
	for series in ("200", "100"):
		templates = loadMultipleTemplates(series, date="2016", addSingle=True, bad=False)
		for key in sorted(templates):
			identifier = str(key)
			if identifier in seen_keys:
				continue
			seen_keys.add(identifier)
			trace = _ensure_1d(np.array(templates[key], copy=True))
			records.append(
				TemplateRecord(
					template_type="DataBL",
					identifier=identifier,
					trace=trace,
				)
			)
	return records


def load_cr_templates() -> List[TemplateRecord]:
	cr_dir = CR_PICKLE_ROOT
	if not cr_dir.is_dir():
		raise FileNotFoundError(f"CR pickle directory not found: {cr_dir}")

	records: List[TemplateRecord] = []
	for pickle_path in sorted(cr_dir.glob("*.pkl")):
		with pickle_path.open("rb") as fin:
			payload = pickle.load(fin)

		events: Iterable[Dict[str, object]] = payload.get("events", [])  # type: ignore[assignment]
		found_any = False
		for event in events:
			traces = event.get("traces", {})  # type: ignore[assignment]
			times = event.get("times", {})  # type: ignore[assignment]
			if not isinstance(traces, dict) or not traces:
				continue
			try:
				trace, time = _select_trace_with_times(traces, times)  # type: ignore[arg-type]
			except ValueError:
				continue
			identifier = f"{pickle_path.stem}_event{event.get('event_id', 'unknown')}"
			records.append(
				TemplateRecord(
					template_type="CR",
					identifier=identifier,
					trace=trace,
					time=time,
					source=pickle_path,
				)
			)
			found_any = True
		if not found_any:
			print(f"No usable events in {pickle_path}")

	return records


def main() -> Dict[str, List[TemplateRecord]]:
	_ensure_dir(PLOT_ROOT)

	loaders = (
		load_rcr_templates,
		load_sim_bl_templates,
		load_data_bl_templates,
		load_cr_templates,
	)

	grouped_records: Dict[str, List[TemplateRecord]] = {}
	for loader in loaders:
		records = loader()
		if not records:
			print(f"Loader {loader.__name__} returned no templates")
		for record in records:
			grouped_records.setdefault(record.template_type, []).append(record)
			_plot_template(record, PLOT_ROOT)

	for template_type, records in grouped_records.items():
		print(f"Loaded {len(records)} templates for {template_type}")

	return grouped_records


if __name__ == "__main__":
	main()
