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
# Ensure repository root is importable for sibling packages. This is
# needed when executing the loaders as scripts.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))


from DeepLearning.D00_helperFunctions import loadMultipleTemplates


RCR_SERIES: Tuple[int, ...] = (100, 200)
SIM_BL_PICKLE_ROOT = Path("TemplateTesting/BL/pickles")
CR_ARCHIVE_ROOT = Path("TemplateTesting/CRs")
PLOT_ROOT = Path("TemplateTesting/plots/templates")
DEFAULT_SAMPLING_RATE_HZ = 2e9


@dataclass
class TemplateRecord:
	template_type: str
	identifier: str
	trace: np.ndarray
	time: Optional[np.ndarray] = None
	source: Optional[Path] = None
	sampling_rate_hz: Optional[float] = None


def _ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def _ensure_1d(trace: np.ndarray) -> np.ndarray:
	arr = np.array(trace, copy=True)
	if arr.ndim == 0:
		return arr.reshape(1)
	if arr.ndim == 1:
		return arr
	reshaped = arr.reshape(arr.shape[0], -1)
	channel_index = int(np.argmax(np.max(np.abs(reshaped), axis=1)))
	return np.array(arr[channel_index], copy=True)


def _infer_sampling_rate_from_times(times: Optional[np.ndarray]) -> Optional[float]:
	if times is None or len(times) < 2:
		return None
	times_arr = np.asarray(times, dtype=float)
	diffs = np.diff(times_arr)
	finite_diffs = diffs[np.isfinite(diffs) & (diffs != 0)]
	if finite_diffs.size == 0:
		return None
	mean_dt_ns = float(np.mean(finite_diffs))
	if mean_dt_ns == 0:
		return None
	return 1e9 / mean_dt_ns


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


def _coerce_channel_mapping(obj: object) -> Optional[Dict[int, np.ndarray]]:
	if isinstance(obj, dict):
		mapping: Dict[int, np.ndarray] = {}
		for idx, (key, value) in enumerate(obj.items()):
			try:
				channel = int(key)
			except (TypeError, ValueError):
				channel = idx
			mapping[channel] = np.array(value, copy=True)
		return mapping or None
	if isinstance(obj, np.ndarray):
		if obj.dtype == object:
			try:
				return _coerce_channel_mapping(obj.item())
			except ValueError:
				return _coerce_channel_mapping(obj.tolist())
		return None
	if isinstance(obj, (list, tuple)):
		mapping = {idx: np.array(value, copy=True) for idx, value in enumerate(obj)}
		return mapping or None
	return None


def _extract_trace_from_objects(
	traces_obj: object,
	times_obj: Optional[object],
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
	channel_traces = _coerce_channel_mapping(traces_obj)
	channel_times = _coerce_channel_mapping(times_obj) if times_obj is not None else None
	if channel_traces is not None:
		try:
			trace_arr, time_arr = _select_trace_with_times(channel_traces, channel_times or {})
		except ValueError:
			pass
		else:
			if (
				time_arr is None
				and channel_times is None
				and isinstance(times_obj, np.ndarray)
				and times_obj.ndim == 1
			):
				time_arr = np.array(times_obj, copy=True)
			return trace_arr, time_arr

	arr = np.array(traces_obj, copy=True)
	try:
		trace_arr = _ensure_1d(arr)
	except Exception:
		return None

	time_arr: Optional[np.ndarray] = None
	if isinstance(times_obj, np.ndarray):
		if times_obj.ndim == 1:
			time_arr = np.array(times_obj, copy=True)
		elif times_obj.ndim > 1:
			time_arr = np.array(times_obj.reshape(times_obj.shape[0], -1)[0], copy=True)

	return trace_arr, time_arr


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
		print(f"Loading RCR series {series} via loadMultipleTemplates")
		try:
			templates = loadMultipleTemplates(series, addSingle=True, bad=False)
		except Exception as exc:  # pragma: no cover - defensive logging
			print(f"  Failed to load series {series}: {exc}")
			continue
		if not templates:
			print(f"  No templates returned for series {series}")
			continue
		for key in sorted(templates):
			trace = _ensure_1d(np.array(templates[key], copy=True))
			identifier = f"{series}_{key}"
			records.append(
				TemplateRecord(
					template_type="RCR",
					identifier=identifier,
					trace=trace,
					sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
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

		sampling_rate = _infer_sampling_rate_from_times(best_time) or DEFAULT_SAMPLING_RATE_HZ

		records.append(
			TemplateRecord(
				template_type="SimBL",
				identifier=selected_path.relative_to(SIM_BL_PICKLE_ROOT).as_posix(),
				trace=best_trace,
				time=best_time,
				source=selected_path,
				sampling_rate_hz=sampling_rate,
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
					sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
				)
			)
	return records


def _iter_event_payloads(obj: object) -> Iterable[Dict[str, object]]:
	if isinstance(obj, dict):
		yield obj
	elif isinstance(obj, (list, tuple)):
		for item in obj:
			yield from _iter_event_payloads(item)
	elif isinstance(obj, np.ndarray):
		if obj.dtype == object:
			for item in obj.flat:
				yield from _iter_event_payloads(item)
		else:
			# Non-object arrays do not carry event dictionaries.
			return


def load_cr_templates() -> List[TemplateRecord]:
	cr_dir = CR_ARCHIVE_ROOT
	if not cr_dir.is_dir():
		raise FileNotFoundError(f"CR archive directory not found: {cr_dir}")

	records: List[TemplateRecord] = []
	archive_paths = sorted(cr_dir.glob("*.npz")) + sorted(cr_dir.glob("*.npy"))
	if not archive_paths:
		print(f"No CR archive files (.npz/.npy) found in {cr_dir}")
	for archive_path in archive_paths:
		print(f"Processing CR archive {archive_path.name}")
		if archive_path.suffix.lower() == ".npz":
			with np.load(archive_path, allow_pickle=True) as npz_file:
				print(f"  Contents: {sorted(npz_file.files)}")
				event_payloads: List[Dict[str, object]] = []
				if "events" in npz_file.files:
					events_obj = npz_file["events"]
					event_payloads.extend(_iter_event_payloads(events_obj))
					print(f"  Extracted {len(event_payloads)} event payloads")

				if not event_payloads:
					traces_obj: Optional[object] = None
					times_obj: Optional[object] = None
					if "traces" in npz_file.files:
						traces_obj = npz_file["traces"]
						if "times" in npz_file.files:
							times_obj = npz_file["times"]
					elif "trace" in npz_file.files:
						traces_obj = npz_file["trace"]
						if "time" in npz_file.files:
							times_obj = npz_file["time"]

					if traces_obj is not None:
						extracted = _extract_trace_from_objects(traces_obj, times_obj)
						if extracted is None:
							print("  No usable trace content in fallback data")
							continue
						trace_arr, time_arr = extracted
						sampling_rate = _infer_sampling_rate_from_times(time_arr) or DEFAULT_SAMPLING_RATE_HZ
						records.append(
							TemplateRecord(
								template_type="CR",
								identifier=archive_path.stem,
								trace=trace_arr,
								time=time_arr,
								source=archive_path,
								sampling_rate_hz=sampling_rate,
							)
						)
						print("  Added fallback trace")
						continue

				found_any = False
				for event in event_payloads:
					traces = event.get("traces", {})  # type: ignore[assignment]
					times = event.get("times", {})  # type: ignore[assignment]
					if not isinstance(traces, dict) or not traces:
						print("    Event skipped: no trace dict")
						continue
					try:
						trace, time = _select_trace_with_times(traces, times)  # type: ignore[arg-type]
					except ValueError:
						print("    Event skipped: trace selection failed")
						continue
					identifier = f"{archive_path.stem}_event{event.get('event_id', 'unknown')}"
					sampling_rate = _infer_sampling_rate_from_times(time) or DEFAULT_SAMPLING_RATE_HZ
					records.append(
						TemplateRecord(
							template_type="CR",
							identifier=identifier,
							trace=trace,
							time=time,
							source=archive_path,
							sampling_rate_hz=sampling_rate,
						)
					)
					found_any = True
				if not found_any:
					print("  No usable events produced a template")
		else:
			arr = np.load(archive_path, allow_pickle=True)
			print(f"  Loaded array: dtype={arr.dtype}, shape={arr.shape}")
			if arr.size == 0:
				print("  Empty array; skipping")
				continue

			def _iter_traces_from_array(data: np.ndarray) -> Iterable[np.ndarray]:
				if data.dtype == object:
					for idx, item in enumerate(data.flat):
						trace = np.array(item, copy=True)
						if trace.size == 0:
							print(f"    Object entry {idx} empty; skipped")
							continue
						yield trace
				elif data.ndim >= 2:
					for idx in range(data.shape[0]):
						trace = np.array(data[idx], copy=True)
						if trace.size == 0:
							print(f"    Row {idx} empty; skipped")
							continue
						yield trace
				else:
					yield np.array(data, copy=True)

			trace_count = 0
			for idx, trace in enumerate(_iter_traces_from_array(arr)):
				trace_arr = _ensure_1d(trace)
				identifier = f"{archive_path.stem}_trace{idx:03d}"
				records.append(
					TemplateRecord(
						template_type="CR",
						identifier=identifier,
						trace=trace_arr,
						source=archive_path,
						sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
					)
				)
				trace_count += 1
			print(f"  Added {trace_count} traces from {archive_path.name}")

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
