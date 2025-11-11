"""Extract triggered station traces per sigma and persist plots/pickles."""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.modules.channelLengthAdjuster import channelLengthAdjuster
from NuRadioReco.modules.io import NuRadioRecoio

TARGET_STATIONS = (13, 17)
TRIGGER_SIGMAS = [50, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
FILE_PREFIX = "HRA_NoiseFalse"
SOURCE_DIR = Path(
	"/dfs8/sbarwick_lab/ariannaproject/rricesmi/HRASimulations/9.29.25/"
)
OUTPUT_ROOT = Path("TemplateTesting/BL/")
PLOT_DIR = OUTPUT_ROOT / "plots"
PICKLE_DIR = OUTPUT_ROOT / "pickles"

GROUP_PATTERN = re.compile(r"(files\d+-\d+)")
PART_PATTERN = re.compile(r"_part(\d+)\.nur$", re.IGNORECASE)


def _parse_group_and_part(nur_file: Path) -> Tuple[str, int]:
	"""Return (group_key, part_number) parsed from the filename."""

	name = nur_file.name
	group_match = GROUP_PATTERN.search(name)
	if not group_match:
		raise ValueError(f"Could not determine files substring for {name}")
	group_key = group_match.group(1)

	part_match = PART_PATTERN.search(name)
	part_number = int(part_match.group(1)) if part_match else 0
	return group_key, part_number


def _group_nur_files(nur_files: List[Path]) -> Dict[str, List[Tuple[int, Path]]]:
	"""Organize files by their shared 'filesA-B' substring."""

	grouped: Dict[str, List[Tuple[int, Path]]] = {}
	for file_path in nur_files:
		try:
			group_key, part_number = _parse_group_and_part(file_path)
		except ValueError as exc:
			print(f"Skipping {file_path.name}: {exc}")
			continue
		grouped.setdefault(group_key, []).append((part_number, file_path))

	for key in grouped:
		grouped[key].sort(key=lambda item: item[0])
	return grouped


def _collect_sigma_events_in_file(
	nur_file: Path,
	group_key: str,
	part_number: int,
	pending_sigmas: Set[int],
) -> Dict[int, Dict[str, object]]:
	"""Scan a .nur file for pending sigmas and return the first hits."""

	if not pending_sigmas:
		return {}

	reader = NuRadioRecoio.NuRadioRecoio(str(nur_file))
	cla = channelLengthAdjuster()
	cla.begin()

	found: Dict[int, Dict[str, object]] = {}
	for event_index, evt in enumerate(reader.get_events()):
		stations_data: Dict[int, Dict[str, object]] = {}
		for station_id in TARGET_STATIONS:
			if station_id not in evt.get_station_ids():
				continue

			station = evt.get_station(station_id)
			if station is None:
				continue

			cla.run(evt, station)

			traces: Dict[int, np.ndarray] = {}
			times: Dict[int, np.ndarray] = {}
			sampling_rates: Dict[int, float] = {}

			for channel in station.iter_channels():
				channel_id = channel.get_id()
				traces[channel_id] = np.array(channel.get_trace(), copy=True)
				times[channel_id] = np.array(channel.get_times(), copy=True)
				sampling_rates[channel_id] = float(channel.get_sampling_rate())

			triggered_sigmas = [
				sigma
				for sigma in TRIGGER_SIGMAS
				if station.has_trigger(f"primary_LPDA_2of4_{sigma}sigma") and station.has_triggered(f"primary_LPDA_2of4_{sigma}sigma")
			]

			stations_data[station_id] = {
				"traces": traces,
				"times": times,
				"sampling_rates": sampling_rates,
				"triggered_sigmas": triggered_sigmas,
			}

		if not stations_data:
			continue

		hit_sigmas = [
			sigma
			for sigma in TRIGGER_SIGMAS
			if sigma in pending_sigmas
			and any(
				sigma in station_info["triggered_sigmas"]
				for station_info in stations_data.values()
			)
		]

		for sigma in hit_sigmas:
			triggering_stations = [
				station_id
				for station_id, station_info in stations_data.items()
				if sigma in station_info["triggered_sigmas"]
			]

			event_payload = {
				"sigma": sigma,
				"group_key": group_key,
				"part_number": part_number,
				"source_file": str(nur_file),
				"event_index": event_index,
				"event_id": int(evt.get_id()),
				"triggering_stations": triggering_stations,
				"stations": stations_data,
			}

			found[sigma] = event_payload
			pending_sigmas.remove(sigma)
			print(
				f"    Found sigma {sigma} in event {event_payload['event_id']} "
				f"(stations {triggering_stations})"
			)

		if not pending_sigmas:
			break

	close_reader = getattr(reader, "close", None)
	if callable(close_reader):
		close_reader()

	return found


def _plot_sigma_event(event_payload: Dict[str, object], output_path: Path) -> None:
	"""Plot all available station traces for a recorded sigma event."""

	stations_data: Dict[int, Dict[str, object]] = event_payload["stations"]  # type: ignore[assignment]
	sigma = event_payload["sigma"]  # type: ignore[assignment]
	total_channels = sum(len(info["traces"]) for info in stations_data.values())
	if total_channels == 0:
		return

	fig, axes = plt.subplots(total_channels, 1, figsize=(12, 2.5 * total_channels))
	if total_channels == 1:
		axes = [axes]  # type: ignore[assignment]

	subplot_index = 0
	for station_id in sorted(stations_data):
		station_info = stations_data[station_id]
		is_triggered = sigma in station_info["triggered_sigmas"]
		color = "C0" if is_triggered else "0.6"

		for channel_id in sorted(station_info["traces"]):
			axis = axes[subplot_index]
			axis.plot(
				station_info["times"][channel_id],
				station_info["traces"][channel_id],
				color=color,
			)
			axis.set_ylabel(f"st {station_id}\nch {channel_id}")
			subplot_index += 1

	axes[-1].set_xlabel("time [ns]")
	triggering = event_payload["triggering_stations"]  # type: ignore[assignment]
	triggering_label = ", ".join(str(station_id) for station_id in triggering) or "none"
	fig.suptitle(
		f"Group {event_payload['group_key']} sigma {sigma}\n"
		f"event {event_payload['event_id']} (stations {triggering_label})"
	)
	fig.tight_layout(rect=[0, 0, 1, 0.94])
	fig.savefig(output_path, dpi=200)
	plt.close(fig)


def _write_outputs(group_key: str, events_by_sigma: Dict[int, Dict[str, object]]) -> None:
	"""Persist collected payloads to plot and pickle files."""

	if not events_by_sigma:
		print(f"  No triggered events recorded for group {group_key}")
		return

	plot_dir = PLOT_DIR / group_key
	pickle_dir = PICKLE_DIR / group_key
	plot_dir.mkdir(parents=True, exist_ok=True)
	pickle_dir.mkdir(parents=True, exist_ok=True)

	for sigma in TRIGGER_SIGMAS:
		event_payload = events_by_sigma.get(sigma)
		if event_payload is None:
			continue

		base_name = f"{group_key}_sigma{sigma}"
		pickle_path = pickle_dir / f"{base_name}.pkl"
		with open(pickle_path, "wb") as fout:
			pickle.dump(event_payload, fout)

		plot_path = plot_dir / f"{base_name}.png"
		_plot_sigma_event(event_payload, plot_path)

		print(f"  Saved sigma {sigma}: {pickle_path.name}, {plot_path.name}")


def main() -> None:
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	PLOT_DIR.mkdir(parents=True, exist_ok=True)
	PICKLE_DIR.mkdir(parents=True, exist_ok=True)

	if not SOURCE_DIR.is_dir():
		raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

	nur_files = sorted(SOURCE_DIR.glob(f"{FILE_PREFIX}*.nur"))
	if not nur_files:
		print(f"No .nur files with prefix {FILE_PREFIX} found in {SOURCE_DIR}")
		return

	grouped_files = _group_nur_files(nur_files)
	if not grouped_files:
		print("No valid file groups found.")
		return

	for group_key in sorted(grouped_files):
		parts = grouped_files[group_key]
		pending_sigmas: Set[int] = set(TRIGGER_SIGMAS)
		events_by_sigma: Dict[int, Dict[str, object]] = {}

		print(f"Processing group {group_key} ({len(parts)} file parts)")
		for part_number, file_path in parts:
			if not pending_sigmas:
				break

			print(f"  Reading {file_path.name}")
			new_events = _collect_sigma_events_in_file(
				file_path,
				group_key,
				part_number,
				pending_sigmas,
			)
			events_by_sigma.update(new_events)

		if pending_sigmas:
			missing = ", ".join(str(sigma) for sigma in sorted(pending_sigmas, reverse=True))
			print(f"  Missing sigmas after scanning group {group_key}: {missing}")
		else:
			print(f"  Found all sigmas for group {group_key}")

		_write_outputs(group_key, events_by_sigma)


if __name__ == "__main__":
	main()
