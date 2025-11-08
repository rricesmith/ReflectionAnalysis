"""Convert Stn51 .nur files into plot and pickle summaries of their traces."""
from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.modules.channelLengthAdjuster import channelLengthAdjuster
from NuRadioReco.modules.io import NuRadioRecoio

STATION_ID = 51
SOURCE_DIR = Path(
    "/dfs8/sbarwick_lab/ariannaproject/Tingwei_liu/CR_Template/CR_NoBL_Template_final/"
)
OUTPUT_ROOT = "TemplateTesting/"
PLOT_DIR = OUTPUT_ROOT + "plots/"
PICKLE_DIR = OUTPUT_ROOT + "pickles/"


def _collect_events(nur_file: Path) -> List[Dict[str, object]]:
    """Gather trace information for every station 51 event in a .nur file."""
    reader = NuRadioRecoio.NuRadioRecoio(str(nur_file))
    cla = channelLengthAdjuster()
    cla.begin()

    events: List[Dict[str, object]] = []
    for event_index, evt in enumerate(reader.get_events()):
        if STATION_ID not in evt.get_station_ids():
            continue

        station = evt.get_station(STATION_ID)
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

        events.append(
            {
                "event_index": event_index,
                "event_id": int(evt.get_id()),
                "triggered": bool(station.has_triggered()),
                "traces": traces,
                "times": times,
                "sampling_rates": sampling_rates,
            }
        )

    return events


def _plot_events(events: List[Dict[str, object]], nur_file: Path, output_path: Path) -> None:
    """Create a multi-channel plot overlaying all traces in the provided events."""
    if not events:
        return

    channel_ids = sorted({ch for event in events for ch in event["traces"].keys()})
    fig, axes = plt.subplots(len(channel_ids), 1, figsize=(12, 3 * len(channel_ids)), sharex=True)
    if len(channel_ids) == 1:
        axes = [axes]

    for axis, channel_id in zip(axes, channel_ids):
        for event in events:
            trace = event["traces"].get(channel_id)
            times = event["times"].get(channel_id)
            if trace is None or times is None:
                continue

            color = "C0" if event["triggered"] else "0.6"
            axis.plot(times, trace, alpha=0.35, color=color)

        axis.set_ylabel(f"ch {channel_id}")

    axes[-1].set_xlabel("time [ns]")
    fig.suptitle(f"Station {STATION_ID} traces\n{nur_file.name}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(PICKLE_DIR, exist_ok=True)

    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    nur_files = sorted(SOURCE_DIR.glob("*.nur"))
    if not nur_files:
        print(f"No .nur files found in {SOURCE_DIR}")
        return

    for nur_file in nur_files:
        print(f"Processing {nur_file.name}...")
        events = _collect_events(nur_file)
        if not events:
            print(f"  No station {STATION_ID} events found; skipping")
            continue

        pickle_payload = {
            "source_file": str(nur_file),
            "station_id": STATION_ID,
            "events": events,
        }

        pickle_path = PICKLE_DIR + f"{nur_file.stem}.pkl"
        with open(pickle_path, "wb") as fout:
            pickle.dump(pickle_payload, fout)

        plot_path = PLOT_DIR + f"{nur_file.stem}.png"
        _plot_events(events, nur_file, plot_path)

        print(f"  Saved {pickle_path.name} and {plot_path.name}")


if __name__ == "__main__":
    main()
