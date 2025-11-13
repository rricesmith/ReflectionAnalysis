"""Utilities to cross-correlate templates with coincidence event traces."""

from __future__ import annotations

import fractions
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from radiotools import helper as hp
from scipy import signal

from TemplateTesting.loadTemplates import DEFAULT_SAMPLING_RATE_HZ, TemplateRecord
from icecream import ic


CHANNEL_PAIRS: Tuple[Tuple[int, int], ...] = ((0, 2), (1, 3))
MATCH_PLOT_ROOT = Path("TemplateTesting/plots/matches")
DEFAULT_TRACE_SAMPLING_HZ = DEFAULT_SAMPLING_RATE_HZ


def _sanitize_identifier(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(text)).strip("_") or "value"


def _prepare_trace(trace: object) -> Optional[np.ndarray]:
    if trace is None:
        return None
    arr = np.asarray(trace, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def _compute_times_ns(length: int, sampling_rate_hz: float) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=float)
    if sampling_rate_hz <= 0:
        spacing = 1.0
    else:
        spacing = 1e9 / float(sampling_rate_hz)
    return np.arange(length, dtype=float) * spacing


def _align_template_to_trace(template: np.ndarray, lag_samples: int, target_length: int) -> np.ndarray:
    template = np.asarray(template, dtype=float)
    if target_length <= 0 or template.size == 0:
        return np.zeros(max(target_length, 0), dtype=float)

    aligned = np.zeros(target_length, dtype=float)
    if lag_samples >= 0:
        start = min(lag_samples, target_length)
        end = min(start + template.size, target_length)
        template_slice_end = end - start
        if template_slice_end > 0:
            aligned[start:end] = template[:template_slice_end]
    else:
        template_start = min(-lag_samples, template.size)
        copy_len = min(template.size - template_start, target_length)
        if copy_len > 0:
            aligned[:copy_len] = template[template_start : template_start + copy_len]
    return aligned


def _match_sampling(ref_template: np.ndarray, resampling_factor: fractions.Fraction) -> np.ndarray:
    arr = np.asarray(ref_template, dtype=float)
    if arr.size == 0:
        return arr
    if resampling_factor.numerator != 1:
        arr = signal.resample(arr, resampling_factor.numerator * len(arr))
    if resampling_factor.denominator != 1 and len(arr) > 0:
        samples = int(len(arr) / resampling_factor.denominator)
        if samples > 0:
            arr = signal.resample(arr, samples)
    return arr


def get_xcorr_for_channel(
    orig_trace,
    template_trace,
    orig_sampling_rate,
    template_sampling_rate,
    times: Iterable[float] | np.ndarray = (),
    debug: bool = False,
    SNR: str = "n/a",
    return_details: bool = False,
):
    orig_arr = np.asarray(orig_trace, dtype=float)
    template_arr = np.asarray(template_trace, dtype=float)
    if orig_arr.ndim > 1:
        orig_arr = orig_arr.reshape(-1)
    if template_arr.ndim > 1:
        template_arr = template_arr.reshape(-1)
    if orig_arr.size == 0 or template_arr.size == 0:
        ic(times)
        times_arr = np.asarray(list(times), dtype=float) if times else None
        if return_details:
            return {
                "xcorr": 0.0,
                "xcorr_trace": np.array([], dtype=float),
                "orig_trace": orig_arr,
                "orig_trace_normalized": orig_arr,
                "resampled_template": None,
                "aligned_template": None,
                "aligned_template_scaled": None,
                "lag_samples": 0,
                "flip": 1.0,
                "scale": 1.0,
                "times_ns": times_arr,
                "template_sampling_rate": template_sampling_rate,
                "resampling_factor": fractions.Fraction(1, 1),
            }
        return 0.0

    orig_abs_max = float(np.max(np.abs(orig_arr))) if orig_arr.size else 1.0
    if orig_abs_max == 0:
        orig_abs_max = 1.0
    template_abs_max = float(np.max(np.abs(template_arr))) if template_arr.size else 1.0
    if template_abs_max == 0:
        template_abs_max = 1.0

    orig_norm = orig_arr / orig_abs_max
    template_norm = template_arr / template_abs_max
    times_arr = np.asarray(list(times), dtype=float) if times else None

    orig_binning = 1.0 / float(template_sampling_rate)
    target_binning = 1.0 / float(orig_sampling_rate)
    resampling_factor = fractions.Fraction(Decimal(orig_binning / target_binning))
    ref_template_resampled = np.asarray(_match_sampling(template_norm, resampling_factor), dtype=float)
    template_norm_max = float(np.max(np.abs(ref_template_resampled))) if ref_template_resampled.size else 1.0
    if template_norm_max == 0:
        template_norm_max = 1.0
    ref_template_resampled = ref_template_resampled / template_norm_max

    orig_norm_work = orig_norm.copy()
    orig_work = orig_arr.copy()
    times_work = times_arr.copy() if times_arr is not None else None

    if ref_template_resampled.size < orig_norm_work.size:
        full_len = int(ref_template_resampled.size)
        orig_max_idx = int(np.argmax(np.abs(orig_norm_work))) if orig_norm_work.size else 0
        cut_start = max(orig_max_idx - 50, 0)
        cut_end = cut_start + full_len
        if cut_end > orig_norm_work.size:
            cut_end = orig_norm_work.size
            cut_start = max(cut_end - full_len, 0)
        orig_norm_work = orig_norm_work[cut_start:cut_end]
        orig_work = orig_work[cut_start:cut_end]
        if times_work is not None and times_work.size >= cut_end:
            times_work = times_work[cut_start:cut_end]
        if orig_norm_work.size != ref_template_resampled.size and orig_norm_work.size > 0:
            ref_template_resampled = signal.resample(ref_template_resampled, orig_norm_work.size)

    if orig_norm_work.size == 0 or ref_template_resampled.size == 0:
        if return_details:
            return {
                "xcorr": 0.0,
                "xcorr_trace": np.array([], dtype=float),
                "orig_trace": orig_work,
                "orig_trace_normalized": orig_norm_work,
                "resampled_template": ref_template_resampled,
                "aligned_template": None,
                "aligned_template_scaled": None,
                "lag_samples": 0,
                "flip": 1.0,
                "scale": float(np.max(np.abs(orig_work))) if orig_work.size else 1.0,
                "times_ns": times_work,
                "template_sampling_rate": template_sampling_rate,
                "resampling_factor": resampling_factor,
            }
        return 0.0

    xcorr_trace = np.asarray(hp.get_normalized_xcorr(orig_norm_work, ref_template_resampled), dtype=float)
    xcorrpos = int(np.argmax(np.abs(xcorr_trace))) if xcorr_trace.size else 0
    xcorr = float(xcorr_trace[xcorrpos]) if xcorr_trace.size else 0.0

    if not (debug or return_details):
        return xcorr

    full_corr = signal.correlate(orig_norm_work, ref_template_resampled, mode="full")
    if full_corr.size:
        lags = signal.correlation_lags(len(orig_norm_work), len(ref_template_resampled), mode="full")
        lag_samples = int(lags[int(np.argmax(np.abs(full_corr)))])
    else:
        lag_samples = 0

    aligned_template = _align_template_to_trace(ref_template_resampled, lag_samples, len(orig_norm_work))
    flip = np.sign(xcorr) if xcorr != 0 else 1.0
    scale = float(np.max(np.abs(orig_work))) if orig_work.size else 1.0
    if scale == 0:
        scale = 1.0
    aligned_abs = float(np.max(np.abs(aligned_template))) if aligned_template.size else 1.0
    if aligned_abs == 0:
        aligned_abs = 1.0
    aligned_template_scaled = flip * aligned_template * (scale / aligned_abs)

    if times_work is None or len(times_work) != len(orig_work):
        times_work = _compute_times_ns(len(orig_work), orig_sampling_rate)

    if debug:
        plot_trace = orig_norm_work
        plt.figure(figsize=(8, 4))
        plt.plot(plot_trace, label="Measured")
        plt.plot(xcorr_trace, label="Template")
        plt.xlabel("samples")
        plt.legend()
        plt.title(f"Xcorr {xcorr}, {SNR}SNR")
        plt.show()

    if not return_details:
        return xcorr

    details = {
        "xcorr": xcorr,
        "xcorr_trace": xcorr_trace,
        "orig_trace": orig_work,
        "orig_trace_normalized": orig_norm_work,
        "resampled_template": ref_template_resampled,
        "aligned_template": aligned_template,
        "aligned_template_scaled": aligned_template_scaled,
        "lag_samples": lag_samples,
        "flip": flip,
        "scale": scale,
        "times_ns": np.asarray(times_work, dtype=float) if times_work is not None else None,
        "template_sampling_rate": template_sampling_rate,
        "resampling_factor": resampling_factor,
    }
    return details


def _evaluate_template_for_trigger(
    trigger_traces: Iterable[object],
    template: TemplateRecord,
    orig_sampling_rate: float,
) -> Optional[Dict[str, object]]:
    if trigger_traces is None:
        sequence: List[object] = []
    else:
        try:
            sequence = list(trigger_traces)
        except TypeError:
            sequence = []

    channel_results: Dict[int, Dict[str, object]] = {}
    for ch_idx, trace in enumerate(sequence):
        arr = _prepare_trace(trace)
        if arr is None:
            continue
        times_ns = _compute_times_ns(len(arr), orig_sampling_rate)
        template_sampling = template.sampling_rate_hz or orig_sampling_rate
        details = get_xcorr_for_channel(
            arr,
            template.trace,
            orig_sampling_rate,
            template_sampling,
            times=times_ns,
            return_details=True,
        )
        if not isinstance(details, dict):
            continue
        channel_results[ch_idx] = details

    if not channel_results:
        return None

    pair_candidates: List[Tuple[float, Tuple[int, int]]] = []
    for pair in CHANNEL_PAIRS:
        if all(ch in channel_results for ch in pair):
            pair_candidates.append(
                (
                    float(np.mean([abs(channel_results[ch]["xcorr"]) for ch in pair])),
                    pair,
                )
            )

    if pair_candidates:
        best_score, best_pair = max(pair_candidates, key=lambda item: item[0])
    else:
        best_score = float(np.mean([abs(info["xcorr"]) for info in channel_results.values()]))
        best_pair = tuple(sorted(channel_results.keys()))

    return {
        "channel_results": channel_results,
        "score": best_score,
        "best_pair": best_pair,
    }


def _plot_template_match(
    event_id: int,
    station_id: object,
    trigger_idx: int,
    template_type: str,
    template: TemplateRecord,
    match_details: Dict[str, object],
    output_root: Path,
) -> Path:
    output_dir = Path(output_root) / template_type / f"event_{event_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes_flat = axes.flatten()
    channel_results: Dict[int, Dict[str, object]] = match_details["channel_results"]

    for idx, ax in enumerate(axes_flat):
        info = channel_results.get(idx)
        if info is None:
            ax.axis("off")
            continue
        times_ns = info.get("times_ns")
        if times_ns is None or len(times_ns) != len(info["orig_trace"]):
            times_ns = _compute_times_ns(len(info["orig_trace"]), DEFAULT_TRACE_SAMPLING_HZ)
        ax.plot(times_ns, info["orig_trace"], label="Trace", color="C0", alpha=0.85)
        overlay = info.get("aligned_template_scaled")
        if overlay is not None:
            ax.plot(times_ns, overlay, label="Template", color="C3", linestyle="--", alpha=0.85)
        ax.set_title(f"Ch {idx} | χ={info['xcorr']:.3f}")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Amplitude")
        if idx == 0:
            ax.legend(loc="upper right")

    station_label = _sanitize_identifier(str(station_id))
    summary = ", ".join(
        f"ch{ch}:{data['xcorr']:.3f}"
        for ch, data in sorted(channel_results.items())
    )
    fig.suptitle(
        f"Event {event_id} | Station {station_label} | Trigger {trigger_idx} | {template_type} match: {template.identifier}\n"
        f"Best pair {match_details['best_pair']} avg χ={match_details['score']:.3f}",
        fontsize=14,
    )
    fig.text(0.5, 0.02, summary, ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    filename = (
        f"event{event_id}_st{station_label}_tr{trigger_idx:02d}_{_sanitize_identifier(template.identifier)}.png"
    )
    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def evaluate_events_against_templates(
    events: Dict[int, Dict[str, object]],
    template_groups: Dict[str, List[TemplateRecord]],
    output_root: Path = MATCH_PLOT_ROOT,
    trace_sampling_rate_hz: float = DEFAULT_TRACE_SAMPLING_HZ,
) -> Dict[int, Dict[str, Dict[str, object]]]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results: Dict[int, Dict[str, Dict[str, object]]] = {}

    for raw_event_id, event_details in events.items():
        try:
            event_id = int(raw_event_id)
        except (TypeError, ValueError):
            event_id = raw_event_id

        if not isinstance(event_details, dict):
            continue
        stations = event_details.get("stations", {})
        if not isinstance(stations, dict):
            continue

        event_matches: Dict[str, Dict[str, object]] = {}
        for template_type, templates in template_groups.items():
            if not templates:
                continue
            best_candidate: Optional[Dict[str, object]] = None
            for template in templates:
                for station_key, station_data in stations.items():
                    if not isinstance(station_data, dict):
                        continue
                    traces_list = station_data.get("Traces", [])
                    try:
                        traces_iterable = list(traces_list)
                    except TypeError:
                        continue
                    for trigger_idx, trigger_traces in enumerate(traces_iterable):
                        match = _evaluate_template_for_trigger(trigger_traces, template, trace_sampling_rate_hz)
                        if match is None:
                            continue
                        candidate = {
                            "event_id": event_id,
                            "station_id": station_key,
                            "trigger_idx": trigger_idx,
                            "template": template,
                            "template_type": template_type,
                            "score": match["score"],
                            "best_pair": match["best_pair"],
                            "channel_results": match["channel_results"],
                        }
                        if best_candidate is None or candidate["score"] > best_candidate["score"]:
                            best_candidate = candidate

            if best_candidate is None:
                print(f"Event {event_id}: no usable matches for {template_type}")
                continue

            plot_path = _plot_template_match(
                event_id,
                best_candidate["station_id"],
                best_candidate["trigger_idx"],
                template_type,
                best_candidate["template"],
                best_candidate,
                output_root,
            )
            best_candidate["plot_path"] = plot_path
            channel_chi = {ch: data["xcorr"] for ch, data in best_candidate["channel_results"].items()}
            print(
                f"Event {event_id} | {template_type}: template {best_candidate['template'].identifier} | "
                f"station {best_candidate['station_id']} trigger {best_candidate['trigger_idx']} | "
                f"best pair {best_candidate['best_pair']} -> avg χ={best_candidate['score']:.3f} | per-channel {channel_chi}"
            )
            event_matches[template_type] = best_candidate

        if event_matches:
            results[event_id] = event_matches

    return results



