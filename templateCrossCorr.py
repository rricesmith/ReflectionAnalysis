"""Utilities to cross-correlate templates with coincidence event traces."""

from __future__ import annotations

import fractions
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from radiotools import helper as hp
from scipy import signal

from TemplateTesting.loadTemplates import DEFAULT_SAMPLING_RATE_HZ, TemplateRecord


CHANNEL_PAIRS: Tuple[Tuple[int, int], ...] = ((0, 2), (1, 3))
MATCH_PLOT_ROOT = Path("TemplateTesting/plots/matches")
DEFAULT_TRACE_SAMPLING_HZ = DEFAULT_SAMPLING_RATE_HZ
DEFAULT_TEMPLATE_ORDER: Tuple[str, ...] = ("RCR", "SimBL", "DataBL", "CR")
EVENT_CATEGORY_COLORS: Dict[str, str] = {
    "Backlobe": "#4c72b0",
    "RCR": "#dd8452",
    "Station 51": "#55a868",
}


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
    if isinstance(times, np.ndarray):
        times_arr: Optional[np.ndarray] = np.asarray(times, dtype=float)
        if times_arr.size == 0:
            times_arr = None
    else:
        try:
            times_list = list(times)
        except TypeError:
            times_list = []
        times_arr = np.asarray(times_list, dtype=float) if times_list else None

    orig_arr = np.asarray(orig_trace, dtype=float)
    template_arr = np.asarray(template_trace, dtype=float)
    if orig_arr.ndim > 1:
        orig_arr = orig_arr.reshape(-1)
    if template_arr.ndim > 1:
        template_arr = template_arr.reshape(-1)
    if orig_arr.size == 0 or template_arr.size == 0:
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
        len_a = len(orig_norm_work)
        len_b = len(ref_template_resampled)
        lags = np.arange(-len_b + 1, len_a, dtype=int)
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

    aggregate_candidates: List[Tuple[float, Tuple[int, ...]]] = []
    for pair in CHANNEL_PAIRS:
        if all(ch in channel_results for ch in pair):
            aggregate_candidates.append(
                (
                    float(np.mean([abs(channel_results[ch]["xcorr"]) for ch in pair])),
                    pair,
                )
            )
    for ch, info in channel_results.items():
        aggregate_candidates.append((abs(info["xcorr"]), (ch,)))

    if aggregate_candidates:
        best_score, best_pair = max(aggregate_candidates, key=lambda item: (item[0], len(item[1])))
    else:
        best_score = float(np.mean([abs(info["xcorr"]) for info in channel_results.values()]))
        best_pair = tuple(sorted(channel_results.keys()))

    return {
        "channel_results": channel_results,
        "score": best_score,
        "best_pair": best_pair,
    }


def _plot_template_match(
    event_id: Union[int, str],
    station_id: object,
    trigger_idx: int,
    template_type: str,
    template: TemplateRecord,
    match_details: Dict[str, object],
    output_root: Path,
) -> Path:
    event_label = _sanitize_identifier(str(event_id))
    output_dir = Path(output_root) / template_type
    output_dir.mkdir(parents=True, exist_ok=True)

    station_sampling = match_details.get("sampling_rate_hz", DEFAULT_TRACE_SAMPLING_HZ)
    try:
        station_sampling = float(station_sampling)
    except (TypeError, ValueError):
        station_sampling = DEFAULT_TRACE_SAMPLING_HZ

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
            times_ns = _compute_times_ns(len(info["orig_trace"]), station_sampling)
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
    best_pair = tuple(match_details.get("best_pair", ()))
    if len(best_pair) == 1:
        best_label = f"Best channel {best_pair[0]}"
    elif best_pair:
        best_label = f"Best pair {best_pair}"
    else:
        best_label = "Best match"
    fig.suptitle(
        f"Event {event_id} | Station {station_label} | Trigger {trigger_idx} | {template_type} match: {template.identifier}\n"
        f"{best_label} avg χ={match_details['score']:.3f}",
        fontsize=14,
    )
    fig.text(0.5, 0.02, summary, ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    filename = (
        f"event{event_label}_st{station_label}_tr{trigger_idx:02d}_"
        f"{_sanitize_identifier(template.identifier)}.png"
    )
    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def evaluate_events_against_templates(
    events: Dict[Union[int, str], Dict[str, object]],
    template_groups: Dict[str, List[TemplateRecord]],
    output_root: Path = MATCH_PLOT_ROOT,
    trace_sampling_rate_hz: float = DEFAULT_TRACE_SAMPLING_HZ,
    prefer_secondary: Optional[Dict[str, Iterable[str]]] = None,
) -> Dict[Union[int, str], Dict[str, Dict[str, object]]]:
    default_output_root = Path(output_root)
    default_output_root.mkdir(parents=True, exist_ok=True)
    results: Dict[Union[int, str], Dict[str, Dict[str, object]]] = {}
    prefer_secondary_map: Dict[str, Set[str]] = {}
    if prefer_secondary:
        prefer_secondary_map = {
            str(category): {str(item) for item in template_types}
            for category, template_types in prefer_secondary.items()
        }

    for raw_event_id, event_details in events.items():
        try:
            event_id = int(raw_event_id)
        except (TypeError, ValueError):
            event_id = raw_event_id

        if not isinstance(event_details, dict):
            continue
        event_output_root_raw = event_details.get("plot_root", default_output_root)
        event_category = event_details.get("category", "Backlobe")
        event_snr = event_details.get("event_snr")
        raw_station_categories = event_details.get("station_categories", {})
        if isinstance(raw_station_categories, dict):
            station_categories = {str(key): str(value) for key, value in raw_station_categories.items()}
        else:
            station_categories = {}
        raw_station_snrs = event_details.get("station_snrs", {})
        station_snrs: Dict[str, float] = {}
        if isinstance(raw_station_snrs, dict):
            for key, value in raw_station_snrs.items():
                try:
                    station_snrs[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        try:
            event_output_root = Path(event_output_root_raw)
        except (TypeError, ValueError):
            event_output_root = default_output_root
        event_output_root.mkdir(parents=True, exist_ok=True)
        stations = event_details.get("stations", {})
        if not isinstance(stations, dict):
            continue

        event_matches: Dict[str, Dict[str, object]] = {}
        station_template_records: Dict[str, Dict[str, Dict[str, object]]] = {}
        for template_type, templates in template_groups.items():
            if not templates:
                continue
            candidates: List[Dict[str, object]] = []
            for template in templates:
                for station_key, station_data in stations.items():
                    if not isinstance(station_data, dict):
                        continue
                    traces_list = station_data.get("Traces", [])
                    station_sampling_rate_raw = station_data.get("sampling_rate_hz", trace_sampling_rate_hz)
                    try:
                        station_sampling_rate = float(station_sampling_rate_raw)
                    except (TypeError, ValueError):
                        station_sampling_rate = trace_sampling_rate_hz
                    try:
                        traces_iterable = list(traces_list)
                    except TypeError:
                        continue
                    for trigger_idx, trigger_traces in enumerate(traces_iterable):
                        match = _evaluate_template_for_trigger(
                            trigger_traces,
                            template,
                            station_sampling_rate,
                        )
                        if match is None:
                            continue
                        station_label = str(station_key)
                        candidates.append(
                            {
                                "event_id": event_id,
                                "station_id": station_key,
                                "trigger_idx": trigger_idx,
                                "template": template,
                                "template_type": template_type,
                                "score": match["score"],
                                "best_pair": match["best_pair"],
                                "channel_results": match["channel_results"],
                                "sampling_rate_hz": station_sampling_rate,
                                "output_root": event_output_root,
                                "station_category": station_categories.get(station_label, event_category),
                                "station_snr": station_snrs.get(station_label),
                                "station_label": station_label,
                                "event_category": event_category,
                                "event_snr": event_snr,
                            }
                        )

            if not candidates:
                print(f"Event {event_id}: no usable matches for {template_type}")
                continue

            per_station_best: Dict[str, Dict[str, object]] = {}
            for candidate in candidates:
                label = candidate.get("station_label")
                if label is None:
                    label = str(candidate.get("station_id"))
                existing = per_station_best.get(label)
                if existing is None or candidate["score"] > existing["score"]:
                    per_station_best[label] = {
                        "score": candidate["score"],
                        "station_category": candidate.get("station_category"),
                        "station_snr": candidate.get("station_snr"),
                        "station_id": candidate.get("station_id"),
                        "trigger_idx": candidate.get("trigger_idx"),
                        "template_identifier": candidate["template"].identifier,
                        "template_type": template_type,
                    }

            for station_label, station_entry in per_station_best.items():
                station_records = station_template_records.setdefault(station_label, {})
                station_records[template_type] = station_entry

            ranked_candidates = [dict(item) for item in sorted(candidates, key=lambda item: item["score"], reverse=True)]
            primary_candidate = ranked_candidates[0]
            primary_candidate["selection_rank"] = 1
            secondary_candidate = None
            if len(ranked_candidates) > 1:
                secondary_candidate = ranked_candidates[1]
                secondary_candidate["selection_rank"] = 2

            category_pref = prefer_secondary_map.get(str(event_category), set())
            use_secondary = secondary_candidate is not None and template_type in category_pref

            if use_secondary:
                selected_candidate = secondary_candidate
                selected_candidate["selection_rank"] = 2
                selected_candidate["primary_candidate"] = dict(primary_candidate)
            else:
                selected_candidate = primary_candidate
                selected_candidate["selection_rank"] = 1
                if secondary_candidate is not None:
                    selected_candidate["secondary_candidate"] = dict(secondary_candidate)

            selected_candidate["top_score"] = primary_candidate["score"]
            selected_candidate["top_template_identifier"] = primary_candidate["template"].identifier
            if secondary_candidate is not None:
                selected_candidate["second_score"] = secondary_candidate["score"]
                selected_candidate["second_template_identifier"] = secondary_candidate["template"].identifier
            else:
                selected_candidate["second_score"] = None
                selected_candidate["second_template_identifier"] = None
            selected_candidate["ranked_scores"] = [
                {
                    "rank": idx + 1,
                    "template_identifier": cand["template"].identifier,
                    "score": cand["score"],
                }
                for idx, cand in enumerate(ranked_candidates[: min(3, len(ranked_candidates))])
            ]

            plot_path = _plot_template_match(
                event_id,
                selected_candidate["station_id"],
                selected_candidate["trigger_idx"],
                template_type,
                selected_candidate["template"],
                selected_candidate,
                selected_candidate.get("output_root", event_output_root),
            )
            selected_candidate["plot_path"] = plot_path
            channel_chi = {ch: data["xcorr"] for ch, data in selected_candidate["channel_results"].items()}
            best_pair = tuple(selected_candidate.get("best_pair", ()))
            if len(best_pair) == 1:
                combo_label = f"best channel {best_pair[0]}"
            elif best_pair:
                combo_label = f"best pair {best_pair}"
            else:
                combo_label = "best match"
            rank_summary = ", ".join(
                f"rank{entry['rank']}:χ={entry['score']:.3f}" for entry in selected_candidate["ranked_scores"]
            )
            selection_label = "second-best" if selected_candidate.get("selection_rank") == 2 else "primary"
            print(
                f"Event {event_id} | {template_type}: template {selected_candidate['template'].identifier} | "
                f"station {selected_candidate['station_id']} trigger {selected_candidate['trigger_idx']} | "
                f"{combo_label} ({selection_label}) -> {rank_summary} | per-channel {channel_chi}"
            )
            event_matches[template_type] = selected_candidate

        if event_matches:
            event_matches["_meta"] = {
                "category": event_category,
                "station_snrs": station_snrs,
                "station_categories": station_categories,
                "snr": event_snr,
                "vrms": event_details.get("vrms"),
            }
            if station_template_records:
                event_matches["_station_matches"] = station_template_records
            results[event_id] = event_matches

    return results


def plot_snr_chi_summary(
    results: Dict[Union[int, str], Dict[str, Dict[str, object]]],
    output_path: Path,
    template_order: Iterable[str] = DEFAULT_TEMPLATE_ORDER,
    template_colors: Optional[Dict[str, str]] = None,
    event_markers: Optional[Dict[str, str]] = None,
    category_filter: Optional[str] = None,
    min_station_snr: Optional[float] = None,
    allowed_categories: Optional[Iterable[str]] = None,
    allowed_templates: Optional[Dict[str, Iterable[str]]] = None,
    category_label_overrides: Optional[Dict[str, str]] = None,
) -> Optional[Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if template_colors is None:
        template_colors = {
            "RCR": "#1f77b4",
            "SimBL": "#ff7f0e",
            "DataBL": "#2ca02c",
            "CR": "#d62728",
        }
    if event_markers is None:
        event_markers = {
            "Backlobe": "o",
            "RCR": "s",
            "Station 51": "^",
        }

    allowed_category_set: Optional[Set[str]] = None
    if allowed_categories is not None:
        allowed_category_set = {str(item) for item in allowed_categories}

    template_allow_map: Optional[Dict[str, Set[str]]] = None
    default_allowed_templates: Optional[Set[str]] = None
    if allowed_templates is not None:
        template_allow_map = {
            str(key): {str(name) for name in value}
            for key, value in allowed_templates.items()
        }
        default_allowed_templates = template_allow_map.get("*")

    category_label_overrides = {str(k): str(v) for k, v in (category_label_overrides or {}).items()}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")
    ax.set_xlim(3, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Station SNR")
    ax.set_ylabel(r"Template match $\chi$")
    ax.set_title("Station SNR vs $\chi$ Summary")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)

    data_plotted = False
    used_templates: Dict[str, str] = {}
    used_categories: Dict[str, str] = {}
    template_order_seq = [str(name) for name in template_order]

    for event_id, matches in results.items():
        if not isinstance(matches, dict):
            continue
        meta = matches.get("_meta", {}) if isinstance(matches, dict) else {}
        event_category = meta.get("category") or "Backlobe"
        raw_station_categories = meta.get("station_categories", {})
        if isinstance(raw_station_categories, dict):
            meta_station_categories = {
                str(key): str(value) for key, value in raw_station_categories.items()
            }
        else:
            meta_station_categories = {}
        raw_station_snrs = meta.get("station_snrs", {})
        meta_station_snrs: Dict[str, float] = {}
        if isinstance(raw_station_snrs, dict):
            for key, value in raw_station_snrs.items():
                try:
                    snr_val = float(value)
                except (TypeError, ValueError):
                    continue
                if snr_val <= 0:
                    continue
                meta_station_snrs[str(key)] = snr_val

        event_category_str = str(event_category)
        if allowed_category_set is not None and event_category_str not in allowed_category_set:
            continue

        station_match_map = matches.get("_station_matches")
        station_groups: Dict[str, Dict[str, object]] = {}

        if isinstance(station_match_map, dict) and station_match_map:
            for station_label, template_map in station_match_map.items():
                if not isinstance(template_map, dict):
                    continue
                station_label_str = str(station_label)
                base_category = meta_station_categories.get(station_label_str, event_category_str)
                station_category = str(base_category)
                base_snr = meta_station_snrs.get(station_label_str)
                snr_val: Optional[float] = None
                if base_snr is not None:
                    snr_val = float(base_snr)

                points: List[Tuple[str, float]] = []
                station_category_candidate = station_category
                for template_name in template_order_seq:
                    if template_allow_map is not None:
                        allowed_set = template_allow_map.get(event_category_str)
                        if allowed_set is None:
                            allowed_set = default_allowed_templates
                        if allowed_set is None or template_name not in allowed_set:
                            continue
                    entry = template_map.get(template_name)
                    if not isinstance(entry, dict):
                        continue
                    chi_val = float(abs(entry.get("score", 0.0)))
                    if not np.isfinite(chi_val):
                        continue
                    chi_val = max(0.0, min(chi_val, 1.0))
                    entry_category = entry.get("station_category")
                    if entry_category:
                        station_category_candidate = str(entry_category)
                    entry_snr = entry.get("station_snr")
                    entry_snr_val: Optional[float] = None
                    if entry_snr is not None:
                        try:
                            entry_snr_val = float(entry_snr)
                        except (TypeError, ValueError):
                            entry_snr_val = None
                    if entry_snr_val is not None and entry_snr_val > 0:
                        if snr_val is None or entry_snr_val > snr_val:
                            snr_val = entry_snr_val
                    target_category = station_category_candidate
                    if allowed_category_set is not None and target_category not in allowed_category_set:
                        continue
                    if category_filter is not None and target_category != category_filter:
                        continue
                    points.append((template_name, chi_val))
                    station_category = target_category

                if not points:
                    continue

                if snr_val is None or snr_val <= 0:
                    continue

                if min_station_snr is not None and snr_val < float(min_station_snr):
                    continue

                if category_filter is not None and station_category != category_filter:
                    continue

                group = station_groups.setdefault(
                    station_label_str,
                    {
                        "snr": snr_val,
                        "category": station_category,
                        "points": [],
                    },
                )
                group_points = group.setdefault("points", [])
                group_points.extend(points)
                if snr_val > group.get("snr", 0.0):
                    group["snr"] = snr_val
                if station_category and not group.get("category"):
                    group["category"] = station_category

        else:
            # Fallback to legacy behaviour if station-level matches are missing.
            for template_name in template_order_seq:
                candidate = matches.get(template_name)
                if not isinstance(candidate, dict):
                    continue
                station_id = candidate.get("station_id")
                station_label = str(station_id)
                chi_val = float(abs(candidate.get("score", 0.0)))
                if not np.isfinite(chi_val):
                    continue
                chi_val = max(0.0, min(chi_val, 1.0))
                station_category_raw = candidate.get("station_category")
                if not station_category_raw:
                    station_category_raw = meta_station_categories.get(station_label, event_category_str)
                station_category = str(station_category_raw)
                station_snr = candidate.get("station_snr")
                if station_snr is None:
                    station_snr = meta_station_snrs.get(station_label)
                try:
                    station_snr = float(station_snr)
                except (TypeError, ValueError):
                    continue
                if station_snr <= 0:
                    continue
                if min_station_snr is not None and station_snr < float(min_station_snr):
                    continue
                if template_allow_map is not None:
                    allowed_set = template_allow_map.get(event_category_str)
                    if allowed_set is None:
                        allowed_set = default_allowed_templates
                    if allowed_set is None or template_name not in allowed_set:
                        continue
                if allowed_category_set is not None and station_category not in allowed_category_set:
                    continue
                if category_filter is not None and station_category != category_filter:
                    continue

                group = station_groups.setdefault(
                    station_label,
                    {
                        "snr": float(station_snr),
                        "category": station_category,
                        "points": [],
                    },
                )
                if station_snr > group["snr"]:
                    group["snr"] = float(station_snr)
                if not group.get("category"):
                    group["category"] = station_category
                group_points = group.setdefault("points", [])
                group_points.append((template_name, chi_val))

        if not station_groups:
            continue

        for station_label, station_info in station_groups.items():
            points = station_info.get("points", [])
            if not points:
                continue
            snr_val = float(station_info.get("snr", 0.0))
            if snr_val <= 0:
                continue
            if min_station_snr is not None and snr_val < float(min_station_snr):
                continue
            station_category = str(station_info.get("category") or event_category or "Backlobe")
            if allowed_category_set is not None and station_category not in allowed_category_set:
                continue
            if category_filter is not None and station_category != category_filter:
                continue
            marker = event_markers.get(station_category, "o")
            used_categories.setdefault(station_category, marker)

            chi_vals = [chi for _, chi in points]
            if len(chi_vals) >= 2:
                sorted_chi = sorted(chi_vals)
                ax.plot(
                    [snr_val] * len(sorted_chi),
                    sorted_chi,
                    color="0.7",
                    linewidth=0.8,
                    alpha=0.8,
                    zorder=1,
                )

            for template_name, chi_val in points:
                color = template_colors.get(template_name, "#444444")
                used_templates.setdefault(template_name, color)
                ax.scatter(
                    snr_val,
                    chi_val,
                    color=color,
                    marker=marker,
                    edgecolors="none",
                    s=64,
                    zorder=2,
                )
            data_plotted = True

    if not data_plotted:
        plt.close(fig)
        return None

    template_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=8,
            label=name,
        )
        for name, color in used_templates.items()
    ]

    event_handles = []
    seen_labels: Set[str] = set()
    for category, marker in used_categories.items():
        label = category_label_overrides.get(category, category)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        handle = Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            label=label,
        )
        event_handles.append(handle)

    legend_handles = template_handles + event_handles
    if legend_handles:
        legend_labels = [handle.get_label() for handle in legend_handles]
        ax.legend(legend_handles, legend_labels, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _extract_best_scores_for_event(
    matches: Dict[str, Dict[str, object]],
    template_order_seq: List[str],
) -> Dict[str, Dict[str, float]]:
    best_by_template: Dict[str, Dict[str, float]] = {name: {} for name in template_order_seq}
    meta = matches.get("_meta", {}) if isinstance(matches, dict) else {}
    event_category = str(meta.get("category") or "Backlobe")
    raw_station_categories = meta.get("station_categories", {})
    if isinstance(raw_station_categories, dict):
        meta_station_categories = {str(key): str(value) for key, value in raw_station_categories.items()}
    else:
        meta_station_categories = {}

    station_match_map = matches.get("_station_matches")
    if isinstance(station_match_map, dict) and station_match_map:
        for station_label, template_map in station_match_map.items():
            if not isinstance(template_map, dict):
                continue
            station_label_str = str(station_label)
            station_category_default = meta_station_categories.get(station_label_str, event_category)
            for template_name in template_order_seq:
                entry = template_map.get(template_name)
                if not isinstance(entry, dict):
                    continue
                score = entry.get("score")
                if score is None:
                    continue
                try:
                    chi_val = float(abs(score))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(chi_val):
                    continue
                station_category = entry.get("station_category")
                if station_category is None:
                    station_category = station_category_default
                station_category = str(station_category or event_category)
                current = best_by_template.setdefault(template_name, {})
                existing = current.get(station_category)
                if existing is None or chi_val > existing:
                    current[station_category] = chi_val
    else:
        for template_name in template_order_seq:
            candidate = matches.get(template_name)
            if not isinstance(candidate, dict):
                continue
            score = candidate.get("score")
            if score is None:
                continue
            try:
                chi_val = float(abs(score))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(chi_val):
                continue
            station_category = candidate.get("station_category")
            if station_category is None:
                station_id = candidate.get("station_id")
                station_category = meta_station_categories.get(str(station_id), event_category)
            station_category = str(station_category or event_category)
            current = best_by_template.setdefault(template_name, {})
            existing = current.get(station_category)
            if existing is None or chi_val > existing:
                current[station_category] = chi_val
    return best_by_template


def _collect_best_template_scores(
    results: Dict[Union[int, str], Dict[str, Dict[str, object]]],
    template_order_seq: List[str],
    min_event_snr: Optional[float] = None,
) -> Tuple[
    Dict[str, Dict[str, List[float]]],
    Dict[str, List[float]],
    Dict[str, Set[Union[int, str]]],
    Set[Union[int, str]],
]:
    template_category_scores: Dict[str, Dict[str, List[float]]] = {name: {} for name in template_order_seq}
    template_overall_scores: Dict[str, List[float]] = {name: [] for name in template_order_seq}
    events_per_category: Dict[str, Set[Union[int, str]]] = {}
    overall_events: Set[Union[int, str]] = set()

    for event_id, matches in results.items():
        if not isinstance(matches, dict):
            continue
        meta = matches.get("_meta", {}) if isinstance(matches, dict) else {}
        event_snr_val: Optional[float] = None
        if isinstance(meta, dict) and "snr" in meta:
            try:
                event_snr_val = float(meta.get("snr"))
            except (TypeError, ValueError):
                event_snr_val = None
        if min_event_snr is not None:
            if event_snr_val is None or event_snr_val < float(min_event_snr):
                continue
        best_by_template = _extract_best_scores_for_event(matches, template_order_seq)
        contributed = False
        for template_name, category_map in best_by_template.items():
            if not category_map:
                continue
            best_val = max(category_map.values())
            template_overall_scores.setdefault(template_name, []).append(best_val)
            for category, value in category_map.items():
                template_category_scores.setdefault(template_name, {}).setdefault(category, []).append(value)
                events_per_category.setdefault(category, set()).add(event_id)
            contributed = True
        if contributed:
            overall_events.add(event_id)

    return template_category_scores, template_overall_scores, events_per_category, overall_events


def plot_template_violin_summary(
    results: Dict[Union[int, str], Dict[str, Dict[str, object]]],
    output_path: Path,
    template_order: Iterable[str] = DEFAULT_TEMPLATE_ORDER,
    template_colors: Optional[Dict[str, str]] = None,
    category_filter: Optional[str] = None,
    min_event_snr: Optional[float] = None,
) -> Optional[Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if template_colors is None:
        template_colors = {
            "RCR": "#1f77b4",
            "SimBL": "#ff7f0e",
            "DataBL": "#2ca02c",
            "CR": "#d62728",
        }

    template_order_seq = [str(name) for name in template_order]
    (
        template_category_scores,
        template_overall_scores,
        events_per_category,
        overall_events,
    ) = _collect_best_template_scores(results, template_order_seq, min_event_snr=min_event_snr)

    labels: List[str] = []
    data_series: List[List[float]] = []
    if category_filter is None:
        for template_name in template_order_seq:
            series = template_overall_scores.get(template_name, [])
            if not series:
                continue
            labels.append(template_name)
            data_series.append(series)
        event_count = len(overall_events)
    else:
        category_key = str(category_filter)
        for template_name in template_order_seq:
            series = template_category_scores.get(template_name, {}).get(category_key, [])
            if not series:
                continue
            labels.append(template_name)
            data_series.append(series)
        event_count = len(events_per_category.get(category_key, set()))

    if not data_series:
        return None

    positions = np.arange(1, len(labels) + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    violins = ax.violinplot(
        data_series,
        positions=positions,
        widths=0.8,
        showmeans=False,
        showextrema=False,
        points=200,
    )

    for body, label in zip(violins["bodies"], labels):
        color = template_colors.get(label, "#888888")
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.5)
        body.set_linewidth(1.2)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Template match $\chi$")
    if category_filter is None:
        ax.set_title("Template match $\chi$ distribution by template type")
    else:
        ax.set_title(f"Template match $\chi$ distribution — {category_filter}")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.tick_params(axis="x", direction="out")

    for position, series, label in zip(positions, data_series, labels):
        color = template_colors.get(label, "#444444")
        quartiles = np.percentile(series, [25, 50, 75])
        whiskers = (np.min(series), np.max(series))
        ax.vlines(position, whiskers[0], whiskers[1], color=color, linewidth=1.0, alpha=0.6)
        ax.hlines(quartiles[1], position - 0.25, position + 0.25, color=color, linewidth=1.6)
        ax.hlines(quartiles[0], position - 0.15, position + 0.15, color=color, linewidth=1.1, alpha=0.8)
        ax.hlines(quartiles[2], position - 0.15, position + 0.15, color=color, linewidth=1.1, alpha=0.8)

    if event_count > 0:
        if category_filter is None:
            label_text = "event" if event_count == 1 else "events"
            annotation = f"{event_count} {label_text}"
        else:
            label_text = "event" if event_count == 1 else "events"
            annotation = f"{event_count} {category_filter} {label_text}"
        ax.text(
            0.02,
            0.95,
            annotation,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_template_box_summary(
    results: Dict[Union[int, str], Dict[str, Dict[str, object]]],
    output_path: Path,
    template_order: Iterable[str] = DEFAULT_TEMPLATE_ORDER,
    category_filter: Optional[str] = None,
    template_colors: Optional[Dict[str, str]] = None,
    min_event_snr: Optional[float] = None,
) -> Optional[Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if template_colors is None:
        template_colors = {
            "RCR": "#1f77b4",
            "SimBL": "#ff7f0e",
            "DataBL": "#2ca02c",
            "CR": "#d62728",
        }
    template_order_seq = [str(name) for name in template_order]
    (
        template_category_scores,
        _template_overall_scores,
        events_per_category,
        overall_events,
    ) = _collect_best_template_scores(
        results,
        template_order_seq,
        min_event_snr=min_event_snr,
    )

    if category_filter is None:
        preferred = ["Backlobe", "RCR", "Station 51"]
        category_sequence = [
            category
            for category in preferred
            if any(template_category_scores.get(template_name, {}).get(category) for template_name in template_order_seq)
        ]
        if not category_sequence:
            category_sequence = sorted(
                {
                    category
                    for template_map in template_category_scores.values()
                    for category, series in template_map.items()
                    if series
                }
            )
    else:
        category_sequence = [str(category_filter)]

    category_sequence = [cat for cat in category_sequence if cat]
    if category_filter is not None:
        category_sequence = [
            cat
            for cat in category_sequence
            if any(template_category_scores.get(template_name, {}).get(cat) for template_name in template_order_seq)
        ]

    if not category_sequence:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    dataset_exists = False
    template_handles: Dict[str, Patch] = {}
    templates_with_data: Set[str] = set()

    if len(template_order_seq) > 1:
        offsets = np.linspace(-0.18, 0.18, len(template_order_seq))
        width = 0.12
    else:
        offsets = np.array([0.0])
        width = 0.2

    for cat_idx, category in enumerate(category_sequence, start=1):
        category_has_data = False
        for offset, template_name in zip(offsets, template_order_seq):
            series = template_category_scores.get(template_name, {}).get(category, [])
            if not series:
                continue
            dataset_exists = True
            category_has_data = True
            templates_with_data.add(template_name)
            scores = np.clip(np.asarray(series, dtype=float), 0.0, 1.0)
            position = cat_idx + offset
            box = ax.boxplot(
                scores.tolist(),
                positions=[position],
                widths=width,
                patch_artist=True,
                showmeans=False,
                showfliers=False,
            )
            color = template_colors.get(template_name, "#888888")
            for patch in box["boxes"]:
                patch.set_facecolor(color)
                patch.set_edgecolor("#333333")
                patch.set_alpha(0.6)
            for element in ("medians", "whiskers", "caps"):
                for artist in box.get(element, []):
                    artist.set_color("#333333")
                    artist.set_linewidth(1.0)
            if template_name not in template_handles:
                template_handles[template_name] = Patch(
                    facecolor=color,
                    edgecolor="#333333",
                    alpha=0.6,
                    label=template_name,
                )
        if not category_has_data:
            continue

    if not dataset_exists:
        plt.close(fig)
        return None

    base_positions = list(range(1, len(category_sequence) + 1))
    ax.set_xticks(base_positions)
    ax.set_xticklabels(category_sequence)
    ax.set_xlabel("Event category")
    ax.set_ylabel(r"Template match $\chi$")
    ax.set_ylim(0, 1)
    ax.set_xlim(0.5, len(category_sequence) + 0.5)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

    legend_handles = [template_handles[name] for name in template_order_seq if name in templates_with_data]
    if legend_handles:
        ax.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="lower right")

    if category_filter is None:
        annotation_lines: List[str] = []
        for category in category_sequence:
            count = len(events_per_category.get(category, set()))
            if count == 0:
                continue
            label_text = "event" if count == 1 else "events"
            annotation_lines.append(f"{count} {category} {label_text}")
        if not annotation_lines:
            total_events = len(overall_events)
            if total_events:
                label_text = "event" if total_events == 1 else "events"
                annotation_lines.append(f"{total_events} events")
        if annotation_lines:
            ax.text(
                0.02,
                0.05,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
    else:
        category_key = str(category_filter)
        count = len(events_per_category.get(category_key, set()))
        if count:
            label_text = "event" if count == 1 else "events"
            ax.text(
                0.02,
                0.05,
                f"{count} {category_key} {label_text}",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path



