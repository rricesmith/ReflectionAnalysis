#!/usr/bin/env python3
"""Produce quick-look plots for single-station RCR simulations.

This script reads the NumPy event summaries written by
``RCRSimulation/S01_RCRSim.py`` and generates:

* Effective area per energy/zenith bin and the zenith-summed projection
* Event-rate equivalents using the Auger parameterisation
* Weighted histograms of reconstructed station zenith and azimuth angles

The implementation borrows the binning and rate-calculation ideas from
``SimpleFootprintSimulation/Stn51RateCalc.py`` but adapts them for the
single-station workflow (trigger fraction derived from all saved events,
per-event weights derived from the event-rate map, etc.).
"""

from __future__ import annotations

import argparse
import configparser
import glob
import math
import re
from pathlib import Path
from typing import Sequence

import astrotools.auger as auger
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

# Ensure the RCRSimEvent class used during pickling is importable
from RCRSimulation.S01_RCRSim import RCRSimEvent


DEFAULT_CONFIG = Path("RCRSimulation/config.ini")
DEFAULT_OUTPUT_SUBDIR = "simple_plots"


def sanitize_component(value: str | None) -> str:
    text = "unknown" if value is None else str(value).strip()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return cleaned or "unknown"


def _format_layer_depth_values(raw_value: str | None) -> tuple[str, str]:
    if raw_value is None:
        return "unknown", "layer-unknown"

    raw_text = str(raw_value).strip()
    if not raw_text:
        return "unknown", "layer-unknown"

    try:
        depth_value = float(raw_text)
    except ValueError:
        text_label = raw_text
        slug_label = f"layer-{sanitize_component(raw_text)}"
        return text_label, slug_label

    if math.isfinite(depth_value):
        if math.isclose(depth_value, round(depth_value)):
            depth_value = int(round(depth_value))
        text_label = f"{depth_value} m"
        slug_core = str(depth_value).replace("+", "")
        slug_label = f"layer_{slug_core}m"
    else:
        text_label = raw_text
        slug_label = f"layer-{sanitize_component(raw_text)}"
    return text_label, slug_label


def _format_layer_db_values(raw_value: str | None) -> tuple[str, str]:
    if raw_value is None:
        return "unknown", "unknown"

    raw_text = str(raw_value).strip()
    if not raw_text:
        return "unknown", "unknown"

    try:
        db_value = float(raw_text)
    except ValueError:
        text_label = raw_text
        slug_label = sanitize_component(f"{raw_text}dB")
        return text_label, slug_label or "unknown"

    if math.isfinite(db_value):
        text_label = f"{db_value:g}"
        slug_label = sanitize_component(f"{text_label}dB")
    else:
        text_label = raw_text
        slug_label = sanitize_component(raw_text)
    return text_label, slug_label or "unknown"


def extract_plot_metadata(config: configparser.ConfigParser) -> dict[str, str]:
    sim_section = config["SIMULATION"] if config.has_section("SIMULATION") else {}

    station_type = sim_section.get("station_type", "unknown") if sim_section else "unknown"
    site = sim_section.get("site", "unknown") if sim_section else "unknown"
    station_depth = sim_section.get("station_depth", "unknown") if sim_section else "unknown"
    layer_raw = None
    if sim_section:
        layer_raw = sim_section.get("layer_depth_m", sim_section.get("layer_depth", None))

    layer_db_raw = sim_section.get("layer_dB") if sim_section else None

    layer_text, layer_slug = _format_layer_depth_values(layer_raw)
    layer_db_text, layer_db_slug = _format_layer_db_values(layer_db_raw)

    annotation_lines = [
        f"Station: {station_type}",
        f"Site: {site}",
        f"Station depth: {station_depth}",
        f"Layer depth: {layer_text}",
    ]
    if layer_db_text == "unknown":
        annotation_lines.append("Layer loss: unknown")
    else:
        annotation_lines.append(f"Layer loss: {layer_db_text} dB")
    annotation_text = "\n".join(annotation_lines)

    filename_suffix = "_".join(
        sanitize_component(component)
        for component in (station_type, site, station_depth, layer_slug, layer_db_slug)
        if component
    )

    return {
        "station_type": station_type,
        "site": site,
        "station_depth": station_depth,
        "layer_depth_text": layer_text,
        "layer_depth_slug": layer_slug,
        "layer_dB_text": layer_db_text,
        "layer_dB_slug": layer_db_slug,
        "annotation_text": annotation_text,
        "filename_suffix": filename_suffix,
    }


def annotate_axes(ax: plt.Axes, annotation: str) -> None:
    if not annotation:
        return
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "boxstyle": "round,pad=0.3"},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate quick plots from RCR simulation outputs.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Path to configuration file (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--events",
        help="Path to an _RCReventList.npy file, directory, or glob pattern. If omitted, all files in the configured numpy folder are used.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override the plot output directory (defaults to save_folder/simple_plots).",
    )
    parser.add_argument(
        "--energy-min",
        type=float,
        default=16.0,
        help="Lower edge of log10(E/eV) binning (default: 16.0).",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=20.0,
        help="Upper edge of log10(E/eV) binning (default: 20.0).",
    )
    parser.add_argument(
        "--energy-step",
        type=float,
        default=0.1,
        help="Step size for log10(E/eV) bins (default: 0.1).",
    )
    parser.add_argument(
        "--sin2-step",
        type=float,
        default=0.2,
        help="Bin width in sin^2(zenith) used for zenith binning (default: 0.1).",
    )
    parser.add_argument(
        "--azimuth-bins",
        type=int,
        default=36,
        help="Number of bins for the azimuth histogram (default: 36).",
    )
    parser.add_argument(
        "--zenith-hist-bins",
        type=int,
        default=18,
        help="Number of bins for the zenith histogram between 0° and 90° (default: 18).",
    )
    return parser.parse_args()


def read_config(config_path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    if not config.read(config_path):
        raise FileNotFoundError(f"Failed to read configuration file at {config_path}")
    return config


def resolve_event_files(
    args: argparse.Namespace,
    config: configparser.ConfigParser,
    layer_db_token: str | None = None,
) -> list[Path]:
    if args.events:
        event_arg = Path(args.events).expanduser()
        if event_arg.exists():
            if event_arg.is_file():
                return [event_arg]
            if event_arg.is_dir():
                files = sorted(event_arg.glob("*_RCReventList.npy"))
                if not files:
                    raise FileNotFoundError(f"No *_RCReventList.npy files found in directory {event_arg}")
                return files
        matches = sorted(Path(p) for p in glob.glob(str(event_arg)))
        if matches:
            return matches
        raise FileNotFoundError(f"Specified event path not found: {event_arg}")

    numpy_folder = Path(config["FOLDERS"]["numpy_folder"]).expanduser()
    if not numpy_folder.exists():
        raise FileNotFoundError(f"Configured numpy_folder does not exist: {numpy_folder}")

    candidates = sorted(numpy_folder.glob("*_RCReventList.npy"))
    if layer_db_token and layer_db_token != "unknown":
        filtered = [path for path in candidates if layer_db_token in path.stem]
        if not filtered:
            raise FileNotFoundError(
                f"No *_RCReventList.npy files matching layer_dB token '{layer_db_token}' found in {numpy_folder}"
            )
        return filtered
    if not candidates:
        raise FileNotFoundError(f"No *_RCReventList.npy files found in {numpy_folder}")
    return candidates


def ensure_output_dir(args: argparse.Namespace, config: configparser.ConfigParser) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    else:
        base = Path(config["FOLDERS"]["save_folder"]).expanduser()
        output_dir = base / DEFAULT_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_events(event_paths: Sequence[Path]) -> Sequence[RCRSimEvent]:
    aggregated: list[RCRSimEvent] = []
    for event_path in event_paths:
        data = np.load(event_path, allow_pickle=True)
        records = data.tolist() if isinstance(data, np.ndarray) else list(data)
        if not records:
            continue
        aggregated.extend(records)
    if not aggregated:
        raise ValueError("No events found in the provided files")
    return aggregated


def energy_zenith_arrays(events: Sequence[RCRSimEvent]) -> dict[str, np.ndarray]:
    energies = np.array([float(evt.energy_eV) for evt in events], dtype=float)
    zenith_deg = np.array([float(evt.zenith_deg) for evt in events], dtype=float)
    azimuth_deg = np.array([float(evt.azimuth_deg) for evt in events], dtype=float)
    triggered = np.array([bool(getattr(evt, "triggered", False)) for evt in events], dtype=bool)
    stn_zenith = np.array([_safe_float(getattr(evt, "stn_zenith", None)) for evt in events], dtype=float)
    stn_azimuth = np.array([_safe_float(getattr(evt, "stn_azimuth", None)) for evt in events], dtype=float)
    return {
        "energies": energies,
        "log_energy": np.log10(energies),
        "zenith_deg": zenith_deg,
        "azimuth_deg": azimuth_deg,
        "sin2_zenith": np.sin(np.deg2rad(zenith_deg)) ** 2,
        "triggered": triggered,
        "stn_zenith": stn_zenith,
        "stn_azimuth": stn_azimuth,
    }


def _safe_float(value) -> float:
    try:
        return float(value) if value is not None else math.nan
    except (TypeError, ValueError):
        return math.nan


def build_bins(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.energy_max <= args.energy_min:
        raise ValueError("energy_max must be greater than energy_min")
    if args.energy_step <= 0:
        raise ValueError("energy_step must be positive")
    energy_bins = np.arange(args.energy_min, args.energy_max + args.energy_step, args.energy_step)
    if len(energy_bins) < 2:
        raise ValueError("Energy bin definition results in fewer than two edges")

    if args.sin2_step <= 0 or args.sin2_step > 1:
        raise ValueError("sin2_step must be in the interval (0, 1]")
    sin2_bins = np.arange(0.0, 1.0 + args.sin2_step, args.sin2_step)
    sin2_bins[-1] = 1.0  # ensure the last edge is exactly 1

    angle_bins = np.rad2deg(np.arcsin(np.sqrt(np.clip(sin2_bins, 0.0, 1.0))))
    return energy_bins, sin2_bins, angle_bins


def bin_events(log_energy: np.ndarray, sin2_zenith: np.ndarray, triggered: np.ndarray,
               energy_bins: np.ndarray, sin2_bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nz = len(sin2_bins) - 1
    ne = len(energy_bins) - 1
    n_total = np.zeros((nz, ne), dtype=int)
    n_trigger = np.zeros((nz, ne), dtype=int)

    e_idx = np.digitize(log_energy, energy_bins) - 1
    z_idx = np.digitize(sin2_zenith, sin2_bins) - 1

    valid = (e_idx >= 0) & (e_idx < ne) & (z_idx >= 0) & (z_idx < nz)
    if not np.any(valid):
        raise ValueError("No events fall inside the chosen energy/zenith bins.")

    np.add.at(n_total, (z_idx[valid], e_idx[valid]), 1)
    trig_valid = valid & triggered
    np.add.at(n_trigger, (z_idx[trig_valid], e_idx[trig_valid]), 1)

    total_counts = n_total.astype(float)
    trigger_fraction = np.divide(n_trigger, total_counts, out=np.zeros_like(total_counts), where=total_counts > 0)

    return n_total, n_trigger, trigger_fraction, (z_idx, e_idx, valid)


def effective_area(trigger_fraction: np.ndarray, distance_km: float) -> np.ndarray:
    if distance_km <= 0:
        raise ValueError("distance_km must be positive to compute an effective area")
    simulation_area = math.pi * (distance_km / 2) ** 2  # km^2
    ic(simulation_area, trigger_fraction)
    return trigger_fraction * simulation_area


def event_rate(aeff: np.ndarray, energy_bins: np.ndarray, angle_bins_deg: np.ndarray) -> np.ndarray:
    nz, ne = aeff.shape
    rate = np.zeros_like(aeff, dtype=float)
    for iz in range(nz):
        z_low = angle_bins_deg[iz]
        z_high = angle_bins_deg[iz + 1]
        for ie in range(ne):
            area = aeff[iz, ie]
            if area <= 0:
                continue
            e_low = energy_bins[ie]
            e_high = energy_bins[ie + 1]
            high_flux = auger.event_rate(e_low, e_high, zmax=z_high, area=area)
            low_flux = auger.event_rate(e_low, e_high, zmax=z_low, area=area)
            value = max(high_flux - low_flux, 0.0)
            rate[iz, ie] = value
    return rate


def per_event_weights(rate_per_bin: np.ndarray, n_trigger: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.divide(rate_per_bin, n_trigger, out=np.zeros_like(rate_per_bin), where=n_trigger > 0)
    return weights


def plot_effective_area(
    energy_bins: np.ndarray,
    angle_bins_deg: np.ndarray,
    aeff: np.ndarray,
    output_dir: Path,
    filename_suffix: str,
    annotation_text: str,
) -> None:
    centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])

    fig, ax = plt.subplots(figsize=(8, 5))
    for iz in range(aeff.shape[0]):
        label = f"{angle_bins_deg[iz]:.1f}-{angle_bins_deg[iz + 1]:.1f}°"
        ax.step(centers, aeff[iz], where="mid", linewidth=1.2, label=label)
    summed = aeff.sum(axis=0)
    ax.step(centers, summed, where="mid", linewidth=2.0, label="Sum", linestyle="--", color="k")
    ax.set_xlabel(r"log$_{10}$(E / eV)")
    ax.set_ylabel(r"Effective area [km$^2$]")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    annotate_axes(ax, annotation_text)
    fig.tight_layout()
    output_path = output_dir / f"aeff_by_zenith_{filename_suffix}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_event_rate(
    energy_bins: np.ndarray,
    angle_bins_deg: np.ndarray,
    rate: np.ndarray,
    output_dir: Path,
    filename_suffix: str,
    annotation_text: str,
) -> None:
    centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])

    fig, ax = plt.subplots(figsize=(8, 5))
    for iz in range(rate.shape[0]):
        row_total = rate[iz].sum()
        label = f"{angle_bins_deg[iz]:.1f}-{angle_bins_deg[iz + 1]:.1f}° ({row_total:.2f} yr$^{-1}$)"
        ax.step(centers, rate[iz], where="mid", linewidth=1.2, label=label)
    summed = rate.sum(axis=0)
    total_rate = summed.sum()
    ax.step(
        centers,
        summed,
        where="mid",
        linewidth=2.0,
        label=f"Sum ({total_rate:.2f} yr$^{-1}$)",
        linestyle="--",
        color="k",
    )
    ax.set_xlabel(r"log$_{10}$(E / eV)")
    ax.set_ylabel("Event rate [yr$^{-1}$]")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    annotate_axes(ax, annotation_text)
    fig.tight_layout()
    output_path = output_dir / f"event_rate_by_zenith_{filename_suffix}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_trigger_counts(
    energy_bins: np.ndarray,
    angle_bins_deg: np.ndarray,
    n_total: np.ndarray,
    n_trigger: np.ndarray,
    output_dir: Path,
    filename_suffix: str,
    annotation_text: str,
) -> None:
    centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])
    fig, ax_counts = plt.subplots(figsize=(8, 5))

    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    colors = prop_cycle.by_key().get("color", []) if prop_cycle else []

    nz = n_total.shape[0]
    for iz in range(nz):
        color = colors[iz % len(colors)] if colors else None
        z_label = f"{angle_bins_deg[iz]:.1f}-{angle_bins_deg[iz + 1]:.1f}°"
        counts = n_total[iz].astype(float)
        triggers = n_trigger[iz].astype(float)
        if not np.any(counts):
            continue
        ax_counts.step(centers, counts, where="mid", linewidth=1.2, label=f"{z_label} throws", color=color)
        ax_counts.step(
            centers,
            triggers,
            where="mid",
            linewidth=1.2,
            linestyle="--",
            label=f"{z_label} triggers",
            color=color,
        )

    total_counts = n_total.sum(axis=0).astype(float)
    total_triggers = n_trigger.sum(axis=0).astype(float)
    if np.any(total_counts):
        ax_counts.step(
            centers,
            total_counts,
            where="mid",
            linewidth=2.0,
            color="k",
            label="Sum throws",
        )
        ax_counts.step(
            centers,
            total_triggers,
            where="mid",
            linewidth=2.0,
            linestyle="--",
            color="k",
            label="Sum triggers",
        )

    ax_counts.set_xlabel(r"log$_{10}$(E / eV)")
    ax_counts.set_ylabel("Event counts")
    ax_counts.grid(alpha=0.3)

    ax_fraction = ax_counts.twinx()
    with np.errstate(divide="ignore", invalid="ignore"):
        overall_fraction = np.divide(
            total_triggers,
            total_counts,
            out=np.zeros_like(total_triggers, dtype=float),
            where=total_counts > 0,
        )

    if np.any(total_counts):
        widths = energy_bins[1:] - energy_bins[:-1]
        ax_fraction.bar(
            centers,
            overall_fraction,
            width=widths * 0.8,
            color="tab:orange",
            alpha=0.35,
            label="Trigger fraction (sum)",
            align="center",
        )

    ax_fraction.set_ylabel("Trigger fraction")
    ax_fraction.set_ylim(0.0, 1.05)

    handles_counts, labels_counts = ax_counts.get_legend_handles_labels()
    handles_fraction, labels_fraction = ax_fraction.get_legend_handles_labels()
    if handles_fraction:
        ax_counts.legend(
            handles_counts + handles_fraction,
            labels_counts + labels_fraction,
            fontsize=8,
            loc="upper left",
            ncol=2,
        )
    else:
        ax_counts.legend(fontsize=8, loc="upper left", ncol=2)

    annotate_axes(ax_counts, annotation_text)
    fig.tight_layout()
    output_path = output_dir / f"trigger_counts_and_fraction_{filename_suffix}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_weighted_histograms(
    zenith_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    weights: np.ndarray,
    azimuth_bins: int,
    zenith_hist_bins: int,
    output_dir: Path,
    filename_suffix: str,
    annotation_text: str,
) -> None:
    triggered_mask = (~np.isnan(zenith_deg)) & (~np.isnan(weights)) & (weights > 0)
    zenith_values = zenith_deg[triggered_mask]
    zenith_weights = weights[triggered_mask]

    if zenith_values.size:
        zenith_edges = np.linspace(0.0, 90.0, zenith_hist_bins + 1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(zenith_values, bins=zenith_edges, weights=zenith_weights, histtype="step", linewidth=1.8)
        ax.set_xlabel("Reconstructed zenith [deg]")
        ax.set_ylabel("Weighted count [yr$^{-1}$]")
        ax.grid(alpha=0.3)
        annotate_axes(ax, annotation_text)
        fig.tight_layout()
        output_path = output_dir / f"triggered_zenith_distribution_{filename_suffix}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    azimuth_mask = (~np.isnan(azimuth_deg)) & (~np.isnan(weights)) & (weights > 0)
    azimuth_values = azimuth_deg[azimuth_mask]
    azimuth_weights = weights[azimuth_mask]

    if azimuth_values.size:
        azimuth_edges = np.linspace(0.0, 360.0, azimuth_bins + 1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(azimuth_values, bins=azimuth_edges, weights=azimuth_weights, histtype="step", linewidth=1.8)
        ax.set_xlabel("Reconstructed azimuth [deg]")
        ax.set_ylabel("Weighted count [yr$^{-1}$]")
        ax.grid(alpha=0.3)
        annotate_axes(ax, annotation_text)
        fig.tight_layout()
        output_path = output_dir / f"triggered_azimuth_distribution_{filename_suffix}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_trigger_scatter(
    energies: np.ndarray,
    zenith_deg: np.ndarray,
    triggered: np.ndarray,
    output_dir: Path,
    filename_suffix: str,
    annotation_text: str,
) -> None:
    if energies.size == 0:
        return

    event_view = np.rec.fromarrays([energies, zenith_deg], names="energy,zenith")
    unique_events, inverse = np.unique(event_view, return_inverse=True)
    total_counts = np.bincount(inverse, minlength=len(unique_events))
    trigger_counts = np.bincount(inverse, weights=triggered.astype(float), minlength=len(unique_events))
    with np.errstate(divide="ignore", invalid="ignore"):
        trigger_fraction = np.divide(
            trigger_counts,
            total_counts,
            out=np.zeros_like(trigger_counts, dtype=float),
            where=total_counts > 0,
        )

    x_values = np.log10(unique_events.energy)
    y_values = unique_events.zenith
    has_triggers = trigger_counts > 0

    fig, ax = plt.subplots(figsize=(7, 5))
    legend_handles: list = []

    if np.any(~has_triggers):
        no_trig = ax.scatter(
            x_values[~has_triggers],
            y_values[~has_triggers],
            facecolors="none",
            edgecolors="tab:gray",
            linewidths=1.0,
            s=45,
            label="No triggers",
        )
        legend_handles.append(no_trig)

    if np.any(has_triggers):
        triggered_scatter = ax.scatter(
            x_values[has_triggers],
            y_values[has_triggers],
            c=trigger_fraction[has_triggers],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=60,
            label="Triggered",
        )
        legend_handles.append(triggered_scatter)
        cbar = plt.colorbar(triggered_scatter, ax=ax)
        cbar.set_label("Trigger fraction")

    ax.set_xlabel(r"log$_{10}$(E / eV)")
    ax.set_ylabel("Zenith [deg]")
    ax.set_ylim(0.0, 90.0)
    ax.grid(alpha=0.3)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")

    annotate_axes(ax, annotation_text)
    fig.tight_layout()
    output_path = output_dir / f"trigger_fraction_scatter_{filename_suffix}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = read_config(Path(args.config).expanduser())

    plot_metadata = extract_plot_metadata(config)
    filename_suffix = plot_metadata["filename_suffix"]
    annotation_text = plot_metadata["annotation_text"]

    layer_db_token = plot_metadata.get("layer_dB_slug")
    event_paths = resolve_event_files(args, config, layer_db_token=layer_db_token)
    output_dir = ensure_output_dir(args, config)

    events = load_events(event_paths)
    print(f"Loaded {len(events)} events from {len(event_paths)} file(s).")
    arrays = energy_zenith_arrays(events)

    energy_bins, sin2_bins, angle_bins_deg = build_bins(args)
    n_total, n_trigger, trigger_fraction, (z_idx, e_idx, valid_mask) = bin_events(
        arrays["log_energy"], arrays["sin2_zenith"], arrays["triggered"], energy_bins, sin2_bins
    )

    distance_km = config.getfloat("SIMULATION", "distance_km", fallback=12.0)
    aeff = effective_area(trigger_fraction, distance_km)
    rate = event_rate(aeff, energy_bins, angle_bins_deg)
    plot_trigger_counts(
        energy_bins,
        angle_bins_deg,
        n_total,
        n_trigger,
        output_dir,
        filename_suffix,
        annotation_text,
    )

    plot_effective_area(energy_bins, angle_bins_deg, aeff, output_dir, filename_suffix, annotation_text)
    plot_event_rate(energy_bins, angle_bins_deg, rate, output_dir, filename_suffix, annotation_text)

    weight_map = per_event_weights(rate, n_trigger)
    per_event_weight = np.zeros(len(events), dtype=float)
    trig_mask = valid_mask & arrays["triggered"]
    trig_indices = np.where(trig_mask)[0]
    if trig_indices.size:
        trig_z = z_idx[trig_indices]
        trig_e = e_idx[trig_indices]
        per_event_weight[trig_indices] = weight_map[trig_z, trig_e]

    plot_weighted_histograms(
        arrays["zenith_deg"],
        arrays["azimuth_deg"],
        per_event_weight,
        args.azimuth_bins,
        args.zenith_hist_bins,
        output_dir,
        filename_suffix,
        annotation_text,
    )
    plot_trigger_scatter(
        arrays["energies"],
        arrays["zenith_deg"],
        arrays["triggered"],
        output_dir,
        filename_suffix,
        annotation_text,
    )

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
