#!/usr/bin/env python3
"""Plotting tools for RCR simulation results.

This module provides visualization functions for analyzing trigger rates,
energy dependencies, and angular distributions for direct vs reflected
cosmic ray signals.

Usage:
    python RCRSimulation/S02_RCRPlots.py [numpy_files...] [--save-prefix PREFIX]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def load_rcr_events(paths: Sequence[Path]) -> np.ndarray:
    """Load and concatenate RCREvent arrays from multiple numpy files."""
    all_events = []
    for path in paths:
        events = np.load(path, allow_pickle=True)
        all_events.extend(events)
    return np.array(all_events, dtype=object)


def extract_trigger_data(events: np.ndarray) -> dict:
    """Extract trigger statistics from RCREvent array.

    Returns dict with keys:
        - energies: array of event energies (eV)
        - zeniths: array of zenith angles (deg)
        - azimuths: array of azimuth angles (deg)
        - direct_triggered: boolean array
        - reflected_triggered: boolean array
        - trigger_names: set of trigger names encountered
    """
    energies = []
    zeniths = []
    azimuths = []
    direct_triggered = []
    reflected_triggered = []
    trigger_names = set()

    for evt in events:
        energies.append(evt.energy)
        zeniths.append(np.rad2deg(evt.zenith))
        azimuths.append(np.rad2deg(evt.azimuth))

        has_direct = False
        has_reflected = False

        for trig_name in evt.all_trigger_names():
            trigger_names.add(trig_name)
            if evt.has_direct_trigger(trig_name):
                has_direct = True
            if evt.has_reflected_trigger(trig_name):
                has_reflected = True

        direct_triggered.append(has_direct)
        reflected_triggered.append(has_reflected)

    return {
        "energies": np.array(energies),
        "zeniths": np.array(zeniths),
        "azimuths": np.array(azimuths),
        "direct_triggered": np.array(direct_triggered),
        "reflected_triggered": np.array(reflected_triggered),
        "trigger_names": trigger_names,
    }


def plot_trigger_rates_vs_energy(
    data: dict,
    energy_bins: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot trigger rate as function of energy for direct and reflected."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    log_energies = np.log10(data["energies"])

    if energy_bins is None:
        energy_bins = np.linspace(log_energies.min(), log_energies.max(), 15)

    direct_counts, _ = np.histogram(log_energies[data["direct_triggered"]], bins=energy_bins)
    reflected_counts, _ = np.histogram(log_energies[data["reflected_triggered"]], bins=energy_bins)
    total_counts, _ = np.histogram(log_energies, bins=energy_bins)

    # Avoid division by zero
    valid = total_counts > 0
    bin_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])

    direct_rate = np.zeros_like(total_counts, dtype=float)
    reflected_rate = np.zeros_like(total_counts, dtype=float)
    direct_rate[valid] = direct_counts[valid] / total_counts[valid]
    reflected_rate[valid] = reflected_counts[valid] / total_counts[valid]

    ax.step(bin_centers, direct_rate * 100, where="mid", label="Direct", color="blue", linewidth=2)
    ax.step(bin_centers, reflected_rate * 100, where="mid", label="Reflected", color="red", linewidth=2)

    ax.set_xlabel(r"log$_{10}$(Energy / eV)", fontsize=12)
    ax.set_ylabel("Trigger Rate (%)", fontsize=12)
    ax.set_title("Trigger Rate vs Energy", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    return ax


def plot_trigger_rates_vs_zenith(
    data: dict,
    zenith_bins: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot trigger rate as function of zenith angle."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if zenith_bins is None:
        zenith_bins = np.linspace(0, 90, 19)

    direct_counts, _ = np.histogram(data["zeniths"][data["direct_triggered"]], bins=zenith_bins)
    reflected_counts, _ = np.histogram(data["zeniths"][data["reflected_triggered"]], bins=zenith_bins)
    total_counts, _ = np.histogram(data["zeniths"], bins=zenith_bins)

    valid = total_counts > 0
    bin_centers = 0.5 * (zenith_bins[:-1] + zenith_bins[1:])

    direct_rate = np.zeros_like(total_counts, dtype=float)
    reflected_rate = np.zeros_like(total_counts, dtype=float)
    direct_rate[valid] = direct_counts[valid] / total_counts[valid]
    reflected_rate[valid] = reflected_counts[valid] / total_counts[valid]

    ax.step(bin_centers, direct_rate * 100, where="mid", label="Direct", color="blue", linewidth=2)
    ax.step(bin_centers, reflected_rate * 100, where="mid", label="Reflected", color="red", linewidth=2)

    ax.set_xlabel("Zenith Angle (deg)", fontsize=12)
    ax.set_ylabel("Trigger Rate (%)", fontsize=12)
    ax.set_title("Trigger Rate vs Zenith Angle", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 100)

    return ax


def plot_footprint_scatter(
    data: dict,
    events: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Scatter plot of trigger positions in CoREAS footprint."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    coreas_x = np.array([evt.coreas_x for evt in events])
    coreas_y = np.array([evt.coreas_y for evt in events])

    # Plot all events as background
    ax.scatter(coreas_x, coreas_y, c="gray", alpha=0.2, s=5, label="All events")

    # Overlay triggered events
    direct_mask = data["direct_triggered"]
    reflected_mask = data["reflected_triggered"]
    both_mask = direct_mask & reflected_mask
    direct_only = direct_mask & ~reflected_mask
    reflected_only = reflected_mask & ~direct_mask

    ax.scatter(coreas_x[direct_only], coreas_y[direct_only], c="blue", alpha=0.6, s=20, label="Direct only")
    ax.scatter(coreas_x[reflected_only], coreas_y[reflected_only], c="red", alpha=0.6, s=20, label="Reflected only")
    ax.scatter(coreas_x[both_mask], coreas_y[both_mask], c="purple", alpha=0.8, s=30, label="Both")

    ax.set_xlabel("CoREAS X (m)", fontsize=12)
    ax.set_ylabel("CoREAS Y (m)", fontsize=12)
    ax.set_title("Trigger Footprint", fontsize=14)
    ax.legend(fontsize=10)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    return ax


def plot_energy_distribution(
    data: dict,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot energy distribution of triggered events."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    log_energies = np.log10(data["energies"])
    bins = np.linspace(log_energies.min(), log_energies.max(), 20)

    ax.hist(log_energies, bins=bins, alpha=0.3, color="gray", label="All events")
    ax.hist(
        log_energies[data["direct_triggered"]],
        bins=bins,
        alpha=0.5,
        color="blue",
        label="Direct triggered",
    )
    ax.hist(
        log_energies[data["reflected_triggered"]],
        bins=bins,
        alpha=0.5,
        color="red",
        label="Reflected triggered",
    )

    ax.set_xlabel(r"log$_{10}$(Energy / eV)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Energy Distribution of Triggers", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return ax


def create_summary_figure(events: np.ndarray, save_path: Optional[Path] = None) -> Figure:
    """Create a 4-panel summary figure of simulation results."""
    data = extract_trigger_data(events)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"RCR Simulation Summary (N={len(events)} events, "
        f"Direct={data['direct_triggered'].sum()}, "
        f"Reflected={data['reflected_triggered'].sum()})",
        fontsize=14,
    )

    plot_trigger_rates_vs_energy(data, ax=axes[0, 0])
    plot_trigger_rates_vs_zenith(data, ax=axes[0, 1])
    plot_footprint_scatter(data, events, ax=axes[1, 0])
    plot_energy_distribution(data, ax=axes[1, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved summary figure to: {save_path}")

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot RCR simulation results."
    )
    parser.add_argument(
        "numpy_files",
        nargs="+",
        type=Path,
        help="Numpy files containing RCREvent lists.",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="Prefix for saved plot files. If not specified, displays interactively.",
    )
    parser.add_argument(
        "--save-folder",
        type=Path,
        default=Path("RCRSimulation/plots/"),
        help="Folder to save plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load events from all files
    events = load_rcr_events(args.numpy_files)
    print(f"Loaded {len(events)} events from {len(args.numpy_files)} files")

    if len(events) == 0:
        print("No events found!")
        return

    # Extract statistics
    data = extract_trigger_data(events)
    print(f"Trigger names: {data['trigger_names']}")
    print(f"Direct triggers: {data['direct_triggered'].sum()}")
    print(f"Reflected triggers: {data['reflected_triggered'].sum()}")

    # Create summary figure
    if args.save_prefix:
        args.save_folder.mkdir(parents=True, exist_ok=True)
        save_path = args.save_folder / f"{args.save_prefix}_summary.png"
    else:
        save_path = None

    fig = create_summary_figure(events, save_path)

    if save_path is None:
        plt.show()


if __name__ == "__main__":
    main()
