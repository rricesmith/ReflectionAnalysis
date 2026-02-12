#!/usr/bin/env python3
"""Chapter 4 multi-panel comparison plots for thesis.

Generates side-by-side comparison plots across detector configurations, sites,
and reflectivity values using multi-dB simulation output.

Usage:
    python RCRSimulation/S04_RCRChapter4Plots.py
    python RCRSimulation/S04_RCRChapter4Plots.py --numpy-folder path/to/numpy --save-folder path/to/plots
"""

from __future__ import annotations

import argparse
import configparser
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
from icecream import ic

from NuRadioReco.utilities import units

from RCRSimulation.RCRAnalysis import (
    getEnergyZenithBins,
    getBinnedTriggerRate,
    getEventRate,
    setEventListRateWeight,
    set_bad_imshow,
    getUniqueEnergyZenithPairs,
    collect_trigger_names,
    filter_real_triggers,
    parse_db_from_trigger_name,
    NOISE_PRECHECK_SIGMA,
)
from RCRSimulation.RCREventObject import RCREvent, REFLECTED_STATION_OFFSET

LOGGER = logging.getLogger(__name__)

# ============================================================================
# Simulation name mappings
# ============================================================================

MB_REFLECTED_SIMS = {
    "HRA": "HRA_MB_576m",
    "Gen2 Shallow": "Gen2_shallow_MB_576m",
    "Gen2 Deep": "Gen2_deep_MB_576m",
}

SP_DEPTHS = ["300m", "500m", "830m"]

# SP sims keyed by (depth, station_type)
SP_REFLECTED_SIMS = {
    ("300m", "shallow"): "Gen2_shallow_SP_300m",
    ("300m", "deep"): "Gen2_deep_SP_300m",
    ("500m", "shallow"): "Gen2_shallow_SP_500m",
    ("500m", "deep"): "Gen2_deep_SP_500m",
    ("830m", "shallow"): "Gen2_shallow_SP_830m",
    ("830m", "deep"): "Gen2_deep_SP_830m",
}

# For direct rates, use the combined sims (which contain both direct and reflected stations).
# These are the "canonical" sims used for direct rates (one per site/station-type).
SP_DIRECT_SIMS = {
    "Gen2 Shallow": "Gen2_shallow_SP_300m",
    "Gen2 Deep": "Gen2_deep_SP_300m",
}

# Expected dB sweep values
MB_DB_VALUES = [0.0, 1.5, 3.0]
SP_DB_VALUES = [40.0, 45.0, 50.0, 55.0]

# MB dB-to-R labels for display
MB_DB_LABELS = {0.0: "R=1.0", 1.5: "R=0.7", 3.0: "R=0.5"}


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_combined_events(numpy_folder: str | Path, sim_name: str) -> np.ndarray | None:
    """Load combined event file for a simulation name.

    Searches for *{sim_name}*combined*RCReventList.npy in numpy_folder.
    Returns the event array, or None if not found.
    """
    folder = Path(numpy_folder)
    # Try exact pattern first
    candidates = list(folder.glob(f"*{sim_name}*combined*RCReventList.npy"))
    if not candidates:
        # Fallback: any file matching sim name
        candidates = list(folder.glob(f"*{sim_name}*RCReventList.npy"))
    if not candidates:
        LOGGER.warning("No event file found for sim '%s' in %s", sim_name, numpy_folder)
        return None
    # Use the first match (should be unique for combined files)
    event_file = candidates[0]
    if len(candidates) > 1:
        LOGGER.info("Multiple matches for '%s', using %s", sim_name, event_file.name)
    events = np.load(event_file, allow_pickle=True)
    LOGGER.info("Loaded %d events from %s", len(events), event_file.name)
    return events


def load_all_sims(numpy_folder: str | Path, sim_names: dict | list) -> Dict[str, np.ndarray]:
    """Load all simulations, returning {sim_name: event_array} for those found."""
    loaded = {}
    names = sim_names.values() if isinstance(sim_names, dict) else sim_names
    for sim_name in set(names):
        events = load_combined_events(numpy_folder, sim_name)
        if events is not None:
            loaded[sim_name] = events
    return loaded


# ============================================================================
# Trigger Discovery Helpers
# ============================================================================

def find_trigger_for_db(event_list: Sequence[RCREvent], db_value: float | None) -> str | None:
    """Find the real (non-noise) trigger name matching a specific dB value.

    db_value=None returns the untagged trigger (for direct or non-multi-dB sims).
    """
    all_triggers = collect_trigger_names(event_list)
    real_triggers = filter_real_triggers(all_triggers)
    for t in real_triggers:
        parsed_db = parse_db_from_trigger_name(t)
        if db_value is None and parsed_db is None:
            return t
        if db_value is not None and parsed_db is not None and abs(parsed_db - db_value) < 0.05:
            return t
    return None


def find_direct_trigger(event_list: Sequence[RCREvent]) -> str | None:
    """Find the untagged (no dB) real trigger for direct stations."""
    return find_trigger_for_db(event_list, db_value=None)


def get_all_db_triggers(event_list: Sequence[RCREvent]) -> Dict[float, str]:
    """Get all dB-tagged real triggers. Returns {db_value: trigger_name}."""
    all_triggers = collect_trigger_names(event_list)
    real_triggers = filter_real_triggers(all_triggers)
    result = {}
    for t in real_triggers:
        db = parse_db_from_trigger_name(t)
        if db is not None:
            result[db] = t
    return result


# ============================================================================
# Plot 1/1b, 2/2b: Multi-Panel Trigger Rate 2D Histograms
# ============================================================================

def plot_trigger_rate_panels(
    event_lists: List[np.ndarray],
    trigger_names: List[str],
    panel_titles: List[str],
    suptitle: str,
    savename: str,
    rate_type: str = "reflected",
):
    """Multi-panel 2D trigger rate histograms with circle overlay.

    Args:
        event_lists: One event array per panel
        trigger_names: Trigger name to query per panel
        panel_titles: Subtitle for each panel
        suptitle: Figure super-title
        savename: Output file path
        rate_type: 'reflected' or 'direct'
    """
    n = len(event_lists)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n + 1, 4.5))
    if n == 1:
        axes = [axes]

    e_bins, z_bins = getEnergyZenithBins()
    e_log = np.log10(np.array(e_bins) / units.eV)
    z_deg = np.array(z_bins) / units.deg
    extent = [e_log[0], e_log[-1], z_deg[0], z_deg[-1]]

    # Collect all rates to share a colorbar
    all_rates = []
    panel_data = []
    for events, trig in zip(event_lists, trigger_names):
        direct_rate, reflected_rate, _ = getBinnedTriggerRate(events, trig)
        rate = reflected_rate if rate_type == "reflected" else direct_rate
        panel_data.append(rate)
        nonzero = rate[rate > 0]
        if nonzero.size > 0:
            all_rates.extend(nonzero.tolist())

    # Shared color normalization
    if all_rates:
        vmin = min(all_rates) * 0.5
        vmax = max(all_rates) * 1.5
    else:
        vmin, vmax = 1e-3, 1.0
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    im = None
    for ax, rate, events, title in zip(axes, panel_data, event_lists, panel_titles):
        masked, cmap = set_bad_imshow(rate, 0)
        im = ax.imshow(
            masked.T, origin="lower", aspect="auto",
            norm=norm, cmap=cmap, extent=extent,
        )
        # Circle overlay
        log_energies, zeniths_deg = getUniqueEnergyZenithPairs(events)
        ax.scatter(
            log_energies, zeniths_deg,
            facecolors="none", edgecolors="black", s=20, linewidths=0.5,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("log$_{10}$(E/eV)")

    axes[0].set_ylabel("Zenith (deg)")
    for ax in axes[1:]:
        ax.set_ylabel("")

    fig.suptitle(suptitle, fontsize=13, y=1.02)

    if im is not None:
        fig.colorbar(im, ax=axes, label="Trigger Rate", shrink=0.8, pad=0.02)

    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches="tight")
    plt.close()
    ic(f"Saved: {savename}")


# ============================================================================
# Plot 3/4: Event Rate Band Plots
# ============================================================================

def plot_event_rate_bands(
    rate_arrays_per_panel: List[List[np.ndarray]],
    panel_titles: List[str],
    suptitle: str,
    savename: str,
):
    """Multi-panel event rate band plots showing min/max across dB values.

    Args:
        rate_arrays_per_panel: [panel_idx][db_idx] -> event_rate 2D array (energy, zenith)
        panel_titles: Subtitle per panel
        suptitle: Figure super-title
        savename: Output file path
    """
    n = len(rate_arrays_per_panel)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    e_bins, z_bins = getEnergyZenithBins()
    energy_centers = (np.array(e_bins[:-1]) + np.array(e_bins[1:])) / 2
    x_vals = np.log10(energy_centers / units.eV)
    n_zenith = len(z_bins) - 1
    colors = plt.cm.rainbow(np.linspace(0, 1, n_zenith))

    for ax, rates_list, title in zip(axes, rate_arrays_per_panel, panel_titles):
        # Stack: shape (n_db, n_energy, n_zenith)
        stacked = np.array([np.nan_to_num(r) for r in rates_list])

        # Per zenith bin band
        for iz in range(n_zenith):
            lo = np.nanmin(stacked[:, :, iz], axis=0)
            hi = np.nanmax(stacked[:, :, iz], axis=0)
            zen_lo = z_bins[iz] / units.deg
            zen_hi = z_bins[iz + 1] / units.deg
            label = f"{zen_lo:.0f}-{zen_hi:.0f}\u00b0"
            ax.fill_between(x_vals, lo, hi, alpha=0.3, color=colors[iz], label=label)

        # Total (sum of zenith bins) band
        totals = np.nansum(stacked, axis=2)  # shape (n_db, n_energy)
        lo_total = np.nanmin(totals, axis=0)
        hi_total = np.nanmax(totals, axis=0)
        ax.fill_between(x_vals, lo_total, hi_total, alpha=0.2, color="black", label="Total")

        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-3)
        ax.set_xlabel("log$_{10}$(E/eV)")
        ax.set_title(title, fontsize=10)

    axes[0].set_ylabel("Event Rate (evts/yr)")
    axes[0].legend(fontsize=7, loc="lower left")

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches="tight")
    plt.close()
    ic(f"Saved: {savename}")


# ============================================================================
# Plot 5: Radii Probability Density
# ============================================================================

def _compute_radii_density(
    event_list: Sequence[RCREvent],
    weight_name: str,
    max_distance: float,
    n_bins: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized probability density histogram of radii.

    Returns (bin_centers, density) where density integrates to ~1.
    """
    radii = []
    weights = []
    for evt in event_list:
        w = evt.get_weight(weight_name)
        if w > 0:
            radii.append(evt.get_radius())
            weights.append(w)

    if not radii:
        bins = np.linspace(0, max_distance, n_bins + 1)
        centers = (bins[:-1] + bins[1:]) / 2
        return centers, np.zeros(n_bins)

    radii = np.array(radii)
    weights = np.array(weights)
    bins = np.linspace(0, max_distance, n_bins + 1)
    hist, _ = np.histogram(radii, bins=bins, weights=weights)

    # Normalize to probability density: integral = 1
    bin_width = bins[1] - bins[0]
    total = hist.sum()
    if total > 0:
        density = hist / (total * bin_width)
    else:
        density = hist

    centers = (bins[:-1] + bins[1:]) / 2
    return centers, density


def plot_radii_density_panels(
    event_lists: List[np.ndarray],
    direct_triggers: List[str],
    reflected_db_triggers: List[Dict[float, str]],
    panel_titles: List[str],
    suptitle: str,
    savename: str,
    max_distance: float,
    db_labels: Dict[float, str] | None = None,
    n_bins: int = 30,
):
    """Multi-panel radii probability density with direct + multi-dB reflected lines.

    Args:
        event_lists: One event array per panel
        direct_triggers: Direct trigger name per panel
        reflected_db_triggers: [{db: trigger_name}] per panel
        panel_titles: Subtitle per panel
        suptitle: Figure super-title
        savename: Output path
        max_distance: Max radius (meters)
        db_labels: Optional {db_value: display_label} for legend
        n_bins: Number of radial bins
    """
    n = len(event_lists)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    e_bins, z_bins = getEnergyZenithBins()

    for ax, events, dir_trig, db_trigs, title in zip(
        axes, event_lists, direct_triggers, reflected_db_triggers, panel_titles
    ):
        events_list = list(events)

        # Direct: set weights and plot density
        if dir_trig:
            direct_rate, _, _ = getBinnedTriggerRate(events_list, dir_trig)
            direct_event_rate = getEventRate(direct_rate, e_bins, z_bins, max_distance)
            setEventListRateWeight(
                events_list, direct_rate, "direct", dir_trig,
                max_distance=max_distance, use_direct=True,
            )
            centers, density = _compute_radii_density(events_list, "direct", max_distance, n_bins)
            ax.plot(centers, density, color="black", linewidth=1.5, label="Direct")

        # Reflected: one line per dB value
        if db_trigs:
            sorted_dbs = sorted(db_trigs.keys())
            cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(sorted_dbs)))
            for db_val, color in zip(sorted_dbs, cmap):
                ref_trig = db_trigs[db_val]
                weight_name = f"reflected_{db_val:.1f}dB"
                _, reflected_rate, _ = getBinnedTriggerRate(events_list, ref_trig)
                setEventListRateWeight(
                    events_list, reflected_rate, weight_name, ref_trig,
                    max_distance=max_distance, use_direct=False,
                )
                centers, density = _compute_radii_density(
                    events_list, weight_name, max_distance, n_bins,
                )
                label = db_labels.get(db_val, f"{db_val:.0f} dB") if db_labels else f"{db_val:.0f} dB"
                ax.plot(centers, density, color=color, linewidth=1.2, linestyle="--", label=label)

        ax.set_xlabel("Radius (m)")
        ax.set_title(title, fontsize=10)

    axes[0].set_ylabel("Probability Density")
    axes[0].legend(fontsize=7)

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches="tight")
    plt.close()
    ic(f"Saved: {savename}")


# ============================================================================
# Output 6: Gen2 SP Rate Table
# ============================================================================

def generate_sp_rate_table(
    loaded: Dict[str, np.ndarray],
    max_distance: float,
    savename: str,
):
    """Generate human-readable text table of Gen2 SP event rates.

    Rows: Direct, 300m, 500m, 830m
    Columns: Shallow, Deep, Combined
    Each reflected cell shows min-max range across dB values.
    """
    e_bins, z_bins = getEnergyZenithBins()

    def total_reflected_rate(events, db_val):
        trig = find_trigger_for_db(events, db_val)
        if trig is None:
            return None
        _, ref_rate, _ = getBinnedTriggerRate(events, trig)
        event_rate = getEventRate(ref_rate, e_bins, z_bins, max_distance)
        return float(np.nansum(event_rate))

    def total_direct_rate(events):
        trig = find_direct_trigger(events)
        if trig is None:
            return None
        dir_rate, _, _ = getBinnedTriggerRate(events, trig)
        event_rate = getEventRate(dir_rate, e_bins, z_bins, max_distance)
        return float(np.nansum(event_rate))

    def format_range(values):
        vals = [v for v in values if v is not None]
        if not vals:
            return "N/A"
        if len(vals) == 1:
            return f"{vals[0]:.3f}"
        return f"{min(vals):.3f} - {max(vals):.3f}"

    def format_single(value):
        return f"{value:.3f}" if value is not None else "N/A"

    lines = []
    lines.append("Gen2 SP Event Rates (evts/yr)")
    lines.append("=" * 70)
    lines.append(f"{'':24s}{'Shallow':>15s}{'Deep':>15s}{'Combined':>15s}")
    lines.append("-" * 70)

    # Direct row
    shallow_dir = loaded.get(SP_DIRECT_SIMS.get("Gen2 Shallow"))
    deep_dir = loaded.get(SP_DIRECT_SIMS.get("Gen2 Deep"))
    r_shallow = total_direct_rate(shallow_dir) if shallow_dir is not None else None
    r_deep = total_direct_rate(deep_dir) if deep_dir is not None else None
    r_combined = None
    if r_shallow is not None and r_deep is not None:
        r_combined = r_shallow + r_deep
    lines.append(
        f"{'Direct':24s}{format_single(r_shallow):>15s}{format_single(r_deep):>15s}{format_single(r_combined):>15s}"
    )

    # Reflected rows per depth
    for depth in SP_DEPTHS:
        shallow_key = SP_REFLECTED_SIMS.get((depth, "shallow"))
        deep_key = SP_REFLECTED_SIMS.get((depth, "deep"))
        shallow_events = loaded.get(shallow_key)
        deep_events = loaded.get(deep_key)

        shallow_rates = []
        deep_rates = []
        combined_rates = []

        for db in SP_DB_VALUES:
            rs = total_reflected_rate(shallow_events, db) if shallow_events is not None else None
            rd = total_reflected_rate(deep_events, db) if deep_events is not None else None
            shallow_rates.append(rs)
            deep_rates.append(rd)
            if rs is not None and rd is not None:
                combined_rates.append(rs + rd)

        row_label = f"{depth} (40-55 dB)"
        lines.append(
            f"{row_label:24s}{format_range(shallow_rates):>15s}{format_range(deep_rates):>15s}{format_range(combined_rates):>15s}"
        )

    lines.append("=" * 70)

    table_text = "\n".join(lines)
    print(table_text)

    with open(savename, "w") as f:
        f.write(table_text + "\n")
    ic(f"Saved: {savename}")


# ============================================================================
# High-Level Plot Generators
# ============================================================================

def generate_mb_trigger_rate_plots(loaded, save_folder, max_distance):
    """Plot 1 + 1b: MB trigger rate panels (reflected at R=0.7, and direct)."""
    labels = list(MB_REFLECTED_SIMS.keys())
    sim_names = [MB_REFLECTED_SIMS[k] for k in labels]
    event_lists = [loaded[s] for s in sim_names if s in loaded]
    if len(event_lists) != len(labels):
        LOGGER.warning("Not all MB reflected sims available, skipping MB trigger rate plots")
        return

    # Reflected at R=0.7 (1.5 dB)
    ref_triggers = [find_trigger_for_db(e, 1.5) for e in event_lists]
    if all(t is not None for t in ref_triggers):
        plot_trigger_rate_panels(
            event_lists, ref_triggers, labels,
            suptitle="MB Reflected Trigger Rate ($R_{power}$=0.7, 1.5 dB)",
            savename=os.path.join(save_folder, "mb_trigger_rate_reflected.png"),
            rate_type="reflected",
        )
    else:
        # Fallback: try single (non-dB) trigger for non-multi-dB data
        ref_triggers = [find_direct_trigger(e) for e in event_lists]
        if all(t is not None for t in ref_triggers):
            plot_trigger_rate_panels(
                event_lists, ref_triggers, labels,
                suptitle="MB Reflected Trigger Rate",
                savename=os.path.join(save_folder, "mb_trigger_rate_reflected.png"),
                rate_type="reflected",
            )

    # Direct
    dir_triggers = [find_direct_trigger(e) for e in event_lists]
    if all(t is not None for t in dir_triggers):
        plot_trigger_rate_panels(
            event_lists, dir_triggers, labels,
            suptitle="MB Direct Trigger Rate",
            savename=os.path.join(save_folder, "mb_trigger_rate_direct.png"),
            rate_type="direct",
        )


def generate_sp_trigger_rate_plots(loaded, save_folder, max_distance):
    """Plot 2 + 2b: SP trigger rate panels (reflected at 40dB, and direct)."""
    labels = ["Gen2 Shallow", "Gen2 Deep"]
    sim_names = [SP_REFLECTED_SIMS[("300m", "shallow")], SP_REFLECTED_SIMS[("300m", "deep")]]
    event_lists = [loaded[s] for s in sim_names if s in loaded]
    if len(event_lists) != 2:
        LOGGER.warning("Not all SP 300m sims available, skipping SP trigger rate plots")
        return

    # Reflected at 40 dB
    ref_triggers = [find_trigger_for_db(e, 40.0) for e in event_lists]
    if all(t is not None for t in ref_triggers):
        plot_trigger_rate_panels(
            event_lists, ref_triggers, labels,
            suptitle="SP Reflected Trigger Rate (300m, 40 dB)",
            savename=os.path.join(save_folder, "sp_trigger_rate_reflected.png"),
            rate_type="reflected",
        )
    else:
        ref_triggers = [find_direct_trigger(e) for e in event_lists]
        if all(t is not None for t in ref_triggers):
            plot_trigger_rate_panels(
                event_lists, ref_triggers, labels,
                suptitle="SP Reflected Trigger Rate (300m)",
                savename=os.path.join(save_folder, "sp_trigger_rate_reflected.png"),
                rate_type="reflected",
            )

    # Direct
    dir_triggers = [find_direct_trigger(e) for e in event_lists]
    if all(t is not None for t in dir_triggers):
        plot_trigger_rate_panels(
            event_lists, dir_triggers, labels,
            suptitle="SP Direct Trigger Rate",
            savename=os.path.join(save_folder, "sp_trigger_rate_direct.png"),
            rate_type="direct",
        )


def generate_mb_event_rate_bands(loaded, save_folder, max_distance):
    """Plot 3: MB event rate bands across R values."""
    e_bins, z_bins = getEnergyZenithBins()
    labels = list(MB_REFLECTED_SIMS.keys())
    sim_names = [MB_REFLECTED_SIMS[k] for k in labels]
    event_lists = [loaded.get(s) for s in sim_names]

    if any(e is None for e in event_lists):
        LOGGER.warning("Not all MB reflected sims available, skipping MB event rate bands")
        return

    rate_arrays_per_panel = []
    for events in event_lists:
        db_triggers = get_all_db_triggers(events)
        db_vals = sorted(db_triggers.keys()) if db_triggers else [None]

        rates_for_dbs = []
        for db in db_vals:
            if db is not None:
                trig = db_triggers[db]
            else:
                # Fallback for non-multi-dB data
                trig = find_direct_trigger(events)
            if trig is None:
                continue
            _, ref_rate, _ = getBinnedTriggerRate(events, trig)
            event_rate = getEventRate(ref_rate, e_bins, z_bins, max_distance)
            rates_for_dbs.append(event_rate)

        if rates_for_dbs:
            rate_arrays_per_panel.append(rates_for_dbs)
        else:
            rate_arrays_per_panel.append([np.zeros_like(getEventRate(
                np.zeros((len(e_bins)-1, len(z_bins)-1)), e_bins, z_bins, max_distance
            ))])

    plot_event_rate_bands(
        rate_arrays_per_panel, labels,
        suptitle="MB Reflected Event Rate ($R_{power}$ = 0.5\u20131.0)",
        savename=os.path.join(save_folder, "mb_event_rate_bands.png"),
    )


def generate_sp_event_rate_bands(loaded, save_folder, max_distance):
    """Plot 4: SP event rate bands — combined Gen2 shallow+deep per depth."""
    e_bins, z_bins = getEnergyZenithBins()

    rate_arrays_per_panel = []
    available_depths = []

    for depth in SP_DEPTHS:
        shallow_key = SP_REFLECTED_SIMS.get((depth, "shallow"))
        deep_key = SP_REFLECTED_SIMS.get((depth, "deep"))
        shallow_events = loaded.get(shallow_key)
        deep_events = loaded.get(deep_key)

        if shallow_events is None or deep_events is None:
            LOGGER.info("Skipping SP depth %s (missing data)", depth)
            continue

        shallow_db_trigs = get_all_db_triggers(shallow_events)
        deep_db_trigs = get_all_db_triggers(deep_events)

        # Use the intersection of available dB values
        if shallow_db_trigs and deep_db_trigs:
            common_dbs = sorted(set(shallow_db_trigs.keys()) & set(deep_db_trigs.keys()))
        else:
            # Fallback for non-multi-dB
            common_dbs = [None]

        rates_for_dbs = []
        for db in common_dbs:
            if db is not None:
                s_trig = shallow_db_trigs.get(db)
                d_trig = deep_db_trigs.get(db)
            else:
                s_trig = find_direct_trigger(shallow_events)
                d_trig = find_direct_trigger(deep_events)

            if s_trig is None or d_trig is None:
                continue

            _, s_ref_rate, _ = getBinnedTriggerRate(shallow_events, s_trig)
            _, d_ref_rate, _ = getBinnedTriggerRate(deep_events, d_trig)
            s_event_rate = getEventRate(s_ref_rate, e_bins, z_bins, max_distance)
            d_event_rate = getEventRate(d_ref_rate, e_bins, z_bins, max_distance)
            combined = np.nan_to_num(s_event_rate) + np.nan_to_num(d_event_rate)
            rates_for_dbs.append(combined)

        if rates_for_dbs:
            rate_arrays_per_panel.append(rates_for_dbs)
            available_depths.append(depth)

    if not available_depths:
        LOGGER.warning("No SP depth data available, skipping SP event rate bands")
        return

    plot_event_rate_bands(
        rate_arrays_per_panel, available_depths,
        suptitle="SP Reflected Event Rate \u2014 Combined Gen2 (dB = 40\u201355)",
        savename=os.path.join(save_folder, "sp_event_rate_bands.png"),
    )


def generate_sp_radii_plots(loaded, save_folder, max_distance):
    """Plot 5a: SP radii probability density."""
    labels = ["Gen2 Shallow", "Gen2 Deep"]
    sim_names = [SP_REFLECTED_SIMS[("300m", "shallow")], SP_REFLECTED_SIMS[("300m", "deep")]]
    event_lists = [loaded.get(s) for s in sim_names]

    if any(e is None for e in event_lists):
        LOGGER.warning("Missing SP 300m data, skipping SP radii plots")
        return

    direct_triggers = [find_direct_trigger(e) for e in event_lists]
    reflected_db_triggers = [get_all_db_triggers(e) for e in event_lists]

    plot_radii_density_panels(
        event_lists, direct_triggers, reflected_db_triggers, labels,
        suptitle="SP Radii Distribution \u2014 Event Rate Weighted Density",
        savename=os.path.join(save_folder, "sp_radii_density.png"),
        max_distance=max_distance,
    )


def generate_mb_radii_plots(loaded, save_folder, max_distance):
    """Plot 5b: MB radii probability density."""
    labels = list(MB_REFLECTED_SIMS.keys())
    sim_names = [MB_REFLECTED_SIMS[k] for k in labels]
    event_lists = [loaded.get(s) for s in sim_names]

    if any(e is None for e in event_lists):
        LOGGER.warning("Missing MB data, skipping MB radii plots")
        return

    direct_triggers = [find_direct_trigger(e) for e in event_lists]
    reflected_db_triggers = [get_all_db_triggers(e) for e in event_lists]

    plot_radii_density_panels(
        event_lists, direct_triggers, reflected_db_triggers, labels,
        suptitle="MB Radii Distribution \u2014 Event Rate Weighted Density",
        savename=os.path.join(save_folder, "mb_radii_density.png"),
        max_distance=max_distance,
        db_labels=MB_DB_LABELS,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate Chapter 4 multi-panel comparison plots"
    )
    parser.add_argument("--config", type=str, default="RCRSimulation/config.ini",
                        help="Path to config.ini")
    parser.add_argument("--numpy-folder", type=str, default=None,
                        help="Override numpy folder from config")
    parser.add_argument("--save-folder", type=str, default=None,
                        help="Override save folder from config")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    numpy_folder = args.numpy_folder or config.get(
        "FOLDERS", "numpy_folder", fallback="RCRSimulation/output/numpy"
    )
    save_folder = args.save_folder or config.get(
        "FOLDERS", "save_folder", fallback="RCRSimulation/plots"
    )
    save_folder = os.path.join(save_folder, "chapter4")
    os.makedirs(save_folder, exist_ok=True)

    max_distance = float(config.get("SIMULATION", "distance_km", fallback="5")) / 2 * units.km

    # Collect all sim names to load (direct rates come from the combined sims)
    all_sim_names = set()
    all_sim_names.update(MB_REFLECTED_SIMS.values())
    all_sim_names.update(SP_REFLECTED_SIMS.values())

    ic("Loading simulations from", numpy_folder)
    loaded = {}
    for sim_name in sorted(all_sim_names):
        events = load_combined_events(numpy_folder, sim_name)
        if events is not None:
            loaded[sim_name] = events

    ic(f"Loaded {len(loaded)}/{len(all_sim_names)} simulations")

    if not loaded:
        LOGGER.error("No simulation data found. Check numpy_folder: %s", numpy_folder)
        return

    # Generate plots — each wrapped so one failure doesn't block others
    plot_generators = [
        ("MB trigger rate plots", lambda: generate_mb_trigger_rate_plots(loaded, save_folder, max_distance)),
        ("SP trigger rate plots", lambda: generate_sp_trigger_rate_plots(loaded, save_folder, max_distance)),
        ("MB event rate bands", lambda: generate_mb_event_rate_bands(loaded, save_folder, max_distance)),
        ("SP event rate bands", lambda: generate_sp_event_rate_bands(loaded, save_folder, max_distance)),
        ("SP radii density", lambda: generate_sp_radii_plots(loaded, save_folder, max_distance)),
        ("MB radii density", lambda: generate_mb_radii_plots(loaded, save_folder, max_distance)),
        ("SP rate table", lambda: generate_sp_rate_table(
            loaded, max_distance, os.path.join(save_folder, "sp_rate_table.txt"),
        )),
    ]

    generated = []
    for name, gen_func in plot_generators:
        try:
            gen_func()
            generated.append(name)
        except Exception as e:
            LOGGER.error("Failed to generate %s: %s", name, e, exc_info=True)

    ic(f"Generated {len(generated)}/{len(plot_generators)} outputs")
    for name in generated:
        ic(f"  OK: {name}")


if __name__ == "__main__":
    main()
