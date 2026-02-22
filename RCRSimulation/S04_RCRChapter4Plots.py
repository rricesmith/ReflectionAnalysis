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
    getErrorEventRates,
    getnThrows,
    setEventListRateWeight,
    set_bad_imshow,
    getUniqueEnergyZenithPairs,
    collect_trigger_names,
    filter_real_triggers,
    parse_db_from_trigger_name,
    parse_r_from_trigger_name,
    get_all_r_triggers,
    get_ab_error_triggers,
    find_trigger_for_r,
    NOISE_PRECHECK_SIGMA,
)
from RCRSimulation.RCREventObject import RCREvent, REFLECTED_STATION_OFFSET
from RCRSimulation.NeutrinoComparisonSimulation.NeutrinoEvent import NeutrinoEvent

LOGGER = logging.getLogger(__name__)

# ============================================================================
# Simulation name mappings
# ============================================================================

MB_REFLECTED_SIMS = {
    "HRA": "HRA_MB_576m",
    "Gen2 Shallow": "Gen2_shallow_MB_576m",
    "Gen2 Deep": "Gen2_deep_MB_576m",
}

SP_DEPTHS = ["300m"]

# SP sims keyed by (depth, station_type)
SP_REFLECTED_SIMS = {
    ("300m", "shallow"): "Gen2_shallow_SP_300m",
    ("300m", "deep"): "Gen2_deep_SP_300m",
}

# For direct rates, use the combined sims (which contain both direct and reflected stations).
# These are the "canonical" sims used for direct rates (one per site/station-type).
SP_DIRECT_SIMS = {
    "Gen2 Shallow": "Gen2_shallow_SP_300m",
    "Gen2 Deep": "Gen2_deep_SP_300m",
}

# MB R-based sweep values (amplitude reflectivity)
MB_R_VALUES = [0.5, 0.75, 0.82, 0.89, 1.0]
MB_R_LABELS = {0.5: "R=0.5", 0.75: "R=0.75", 0.82: "R=0.82", 0.89: "R=0.89", 1.0: "R=1.0"}

# SP dB sweep values (kept for SP compatibility)
SP_DB_VALUES = [40.0, 45.0, 50.0, 55.0]

# Index of refraction for Snell's law (air → ice)
N_ICE = 1.78

# Energy mask: minimum log10(E/eV) for specific panels (reduces noise in low-stats regions)
# Applied to Gen2 Deep reflected at MB and SP
MIN_LOG_ENERGY = {"Gen2 Deep": 18.5}


def _apply_energy_mask(rate_array: np.ndarray, e_bins: np.ndarray, panel_label: str,
                       rate_type: str = "reflected") -> np.ndarray:
    """Zero out energy bins below the minimum for panels that need masking.

    Only applies to reflected rate_type and panels listed in MIN_LOG_ENERGY.
    Returns a copy with low-energy bins zeroed.
    """
    if rate_type != "reflected" or panel_label not in MIN_LOG_ENERGY:
        return rate_array
    min_log_e = MIN_LOG_ENERGY[panel_label]
    e_log = np.log10(np.array(e_bins) / units.eV)
    # e_log has len(e_bins) edges; rate_array has len(e_bins)-1 rows
    # Bin i spans e_log[i] to e_log[i+1]; mask if bin center < min_log_e
    e_centers = (e_log[:-1] + e_log[1:]) / 2
    masked = rate_array.copy()
    masked[e_centers < min_log_e] = 0
    return masked


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
# Neutrino Data Loading
# ============================================================================

def load_neutrino_events(neutrino_folder: str | Path) -> Dict[str, list]:
    """Load neutrino comparison events from numpy files.

    Searches for *neutrino_events.npy files in the folder.
    Returns {site: [NeutrinoEvent, ...]} where site is 'MB' or 'SP'.
    """
    folder = Path(neutrino_folder)
    if not folder.exists():
        LOGGER.info("Neutrino folder %s does not exist, skipping", neutrino_folder)
        return {}

    result = {}
    for site in ["MB", "SP"]:
        pattern = f"*{site}*neutrino_events.npy"
        files = sorted(folder.glob(pattern))
        if not files:
            continue
        events = []
        for f in files:
            arr = np.load(f, allow_pickle=True)
            events.extend(arr.tolist())
        result[site] = events
        LOGGER.info("Loaded %d neutrino events for %s (%d files)", len(events), site, len(files))

    return result


def _compute_neutrino_arrival_density(
    nu_events: list,
    trigger_type: str,
    bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute arrival angle density for neutrino events.

    Uses signal arrival zenith (from NuRadioMC ray tracing) directly.

    Args:
        nu_events: List of NeutrinoEvent objects
        trigger_type: 'lpda' or 'pa'
        bins: Bin edges in degrees
    """
    vals, ws = [], []
    for evt in nu_events:
        if trigger_type == "lpda" and evt.has_lpda_trigger():
            zen = evt.lpda_arrival_zenith
        elif trigger_type == "pa" and evt.has_pa_trigger():
            zen = evt.pa_arrival_zenith
        else:
            continue
        if zen is not None:
            vals.append(np.rad2deg(zen))
            ws.append(evt.weight)

    return _compute_weighted_density(np.array(vals), np.array(ws), bins)


def _compute_neutrino_polarization_density(
    nu_events: list,
    trigger_type: str,
    bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute polarization angle density for neutrino events.

    Args:
        nu_events: List of NeutrinoEvent objects
        trigger_type: 'lpda' or 'pa'
        bins: Bin edges in degrees
    """
    vals, ws = [], []
    for evt in nu_events:
        if trigger_type == "lpda" and evt.has_lpda_trigger():
            pol = evt.lpda_polarization
        elif trigger_type == "pa" and evt.has_pa_trigger():
            pol = evt.pa_polarization
        else:
            continue
        if pol is not None:
            vals.append(np.rad2deg(pol))
            ws.append(evt.weight)

    return _compute_weighted_density(np.array(vals), np.array(ws), bins)


# Panel label -> neutrino trigger type mapping
NEUTRINO_TRIGGER_MAP = {
    "HRA": "lpda",
    "Gen2 Shallow": "lpda",
    "Gen2 Deep": "pa",
}


def _build_neutrino_densities(
    nu_events_site: list,
    panel_titles: List[str],
    density_fn,
    bins: np.ndarray,
) -> List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Compute neutrino density per panel based on trigger type mapping.

    Returns a list with one (centers, density, error) tuple per panel,
    or None for panels where no neutrino trigger type is mapped.
    """
    densities = []
    for title in panel_titles:
        ttype = NEUTRINO_TRIGGER_MAP.get(title)
        if ttype is None:
            densities.append(None)
            continue
        densities.append(density_fn(nu_events_site, ttype, bins))
    return densities


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
    info_texts: Optional[List[str]] = None,
):
    """Multi-panel 2D trigger rate histograms with circle overlay.

    Y-axis uses cos(zenith) so that bins (uniform in cos-space) display at
    correct visual widths.

    Args:
        event_lists: One event array per panel
        trigger_names: Trigger name to query per panel
        panel_titles: Subtitle for each panel
        suptitle: Figure super-title
        savename: Output file path
        rate_type: 'reflected' or 'direct'
        info_texts: Optional info string per panel (upper-left annotation)
    """
    n = len(event_lists)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n + 1, 4.5))
    if n == 1:
        axes = [axes]

    e_bins, z_bins = getEnergyZenithBins()
    e_log = np.log10(np.array(e_bins) / units.eV)
    # Bins are uniform in cos(zenith) space — use that for the y-axis
    z_cos = np.cos(np.array(z_bins))
    z_cos_sorted = np.sort(z_cos)  # [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    extent = [e_log[0], e_log[-1], z_cos_sorted[0], z_cos_sorted[-1]]

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
    for ip, (ax, rate, events, title) in enumerate(zip(axes, panel_data, event_lists, panel_titles)):
        masked, cmap = set_bad_imshow(rate, 0)
        # Flip zenith axis so row 0 (smallest zenith = largest cos) maps to top
        im = ax.imshow(
            masked.T[::-1], origin="lower", aspect="auto",
            norm=norm, cmap=cmap, extent=extent,
        )
        # Circle overlay — convert zenith degrees to cos(zenith)
        log_energies, zeniths_deg = getUniqueEnergyZenithPairs(events)
        cos_zeniths = np.cos(np.deg2rad(zeniths_deg))
        ax.scatter(
            log_energies, cos_zeniths,
            facecolors="none", edgecolors="black", s=20, linewidths=0.5,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("log$_{10}$(E/eV)")

        if info_texts and ip < len(info_texts):
            ax.text(0.03, 0.97, info_texts[ip], transform=ax.transAxes,
                    fontsize=6, verticalalignment="top", family="monospace", color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

    axes[0].set_ylabel("cos(zenith)")
    for ax in axes[1:]:
        ax.set_ylabel("")
    # Set y-axis ticks at 0.25 spacing to match zenith bins
    for ax in axes:
        ax.set_yticks(np.arange(0, 1.01, 0.25))

    fig.suptitle(suptitle, fontsize=13, y=1.02)

    if im is not None:
        # Attach colorbar to all axes so all panels share equal width
        fig.colorbar(im, ax=list(axes), label="Trigger Rate", shrink=0.8, pad=0.04)

    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches="tight")
    plt.close()
    ic(f"Saved: {savename}")


def plot_event_rate_panels(
    event_lists: List[np.ndarray],
    trigger_names: List[str],
    panel_titles: List[str],
    suptitle: str,
    savename: str,
    max_distance: float,
    rate_type: str = "reflected",
    info_texts: Optional[List[str]] = None,
):
    """Multi-panel 2D event rate histograms with circle overlay.

    Same layout as trigger rate panels but converts trigger rates to event rates
    (evts/yr) using the cosmic ray spectrum.

    Args:
        event_lists: One event array per panel
        trigger_names: Trigger name to query per panel
        panel_titles: Subtitle for each panel
        suptitle: Figure super-title
        savename: Output file path
        max_distance: Max throw distance for event rate calculation
        rate_type: 'reflected' or 'direct'
        info_texts: Optional info string per panel (upper-left annotation)
    """
    n = len(event_lists)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n + 1, 4.5))
    if n == 1:
        axes = [axes]

    e_bins, z_bins = getEnergyZenithBins()
    e_log = np.log10(np.array(e_bins) / units.eV)
    z_cos = np.cos(np.array(z_bins))
    z_cos_sorted = np.sort(z_cos)
    extent = [e_log[0], e_log[-1], z_cos_sorted[0], z_cos_sorted[-1]]

    all_rates = []
    panel_data = []
    for events, trig, ptitle in zip(event_lists, trigger_names, panel_titles):
        direct_rate, reflected_rate, _ = getBinnedTriggerRate(events, trig)
        trig_rate = reflected_rate if rate_type == "reflected" else direct_rate
        event_rate = getEventRate(trig_rate, e_bins, z_bins, max_distance)
        event_rate = _apply_energy_mask(event_rate, e_bins, ptitle, rate_type)
        panel_data.append(event_rate)
        nonzero = event_rate[event_rate > 0]
        if nonzero.size > 0:
            all_rates.extend(nonzero.tolist())

    if all_rates:
        vmin = min(all_rates) * 0.5
        vmax = max(all_rates) * 1.5
    else:
        vmin, vmax = 1e-6, 1.0
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    im = None
    for ip, (ax, rate, events, title) in enumerate(zip(axes, panel_data, event_lists, panel_titles)):
        masked, cmap = set_bad_imshow(rate, 0)
        im = ax.imshow(
            masked.T[::-1], origin="lower", aspect="auto",
            norm=norm, cmap=cmap, extent=extent,
        )
        log_energies, zeniths_deg = getUniqueEnergyZenithPairs(events)
        cos_zeniths = np.cos(np.deg2rad(zeniths_deg))
        ax.scatter(
            log_energies, cos_zeniths,
            facecolors="none", edgecolors="black", s=20, linewidths=0.5,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("log$_{10}$(E/eV)")

        if info_texts and ip < len(info_texts):
            ax.text(0.03, 0.97, info_texts[ip], transform=ax.transAxes,
                    fontsize=6, verticalalignment="top", family="monospace", color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

    axes[0].set_ylabel("cos(zenith)")
    for ax in axes[1:]:
        ax.set_ylabel("")
    for ax in axes:
        ax.set_yticks(np.arange(0, 1.01, 0.25))

    fig.suptitle(suptitle, fontsize=13, y=1.02)

    if im is not None:
        fig.colorbar(im, ax=list(axes), label="Event Rate (evts/yr)", shrink=0.8, pad=0.04)

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
    error_arrays_per_panel: Optional[List[List[np.ndarray]]] = None,
    info_texts: Optional[List[str]] = None,
):
    """Multi-panel event rate band plots showing min/max across dB values.

    When error arrays are provided, bands extend from (min - 1sigma) to
    (max + 1sigma) across dB values.

    Args:
        rate_arrays_per_panel: [panel_idx][db_idx] -> event_rate 2D array (energy, zenith)
        panel_titles: Subtitle per panel
        suptitle: Figure super-title
        savename: Output file path
        error_arrays_per_panel: Optional matching error arrays
        info_texts: Optional info string per panel (upper-left annotation)
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

    # Track global bounds for consistent axes across all panels
    global_ymax = 0
    data_x_mask = np.zeros(len(x_vals), dtype=bool)

    for ip, (ax, rates_list, title) in enumerate(zip(axes, rate_arrays_per_panel, panel_titles)):
        # Stack: shape (n_db, n_energy, n_zenith)
        stacked = np.array([np.nan_to_num(r) for r in rates_list])
        has_errors = (error_arrays_per_panel is not None and ip < len(error_arrays_per_panel))
        if has_errors:
            err_stacked = np.array([np.nan_to_num(e) for e in error_arrays_per_panel[ip]])

        # Per zenith bin band
        for iz in range(n_zenith):
            if has_errors:
                lo = np.nanmin(stacked[:, :, iz] - err_stacked[:, :, iz], axis=0)
                hi = np.nanmax(stacked[:, :, iz] + err_stacked[:, :, iz], axis=0)
                lo = np.maximum(lo, 0)
            else:
                lo = np.nanmin(stacked[:, :, iz], axis=0)
                hi = np.nanmax(stacked[:, :, iz], axis=0)
            zen_lo = z_bins[iz] / units.deg
            zen_hi = z_bins[iz + 1] / units.deg
            label = f"{zen_lo:.0f}-{zen_hi:.0f}\u00b0"
            ax.fill_between(x_vals, lo, hi, alpha=0.3, color=colors[iz], label=label)

        # Total (sum of zenith bins) band
        total_rates = np.nansum(stacked, axis=2)  # shape (n_db, n_energy)
        if has_errors:
            total_errors = np.sqrt(np.nansum(err_stacked**2, axis=2))
            lo_total = np.nanmin(total_rates - total_errors, axis=0)
            hi_total = np.nanmax(total_rates + total_errors, axis=0)
            lo_total = np.maximum(lo_total, 0)
        else:
            lo_total = np.nanmin(total_rates, axis=0)
            hi_total = np.nanmax(total_rates, axis=0)
        ax.fill_between(x_vals, lo_total, hi_total, alpha=0.2, color="black", label="Total")

        # Track global y-max and x-range where data exists
        if hi_total.max() > global_ymax:
            global_ymax = hi_total.max()
        data_x_mask |= (hi_total > 0)

        ax.set_yscale("log")
        ax.set_xlabel("log$_{10}$(E/eV)")
        ax.set_title(title, fontsize=10)

        # Info text annotation
        if info_texts and ip < len(info_texts):
            ax.text(0.03, 0.97, info_texts[ip], transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Set consistent axis limits across all panels
    if data_x_mask.any():
        x_data = x_vals[data_x_mask]
        x_pad = 0.25  # half a bin width in log-energy space
        x_lo, x_hi = x_data.min() - x_pad, x_data.max() + x_pad
    else:
        x_lo, x_hi = x_vals[0], x_vals[-1]

    y_top = global_ymax * 3 if global_ymax > 0 else 1.0
    for ax in axes:
        ax.set_ylim(bottom=1e-3, top=y_top)
        ax.set_xlim(x_lo, x_hi)

    axes[0].set_ylabel("Event Rate (evts/yr)")
    axes[0].legend(fontsize=7, loc="lower left")

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches="tight")
    plt.close()
    ic(f"Saved: {savename}")


def plot_event_rate_bands_single_db(
    rate_arrays_per_panel: List[np.ndarray],
    panel_titles: List[str],
    suptitle: str,
    savename: str,
    error_arrays_per_panel: Optional[List[np.ndarray]] = None,
    info_texts: Optional[List[str]] = None,
):
    """Multi-panel event rate band plot for a single dB value.

    Shows the statistical uncertainty band (rate +/- 1sigma) with a central
    dashed line representing the rate without statistical uncertainty.

    Args:
        rate_arrays_per_panel: [panel_idx] -> event_rate 2D array (energy, zenith)
        panel_titles: Subtitle per panel
        suptitle: Figure super-title
        savename: Output file path
        error_arrays_per_panel: Optional matching error arrays (same shape)
        info_texts: Optional info string per panel (upper-left annotation)
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

    global_ymax = 0
    data_x_mask = np.zeros(len(x_vals), dtype=bool)

    for ip, (ax, rate, title) in enumerate(zip(axes, rate_arrays_per_panel, panel_titles)):
        rate = np.nan_to_num(rate)
        has_errors = (error_arrays_per_panel is not None and ip < len(error_arrays_per_panel))
        if has_errors:
            err = np.nan_to_num(error_arrays_per_panel[ip])

        # Per zenith bin band
        for iz in range(n_zenith):
            zen_lo = z_bins[iz] / units.deg
            zen_hi = z_bins[iz + 1] / units.deg
            label = f"{zen_lo:.0f}-{zen_hi:.0f}\u00b0"
            central = rate[:, iz]
            # Dashed central line (no uncertainty)
            ax.plot(x_vals, central, color=colors[iz], linewidth=1, linestyle="--", alpha=0.7)
            if has_errors:
                lo = np.maximum(central - err[:, iz], 0)
                hi = central + err[:, iz]
            else:
                lo = central
                hi = central
            ax.fill_between(x_vals, lo, hi, alpha=0.3, color=colors[iz], label=label)

        # Total (sum of zenith bins) band
        total_rate = np.nansum(rate, axis=1)
        ax.plot(x_vals, total_rate, color="black", linewidth=1, linestyle="--", alpha=0.7)
        if has_errors:
            total_err = np.sqrt(np.nansum(err**2, axis=1))
            lo_total = np.maximum(total_rate - total_err, 0)
            hi_total = total_rate + total_err
        else:
            lo_total = total_rate
            hi_total = total_rate
        ax.fill_between(x_vals, lo_total, hi_total, alpha=0.2, color="black", label="Total")

        if hi_total.max() > global_ymax:
            global_ymax = hi_total.max()
        data_x_mask |= (hi_total > 0)

        ax.set_yscale("log")
        ax.set_xlabel("log$_{10}$(E/eV)")
        ax.set_title(title, fontsize=10)

        if info_texts and ip < len(info_texts):
            ax.text(0.03, 0.97, info_texts[ip], transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    if data_x_mask.any():
        x_data = x_vals[data_x_mask]
        x_pad = 0.25
        x_lo, x_hi = x_data.min() - x_pad, x_data.max() + x_pad
    else:
        x_lo, x_hi = x_vals[0], x_vals[-1]

    y_top = global_ymax * 3 if global_ymax > 0 else 1.0
    for ax in axes:
        ax.set_ylim(bottom=1e-3, top=y_top)
        ax.set_xlim(x_lo, x_hi)

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

def _compute_weighted_density(
    values: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalized probability density histogram with error.

    Returns (bin_centers, density, density_error).
    Error uses sqrt(sum(w^2)) per bin, normalized the same as density.
    """
    centers = (bins[:-1] + bins[1:]) / 2
    n_bins = len(centers)
    bin_width = bins[1] - bins[0]

    if len(values) == 0:
        return centers, np.zeros(n_bins), np.zeros(n_bins)

    hist, _ = np.histogram(values, bins=bins, weights=weights)
    hist_w2, _ = np.histogram(values, bins=bins, weights=weights**2)

    total = hist.sum()
    if total > 0:
        density = hist / (total * bin_width)
        density_error = np.sqrt(hist_w2) / (total * bin_width)
    else:
        density = np.zeros(n_bins)
        density_error = np.zeros(n_bins)

    return centers, density, density_error


def _collect_weighted_values(
    event_list: Sequence[RCREvent],
    weight_name: str,
    value_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect (values, weights) from events using a value extraction function.

    Args:
        event_list: Events to iterate
        weight_name: Name of weight to query
        value_fn: Callable(event) -> float, extracts the x-axis value

    Returns:
        (values, weights) arrays for events with positive weight
    """
    vals = []
    ws = []
    for evt in event_list:
        w = evt.get_weight(weight_name)
        if w > 0:
            vals.append(value_fn(evt))
            ws.append(w)
    return np.array(vals), np.array(ws)


def _compute_radii_density(
    event_list: Sequence[RCREvent],
    weight_name: str,
    max_distance: float,
    n_bins: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalized probability density histogram of radii.

    Returns (bin_centers, density, density_error).
    """
    bins = np.linspace(0, max_distance, n_bins + 1)
    vals, ws = _collect_weighted_values(event_list, weight_name, lambda e: e.get_radius())
    return _compute_weighted_density(vals, ws, bins)


def _compute_arrival_angle_density(
    event_list: Sequence[RCREvent],
    weight_name: str,
    n_bins: int = 15,
    max_angle_deg: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalized probability density of refracted arrival angle.

    Applies Snell's law: arrival_angle = arcsin(sin(zenith) / N_ICE).
    Returns (bin_centers_deg, density, density_error).

    Args:
        max_angle_deg: Upper edge of histogram in degrees. Defaults to
            Snell's law limit (~34.2 deg). Set to 90 when overlaying
            neutrino distributions.
    """
    if max_angle_deg is None:
        max_angle_deg = np.rad2deg(np.arcsin(1.0 / N_ICE))  # ~34.2 deg
    bins = np.linspace(0, max_angle_deg, n_bins + 1)

    def arrival_angle_deg(evt):
        sin_refr = np.sin(evt.zenith) / N_ICE
        sin_refr = np.clip(sin_refr, -1, 1)
        return np.rad2deg(np.arcsin(sin_refr))

    vals, ws = _collect_weighted_values(event_list, weight_name, arrival_angle_deg)
    return _compute_weighted_density(vals, ws, bins)


def _compute_polarization_angle_density(
    event_list: Sequence[RCREvent],
    weight_name: str,
    n_bins: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute normalized probability density of polarization angle.

    Uses direct station polarization when weight_name starts with 'direct',
    otherwise uses reflected station polarization.
    Returns (bin_centers_deg, density, density_error).
    """
    bins = np.linspace(0, 180, n_bins + 1)
    use_direct = weight_name.startswith("direct")

    def pol_angle_deg(evt):
        if use_direct:
            a = evt.get_direct_polarization_angle()
        else:
            a = evt.get_reflected_polarization_angle()
        if a is None:
            return np.nan
        return np.rad2deg(a)

    vals, ws = _collect_weighted_values(event_list, weight_name, pol_angle_deg)
    # Filter out events with no polarization data
    if len(vals) > 0:
        mask = np.isfinite(vals)
        vals, ws = vals[mask], ws[mask]
    return _compute_weighted_density(vals, ws, bins)


def _plot_density_panels(
    event_lists: List[np.ndarray],
    direct_triggers: List[str],
    reflected_db_triggers: List[Dict[float, str]],
    panel_titles: List[str],
    suptitle: str,
    savename: str,
    max_distance: float,
    density_fn,
    xlabel: str,
    ylabel: str = "Probability Density",
    db_labels: Dict[float, str] | None = None,
    n_bins: int = 15,
    info_texts: Optional[List[str]] = None,
    neutrino_densities: Optional[List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = None,
):
    """Generic multi-panel density plot with direct band + reflected band.

    Direct lines become bands (±1sigma). Reflected bands extend from
    (min - 1sigma) to (max + 1sigma) across dB values.

    Args:
        event_lists: One event array per panel
        direct_triggers: Direct trigger name per panel
        reflected_db_triggers: [{db: trigger_name}] per panel
        panel_titles: Subtitle per panel
        suptitle: Figure super-title
        savename: Output path
        max_distance: Max radius (meters), used for weight computation
        density_fn: Callable(event_list, weight_name, n_bins) -> (centers, density, error)
        xlabel: X-axis label
        ylabel: Y-axis label
        db_labels: Optional {db_value: display_label} for legend
        n_bins: Number of bins
        info_texts: Optional info string per panel (upper-left annotation)
        neutrino_densities: Optional list of (centers, density, error) tuples per panel.
            When provided, overlays the neutrino signal distribution on each panel.
    """
    n = len(event_lists)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    e_bins, z_bins = getEnergyZenithBins()

    for ip, (ax, events, dir_trig, db_trigs, title) in enumerate(zip(
        axes, event_lists, direct_triggers, reflected_db_triggers, panel_titles
    )):
        events_list = list(events)

        # Direct: step histogram outline with ±1sigma shading
        if dir_trig:
            direct_rate, _, _ = getBinnedTriggerRate(events_list, dir_trig)
            setEventListRateWeight(
                events_list, direct_rate, "direct", dir_trig,
                max_distance=max_distance, use_direct=True,
            )
            centers, density, density_err = density_fn(events_list, "direct", n_bins)
            ax.fill_between(centers, np.maximum(density - density_err, 0), density + density_err,
                            alpha=0.3, color="black", step="mid")
            ax.step(centers, density, color="black", linewidth=1.5, label="Direct", where="mid")

        # Reflected: step histogram band from (min - 1sigma) to (max + 1sigma)
        if db_trigs:
            sorted_dbs = sorted(db_trigs.keys())
            all_densities = []
            all_errors = []
            for sweep_val in sorted_dbs:
                ref_trig = db_trigs[sweep_val]
                weight_name = f"reflected_{sweep_val}"
                _, reflected_rate, _ = getBinnedTriggerRate(events_list, ref_trig)
                setEventListRateWeight(
                    events_list, reflected_rate, weight_name, ref_trig,
                    max_distance=max_distance, use_direct=False,
                )
                centers, density, density_err = density_fn(events_list, weight_name, n_bins)
                all_densities.append(density)
                all_errors.append(density_err)

            if all_densities:
                all_densities = np.array(all_densities)
                all_errors = np.array(all_errors)
                lo_idx = np.argmin(all_densities, axis=0)
                hi_idx = np.argmax(all_densities, axis=0)
                lo = np.array([all_densities[lo_idx[i], i] - all_errors[lo_idx[i], i]
                               for i in range(len(centers))])
                hi = np.array([all_densities[hi_idx[i], i] + all_errors[hi_idx[i], i]
                               for i in range(len(centers))])
                lo = np.maximum(lo, 0)
                if db_labels:
                    labels_sorted = [db_labels.get(d, f"{d:.0f} dB") for d in sorted_dbs]
                    band_label = f"Reflected ({labels_sorted[0]}\u2013{labels_sorted[-1]})"
                else:
                    band_label = f"Reflected ({sorted_dbs[0]:.0f}\u2013{sorted_dbs[-1]:.0f} dB)"
                ax.fill_between(centers, lo, hi, alpha=0.3, color="tab:blue",
                                label=band_label, step="mid")

        # Neutrino overlay
        if neutrino_densities and ip < len(neutrino_densities) and neutrino_densities[ip] is not None:
            nu_centers, nu_density, nu_error = neutrino_densities[ip]
            if nu_density.sum() > 0:
                ax.fill_between(nu_centers, np.maximum(nu_density - nu_error, 0),
                                nu_density + nu_error, alpha=0.25, color="tab:orange",
                                step="mid")
                ax.step(nu_centers, nu_density, color="tab:orange", linewidth=1.5,
                        label="Neutrino Signal", where="mid")

        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)

        if info_texts and ip < len(info_texts):
            ax.text(0.03, 0.97, info_texts[ip], transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=7)

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches="tight")
    plt.close()
    ic(f"Saved: {savename}")


def plot_radii_density_panels(
    event_lists, direct_triggers, reflected_db_triggers, panel_titles,
    suptitle, savename, max_distance, db_labels=None, n_bins=15, info_texts=None,
):
    """Multi-panel radii probability density with direct band + reflected band."""
    def radii_density_fn(events_list, weight_name, nbins):
        return _compute_radii_density(events_list, weight_name, max_distance, nbins)

    _plot_density_panels(
        event_lists, direct_triggers, reflected_db_triggers, panel_titles,
        suptitle, savename, max_distance, radii_density_fn,
        xlabel="Radius (m)", db_labels=db_labels, n_bins=n_bins, info_texts=info_texts,
    )


def plot_arrival_angle_panels(
    event_lists, direct_triggers, reflected_db_triggers, panel_titles,
    suptitle, savename, max_distance, db_labels=None, n_bins=12, info_texts=None,
    neutrino_densities=None,
):
    """Multi-panel refracted arrival angle density with direct band + reflected band."""
    max_angle = 90.0 if neutrino_densities else None

    def arrival_density_fn(events_list, weight_name, nbins):
        return _compute_arrival_angle_density(events_list, weight_name, nbins,
                                              max_angle_deg=max_angle)

    _plot_density_panels(
        event_lists, direct_triggers, reflected_db_triggers, panel_titles,
        suptitle, savename, max_distance, arrival_density_fn,
        xlabel="Arrival Angle (deg)", db_labels=db_labels, n_bins=n_bins,
        info_texts=info_texts, neutrino_densities=neutrino_densities,
    )


def plot_polarization_angle_panels(
    event_lists, direct_triggers, reflected_db_triggers, panel_titles,
    suptitle, savename, max_distance, db_labels=None, n_bins=15, info_texts=None,
    neutrino_densities=None,
):
    """Multi-panel polarization angle density with direct band + reflected band."""
    def pol_density_fn(events_list, weight_name, nbins):
        return _compute_polarization_angle_density(events_list, weight_name, nbins)

    _plot_density_panels(
        event_lists, direct_triggers, reflected_db_triggers, panel_titles,
        suptitle, savename, max_distance, pol_density_fn,
        xlabel="Polarization Angle (deg)", db_labels=db_labels, n_bins=n_bins,
        info_texts=info_texts, neutrino_densities=neutrino_densities,
    )


# ============================================================================
# Output 6: Gen2 SP Rate Table
# ============================================================================

def generate_rate_table(
    loaded: Dict[str, np.ndarray],
    max_distance: float,
    savename: str,
):
    """Generate human-readable text table of event rates for both MB and SP.

    MB table: Rows = Direct, 576m (R=0.5-1.0). Columns = HRA, Gen2 Shallow, Gen2 Deep.
    SP table: Rows = Direct, 300m. Columns = Shallow, Deep.
    Each reflected cell shows min-max range across dB values.
    """
    e_bins, z_bins = getEnergyZenithBins()

    def total_reflected_rate_db(events, db_val, panel_label=""):
        trig = find_trigger_for_db(events, db_val)
        if trig is None:
            return None
        _, ref_rate, _ = getBinnedTriggerRate(events, trig)
        event_rate = getEventRate(ref_rate, e_bins, z_bins, max_distance)
        event_rate = _apply_energy_mask(event_rate, e_bins, panel_label, "reflected")
        return float(np.nansum(event_rate))

    def total_reflected_rate_r(events, r_val, panel_label=""):
        trig = find_trigger_for_r(events, r_val)
        if trig is None:
            return None
        _, ref_rate, _ = getBinnedTriggerRate(events, trig)
        event_rate = getEventRate(ref_rate, e_bins, z_bins, max_distance)
        event_rate = _apply_energy_mask(event_rate, e_bins, panel_label, "reflected")
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

    # ---- MB Table ----
    lines.append("MB Event Rates (evts/yr)")
    lines.append("=" * 70)
    mb_col_labels = list(MB_REFLECTED_SIMS.keys())  # HRA, Gen2 Shallow, Gen2 Deep
    header = f"{'':24s}" + "".join(f"{c:>15s}" for c in mb_col_labels)
    lines.append(header)
    lines.append("-" * 70)

    # MB Direct row
    mb_direct_vals = []
    for label in mb_col_labels:
        sim_name = MB_REFLECTED_SIMS[label]
        events = loaded.get(sim_name)
        r = total_direct_rate(events) if events is not None else None
        mb_direct_vals.append(format_single(r))
    lines.append(f"{'Direct':24s}" + "".join(f"{v:>15s}" for v in mb_direct_vals))

    # MB Reflected row (576m, R=0.5-1.0)
    mb_refl_vals = []
    for label in mb_col_labels:
        sim_name = MB_REFLECTED_SIMS[label]
        events = loaded.get(sim_name)
        rates = []
        if events is not None:
            # Try R-based triggers first
            r_triggers = get_all_r_triggers(events)
            if r_triggers:
                for r_val in sorted(r_triggers.keys()):
                    r = total_reflected_rate_r(events, r_val, panel_label=label)
                    rates.append(r)
            else:
                # Fallback to dB-based triggers
                for db in [0.0, 1.5, 3.0]:
                    r = total_reflected_rate_db(events, db, panel_label=label)
                    rates.append(r)
        mb_refl_vals.append(format_range(rates))
    lines.append(f"{'576m (R=0.5-1.0)':24s}" + "".join(f"{v:>15s}" for v in mb_refl_vals))

    lines.append("=" * 70)
    lines.append("")

    # ---- SP Table ----
    lines.append("Gen2 SP Event Rates (evts/yr)")
    lines.append("=" * 55)
    lines.append(f"{'':24s}{'Shallow':>15s}{'Deep':>15s}")
    lines.append("-" * 55)

    # SP Direct row
    shallow_dir = loaded.get(SP_DIRECT_SIMS.get("Gen2 Shallow"))
    deep_dir = loaded.get(SP_DIRECT_SIMS.get("Gen2 Deep"))
    r_shallow = total_direct_rate(shallow_dir) if shallow_dir is not None else None
    r_deep = total_direct_rate(deep_dir) if deep_dir is not None else None
    lines.append(
        f"{'Direct':24s}{format_single(r_shallow):>15s}{format_single(r_deep):>15s}"
    )

    # SP Reflected rows per depth
    for depth in SP_DEPTHS:
        shallow_key = SP_REFLECTED_SIMS.get((depth, "shallow"))
        deep_key = SP_REFLECTED_SIMS.get((depth, "deep"))
        shallow_events = loaded.get(shallow_key)
        deep_events = loaded.get(deep_key)

        shallow_rates = []
        deep_rates = []

        for db in SP_DB_VALUES:
            rs = total_reflected_rate_db(shallow_events, db, panel_label="Gen2 Shallow") if shallow_events is not None else None
            rd = total_reflected_rate_db(deep_events, db, panel_label="Gen2 Deep") if deep_events is not None else None
            shallow_rates.append(rs)
            deep_rates.append(rd)

        row_label = f"{depth} (40-55 dB)"
        lines.append(
            f"{row_label:24s}{format_range(shallow_rates):>15s}{format_range(deep_rates):>15s}"
        )

    lines.append("=" * 55)

    if MIN_LOG_ENERGY:
        lines.append("")
        lines.append("Notes:")
        for panel, min_e in MIN_LOG_ENERGY.items():
            lines.append(f"  {panel} reflected: rates masked below 10^{min_e} eV")

    table_text = "\n".join(lines)
    print(table_text)

    with open(savename, "w") as f:
        f.write(table_text + "\n")
    ic(f"Saved: {savename}")


def generate_mb_error_breakdown_table(
    loaded: Dict[str, np.ndarray],
    max_distance: float,
    savename: str,
):
    """Generate detailed MB error breakdown table showing each error component.

    For each MB sim and R value, shows: rate, stat error, δ_A, δ_B, combined error.
    At the end, shows the final reported rate as "rate ± combined_error" at each R endpoint.
    """
    e_bins, z_bins = getEnergyZenithBins()
    lines = []
    lines.append("MB Error Breakdown (evts/yr)")
    lines.append("=" * 100)

    for label in MB_REFLECTED_SIMS:
        sim_name = MB_REFLECTED_SIMS[label]
        events = loaded.get(sim_name)
        if events is None:
            continue

        events_list = list(events)
        r_triggers = get_all_r_triggers(events_list)
        if not r_triggers:
            lines.append(f"\n{label}: No R-based triggers found (dB-based data, no A/B breakdown)")
            continue

        lines.append(f"\n{label}")
        lines.append("-" * 100)
        lines.append(f"{'R':>6s}  {'Rate':>10s}  {'σ_stat':>10s}  {'δ_A':>10s}  {'δ_B':>10s}  {'σ_comb':>10s}  {'Rate ± σ':>22s}")
        lines.append("-" * 100)

        r_vals = sorted(r_triggers.keys())
        r_min = min(r_vals)
        r_max = max(r_vals)
        ab_triggers = get_ab_error_triggers(events_list)

        for r_val in r_vals:
            trig = r_triggers[r_val]
            _, ref_rate, _ = getBinnedTriggerRate(events_list, trig)
            event_rate = getEventRate(ref_rate, e_bins, z_bins, max_distance)
            event_rate = _apply_energy_mask(event_rate, e_bins, label, "reflected")
            stat_error = getErrorEventRates(ref_rate, events_list, max_distance)
            stat_error = _apply_energy_mask(stat_error, e_bins, label, "reflected")

            total_rate = float(np.nansum(event_rate))
            total_stat = float(np.sqrt(np.nansum(stat_error**2)))

            # A/B breakdown only at endpoints
            delta_A_total = 0.0
            delta_B_total = 0.0
            if abs(r_val - r_max) < 0.005 or abs(r_val - r_min) < 0.005:
                # Compute per-variant rates
                def _variant_total(variant_tag):
                    trig_v = ab_triggers.get((r_val, variant_tag))
                    if trig_v is None:
                        return None
                    _, rr, _ = getBinnedTriggerRate(events_list, trig_v)
                    er = getEventRate(rr, e_bins, z_bins, max_distance)
                    er = _apply_energy_mask(er, e_bins, label, "reflected")
                    return float(np.nansum(er))

                rate_Ap = _variant_total("Ap")
                rate_Am = _variant_total("Am")
                rate_Bp = _variant_total("Bp")
                rate_Bm = _variant_total("Bm")

                a_devs = [abs(r - total_rate) for r in [rate_Ap, rate_Am] if r is not None]
                b_devs = [abs(r - total_rate) for r in [rate_Bp, rate_Bm] if r is not None]
                delta_A_total = max(a_devs) if a_devs else 0.0
                delta_B_total = max(b_devs) if b_devs else 0.0

            combined = np.sqrt(total_stat**2 + delta_A_total**2 + delta_B_total**2)

            # Format the row
            if delta_A_total > 0 or delta_B_total > 0:
                lines.append(
                    f"{r_val:6.2f}  {total_rate:10.4f}  {total_stat:10.4f}  "
                    f"{delta_A_total:10.4f}  {delta_B_total:10.4f}  {combined:10.4f}  "
                    f"{total_rate:8.4f} ± {combined:.4f}"
                )
            else:
                lines.append(
                    f"{r_val:6.2f}  {total_rate:10.4f}  {total_stat:10.4f}  "
                    f"{'--':>10s}  {'--':>10s}  {total_stat:10.4f}  "
                    f"{total_rate:8.4f} ± {total_stat:.4f}"
                )

        lines.append("")

    lines.append("=" * 100)
    lines.append("Notes:")
    lines.append("  σ_stat  = Poisson statistical error (sqrt of weighted counts)")
    lines.append("  δ_A     = max |rate(A±σ) - rate(nom)| at R endpoints; A = 460 ± 20 m")
    lines.append("  δ_B     = max |rate(B±σ) - rate(nom)| at R endpoints; B = 180 ± 40 m/GHz")
    lines.append("  σ_comb  = sqrt(σ_stat² + δ_A² + δ_B²)")
    lines.append("  '--' indicates middle R values where only stat error is computed")
    if MIN_LOG_ENERGY:
        for panel, min_e in MIN_LOG_ENERGY.items():
            lines.append(f"  {panel}: rates masked below 10^{min_e} eV")

    table_text = "\n".join(lines)
    print(table_text)

    with open(savename, "w") as f:
        f.write(table_text + "\n")
    ic(f"Saved: {savename}")


# ============================================================================
# Info Text Builder
# ============================================================================

# Display name overrides for trigger names in plot annotations
TRIGGER_DISPLAY_NAMES = {
    "Gen2 Shallow": "LPDA_2of4_100Hz",
}


def _build_info(site: str, station: str, trigger: str | None = None,
                reflectivity: str | None = None) -> str:
    """Build info annotation string for a plot panel."""
    site_full = "Moore's Bay" if site == "MB" else "South Pole"
    parts = [f"Site: {site_full}", f"Station: {station}"]
    if trigger:
        display_name = TRIGGER_DISPLAY_NAMES.get(station, trigger)
        parts.append(f"Trigger: {display_name}")
    if reflectivity:
        parts.append(reflectivity)
    return "\n".join(parts)


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

    # Reflected at R=0.82 (nominal), fallback to R=0.7 (1.5 dB)
    ref_triggers = [find_trigger_for_r(e, 0.82) for e in event_lists]
    ref_label = "R=0.82"
    if not all(t is not None for t in ref_triggers):
        ref_triggers = [find_trigger_for_db(e, 1.5) for e in event_lists]
        ref_label = "R=0.7 (1.5 dB)"
    if all(t is not None for t in ref_triggers):
        infos = [_build_info("MB", lab, trig, ref_label)
                 for lab, trig in zip(labels, ref_triggers)]
        plot_trigger_rate_panels(
            event_lists, ref_triggers, labels,
            suptitle=f"MB Reflected Trigger Rate ({ref_label})",
            savename=os.path.join(save_folder, "mb_trigger_rate_reflected.png"),
            rate_type="reflected", info_texts=infos,
        )

    # Direct
    dir_triggers = [find_direct_trigger(e) for e in event_lists]
    if all(t is not None for t in dir_triggers):
        infos = [_build_info("MB", lab, trig) for lab, trig in zip(labels, dir_triggers)]
        plot_trigger_rate_panels(
            event_lists, dir_triggers, labels,
            suptitle="MB Direct Trigger Rate",
            savename=os.path.join(save_folder, "mb_trigger_rate_direct.png"),
            rate_type="direct", info_texts=infos,
        )


def _generate_sp_trigger_rate_for_depth(loaded, save_folder, max_distance, depth):
    """SP trigger rate panels for a specific depth (reflected at 40dB, and direct)."""
    labels = ["Gen2 Shallow", "Gen2 Deep"]
    sim_names = [SP_REFLECTED_SIMS.get((depth, "shallow")), SP_REFLECTED_SIMS.get((depth, "deep"))]
    event_lists = [loaded.get(s) for s in sim_names]
    if any(e is None for e in event_lists):
        LOGGER.warning("Not all SP %s sims available, skipping SP trigger rate plots for %s", depth, depth)
        return

    suffix = f"_{depth}" if depth != "300m" else ""

    # Reflected at 40 dB
    ref_triggers = [find_trigger_for_db(e, 40.0) for e in event_lists]
    if all(t is not None for t in ref_triggers):
        infos = [_build_info("SP", lab, trig, f"{depth}, 40 dB")
                 for lab, trig in zip(labels, ref_triggers)]
        plot_trigger_rate_panels(
            event_lists, ref_triggers, labels,
            suptitle=f"SP Reflected Trigger Rate ({depth}, 40 dB)",
            savename=os.path.join(save_folder, f"sp_trigger_rate_reflected{suffix}.png"),
            rate_type="reflected", info_texts=infos,
        )

    # Direct
    dir_triggers = [find_direct_trigger(e) for e in event_lists]
    if all(t is not None for t in dir_triggers):
        infos = [_build_info("SP", lab, trig) for lab, trig in zip(labels, dir_triggers)]
        plot_trigger_rate_panels(
            event_lists, dir_triggers, labels,
            suptitle=f"SP Direct Trigger Rate ({depth})",
            savename=os.path.join(save_folder, f"sp_trigger_rate_direct{suffix}.png"),
            rate_type="direct", info_texts=infos,
        )


def _generate_sp_event_rate_2d_for_depth(loaded, save_folder, max_distance, depth):
    """SP 2D event rate histograms for a specific depth (reflected at 40dB, and direct)."""
    labels = ["Gen2 Shallow", "Gen2 Deep"]
    sim_names = [SP_REFLECTED_SIMS.get((depth, "shallow")), SP_REFLECTED_SIMS.get((depth, "deep"))]
    event_lists = [loaded.get(s) for s in sim_names]
    if any(e is None for e in event_lists):
        LOGGER.warning("Not all SP %s sims available, skipping SP event rate 2D for %s", depth, depth)
        return

    suffix = f"_{depth}" if depth != "300m" else ""

    # Reflected at 40 dB
    ref_triggers = [find_trigger_for_db(e, 40.0) for e in event_lists]
    if all(t is not None for t in ref_triggers):
        infos = [_build_info("SP", lab, trig, f"{depth}, 40 dB")
                 for lab, trig in zip(labels, ref_triggers)]
        plot_event_rate_panels(
            event_lists, ref_triggers, labels,
            suptitle=f"SP Reflected Event Rate ({depth}, 40 dB)",
            savename=os.path.join(save_folder, f"sp_event_rate_reflected_2d{suffix}.png"),
            max_distance=max_distance, rate_type="reflected", info_texts=infos,
        )

    # Direct
    dir_triggers = [find_direct_trigger(e) for e in event_lists]
    if all(t is not None for t in dir_triggers):
        infos = [_build_info("SP", lab, trig) for lab, trig in zip(labels, dir_triggers)]
        plot_event_rate_panels(
            event_lists, dir_triggers, labels,
            suptitle=f"SP Direct Event Rate ({depth})",
            savename=os.path.join(save_folder, f"sp_event_rate_direct_2d{suffix}.png"),
            max_distance=max_distance, rate_type="direct", info_texts=infos,
        )


def generate_sp_trigger_rate_plots(loaded, save_folder, max_distance):
    """SP trigger rate panels for 300m (reflected at 40dB, and direct)."""
    _generate_sp_trigger_rate_for_depth(loaded, save_folder, max_distance, "300m")



def generate_mb_event_rate_2d(loaded, save_folder, max_distance):
    """MB 2D event rate histograms (reflected at R=0.7, and direct)."""
    labels = list(MB_REFLECTED_SIMS.keys())
    sim_names = [MB_REFLECTED_SIMS[k] for k in labels]
    event_lists = [loaded[s] for s in sim_names if s in loaded]
    if len(event_lists) != len(labels):
        LOGGER.warning("Not all MB reflected sims available, skipping MB event rate 2D")
        return

    # Reflected at R=0.82 (nominal), fallback to 1.5 dB
    ref_triggers = [find_trigger_for_r(e, 0.82) for e in event_lists]
    ref_label = "R=0.82"
    if not all(t is not None for t in ref_triggers):
        ref_triggers = [find_trigger_for_db(e, 1.5) for e in event_lists]
        ref_label = "R=0.7 (1.5 dB)"
    if all(t is not None for t in ref_triggers):
        infos = [_build_info("MB", lab, trig, ref_label)
                 for lab, trig in zip(labels, ref_triggers)]
        plot_event_rate_panels(
            event_lists, ref_triggers, labels,
            suptitle=f"MB Reflected Event Rate ({ref_label})",
            savename=os.path.join(save_folder, "mb_event_rate_reflected_2d.png"),
            max_distance=max_distance, rate_type="reflected", info_texts=infos,
        )

    # Direct
    dir_triggers = [find_direct_trigger(e) for e in event_lists]
    if all(t is not None for t in dir_triggers):
        infos = [_build_info("MB", lab, trig) for lab, trig in zip(labels, dir_triggers)]
        plot_event_rate_panels(
            event_lists, dir_triggers, labels,
            suptitle="MB Direct Event Rate",
            savename=os.path.join(save_folder, "mb_event_rate_direct_2d.png"),
            max_distance=max_distance, rate_type="direct", info_texts=infos,
        )


def generate_sp_event_rate_2d(loaded, save_folder, max_distance):
    """SP 2D event rate histograms for 300m (reflected at 40dB, and direct)."""
    _generate_sp_event_rate_2d_for_depth(loaded, save_folder, max_distance, "300m")



def _compute_ab_combined_error(events, nominal_rate, r_value, stat_error, e_bins, z_bins, max_distance):
    """Compute combined error from A/B variants and statistical error at a given R.

    For upper bound (high R): takes max(rate(Ap), rate(Am)) - nominal for δ_A, similarly for B.
    For lower bound (low R): takes nominal - min(rate(Ap), rate(Am)) for δ_A, similarly for B.
    Returns combined error = sqrt(δ_A² + δ_B² + σ_stat²).
    """
    ab_triggers = get_ab_error_triggers(events)

    def _rate_for_variant(variant_tag):
        trig = ab_triggers.get((r_value, variant_tag))
        if trig is None:
            return None
        _, ref_rate, _ = getBinnedTriggerRate(events, trig)
        return getEventRate(ref_rate, e_bins, z_bins, max_distance)

    rate_Ap = _rate_for_variant("Ap")
    rate_Am = _rate_for_variant("Am")
    rate_Bp = _rate_for_variant("Bp")
    rate_Bm = _rate_for_variant("Bm")

    nom = np.nan_to_num(nominal_rate)

    # δ_A: max deviation from nominal across both A directions
    a_rates = [np.nan_to_num(r) for r in [rate_Ap, rate_Am] if r is not None]
    if a_rates:
        delta_A = np.max(np.abs(np.array(a_rates) - nom), axis=0)
    else:
        delta_A = np.zeros_like(nom)

    # δ_B: max deviation from nominal across both B directions
    b_rates = [np.nan_to_num(r) for r in [rate_Bp, rate_Bm] if r is not None]
    if b_rates:
        delta_B = np.max(np.abs(np.array(b_rates) - nom), axis=0)
    else:
        delta_B = np.zeros_like(nom)

    return np.sqrt(delta_A**2 + delta_B**2 + np.nan_to_num(stat_error)**2)


def generate_mb_event_rate_bands(loaded, save_folder, max_distance):
    """Plot 3: MB event rate bands across R values, with A/B + statistical uncertainty.

    For each R value, computes event rate and combined error (A/B + stat in quadrature).
    The band plot shows the envelope from R=0.5 to R=1.0 with combined errors.
    """
    e_bins, z_bins = getEnergyZenithBins()
    labels = list(MB_REFLECTED_SIMS.keys())
    sim_names = [MB_REFLECTED_SIMS[k] for k in labels]
    event_lists = [loaded.get(s) for s in sim_names]

    if any(e is None for e in event_lists):
        LOGGER.warning("Not all MB reflected sims available, skipping MB event rate bands")
        return

    rate_arrays_per_panel = []
    error_arrays_per_panel = []
    info_texts = []
    for label, events in zip(labels, event_lists):
        # Try R-based triggers first, fall back to dB-based
        r_triggers = get_all_r_triggers(events)

        if r_triggers:
            # R-based sweep
            r_vals = sorted(r_triggers.keys())
            rates_for_rs = []
            errors_for_rs = []
            trig_name = None
            r_min = min(r_vals)
            r_max = max(r_vals)

            for r_val in r_vals:
                trig = r_triggers[r_val]
                if trig_name is None:
                    trig_name = trig
                _, ref_rate, _ = getBinnedTriggerRate(events, trig)
                event_rate = getEventRate(ref_rate, e_bins, z_bins, max_distance)
                event_rate = _apply_energy_mask(event_rate, e_bins, label, "reflected")
                stat_error = getErrorEventRates(ref_rate, events, max_distance)
                stat_error = _apply_energy_mask(stat_error, e_bins, label, "reflected")

                # Compute combined error at R endpoints (A/B + stat), stat-only for middle
                if abs(r_val - r_max) < 0.005 or abs(r_val - r_min) < 0.005:
                    combined_error = _compute_ab_combined_error(
                        events, event_rate, r_val, stat_error, e_bins, z_bins, max_distance)
                    combined_error = _apply_energy_mask(combined_error, e_bins, label, "reflected")
                else:
                    combined_error = stat_error

                rates_for_rs.append(event_rate)
                errors_for_rs.append(combined_error)

            if rates_for_rs:
                rate_arrays_per_panel.append(rates_for_rs)
                error_arrays_per_panel.append(errors_for_rs)
            else:
                zero = np.zeros((len(e_bins)-1, len(z_bins)-1))
                rate_arrays_per_panel.append([getEventRate(zero, e_bins, z_bins, max_distance)])
                error_arrays_per_panel.append([getEventRate(zero, e_bins, z_bins, max_distance)])

            info_texts.append(_build_info("MB", label, trig_name, f"R={r_min}\u2013{r_max}"))
        else:
            # Fallback: dB-based triggers (backward compatibility)
            db_triggers = get_all_db_triggers(events)
            db_vals = sorted(db_triggers.keys()) if db_triggers else [None]

            rates_for_dbs = []
            errors_for_dbs = []
            trig_name = None
            for db in db_vals:
                if db is not None:
                    trig = db_triggers[db]
                else:
                    trig = find_direct_trigger(events)
                if trig is None:
                    continue
                if trig_name is None:
                    trig_name = trig
                _, ref_rate, _ = getBinnedTriggerRate(events, trig)
                event_rate = getEventRate(ref_rate, e_bins, z_bins, max_distance)
                event_rate = _apply_energy_mask(event_rate, e_bins, label, "reflected")
                error_rate = getErrorEventRates(ref_rate, events, max_distance)
                error_rate = _apply_energy_mask(error_rate, e_bins, label, "reflected")
                rates_for_dbs.append(event_rate)
                errors_for_dbs.append(error_rate)

            if rates_for_dbs:
                rate_arrays_per_panel.append(rates_for_dbs)
                error_arrays_per_panel.append(errors_for_dbs)
            else:
                zero = np.zeros((len(e_bins)-1, len(z_bins)-1))
                rate_arrays_per_panel.append([getEventRate(zero, e_bins, z_bins, max_distance)])
                error_arrays_per_panel.append([getEventRate(zero, e_bins, z_bins, max_distance)])

            info_texts.append(_build_info("MB", label, trig_name, "R=0.5\u20131.0"))

    plot_event_rate_bands(
        rate_arrays_per_panel, labels,
        suptitle="MB Reflected Event Rate (R = 0.5\u20131.0)",
        savename=os.path.join(save_folder, "mb_event_rate_bands.png"),
        error_arrays_per_panel=error_arrays_per_panel,
        info_texts=info_texts,
    )


def generate_sp_event_rate_bands(loaded, save_folder, max_distance):
    """Plot 4: SP event rate bands — side-by-side Gen2 shallow and deep, with uncertainty."""
    e_bins, z_bins = getEnergyZenithBins()
    depth = SP_DEPTHS[0]  # 300m

    rate_arrays_per_panel = []
    error_arrays_per_panel = []
    panel_labels = []
    info_texts = []

    for stype, slabel in [("shallow", "Gen2 Shallow"), ("deep", "Gen2 Deep")]:
        sim_key = SP_REFLECTED_SIMS.get((depth, stype))
        events = loaded.get(sim_key)
        if events is None:
            LOGGER.info("Skipping SP %s %s (missing data)", depth, stype)
            continue

        db_trigs = get_all_db_triggers(events)
        if not db_trigs:
            continue

        sorted_dbs = sorted(db_trigs.keys())
        rates_for_dbs = []
        errors_for_dbs = []
        trig_name = None
        for db in sorted_dbs:
            trig = db_trigs[db]
            if trig_name is None:
                trig_name = trig
            _, ref_rate, _ = getBinnedTriggerRate(events, trig)
            event_rate = getEventRate(ref_rate, e_bins, z_bins, max_distance)
            event_rate = _apply_energy_mask(event_rate, e_bins, slabel, "reflected")
            error_rate = getErrorEventRates(ref_rate, events, max_distance)
            error_rate = _apply_energy_mask(error_rate, e_bins, slabel, "reflected")
            rates_for_dbs.append(event_rate)
            errors_for_dbs.append(error_rate)

        if rates_for_dbs:
            rate_arrays_per_panel.append(rates_for_dbs)
            error_arrays_per_panel.append(errors_for_dbs)
            panel_labels.append(slabel)
            info_texts.append(_build_info("SP", slabel, trig_name, f"{depth}, 40\u201355 dB"))

    if not panel_labels:
        LOGGER.warning("No SP data available, skipping SP event rate bands")
        return

    plot_event_rate_bands(
        rate_arrays_per_panel, panel_labels,
        suptitle=f"SP Reflected Event Rate \u2014 {depth} (dB = 40\u201355)",
        savename=os.path.join(save_folder, "sp_event_rate_bands.png"),
        error_arrays_per_panel=error_arrays_per_panel,
        info_texts=info_texts,
    )


def generate_sp_event_rate_bands_40dB(loaded, save_folder, max_distance):
    """SP event rate bands — side-by-side Gen2 shallow and deep, 40 dB only.

    Shows statistical uncertainty band with central dashed line (no uncertainty).
    """
    e_bins, z_bins = getEnergyZenithBins()
    db_value = 40.0
    depth = SP_DEPTHS[0]  # 300m

    rate_arrays_per_panel = []
    error_arrays_per_panel = []
    panel_labels = []
    info_texts = []

    for stype, slabel in [("shallow", "Gen2 Shallow"), ("deep", "Gen2 Deep")]:
        sim_key = SP_REFLECTED_SIMS.get((depth, stype))
        events = loaded.get(sim_key)
        if events is None:
            LOGGER.info("Skipping SP %s %s (missing data) for 40dB plot", depth, stype)
            continue

        trig = find_trigger_for_db(events, db_value)
        if trig is None:
            LOGGER.info("No 40 dB trigger found for SP %s %s, skipping", depth, stype)
            continue

        _, ref_rate, _ = getBinnedTriggerRate(events, trig)
        event_rate = getEventRate(ref_rate, e_bins, z_bins, max_distance)
        event_rate = _apply_energy_mask(event_rate, e_bins, slabel, "reflected")
        error_rate = getErrorEventRates(ref_rate, events, max_distance)
        error_rate = _apply_energy_mask(error_rate, e_bins, slabel, "reflected")

        rate_arrays_per_panel.append(event_rate)
        error_arrays_per_panel.append(error_rate)
        panel_labels.append(slabel)
        info_texts.append(_build_info("SP", slabel, trig, f"{depth}, 40 dB"))

    if not panel_labels:
        LOGGER.warning("No SP data available for 40 dB, skipping")
        return

    plot_event_rate_bands_single_db(
        rate_arrays_per_panel, panel_labels,
        suptitle=f"SP Reflected Event Rate \u2014 {depth} (40 dB)",
        savename=os.path.join(save_folder, "sp_event_rate_bands_40dB.png"),
        error_arrays_per_panel=error_arrays_per_panel,
        info_texts=info_texts,
    )


def generate_sp_radii_plots(loaded, save_folder, max_distance):
    """Plot 5a: SP radii probability density — 300m depth.

    Two panels: Gen2 Shallow, Gen2 Deep.
    Each panel shows direct (black band) + reflected band for 300m.
    """
    station_types = [("shallow", "Gen2 Shallow"), ("deep", "Gen2 Deep")]
    depth_colors = {"300m": "tab:blue"}

    e_bins, z_bins = getEnergyZenithBins()
    n_bins = 15

    panels_data = []
    panel_titles = []
    info_texts = []

    for stype, slabel in station_types:
        # Direct trigger from the 300m sim (direct is depth-independent)
        direct_sim = SP_REFLECTED_SIMS.get(("300m", stype))
        direct_events = loaded.get(direct_sim)
        if direct_events is None:
            LOGGER.warning("Missing SP %s 300m data, skipping SP radii panel", stype)
            continue

        panels_data.append((stype, slabel, direct_events))
        panel_titles.append(slabel)

        dir_trig = find_direct_trigger(direct_events)
        info_texts.append(_build_info("SP", slabel, dir_trig, "300m, 40\u201355 dB"))

    if not panels_data:
        return

    n = len(panels_data)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ip, (ax, (stype, slabel, direct_events)) in enumerate(zip(axes, panels_data)):
        events_list = list(direct_events)

        # Direct band
        dir_trig = find_direct_trigger(direct_events)
        if dir_trig:
            direct_rate, _, _ = getBinnedTriggerRate(events_list, dir_trig)
            setEventListRateWeight(
                events_list, direct_rate, "direct", dir_trig,
                max_distance=max_distance, use_direct=True,
            )
            bins_r = np.linspace(0, max_distance, n_bins + 1)
            vals, ws = _collect_weighted_values(events_list, "direct", lambda e: e.get_radius())
            centers, density, derr = _compute_weighted_density(vals, ws, bins_r)
            ax.fill_between(centers, np.maximum(density - derr, 0), density + derr,
                            alpha=0.3, color="black", step="mid")
            ax.step(centers, density, color="black", linewidth=1.5, label="Direct", where="mid")

        # One band per depth
        for depth in SP_DEPTHS:
            sim_key = SP_REFLECTED_SIMS.get((depth, stype))
            depth_events = loaded.get(sim_key)
            if depth_events is None:
                continue

            depth_events_list = list(depth_events)
            db_trigs = get_all_db_triggers(depth_events_list)
            if not db_trigs:
                continue

            sorted_dbs = sorted(db_trigs.keys())
            all_densities = []
            all_errors = []
            for db_val in sorted_dbs:
                ref_trig = db_trigs[db_val]
                weight_name = f"refl_{depth}_{db_val:.1f}dB"
                _, ref_rate, _ = getBinnedTriggerRate(depth_events_list, ref_trig)
                setEventListRateWeight(
                    depth_events_list, ref_rate, weight_name, ref_trig,
                    max_distance=max_distance, use_direct=False,
                )
                bins_r = np.linspace(0, max_distance, n_bins + 1)
                vals, ws = _collect_weighted_values(depth_events_list, weight_name, lambda e: e.get_radius())
                centers, dens, derr = _compute_weighted_density(vals, ws, bins_r)
                all_densities.append(dens)
                all_errors.append(derr)

            if all_densities:
                all_densities = np.array(all_densities)
                all_errors = np.array(all_errors)
                lo_idx = np.argmin(all_densities, axis=0)
                hi_idx = np.argmax(all_densities, axis=0)
                lo = np.array([all_densities[lo_idx[i], i] - all_errors[lo_idx[i], i]
                               for i in range(len(centers))])
                hi = np.array([all_densities[hi_idx[i], i] + all_errors[hi_idx[i], i]
                               for i in range(len(centers))])
                lo = np.maximum(lo, 0)
                color = depth_colors.get(depth, "tab:gray")
                ax.fill_between(centers, lo, hi, alpha=0.3, color=color, label=f"{depth}",
                                step="mid")

        ax.set_xlabel("Radius (m)")
        ax.set_title(slabel, fontsize=10)

        if ip < len(info_texts):
            ax.text(0.03, 0.97, info_texts[ip], transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[0].set_ylabel("Probability Density")
    axes[0].legend(fontsize=7)
    fig.suptitle("SP Radii Distribution \u2014 300m", fontsize=13, y=1.02)
    plt.tight_layout()
    savename = os.path.join(save_folder, "sp_radii_density.png")
    plt.savefig(savename, dpi=150, bbox_inches="tight")
    plt.close()
    ic(f"Saved: {savename}")


def generate_mb_radii_plots(loaded, save_folder, max_distance):
    """Plot 5b: MB radii probability density."""
    labels = list(MB_REFLECTED_SIMS.keys())
    sim_names = [MB_REFLECTED_SIMS[k] for k in labels]
    event_lists = [loaded.get(s) for s in sim_names]

    if any(e is None for e in event_lists):
        LOGGER.warning("Missing MB data, skipping MB radii plots")
        return

    direct_triggers = [find_direct_trigger(e) for e in event_lists]
    # Try R-based triggers first, fallback to dB-based
    r_trigs = [get_all_r_triggers(e) for e in event_lists]
    if any(rt for rt in r_trigs):
        reflected_triggers = r_trigs
        sweep_labels = MB_R_LABELS
        info_label = "R=0.5\u20131.0"
    else:
        reflected_triggers = [get_all_db_triggers(e) for e in event_lists]
        sweep_labels = {0.0: "R=1.0", 1.5: "R=0.7", 3.0: "R=0.5"}
        info_label = "R=0.5\u20131.0"
    info_texts = [_build_info("MB", lab, dt, info_label)
                  for lab, dt in zip(labels, direct_triggers)]

    plot_radii_density_panels(
        event_lists, direct_triggers, reflected_triggers, labels,
        suptitle="MB Radii Distribution \u2014 Event Rate Weighted Density",
        savename=os.path.join(save_folder, "mb_radii_density.png"),
        max_distance=max_distance, db_labels=sweep_labels, info_texts=info_texts,
    )


def generate_mb_arrival_angle_plots(loaded, save_folder, max_distance, nu_events=None):
    """MB arrival angle (refracted zenith) density."""
    labels = list(MB_REFLECTED_SIMS.keys())
    sim_names = [MB_REFLECTED_SIMS[k] for k in labels]
    event_lists = [loaded.get(s) for s in sim_names]

    if any(e is None for e in event_lists):
        LOGGER.warning("Missing MB data, skipping MB arrival angle plots")
        return

    direct_triggers = [find_direct_trigger(e) for e in event_lists]
    r_trigs = [get_all_r_triggers(e) for e in event_lists]
    if any(rt for rt in r_trigs):
        reflected_triggers = r_trigs
        sweep_labels = MB_R_LABELS
    else:
        reflected_triggers = [get_all_db_triggers(e) for e in event_lists]
        sweep_labels = {0.0: "R=1.0", 1.5: "R=0.7", 3.0: "R=0.5"}
    info_texts = [_build_info("MB", lab, dt, "R=0.5\u20131.0")
                  for lab, dt in zip(labels, direct_triggers)]

    nu_densities = None
    if nu_events and "MB" in nu_events:
        n_bins = 12
        bins = np.linspace(0, 90, n_bins + 1)
        nu_densities = _build_neutrino_densities(
            nu_events["MB"], labels, _compute_neutrino_arrival_density, bins)

    plot_arrival_angle_panels(
        event_lists, direct_triggers, reflected_triggers, labels,
        suptitle="MB Arrival Angle Distribution (Snell's Law, n$_{ice}$=1.78)",
        savename=os.path.join(save_folder, "mb_arrival_angle.png"),
        max_distance=max_distance, db_labels=sweep_labels, info_texts=info_texts,
        neutrino_densities=nu_densities,
    )


def generate_sp_arrival_angle_plots(loaded, save_folder, max_distance, nu_events=None):
    """SP arrival angle (refracted zenith) density."""
    labels = ["Gen2 Shallow", "Gen2 Deep"]
    sim_names = [SP_REFLECTED_SIMS[("300m", "shallow")], SP_REFLECTED_SIMS[("300m", "deep")]]
    event_lists = [loaded.get(s) for s in sim_names]

    if any(e is None for e in event_lists):
        LOGGER.warning("Missing SP 300m data, skipping SP arrival angle plots")
        return

    direct_triggers = [find_direct_trigger(e) for e in event_lists]
    reflected_db_triggers = [get_all_db_triggers(e) for e in event_lists]
    info_texts = [_build_info("SP", lab, dt, "300m, 40\u201355 dB")
                  for lab, dt in zip(labels, direct_triggers)]

    nu_densities = None
    if nu_events and "SP" in nu_events:
        n_bins = 12
        bins = np.linspace(0, 90, n_bins + 1)
        nu_densities = _build_neutrino_densities(
            nu_events["SP"], labels, _compute_neutrino_arrival_density, bins)

    plot_arrival_angle_panels(
        event_lists, direct_triggers, reflected_db_triggers, labels,
        suptitle="SP Arrival Angle Distribution (Snell's Law, n$_{ice}$=1.78)",
        savename=os.path.join(save_folder, "sp_arrival_angle.png"),
        max_distance=max_distance, info_texts=info_texts,
        neutrino_densities=nu_densities,
    )


def generate_mb_polarization_angle_plots(loaded, save_folder, max_distance, nu_events=None):
    """MB polarization angle density."""
    labels = list(MB_REFLECTED_SIMS.keys())
    sim_names = [MB_REFLECTED_SIMS[k] for k in labels]
    event_lists = [loaded.get(s) for s in sim_names]

    if any(e is None for e in event_lists):
        LOGGER.warning("Missing MB data, skipping MB polarization angle plots")
        return

    direct_triggers = [find_direct_trigger(e) for e in event_lists]
    r_trigs = [get_all_r_triggers(e) for e in event_lists]
    if any(rt for rt in r_trigs):
        reflected_triggers = r_trigs
        sweep_labels = MB_R_LABELS
    else:
        reflected_triggers = [get_all_db_triggers(e) for e in event_lists]
        sweep_labels = {0.0: "R=1.0", 1.5: "R=0.7", 3.0: "R=0.5"}
    info_texts = [_build_info("MB", lab, dt, "R=0.5\u20131.0")
                  for lab, dt in zip(labels, direct_triggers)]

    nu_densities = None
    if nu_events and "MB" in nu_events:
        n_bins = 15
        bins = np.linspace(0, 180, n_bins + 1)
        nu_densities = _build_neutrino_densities(
            nu_events["MB"], labels, _compute_neutrino_polarization_density, bins)

    plot_polarization_angle_panels(
        event_lists, direct_triggers, reflected_triggers, labels,
        suptitle="MB Polarization Angle Distribution",
        savename=os.path.join(save_folder, "mb_polarization_angle.png"),
        max_distance=max_distance, db_labels=sweep_labels, info_texts=info_texts,
        neutrino_densities=nu_densities,
    )


def generate_sp_polarization_angle_plots(loaded, save_folder, max_distance, nu_events=None):
    """SP polarization angle density."""
    labels = ["Gen2 Shallow", "Gen2 Deep"]
    sim_names = [SP_REFLECTED_SIMS[("300m", "shallow")], SP_REFLECTED_SIMS[("300m", "deep")]]
    event_lists = [loaded.get(s) for s in sim_names]

    if any(e is None for e in event_lists):
        LOGGER.warning("Missing SP 300m data, skipping SP polarization angle plots")
        return

    direct_triggers = [find_direct_trigger(e) for e in event_lists]
    reflected_db_triggers = [get_all_db_triggers(e) for e in event_lists]
    info_texts = [_build_info("SP", lab, dt, "300m, 40\u201355 dB")
                  for lab, dt in zip(labels, direct_triggers)]

    nu_densities = None
    if nu_events and "SP" in nu_events:
        n_bins = 15
        bins = np.linspace(0, 180, n_bins + 1)
        nu_densities = _build_neutrino_densities(
            nu_events["SP"], labels, _compute_neutrino_polarization_density, bins)

    plot_polarization_angle_panels(
        event_lists, direct_triggers, reflected_db_triggers, labels,
        suptitle="SP Polarization Angle Distribution",
        savename=os.path.join(save_folder, "sp_polarization_angle.png"),
        max_distance=max_distance, info_texts=info_texts,
        neutrino_densities=nu_densities,
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
    parser.add_argument("--neutrino-folder", type=str, default=None,
                        help="Folder containing neutrino comparison numpy files")
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

    # Load neutrino comparison data if available
    nu_events = None
    if args.neutrino_folder:
        nu_events = load_neutrino_events(args.neutrino_folder)
        if nu_events:
            for site, evts in nu_events.items():
                ic(f"Neutrino data: {site} = {len(evts)} events")
        else:
            ic("No neutrino data found in", args.neutrino_folder)

    # Generate plots — each wrapped so one failure doesn't block others
    plot_generators = [
        ("MB trigger rate plots", lambda: generate_mb_trigger_rate_plots(loaded, save_folder, max_distance)),
        ("SP trigger rate plots", lambda: generate_sp_trigger_rate_plots(loaded, save_folder, max_distance)),
        ("MB event rate 2D", lambda: generate_mb_event_rate_2d(loaded, save_folder, max_distance)),
        ("SP event rate 2D", lambda: generate_sp_event_rate_2d(loaded, save_folder, max_distance)),
        ("MB event rate bands", lambda: generate_mb_event_rate_bands(loaded, save_folder, max_distance)),
        ("SP event rate bands", lambda: generate_sp_event_rate_bands(loaded, save_folder, max_distance)),
        ("SP event rate bands (40 dB)", lambda: generate_sp_event_rate_bands_40dB(loaded, save_folder, max_distance)),
        ("SP radii density", lambda: generate_sp_radii_plots(loaded, save_folder, max_distance)),
        ("MB radii density", lambda: generate_mb_radii_plots(loaded, save_folder, max_distance)),
        ("MB arrival angle", lambda: generate_mb_arrival_angle_plots(loaded, save_folder, max_distance, nu_events)),
        ("SP arrival angle", lambda: generate_sp_arrival_angle_plots(loaded, save_folder, max_distance, nu_events)),
        ("MB polarization angle", lambda: generate_mb_polarization_angle_plots(loaded, save_folder, max_distance, nu_events)),
        ("SP polarization angle", lambda: generate_sp_polarization_angle_plots(loaded, save_folder, max_distance, nu_events)),
        ("Rate table", lambda: generate_rate_table(
            loaded, max_distance, os.path.join(save_folder, "rate_table.txt"),
        )),
        ("MB error breakdown", lambda: generate_mb_error_breakdown_table(
            loaded, max_distance, os.path.join(save_folder, "mb_error_breakdown.txt"),
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
