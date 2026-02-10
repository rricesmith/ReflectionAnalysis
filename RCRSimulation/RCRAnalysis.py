"""RCR Simulation Analysis and Plotting Module.

This module provides analysis functions for RCR (Reflected Cosmic Ray) simulations,
including event rate calculations, trigger rate binning, and various plotting functions.

Key functions:
- getEnergyZenithBins(): Define standard energy-zenith bins
- getEventRateArray(): Calculate cosmic ray event rates per bin
- getBinnedTriggerRate(): Compute trigger rates in energy-zenith bins
- setEventListRateWeight(): Assign event rate weights to events
- plotRateWithError(): Plot event rate vs energy for zenith bins
- imshowRate(): 2D histogram of trigger/event rate
- histRadiusRate(): Event rate vs radius from station
"""

from __future__ import annotations

import os
import configparser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from icecream import ic

from NuRadioReco.utilities import units
import astrotools.auger as auger

# Must match NOISE_TRIGGER_SIGMA in S01_RCRSim.py — used to filter out
# noise pre-check triggers when auto-selecting the analysis trigger name
NOISE_PRECHECK_SIGMA = 2.0

from RCRSimulation.RCREventObject import RCREvent, REFLECTED_STATION_OFFSET


# =============================================================================
# Energy-Zenith Binning
# =============================================================================

def getEnergyZenithBins() -> Tuple[np.ndarray, np.ndarray]:
    """Define standard energy and zenith bins for analysis.

    Returns:
        Tuple of (energy_bins, zenith_bins) in eV and radians respectively.
        Energy bins: log10(E/eV) from 16 to 20 in 0.5 steps
        Zenith bins: cos(zenith) from 0 to 1 in 0.2 steps, converted to radians
    """
    min_energy = 16.0
    max_energy = 20.1
    e_bins = 10**np.arange(min_energy, max_energy, 0.5) * units.eV

    # Zenith bins in cos(zenith) space, then convert to radians
    z_bins = np.arange(0, 1.01, 0.2)
    z_bins = np.arccos(z_bins)
    z_bins[np.isnan(z_bins)] = 0
    z_bins = z_bins * units.rad
    z_bins = np.sort(z_bins)

    return e_bins, z_bins


def getEnergyZenithArray() -> np.ndarray:
    """Create empty 2D array matching energy-zenith bin dimensions.

    Returns:
        Zero-filled numpy array of shape (n_energy_bins-1, n_zenith_bins-1)
    """
    e_bins, z_bins = getEnergyZenithBins()
    return np.zeros((len(e_bins) - 1, len(z_bins) - 1))


# =============================================================================
# Event Counting and Trigger Rates
# =============================================================================

def getnThrows(event_list: Sequence[RCREvent]) -> np.ndarray:
    """Count number of events (throws) in each energy-zenith bin.

    Args:
        event_list: List of RCREvent objects

    Returns:
        2D array of event counts per bin
    """
    e_bins, z_bins = getEnergyZenithBins()
    n_throws = getEnergyZenithArray()

    for event in event_list:
        energy_bin = np.digitize(event.get_energy() * units.eV, e_bins) - 1
        zenith_bin = np.digitize(event.zenith * units.rad, z_bins) - 1

        if energy_bin < 0 or zenith_bin < 0:
            continue
        if energy_bin >= len(e_bins) - 1 or zenith_bin >= len(z_bins) - 1:
            continue

        n_throws[energy_bin][zenith_bin] += 1

    return n_throws


def getBinnedTriggerRate(
    event_list: Sequence[RCREvent],
    trigger_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate trigger rates in energy-zenith bins for direct and reflected.

    Args:
        event_list: List of RCREvent objects
        trigger_name: Name of trigger to check

    Returns:
        Tuple of (direct_trigger_rate, reflected_trigger_rate, n_throws)
        Each rate array is normalized by number of throws per bin.
    """
    e_bins, z_bins = getEnergyZenithBins()
    n_throws = getnThrows(event_list)

    direct_triggers = getEnergyZenithArray()
    reflected_triggers = getEnergyZenithArray()

    for event in event_list:
        energy_bin = np.digitize(event.get_energy() * units.eV, e_bins) - 1
        zenith_bin = np.digitize(event.zenith * units.rad, z_bins) - 1

        if energy_bin < 0 or zenith_bin < 0:
            continue
        if energy_bin >= len(e_bins) - 1 or zenith_bin >= len(z_bins) - 1:
            continue

        # Count direct triggers
        if event.has_direct_trigger(trigger_name):
            direct_triggers[energy_bin][zenith_bin] += 1

        # Count reflected triggers
        if event.has_reflected_trigger(trigger_name):
            reflected_triggers[energy_bin][zenith_bin] += 1

    # Normalize by number of throws
    with np.errstate(divide='ignore', invalid='ignore'):
        direct_rate = direct_triggers / n_throws
        reflected_rate = reflected_triggers / n_throws

    direct_rate[np.isnan(direct_rate)] = 0
    reflected_rate[np.isnan(reflected_rate)] = 0

    return direct_rate, reflected_rate, n_throws


# =============================================================================
# Event Rate Calculation
# =============================================================================

def getEventRateArray(e_bins: np.ndarray, z_bins: np.ndarray) -> np.ndarray:
    """Calculate cosmic ray event rate in each energy-zenith bin.

    Uses astrotools.auger.event_rate for cosmic ray flux calculation.

    Args:
        e_bins: Energy bin edges in eV (or log10(E/eV) if values < 100)
        z_bins: Zenith bin edges in radians

    Returns:
        2D array of event rates in evts/km^2/yr per bin
    """
    # Convert to log10 if not already
    if e_bins[0] > 100:
        logE_bins = np.log10(e_bins / units.eV)
    else:
        logE_bins = e_bins

    event_rate_array = getEnergyZenithArray()

    for i in range(len(e_bins) - 1):
        for j in range(len(z_bins) - 1):
            # Calculate flux contribution from this bin
            # Using geometric exposure factor: 0.5*(1+cos(zenith))
            high_flux = auger.event_rate(
                logE_bins[i], logE_bins[i + 1],
                zmax=z_bins[j + 1] / units.deg,
                area=1 * 0.5 * (1 + np.cos(z_bins[j + 1]))
            )
            low_flux = auger.event_rate(
                logE_bins[i], logE_bins[i + 1],
                zmax=z_bins[j] / units.deg,
                area=1 * 0.5 * (1 + np.cos(z_bins[j]))
            )
            event_rate_array[i][j] = high_flux - low_flux

    return event_rate_array


def getEventRate(
    trigger_rate: np.ndarray,
    e_bins: np.ndarray,
    z_bins: np.ndarray,
    max_distance: float = 6.0 * units.km,
) -> np.ndarray:
    """Convert trigger rate to event rate.

    Args:
        trigger_rate: 2D array of trigger rates per bin
        e_bins: Energy bin edges
        z_bins: Zenith bin edges
        max_distance: Maximum throw distance (defines area)

    Returns:
        2D array of event rates in evts/yr
    """
    area = np.pi * max_distance**2
    event_rate_array = getEventRateArray(e_bins, z_bins)
    return event_rate_array * trigger_rate * area / units.km**2


def getErrorEventRates(
    trigger_rate: np.ndarray,
    event_list: Sequence[RCREvent],
    max_distance: float = 6.0 * units.km,
) -> np.ndarray:
    """Calculate error on event rates from Poisson statistics.

    Args:
        trigger_rate: 2D array of trigger rates per bin
        event_list: List of events (used to get n_throws)
        max_distance: Maximum throw distance

    Returns:
        2D array of event rate errors in evts/yr
    """
    e_bins, z_bins = getEnergyZenithBins()
    n_throws = getnThrows(event_list)

    # Get number of triggers per bin
    n_trig = trigger_rate * n_throws

    # Poisson error on trigger rate
    with np.errstate(divide='ignore', invalid='ignore'):
        trig_rate_error = np.sqrt(n_trig) / n_throws

    trig_rate_error[np.isnan(trig_rate_error)] = 0

    return getEventRate(trig_rate_error, e_bins, z_bins, max_distance=max_distance)


# =============================================================================
# Event Weight Assignment
# =============================================================================

def setEventListRateWeight(
    event_list: Sequence[RCREvent],
    trigger_rate_array: np.ndarray,
    weight_name: str,
    trigger_name: str,
    max_distance: float = 6.0 * units.km,
    use_direct: bool = True,
) -> None:
    """Assign event rate weights to events for weighted analysis.

    Each triggered event receives weight = event_rate / n_triggered in its bin.

    Args:
        event_list: List of RCREvent objects (modified in place)
        trigger_rate_array: 2D array of trigger rates
        weight_name: Name for this weight category
        trigger_name: Trigger name to check for
        max_distance: Maximum throw distance
        use_direct: If True, check direct triggers; if False, check reflected
    """
    e_bins, z_bins = getEnergyZenithBins()
    event_rate_array = getEventRateArray(e_bins, z_bins)
    n_throws = getnThrows(event_list)
    area = np.pi * max_distance**2

    for event in event_list:
        # Check if event triggered for this weight category
        if use_direct:
            triggered = event.has_direct_trigger(trigger_name)
        else:
            triggered = event.has_reflected_trigger(trigger_name)

        if not triggered:
            event.set_weight(0, weight_name)
            continue

        energy_bin = np.digitize(event.get_energy() * units.eV, e_bins) - 1
        zenith_bin = np.digitize(event.zenith * units.rad, z_bins) - 1

        if energy_bin < 0 or zenith_bin < 0:
            event.set_weight(0, weight_name)
            continue
        if energy_bin >= len(e_bins) - 1 or zenith_bin >= len(z_bins) - 1:
            event.set_weight(0, weight_name)
            continue

        event_rate = event_rate_array[energy_bin][zenith_bin]
        n_trig = trigger_rate_array[energy_bin][zenith_bin] * n_throws[energy_bin][zenith_bin]

        if n_trig == 0:
            event.set_weight(0, weight_name)
        else:
            weight = event_rate * trigger_rate_array[energy_bin][zenith_bin] * (area / units.km**2) / n_trig
            event.set_weight(weight, weight_name)


# =============================================================================
# Plotting Utilities
# =============================================================================

def set_bad_imshow(array: np.ndarray, value: float):
    """Create masked array and colormap with white for bad values.

    Args:
        array: Input array
        value: Value to mask (typically 0)

    Returns:
        Tuple of (masked_array, colormap)
    """
    ma = np.ma.masked_where(array == value, array)
    cmap = matplotlib.cm.get_cmap('viridis')
    cmap.set_bad(color='white')
    return ma, cmap


def getUniqueEnergyZenithPairs(
    event_list: Sequence[RCREvent],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract unique energy-zenith combinations from event list.

    Args:
        event_list: List of RCREvent objects

    Returns:
        Tuple of (log_energies, zeniths_deg) arrays
    """
    energies = []
    zeniths = []

    for event in event_list:
        energies.append(event.get_energy() * units.eV)
        zeniths.append(event.zenith * units.rad)

    energies = np.array(energies)
    zeniths = np.array(zeniths)

    # Create pairs and find unique ones
    pairs = np.column_stack((energies, zeniths))
    unique_pairs = np.unique(pairs, axis=0)

    log_energies = np.log10(unique_pairs[:, 0] / units.eV)
    zeniths_deg = unique_pairs[:, 1] / units.deg

    return log_energies, zeniths_deg


# =============================================================================
# Plotting Functions
# =============================================================================

def plotRateWithError(
    event_rate: np.ndarray,
    error_rate: np.ndarray,
    savename: str,
    title: str,
) -> None:
    """Plot event rate vs energy with error bands for each zenith bin.

    Creates plot showing event rate as function of energy with colored
    bands for each zenith bin, plus total rate with black band.

    Args:
        event_rate: 2D array of event rates [energy, zenith]
        error_rate: 2D array of rate errors [energy, zenith]
        savename: Output file path
        title: Plot title
    """
    e_bins, z_bins = getEnergyZenithBins()
    e_bins = np.log10(e_bins / units.eV)

    fig, ax = plt.subplots()
    color = plt.cm.rainbow(np.linspace(0, 1, len(z_bins) - 1))

    event_rate[np.isnan(event_rate)] = 0
    error_rate[np.isnan(error_rate)] = 0

    # Plot total rate
    total_rate = np.nansum(event_rate, axis=1)
    total_error = np.sqrt(np.nansum(error_rate**2, axis=1))
    e_centers = (e_bins[1:] + e_bins[:-1]) / 2

    ax.fill_between(
        e_centers,
        total_rate - total_error,
        total_rate + total_error,
        alpha=0.5,
        label=f'{np.nansum(event_rate):.2f} +/- {np.sqrt(np.nansum(error_rate**2)):.2f} Evts/Yr',
        color='black'
    )
    ax.plot(e_centers, total_rate, color='black', linestyle='--')

    # Plot each zenith bin
    for iZ in range(len(z_bins) - 1):
        ax.fill_between(
            e_centers,
            event_rate[:, iZ] - error_rate[:, iZ],
            event_rate[:, iZ] + error_rate[:, iZ],
            alpha=0.5,
            label=f'{z_bins[iZ]/units.deg:.1f}-{z_bins[iZ+1]/units.deg:.1f}deg',
            color=color[iZ]
        )
        ax.plot(e_centers, event_rate[:, iZ], color=color[iZ], linestyle='--')

    ax.set_xlabel('log10(E/eV)')
    ax.set_ylabel('Evts/Yr')
    ax.set_yscale('log')
    ax.set_ylim(bottom=10**-3)
    ax.legend(loc='lower left', fontsize=8)
    ax.set_title(title)

    fig.savefig(savename, dpi=150, bbox_inches='tight')
    ic(f'Saved {savename}')
    plt.close(fig)


def imshowRate(
    rate: np.ndarray,
    title: str,
    savename: str,
    colorbar_label: str = 'Evts/yr',
    event_list: Optional[Sequence[RCREvent]] = None,
) -> None:
    """Create 2D histogram showing rate in energy-zenith space.

    Args:
        rate: 2D array of rates [energy, zenith]
        title: Plot title
        savename: Output file path
        colorbar_label: Label for colorbar
        event_list: Optional event list to overlay unique energy-zenith points
    """
    e_bins, z_bins = getEnergyZenithBins()
    e_bins = np.log10(e_bins / units.eV)
    z_bins_deg = z_bins / units.deg

    # No flip: origin='lower' puts row 0 (lowest zenith bin) at bottom → 0° at bottom, 90° at top
    rate_masked, cmap = set_bad_imshow(rate.T, 0)

    fig, ax = plt.subplots()

    im = ax.imshow(
        rate_masked,
        aspect='auto',
        origin='lower',
        extent=[min(e_bins), max(e_bins), min(z_bins_deg), max(z_bins_deg)],
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap
    )

    ax.set_yticks(z_bins_deg)
    ax.set_yticklabels([f'{z:.0f}' for z in z_bins_deg])
    ax.set_ylabel('Zenith (deg)')
    ax.set_xlabel('log(Energy (eV))')

    # Overlay unique energy-zenith pairs
    if event_list is not None:
        log_energies, zeniths_deg = getUniqueEnergyZenithPairs(event_list)
        ax.scatter(
            log_energies, zeniths_deg,
            facecolors='none', edgecolors='black',
            s=20, linewidths=0.5
        )

    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(title)

    fig.savefig(savename, dpi=150, bbox_inches='tight')
    ic(f'Saved {savename}')
    plt.close(fig)


def histRadiusRate(
    event_list: Sequence[RCREvent],
    weight_name: str,
    title: str,
    savename: str,
    max_distance: float = 6.0 * units.km,
    n_bins: int = 30,
) -> None:
    """Plot event rate as function of radial distance from station.

    Args:
        event_list: List of RCREvent objects with weights set
        weight_name: Name of weight to use
        title: Plot title
        savename: Output file path
        max_distance: Maximum distance for binning
        n_bins: Number of radial bins
    """
    radius_bins = np.linspace(0, max_distance / units.m, n_bins + 1)
    bin_centers = (radius_bins[1:] + radius_bins[:-1]) / 2

    # Collect radii and weights
    radii = []
    weights = []

    for event in event_list:
        radii.append(event.get_radius())
        weights.append(event.get_weight(weight_name))

    radii = np.array(radii)
    weights = np.array(weights)

    fig, ax = plt.subplots()

    # Create weighted histogram
    hist, _ = np.histogram(radii, bins=radius_bins, weights=weights)

    # Plot as bar chart
    ax.bar(bin_centers, hist, width=radius_bins[1] - radius_bins[0], alpha=0.7, edgecolor='black')

    ax.set_xlabel('Distance from Station (m)')
    ax.set_ylabel('Event Rate (Evts/Yr)')
    ax.set_yscale('log')
    ax.set_title(title)

    # Add total rate annotation
    total_rate = np.sum(hist)
    ax.annotate(
        f'Total: {total_rate:.3f} Evts/Yr',
        xy=(0.95, 0.95), xycoords='axes fraction',
        ha='right', va='top',
        fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    fig.savefig(savename, dpi=150, bbox_inches='tight')
    ic(f'Saved {savename}')
    plt.close(fig)


def histAreaRate(
    event_list: Sequence[RCREvent],
    weight_name: str,
    title: str,
    savename: str,
    max_distance: float = 6.0 * units.km,
    colorbar_label: str = 'Evts/yr',
) -> None:
    """Create 2D histogram of event rate in x-y space.

    Args:
        event_list: List of RCREvent objects with weights set
        weight_name: Name of weight to use
        title: Plot title
        savename: Output file path
        max_distance: Maximum distance for binning
        colorbar_label: Label for colorbar
    """
    x_bins = np.linspace(-max_distance / units.m, max_distance / units.m, 50)
    y_bins = np.linspace(-max_distance / units.m, max_distance / units.m, 50)

    # Collect positions and weights
    x_vals = []
    y_vals = []
    weights = []

    for event in event_list:
        x, y = event.get_coreas_position()
        x_vals.append(x)
        y_vals.append(y)
        weights.append(event.get_weight(weight_name))

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    weights = np.array(weights)

    fig, ax = plt.subplots()

    # Filter to non-zero weights for log norm
    nonzero_weights = weights[weights > 0]
    if len(nonzero_weights) > 0:
        norm = matplotlib.colors.LogNorm(
            vmin=np.min(nonzero_weights),
            vmax=np.max(nonzero_weights) * 5
        )
    else:
        norm = None

    h, xedges, yedges, im = ax.hist2d(
        x_vals, y_vals,
        bins=(x_bins, y_bins),
        weights=weights,
        cmap='viridis',
        norm=norm
    )

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(title)

    fig.savefig(savename, dpi=150, bbox_inches='tight')
    ic(f'Saved {savename}')
    plt.close(fig)


def plotCombinedRateWithError(
    direct_rate: np.ndarray,
    direct_error: np.ndarray,
    reflected_rate: np.ndarray,
    reflected_error: np.ndarray,
    savename: str,
    title: str,
) -> None:
    """Plot direct and reflected event rates overlaid on the same axes.

    Colors represent zenith bins, line style distinguishes direct (solid) vs
    reflected (dashed). Two legends: one for zenith colors, one for line styles.

    Args:
        direct_rate: 2D array of direct event rates [energy, zenith]
        direct_error: 2D array of direct rate errors [energy, zenith]
        reflected_rate: 2D array of reflected event rates [energy, zenith]
        reflected_error: 2D array of reflected rate errors [energy, zenith]
        savename: Output file path
        title: Plot title
    """
    e_bins, z_bins = getEnergyZenithBins()
    e_bins = np.log10(e_bins / units.eV)
    e_centers = (e_bins[1:] + e_bins[:-1]) / 2

    fig, ax = plt.subplots(figsize=(10, 6))
    color = plt.cm.rainbow(np.linspace(0, 1, len(z_bins) - 1))

    direct_rate = np.nan_to_num(direct_rate)
    direct_error = np.nan_to_num(direct_error)
    reflected_rate = np.nan_to_num(reflected_rate)
    reflected_error = np.nan_to_num(reflected_error)

    # Plot each zenith bin for both direct and reflected
    for iZ in range(len(z_bins) - 1):
        zen_label = f'{z_bins[iZ]/units.deg:.1f}-{z_bins[iZ+1]/units.deg:.1f}deg'

        # Direct: solid line + shaded error
        ax.fill_between(
            e_centers,
            direct_rate[:, iZ] - direct_error[:, iZ],
            direct_rate[:, iZ] + direct_error[:, iZ],
            alpha=0.2, color=color[iZ]
        )
        ax.plot(e_centers, direct_rate[:, iZ], color=color[iZ], linestyle='-',
                label=zen_label)

        # Reflected: dashed line + shaded error
        ax.fill_between(
            e_centers,
            reflected_rate[:, iZ] - reflected_error[:, iZ],
            reflected_rate[:, iZ] + reflected_error[:, iZ],
            alpha=0.1, color=color[iZ]
        )
        ax.plot(e_centers, reflected_rate[:, iZ], color=color[iZ], linestyle='--')

    # Plot totals
    direct_total = np.nansum(direct_rate, axis=1)
    reflected_total = np.nansum(reflected_rate, axis=1)
    ax.plot(e_centers, direct_total, color='black', linestyle='-', linewidth=2,
            label=f'Direct total: {np.sum(direct_rate):.2f} Evts/Yr')
    ax.plot(e_centers, reflected_total, color='black', linestyle='--', linewidth=2,
            label=f'Reflected total: {np.sum(reflected_rate):.2f} Evts/Yr')

    ax.set_xlabel('log10(E/eV)')
    ax.set_ylabel('Evts/Yr')
    ax.set_yscale('log')
    ax.set_ylim(bottom=10**-3)

    # Dual legend: zenith colors + line style
    zenith_legend = ax.legend(loc='lower left', fontsize=7, ncol=2)
    ax.add_artist(zenith_legend)
    # Add style legend with explicit dash pattern for visibility
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='Direct'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2,
               dashes=(5, 3), label='Reflected'),
    ]
    ax.legend(handles=style_handles, loc='upper right', fontsize=9, handlelength=3.5)

    ax.set_title(title)
    fig.savefig(savename, dpi=150, bbox_inches='tight')
    ic(f'Saved {savename}')
    plt.close(fig)


def plotCombinedRadiusRate(
    event_list: Sequence[RCREvent],
    title: str,
    savename: str,
    max_distance: float = 6.0 * units.km,
    n_bins: int = 30,
) -> None:
    """Plot direct and reflected event rate vs radius on same axes.

    Args:
        event_list: List of RCREvent objects with weights set
        title: Plot title
        savename: Output file path
        max_distance: Maximum distance for binning
        n_bins: Number of radial bins
    """
    radius_bins = np.linspace(0, max_distance / units.m, n_bins + 1)
    bin_centers = (radius_bins[1:] + radius_bins[:-1]) / 2
    bin_width = radius_bins[1] - radius_bins[0]

    radii = np.array([event.get_radius() for event in event_list])
    direct_weights = np.array([event.get_weight('direct') for event in event_list])
    reflected_weights = np.array([event.get_weight('reflected') for event in event_list])

    direct_hist, _ = np.histogram(radii, bins=radius_bins, weights=direct_weights)
    reflected_hist, _ = np.histogram(radii, bins=radius_bins, weights=reflected_weights)

    fig, ax = plt.subplots()
    ax.bar(bin_centers - bin_width * 0.2, direct_hist, width=bin_width * 0.4,
           alpha=0.7, edgecolor='black', label=f'Direct: {np.sum(direct_hist):.3f} Evts/Yr')
    ax.bar(bin_centers + bin_width * 0.2, reflected_hist, width=bin_width * 0.4,
           alpha=0.7, edgecolor='black', color='red',
           label=f'Reflected: {np.sum(reflected_hist):.3f} Evts/Yr')

    ax.set_xlabel('Distance from Station (m)')
    ax.set_ylabel('Event Rate (Evts/Yr)')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend(fontsize=9)

    fig.savefig(savename, dpi=150, bbox_inches='tight')
    ic(f'Saved {savename}')
    plt.close(fig)


def plotWeightedHistograms(
    event_list: Sequence[RCREvent],
    savename: str,
    title: str = 'Weighted Distributions',
) -> None:
    """Plot 3 side-by-side histograms of SNR, Azimuth, Zenith weighted by event rate.

    Each histogram overlays direct (solid) and reflected (dashed/hatched).

    Args:
        event_list: List of RCREvent objects with weights and SNR set
        savename: Output file path
        title: Plot super-title
    """
    # Collect data
    direct_snr, reflected_snr = [], []
    direct_azimuth, reflected_azimuth = [], []
    direct_zenith, reflected_zenith = [], []
    direct_snr_w, reflected_snr_w = [], []
    direct_az_w, reflected_az_w = [], []
    direct_zen_w, reflected_zen_w = [], []

    for event in event_list:
        zen_deg = np.rad2deg(event.zenith)
        az_deg = np.rad2deg(event.azimuth)

        d_w = event.get_weight('direct')
        r_w = event.get_weight('reflected')

        if d_w > 0:
            direct_zenith.append(zen_deg)
            direct_azimuth.append(az_deg)
            direct_zen_w.append(d_w)
            direct_az_w.append(d_w)
            # SNR: check all direct stations (id < 100)
            for sid, snr_val in event.station_snr.items():
                if sid < 100:
                    direct_snr.append(snr_val)
                    direct_snr_w.append(d_w)

        if r_w > 0:
            reflected_zenith.append(zen_deg)
            reflected_azimuth.append(az_deg)
            reflected_zen_w.append(r_w)
            reflected_az_w.append(r_w)
            # SNR: check all reflected stations (id >= 100)
            for sid, snr_val in event.station_snr.items():
                if sid >= 100:
                    reflected_snr.append(snr_val)
                    reflected_snr_w.append(r_w)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)

    # Panel 1: SNR (20 bins, ~50% wider than original 30)
    ax = axes[0]
    has_snr = len(direct_snr) > 0 or len(reflected_snr) > 0
    if has_snr:
        all_snr = direct_snr + reflected_snr
        snr_bins = np.linspace(0, min(max(all_snr) * 1.1, 50), 20)
        if direct_snr:
            ax.hist(direct_snr, bins=snr_bins, weights=direct_snr_w,
                    alpha=0.6, label='Direct', edgecolor='black')
        if reflected_snr:
            ax.hist(reflected_snr, bins=snr_bins, weights=reflected_snr_w,
                    alpha=0.6, label='Reflected', edgecolor='black',
                    histtype='step', linewidth=2, linestyle='--')
    else:
        ax.text(0.5, 0.5, 'No SNR data', transform=ax.transAxes, ha='center', va='center')
    ax.set_xlabel('SNR')
    ax.set_ylabel('Weighted Count (Evts/Yr)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.set_title('SNR')

    # Panel 2: Azimuth (24 bins of 15°, ~50% wider than original 36 bins of 10°)
    ax = axes[1]
    az_bins = np.linspace(0, 360, 25)
    if direct_azimuth:
        ax.hist(direct_azimuth, bins=az_bins, weights=direct_az_w,
                alpha=0.6, label='Direct', edgecolor='black')
    if reflected_azimuth:
        ax.hist(reflected_azimuth, bins=az_bins, weights=reflected_az_w,
                alpha=0.6, label='Reflected', edgecolor='black',
                histtype='step', linewidth=2, linestyle='--')
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Weighted Count (Evts/Yr)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.set_title('Azimuth')

    # Panel 3: Zenith (12 bins of 7.5°, ~50% wider than original 18 bins of 5°)
    ax = axes[2]
    zen_bins = np.linspace(0, 90, 13)
    if direct_zenith:
        ax.hist(direct_zenith, bins=zen_bins, weights=direct_zen_w,
                alpha=0.6, label='Direct', edgecolor='black')
    if reflected_zenith:
        ax.hist(reflected_zenith, bins=zen_bins, weights=reflected_zen_w,
                alpha=0.6, label='Reflected', edgecolor='black',
                histtype='step', linewidth=2, linestyle='--')
    ax.set_xlabel('Zenith (deg)')
    ax.set_ylabel('Weighted Count (Evts/Yr)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.set_title('Zenith')

    plt.tight_layout()
    fig.savefig(savename, dpi=150, bbox_inches='tight')
    ic(f'Saved {savename}')
    plt.close(fig)


# =============================================================================
# Analysis Runner
# =============================================================================

def runAnalysis(
    event_list: Sequence[RCREvent],
    trigger_name: str,
    save_folder: str,
    max_distance: float = 5.0 * units.km,
    label: str = "",
) -> Dict[str, np.ndarray]:
    """Run full analysis pipeline on event list.

    Args:
        event_list: List of RCREvent objects
        trigger_name: Name of trigger to analyze
        save_folder: Directory for output plots
        max_distance: Maximum throw distance
        label: Optional label prefix for output files

    Returns:
        Dictionary with analysis results
    """
    os.makedirs(save_folder, exist_ok=True)
    prefix = f"{label}_" if label else ""

    e_bins, z_bins = getEnergyZenithBins()

    # Get trigger rates
    direct_rate, reflected_rate, n_throws = getBinnedTriggerRate(event_list, trigger_name)

    # Calculate event rates
    direct_event_rate = getEventRate(direct_rate, e_bins, z_bins, max_distance)
    reflected_event_rate = getEventRate(reflected_rate, e_bins, z_bins, max_distance)

    # Calculate errors
    direct_error = getErrorEventRates(direct_rate, event_list, max_distance)
    reflected_error = getErrorEventRates(reflected_rate, event_list, max_distance)

    # Set weights for spatial analysis
    setEventListRateWeight(
        event_list, direct_rate, 'direct', trigger_name,
        max_distance=max_distance, use_direct=True
    )
    setEventListRateWeight(
        event_list, reflected_rate, 'reflected', trigger_name,
        max_distance=max_distance, use_direct=False
    )

    # Create plots
    # 3a. Event rate vs energy for zenith bins
    plotRateWithError(
        direct_event_rate, direct_error,
        os.path.join(save_folder, f'{prefix}direct_event_rate_vs_energy.png'),
        f'Direct Event Rate - {label}'
    )
    plotRateWithError(
        reflected_event_rate, reflected_error,
        os.path.join(save_folder, f'{prefix}reflected_event_rate_vs_energy.png'),
        f'Reflected Event Rate - {label}'
    )

    # 3b. 2D histogram trigger/event rate
    imshowRate(
        direct_rate,
        f'Direct Trigger Rate - {label}',
        os.path.join(save_folder, f'{prefix}direct_trigger_rate_2d.png'),
        colorbar_label='Trigger Rate',
        event_list=event_list
    )
    imshowRate(
        reflected_rate,
        f'Reflected Trigger Rate - {label}',
        os.path.join(save_folder, f'{prefix}reflected_trigger_rate_2d.png'),
        colorbar_label='Trigger Rate',
        event_list=event_list
    )
    imshowRate(
        direct_event_rate,
        f'Direct Event Rate - {label}',
        os.path.join(save_folder, f'{prefix}direct_event_rate_2d.png'),
        colorbar_label=f'Evts/yr, Sum {np.nansum(direct_event_rate):.3f}',
        event_list=event_list
    )
    imshowRate(
        reflected_event_rate,
        f'Reflected Event Rate - {label}',
        os.path.join(save_folder, f'{prefix}reflected_event_rate_2d.png'),
        colorbar_label=f'Evts/yr, Sum {np.nansum(reflected_event_rate):.3f}',
        event_list=event_list
    )

    # 3c. Event rate vs radius
    histRadiusRate(
        event_list, 'direct',
        f'Direct Event Rate vs Distance - {label}',
        os.path.join(save_folder, f'{prefix}direct_rate_vs_radius.png'),
        max_distance=max_distance
    )
    histRadiusRate(
        event_list, 'reflected',
        f'Reflected Event Rate vs Distance - {label}',
        os.path.join(save_folder, f'{prefix}reflected_rate_vs_radius.png'),
        max_distance=max_distance
    )

    # Area rate plots
    histAreaRate(
        event_list, 'direct',
        f'Direct Event Rate - {label}',
        os.path.join(save_folder, f'{prefix}direct_rate_area.png'),
        max_distance=max_distance
    )
    histAreaRate(
        event_list, 'reflected',
        f'Reflected Event Rate - {label}',
        os.path.join(save_folder, f'{prefix}reflected_rate_area.png'),
        max_distance=max_distance
    )

    # Combined direct+reflected overlays
    plotCombinedRateWithError(
        direct_event_rate, direct_error,
        reflected_event_rate, reflected_error,
        os.path.join(save_folder, f'{prefix}combined_event_rate_vs_energy.png'),
        f'Combined Event Rate - {label}'
    )
    plotCombinedRadiusRate(
        event_list,
        f'Combined Event Rate vs Distance - {label}',
        os.path.join(save_folder, f'{prefix}combined_rate_vs_radius.png'),
        max_distance=max_distance
    )

    # Weighted distribution histograms (SNR, Azimuth, Zenith)
    plotWeightedHistograms(
        event_list,
        os.path.join(save_folder, f'{prefix}weighted_distributions.png'),
        title=f'Weighted Distributions - {label}'
    )

    return {
        'direct_trigger_rate': direct_rate,
        'reflected_trigger_rate': reflected_rate,
        'direct_event_rate': direct_event_rate,
        'reflected_event_rate': reflected_event_rate,
        'direct_error': direct_error,
        'reflected_error': reflected_error,
        'n_throws': n_throws,
        'e_bins': e_bins,
        'z_bins': z_bins,
    }


# =============================================================================
# Event Loading and Merging
# =============================================================================

def loadEventsFromFolder(
    folder: str | Path,
    pattern: str = "*_RCReventList.npy",
) -> List[RCREvent]:
    """Load all RCREvent objects from a folder.

    Args:
        folder: Path to folder containing .npy event files
        pattern: Glob pattern for event files

    Returns:
        List of RCREvent objects from all matching files
    """
    folder_path = Path(folder)
    event_files = list(folder_path.glob(pattern))

    all_events: List[RCREvent] = []
    for event_file in event_files:
        try:
            events = np.load(event_file, allow_pickle=True)
            all_events.extend(events)
        except Exception as e:
            print(f"Warning: Failed to load {event_file}: {e}")

    return all_events


def loadSeparatedSimulations(
    direct_folder: str | Path,
    reflected_folder: str | Path,
    direct_pattern: str = "*direct*_RCReventList.npy",
    reflected_pattern: str = "*_RCReventList.npy",
) -> List[RCREvent]:
    """Load and merge separately-run direct and reflected simulations.

    Combines trigger information from direct-only and reflected simulations.
    For events that exist in both, takes direct triggers from direct sim
    and reflected triggers from reflected sim.

    Args:
        direct_folder: Path to folder with direct simulation outputs
        reflected_folder: Path to folder with reflected simulation outputs
        direct_pattern: Glob pattern for direct event files
        reflected_pattern: Glob pattern for reflected event files

    Returns:
        List of merged RCREvent objects
    """
    direct_events = loadEventsFromFolder(direct_folder, direct_pattern)
    reflected_events = loadEventsFromFolder(reflected_folder, reflected_pattern)

    return mergeSeparatedEvents(direct_events, reflected_events)


def mergeSeparatedEvents(
    direct_events: List[RCREvent],
    reflected_events: List[RCREvent],
    position_tolerance: float = 1.0,
) -> List[RCREvent]:
    """Merge trigger information from separate direct and reflected runs.

    Matches events by (event_id, coreas_x, coreas_y) within tolerance.
    For matched events, combines triggers from both sources.
    Unmatched reflected events are added with no direct trigger.
    Unmatched direct events are added with no reflected trigger.

    Args:
        direct_events: Events from direct-only simulation (smaller throw area)
        reflected_events: Events from reflected simulation (full area)
        position_tolerance: Max distance (m) for matching event positions

    Returns:
        List of merged RCREvent objects
    """
    # Build lookup for direct events by (event_id, rounded position)
    direct_lookup: Dict[Tuple[int, float, float], RCREvent] = {}
    for evt in direct_events:
        key = (evt.event_id, round(evt.coreas_x, 0), round(evt.coreas_y, 0))
        direct_lookup[key] = evt

    merged_events: List[RCREvent] = []
    matched_direct_keys = set()

    # Process reflected events, merging with matching direct events
    for ref_evt in reflected_events:
        key = (ref_evt.event_id, round(ref_evt.coreas_x, 0), round(ref_evt.coreas_y, 0))

        if key in direct_lookup:
            # Merge: use direct event as base, add reflected triggers
            direct_evt = direct_lookup[key]
            matched_direct_keys.add(key)

            # Copy direct event and add reflected triggers
            merged_evt = RCREvent(
                event_id=direct_evt.event_id,
                energy=direct_evt.energy,
                zenith=direct_evt.zenith,
                azimuth=direct_evt.azimuth,
                coreas_x=direct_evt.coreas_x,
                coreas_y=direct_evt.coreas_y,
                station_triggers=dict(direct_evt.station_triggers),
                layer_dB=direct_evt.layer_dB,
            )
            # Add reflected triggers
            for trigger_name, station_ids in ref_evt.station_triggers.items():
                for stn_id in station_ids:
                    if stn_id >= REFLECTED_STATION_OFFSET:
                        merged_evt.add_trigger(trigger_name, stn_id)

            merged_events.append(merged_evt)
        else:
            # Reflected event outside direct throw area - add as-is
            merged_events.append(ref_evt)

    # Add unmatched direct events (have direct but no reflected)
    for key, direct_evt in direct_lookup.items():
        if key not in matched_direct_keys:
            merged_events.append(direct_evt)

    return merged_events


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Example usage
    config = configparser.ConfigParser()
    config.read('RCRSimulation/config.ini')

    numpy_folder = config.get('FOLDERS', 'numpy_folder', fallback='RCRSimulation/output/numpy')
    save_folder = config.get('FOLDERS', 'save_folder', fallback='RCRSimulation/plots')
    max_distance = float(config.get('SIMULATION', 'distance_km', fallback='5')) * units.km

    os.makedirs(save_folder, exist_ok=True)

    # Look for event files
    numpy_path = Path(numpy_folder)
    event_files = list(numpy_path.glob('*combined_RCReventList.npy'))
    # event_files = list(numpy_path.glob('*_RCReventList.npy'))

    if not event_files:
        ic('No combined event files found in', numpy_folder)
    else:
        for event_file in event_files:
            ic(f'Processing {event_file}')
            event_list = np.load(event_file, allow_pickle=True)

            # Get label from filename
            label = event_file.stem.replace('_RCReventList', '')

            # Determine trigger name by scanning events until we find one that triggered
            if len(event_list) > 0:
                trigger_names = set()
                for evt in event_list:
                    trigger_names.update(evt.all_trigger_names())
                    if trigger_names:
                        break

                if trigger_names:
                    # Filter out the noise pre-check trigger (2σ screening threshold)
                    # so we use the actual physics trigger for rate calculations
                    noise_tag = f"{NOISE_PRECHECK_SIGMA:g}sigma"
                    real_triggers = [t for t in trigger_names if noise_tag not in t]
                    if real_triggers:
                        trigger_name = sorted(real_triggers)[0]
                    else:
                        trigger_name = sorted(trigger_names)[0]
                    ic(f'Using trigger: {trigger_name}')

                    results = runAnalysis(
                        event_list,
                        trigger_name,
                        os.path.join(save_folder, label),
                        max_distance=max_distance,
                        label=label
                    )

                    ic(f'Direct event rate: {np.nansum(results["direct_event_rate"]):.3f} evts/yr')
                    ic(f'Reflected event rate: {np.nansum(results["reflected_event_rate"]):.3f} evts/yr')
                else:
                    ic(f'No triggered events found in {event_file}')
