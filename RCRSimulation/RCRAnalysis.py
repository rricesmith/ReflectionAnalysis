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
    cmap = matplotlib.cm.viridis.copy()
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
    cos_bins = np.cos(z_bins)

    rate_masked, cmap = set_bad_imshow(rate.T, 0)

    fig, ax = plt.subplots()

    im = ax.imshow(
        rate_masked,
        aspect='auto',
        origin='lower',
        extent=[min(e_bins), max(e_bins), min(cos_bins), max(cos_bins)],
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap
    )

    # Set y-axis labels to zenith in degrees
    ax_labels = [f'{z/units.deg:.0f}' for z in z_bins]
    ax_labels.reverse()
    ax.set_yticks(cos_bins)
    ax.set_yticklabels(ax_labels)
    ax.set_ylabel('Zenith (deg)')
    ax.set_xlabel('log(Energy (eV))')

    # Overlay unique energy-zenith pairs
    if event_list is not None:
        log_energies, zeniths_deg = getUniqueEnergyZenithPairs(event_list)
        cos_zeniths = np.cos(np.deg2rad(zeniths_deg))
        ax.scatter(
            log_energies, cos_zeniths,
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
    event_files = list(numpy_path.glob('*_RCReventList.npy'))

    if not event_files:
        ic('No event files found in', numpy_folder)
    else:
        for event_file in event_files:
            ic(f'Processing {event_file}')
            event_list = np.load(event_file, allow_pickle=True)

            # Get label from filename
            label = event_file.stem.replace('_RCReventList', '')

            # Determine trigger name from first event
            if len(event_list) > 0:
                trigger_names = event_list[0].all_trigger_names()
                if trigger_names:
                    trigger_name = trigger_names[0]
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
