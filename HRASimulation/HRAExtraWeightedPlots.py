import argparse
import configparser
import itertools
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running this file directly from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import h5py
import pickle


def load_hra_from_h5(filename):
    """Load the pickled HRAevent objects from the HDF5 file written by S02_HRANurToNpy.

    Inlined here to avoid importing the full S02 pipeline (which can require
    optional dependencies not needed for plotting).
    """
    HRAeventList = []
    with h5py.File(filename, 'r') as hf:
        keys = list(hf.keys())
        for i in range(len(keys)):
            dataset = hf[f'object_{i}']
            if isinstance(dataset, h5py.Dataset) and dataset.dtype != h5py.special_dtype(vlen=np.dtype('uint8')):
                obj = dataset[...]
            else:
                obj_bytes = dataset[0]
                obj = pickle.loads(obj_bytes.tobytes())
            HRAeventList.append(obj)
    return HRAeventList


# Plot binning defaults (coarser than prior version)
DEFAULT_ZEN_BINS_DEG = np.linspace(0, 90, 31)  # ~3 deg bins
DEFAULT_AZ_BINS_DEG = np.linspace(0, 360, 37)  # 10 deg bins
DEFAULT_SNR_BINS = np.linspace(0, 30, 31)      # 1 SNR bins

# Requested 2D angle binning
ANGLE2D_ZEN_BINS_DEG = np.arange(0, 90 + 5, 5)
ANGLE2D_AZ_BINS_DEG = np.arange(0, 360 + 30, 30)


def _positive_floor(values, abs_floor=1e-12):
    """Choose a positive floor for log-scale plotting."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    pos = arr[arr > 0]
    if pos.size == 0:
        return abs_floor
    return max(abs_floor, float(np.min(pos)) * 0.5)


def _apply_log_y(ax, values_for_floor, abs_floor=1e-12):
    floor = _positive_floor(values_for_floor, abs_floor=abs_floor)
    ax.set_yscale('log')
    ax.set_ylim(bottom=floor)
    return floor


def _as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_weight(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return v


def _resolve_path(path):
    if path is None:
        return None
    path = str(path).strip()
    if path == "":
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(str(_REPO_ROOT), path)


def get_triggered_station_ids(event, sigma=4.5, bad_stations=None, station_mode='all'):
    """Return triggered station IDs for the given sigma (ints only).

    Filters out synthetic string triggers (e.g. 'combined_direct').

    station_mode: 'all' | 'direct' (<100) | 'reflected' (>=100)
    """
    triggered = []
    for key in event.station_triggers.get(sigma, []):
        if isinstance(key, str):
            continue
        try:
            st_id = int(key)
        except (TypeError, ValueError):
            continue
        if bad_stations is not None and st_id in bad_stations:
            continue
        if station_mode == 'direct' and st_id >= 100:
            continue
        if station_mode == 'reflected' and st_id < 100:
            continue
        triggered.append(st_id)
    return triggered


def get_effective_bases_for_event(event, base_stations, sigma=4.5, bad_stations=None):
    """Compute effective base-station set for an event.

    Uses the same notion as HRAAnalysis.categorize_events_by_coincidence:
    - direct triggers contribute base station ID
    - reflected triggers contribute (station_id - 100) base station ID
    """
    base_set = set(base_stations)
    triggered = get_triggered_station_ids(event, sigma=sigma, bad_stations=bad_stations, station_mode='all')
    direct = {st for st in triggered if st in base_set}
    reflected = {st - 100 for st in triggered if (st - 100) in base_set and st >= 100}
    return direct.union(reflected)


def get_direct_and_reflected_bases_for_event(event, base_stations, sigma=4.5, bad_stations=None):
    base_set = set(base_stations)
    triggered = get_triggered_station_ids(event, sigma=sigma, bad_stations=bad_stations, station_mode='all')
    direct = {st for st in triggered if st in base_set}
    refl = {st - 100 for st in triggered if st >= 100 and (st - 100) in base_set}
    return direct, refl


def event_passes_mode(event, base_stations, mode, sigma=4.5, bad_stations=None):
    """Mode filter used for 'direct-only' vs 'reflection-required' selections."""
    direct_bases, refl_bases = get_direct_and_reflected_bases_for_event(
        event, base_stations, sigma=sigma, bad_stations=bad_stations
    )
    if mode == 'direct_only':
        return len(direct_bases) > 0 and len(refl_bases) == 0
    if mode == 'reflection_required':
        return len(direct_bases) > 0 and len(refl_bases) > 0
    if mode == 'any':
        return (len(direct_bases) + len(refl_bases)) > 0
    raise ValueError(f"Unknown mode '{mode}'")


def get_event_reco_angles_deg(event, sigma=4.5, bad_stations=None, station_mode='all'):
    """Return (zen_deg, az_deg) averaged across triggered stations with reco available."""
    trig = get_triggered_station_ids(event, sigma=sigma, bad_stations=bad_stations, station_mode=station_mode)
    if len(trig) == 0:
        return None
    z_list = []
    a_list = []
    for st in trig:
        z = event.recon_zenith.get(st)
        a = event.recon_azimuth.get(st)
        if z is None or a is None:
            continue
        try:
            z = float(z)
            a = float(a)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(z) and np.isfinite(a)):
            continue
        z_list.append(z)
        a_list.append(a)
    if len(z_list) == 0:
        return None
    zen_deg = float(np.rad2deg(np.mean(z_list)))
    az_deg = float(np.rad2deg(np.mean(a_list))) % 360.0
    return zen_deg, az_deg


def get_net_event_weight(event, weight_names, sigma=4.5):
    w = 0.0
    for name in weight_names:
        if not event.hasWeight(name, sigma=sigma):
            continue
        w += _safe_weight(event.getWeight(name, sigma=sigma))
    return w


def get_best_available_event_weight(event, weight_names_descending, sigma=4.5):
    """Return the first (highest-priority) available weight from a list.

    This is useful for inclusive (e.g., n>=2) sums where events may carry multiple
    coincidence weights; we want to count each event once using its highest-n weight.
    """
    for name in weight_names_descending:
        if not event.hasWeight(name, sigma=sigma):
            continue
        return _safe_weight(event.getWeight(name, sigma=sigma))
    return 0.0


def weighted_true_angle_distribution(HRAeventList, outdir, weight_names, sigma=4.5, az_bins=None, zen_bins=None):
    az_deg = []
    zen_deg = []
    weights = []

    for ev in HRAeventList:
        w = get_net_event_weight(ev, weight_names, sigma=sigma)
        if w <= 0:
            continue
        zen, az = ev.getAngles()
        zen_deg.append(float(np.rad2deg(zen)))
        az_deg.append(float(np.rad2deg(az)) % 360.0)
        weights.append(w)

    if len(weights) == 0:
        return None

    fig = plt.figure(figsize=(8, 6))
    bins_az = DEFAULT_AZ_BINS_DEG if az_bins is None else az_bins
    bins_zen = DEFAULT_ZEN_BINS_DEG if zen_bins is None else zen_bins
    plt.hist2d(az_deg, zen_deg, bins=[bins_az, bins_zen], weights=weights)
    plt.xlabel('True Azimuth [deg]')
    plt.ylabel('True Zenith [deg]')
    plt.title(f"Weighted True Angular Distribution ({', '.join(weight_names)})")
    plt.colorbar(label='Rate [1/yr]')
    plt.tight_layout()
    savename = os.path.join(outdir, 'weighted_true_angles_avg_station_rate.png')
    plt.savefig(savename)
    plt.close(fig)
    return savename


def weighted_angle_distribution_2d_by_mode(
    HRAeventList,
    outdir,
    weight_names,
    base_stations,
    mode,
    sigma=4.5,
    bad_stations=None,
    use_reco=False,
    az_bins=None,
    zen_bins=None,
):
    """2D histogram of angles for a given selection mode.

    mode: 'direct_only' or 'reflection_required'
    use_reco: True -> reconstructed (event-avg) angles, False -> true angles
    """
    az_deg = []
    zen_deg = []
    weights = []

    for ev in HRAeventList:
        if not event_passes_mode(ev, base_stations, mode, sigma=sigma, bad_stations=bad_stations):
            continue
        w = get_net_event_weight(ev, weight_names, sigma=sigma)
        if w <= 0:
            continue

        if use_reco:
            # For reco, average across all triggered stations (direct+reflected) since the mode is event-level.
            reco = get_event_reco_angles_deg(ev, sigma=sigma, bad_stations=bad_stations, station_mode='all')
            if reco is None:
                continue
            zdeg, adeg = reco
        else:
            zen, az = ev.getAngles()
            zdeg = float(np.rad2deg(zen))
            adeg = float(np.rad2deg(az)) % 360.0

        zen_deg.append(zdeg)
        az_deg.append(adeg)
        weights.append(w)

    if len(weights) == 0:
        return None

    bins_az = ANGLE2D_AZ_BINS_DEG if az_bins is None else az_bins
    bins_zen = ANGLE2D_ZEN_BINS_DEG if zen_bins is None else zen_bins

    fig = plt.figure(figsize=(8, 6))
    plt.hist2d(az_deg, zen_deg, bins=[bins_az, bins_zen], weights=weights)
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Zenith [deg]')
    angle_tag = 'reco' if use_reco else 'true'
    plt.title(f"Weighted {angle_tag} angular distribution ({mode})")
    plt.colorbar(label='Rate [1/yr]')
    plt.tight_layout()
    savename = os.path.join(outdir, f'weighted_{angle_tag}_angles_2d_{mode}.png')
    plt.savefig(savename)
    plt.close(fig)
    return savename


def weighted_true_angle_1d_hists(
    HRAeventList,
    outdir,
    weight_names,
    tag,
    sigma=4.5,
    az_bins=None,
    zen_bins=None,
    logy=True,
):
    az_deg = []
    zen_deg = []
    weights = []
    for ev in HRAeventList:
        w = get_net_event_weight(ev, weight_names, sigma=sigma)
        if w <= 0:
            continue
        zen, az = ev.getAngles()
        zen_deg.append(float(np.rad2deg(zen)))
        az_deg.append(float(np.rad2deg(az)) % 360.0)
        weights.append(w)

    if len(weights) == 0:
        return []

    bins_az = DEFAULT_AZ_BINS_DEG if az_bins is None else az_bins
    bins_zen = DEFAULT_ZEN_BINS_DEG if zen_bins is None else zen_bins

    saved = []
    fig, ax = plt.subplots(figsize=(7, 5))
    counts, _ = np.histogram(zen_deg, bins=bins_zen, weights=weights)
    ax.hist(zen_deg, bins=bins_zen, weights=weights)
    ax.set_xlabel('True Zenith [deg]')
    ax.set_ylabel('Rate [1/yr]')
    ax.set_title(f'True Zenith (weighted) - {tag}')
    if logy:
        _apply_log_y(ax, counts)
    fig.tight_layout()
    savename = os.path.join(outdir, f'weighted_true_zenith_hist_{tag}.png')
    fig.savefig(savename)
    plt.close(fig)
    saved.append(savename)

    fig, ax = plt.subplots(figsize=(7, 5))
    counts, _ = np.histogram(az_deg, bins=bins_az, weights=weights)
    ax.hist(az_deg, bins=bins_az, weights=weights)
    ax.set_xlabel('True Azimuth [deg]')
    ax.set_ylabel('Rate [1/yr]')
    ax.set_title(f'True Azimuth (weighted) - {tag}')
    if logy:
        _apply_log_y(ax, counts)
    fig.tight_layout()
    savename = os.path.join(outdir, f'weighted_true_azimuth_hist_{tag}.png')
    fig.savefig(savename)
    plt.close(fig)
    saved.append(savename)

    return saved


def compute_n2_pair_rates(
    HRAeventList,
    base_stations,
    weight_names,
    sigma=4.5,
    bad_stations=None,
    pair_mode='any',
):
    """Return dict { (i,j): rate } for effective n=2 (two base stations) events.

    pair_mode:
      - 'any': no extra requirement beyond effective bases == 2
      - 'direct_only': require direct triggers only (no reflected bases)
      - 'reflection_required': require at least one direct and one reflected base
    """
    rates = defaultdict(float)
    for ev in HRAeventList:
        w = get_net_event_weight(ev, weight_names, sigma=sigma)
        if w <= 0:
            continue
        bases = sorted(get_effective_bases_for_event(ev, base_stations, sigma=sigma, bad_stations=bad_stations))
        if len(bases) != 2:
            continue

        if pair_mode != 'any':
            direct_bases, refl_bases = get_direct_and_reflected_bases_for_event(
                ev, base_stations, sigma=sigma, bad_stations=bad_stations
            )
            if pair_mode == 'direct_only':
                if len(refl_bases) != 0:
                    continue
                if len(direct_bases) == 0:
                    continue
            elif pair_mode == 'reflection_required':
                if len(refl_bases) == 0 or len(direct_bases) == 0:
                    continue

        rates[(bases[0], bases[1])] += w
    return dict(rates)


def compute_alln_pair_rates(
    HRAeventList,
    base_stations,
    weight_names_descending,
    sigma=4.5,
    bad_stations=None,
    pair_mode='any',
    min_n=2,
):
    """Return dict {(i,j): rate} including events with n>=min_n.

    Each event with effective base-station multiplicity >= min_n contributes its
    event weight to *every* pair (i,j) present in that event.

    weight_names_descending should be ordered highest-n -> lowest-n (e.g., 7..2)
    so we can take the event's highest available coincidence weight.
    """
    rates = defaultdict(float)
    for ev in HRAeventList:
        w = get_best_available_event_weight(ev, weight_names_descending, sigma=sigma)
        if w <= 0:
            continue

        bases = sorted(get_effective_bases_for_event(ev, base_stations, sigma=sigma, bad_stations=bad_stations))
        if len(bases) < min_n:
            continue

        if pair_mode != 'any':
            direct_bases, refl_bases = get_direct_and_reflected_bases_for_event(
                ev, base_stations, sigma=sigma, bad_stations=bad_stations
            )
            if pair_mode == 'direct_only':
                if len(refl_bases) != 0:
                    continue
                if len(direct_bases) == 0:
                    continue
            elif pair_mode == 'reflection_required':
                if len(refl_bases) == 0 or len(direct_bases) == 0:
                    continue

        for a, b in itertools.combinations(bases, 2):
            rates[(a, b)] += w

    return dict(rates)


def plot_pair_rate_bars(pair_rates, outdir, filename, title, logy=True):
    if not pair_rates:
        return None
    items = sorted(pair_rates.items(), key=lambda kv: kv[1], reverse=True)
    labels = [f"{a}-{b}" for (a, b), _ in items]
    values = [v for _, v in items]

    fig = plt.figure(figsize=(max(10, 0.35 * len(labels)), 5))
    x = np.arange(len(labels))
    y = np.asarray(values, dtype=float)
    y_plot = np.maximum(y, _positive_floor(y)) if logy else y
    plt.bar(x, y_plot)
    plt.xticks(x, labels, rotation=90)
    plt.ylabel('Rate [1/yr]')
    plt.title(title)
    if logy:
        _apply_log_y(plt.gca(), y)
    plt.tight_layout()
    savename = os.path.join(outdir, filename)
    plt.savefig(savename)
    plt.close(fig)
    return savename


def plot_pair_rate_bars_dual_axis(
    pair_rates_left,
    pair_rates_right,
    outdir,
    filename,
    title,
    left_label='Direct-only',
    right_label='Reflection-required',
):
    keys = sorted(set(pair_rates_left.keys()).union(pair_rates_right.keys()))
    if not keys:
        return None

    keys = sorted(keys, key=lambda k: max(pair_rates_left.get(k, 0.0), pair_rates_right.get(k, 0.0)), reverse=True)
    labels = [f"{a}-{b}" for (a, b) in keys]
    left_vals = np.array([float(pair_rates_left.get(k, 0.0)) for k in keys], dtype=float)
    right_vals = np.array([float(pair_rates_right.get(k, 0.0)) for k in keys], dtype=float)

    x = np.arange(len(keys))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(max(10, 0.35 * len(labels)), 5))
    ax2 = ax1.twinx()

    left_plot = np.maximum(left_vals, _positive_floor(left_vals))
    right_plot = np.maximum(right_vals, _positive_floor(right_vals))

    ax1.bar(x - width / 2, left_plot, width=width, label=left_label)
    ax2.bar(x + width / 2, right_plot, width=width, label=right_label, alpha=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90)
    ax1.set_title(title)
    ax1.set_ylabel(f'Rate [1/yr] ({left_label})')
    ax2.set_ylabel(f'Rate [1/yr] ({right_label})')

    _apply_log_y(ax1, left_vals)
    _apply_log_y(ax2, right_vals)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    savename = os.path.join(outdir, filename)
    fig.savefig(savename)
    plt.close(fig)
    return savename


def plot_pair_categories(
    pair_rates,
    outdir,
    categories,
    category_distances_km,
):
    saved = []

    # 3) per-category pair bar charts (sorted)
    for cat_name, pairs in categories.items():
        cat_rates = {}
        for a, b in pairs:
            key = tuple(sorted((int(a), int(b))))
            if key in pair_rates:
                cat_rates[key] = pair_rates[key]
            else:
                cat_rates[key] = 0.0
        sav = plot_pair_rate_bars(
            cat_rates,
            outdir,
            filename=f"weighted_n2_pair_rates_{cat_name}.png",
            title=f"Weighted n=2 Pair Rates ({cat_name})",
        )
        if sav is not None:
            saved.append(sav)

    # 4) category totals as a group histogram (more useful than distance scatter)
    cat_names = list(categories.keys())
    cat_total = []
    for cat_name, pairs in categories.items():
        tot = 0.0
        for a, b in pairs:
            key = tuple(sorted((int(a), int(b))))
            tot += float(pair_rates.get(key, 0.0))
        cat_total.append(tot)

    if len(cat_total) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        y = np.asarray(cat_total, dtype=float)
        ax.bar(np.arange(len(cat_names)), np.maximum(y, _positive_floor(y)))
        ax.set_xticks(np.arange(len(cat_names)))
        ax.set_xticklabels(cat_names)
        ax.set_ylabel('Total rate [1/yr]')
        ax.set_title('Weighted n=2 Category Total Rate')
        _apply_log_y(ax, y)
        fig.tight_layout()
        savename = os.path.join(outdir, 'weighted_n2_category_rate_by_group.png')
        fig.savefig(savename)
        plt.close(fig)
        saved.append(savename)

    return saved


def plot_pair_categories_dual_axis(
    pair_rates_left,
    pair_rates_right,
    outdir,
    categories,
    category_distances_km,
    left_label='Direct-only',
    right_label='Reflection-required',
):
    saved = []

    for cat_name, pairs in categories.items():
        left = {}
        right = {}
        for a, b in pairs:
            key = tuple(sorted((int(a), int(b))))
            left[key] = float(pair_rates_left.get(key, 0.0))
            right[key] = float(pair_rates_right.get(key, 0.0))

        sav = plot_pair_rate_bars_dual_axis(
            left,
            right,
            outdir,
            filename=f"weighted_n2_pair_rates_{cat_name}_dual_axis.png",
            title=f"Weighted n=2 Pair Rates ({cat_name})",
            left_label=left_label,
            right_label=right_label,
        )
        if sav is not None:
            saved.append(sav)

    # category totals as group histogram with dual y-axis
    cat_names = list(categories.keys())
    left_totals = []
    right_totals = []
    for cat_name, pairs in categories.items():
        lt = 0.0
        rt = 0.0
        for a, b in pairs:
            key = tuple(sorted((int(a), int(b))))
            lt += float(pair_rates_left.get(key, 0.0))
            rt += float(pair_rates_right.get(key, 0.0))
        left_totals.append(lt)
        right_totals.append(rt)

    if len(cat_names) > 0:
        left_totals = np.asarray(left_totals, dtype=float)
        right_totals = np.asarray(right_totals, dtype=float)

        # Friendly labels for plotting
        print_label = {
            'horizontal': 'Horizontal',
            'forward_diag': '/ diag.',
            'backward_diag': '\\ diag.',
            'other': 'All Other',
        }
        cat_labels = [print_label.get(name, name) for name in cat_names]

        x = np.arange(len(cat_names))
        width = 0.4
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax2 = ax1.twinx()

        # Ensure legend (on ax1) draws above ax2 artists.
        # Make ax1 the top axis but keep its background transparent so ax2 bars remain visible.
        ax2.set_zorder(0)
        ax1.set_zorder(1)
        ax1.patch.set_visible(False)

        ax1.bar(x - width / 2, np.maximum(left_totals, _positive_floor(left_totals)), width=width, label=left_label)
        ax2.bar(x + width / 2, np.maximum(right_totals, _positive_floor(right_totals)), width=width, label=right_label, alpha=0.7)

        ax1.set_xticks(x)
        ax1.set_xticklabels(cat_labels)
        ax1.set_ylabel(f'Total rate [1/yr] ({left_label})')
        ax2.set_ylabel(f'Total rate [1/yr] ({right_label})')
        ax1.set_title('Weighted n=2 Category Total Rate (dual axis)')

        _apply_log_y(ax1, left_totals)
        _apply_log_y(ax2, right_totals)

        # Single combined legend (bars are on different axes)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        leg = ax1.legend(h1 + h2, l1 + l2, loc='lower left')
        leg.set_zorder(10)
        if leg.get_frame() is not None:
            leg.get_frame().set_alpha(0.9)

        fig.tight_layout()
        savename = os.path.join(outdir, 'weighted_n2_category_rate_by_group_dual_axis.png')
        fig.savefig(savename)
        plt.close(fig)
        saved.append(savename)

    return saved


def weighted_snr_amplitude_histogram(
    HRAeventList,
    outdir,
    weight_names,
    sigma=4.5,
    bad_stations=None,
    station_mode='all',
    bins=None,
    logy=True,
):
    snr_vals = []
    snr_w = []

    for ev in HRAeventList:
        w = get_net_event_weight(ev, weight_names, sigma=sigma)
        if w <= 0:
            continue
        trig = get_triggered_station_ids(ev, sigma=sigma, bad_stations=bad_stations, station_mode=station_mode)
        if len(trig) == 0:
            continue
        # split the event weight across contributing station amplitudes
        per = w / len(trig)
        for st in trig:
            s = ev.getSNR(st)
            if s is None:
                continue
            s = float(s)
            if not np.isfinite(s):
                continue
            snr_vals.append(s)
            snr_w.append(per)

    if len(snr_vals) == 0:
        return None

    if bins is None:
        bins = DEFAULT_SNR_BINS

    fig, ax = plt.subplots(figsize=(7, 5))
    counts, _ = np.histogram(snr_vals, bins=bins, weights=snr_w)
    ax.hist(snr_vals, bins=bins, weights=snr_w)
    ax.set_xlabel('Max station SNR (triggered)')
    ax.set_ylabel('Rate [1/yr]')
    ax.set_title(f"Weighted SNR Amplitude ({station_mode})")
    if logy:
        _apply_log_y(ax, counts)
    fig.tight_layout()
    savename = os.path.join(outdir, f'weighted_station_snr_amplitude_hist_{station_mode}.png')
    fig.savefig(savename)
    plt.close(fig)
    return savename


def weighted_snr_amplitude_histogram_dual_axis(
    HRAeventList,
    outdir,
    weight_names_left,
    weight_names_right,
    sigma=4.5,
    bad_stations=None,
    bins=None,
    left_label='Direct',
    right_label='Reflected',
):
    if bins is None:
        bins = DEFAULT_SNR_BINS

    def _collect(mode, weight_names):
        vals, wts = [], []
        for ev in HRAeventList:
            w = get_net_event_weight(ev, weight_names, sigma=sigma)
            if w <= 0:
                continue
            trig = get_triggered_station_ids(ev, sigma=sigma, bad_stations=bad_stations, station_mode=mode)
            if len(trig) == 0:
                continue
            per = w / len(trig)
            for st in trig:
                s = ev.getSNR(st)
                if s is None:
                    continue
                s = float(s)
                if not np.isfinite(s):
                    continue
                vals.append(s)
                wts.append(per)
        return np.asarray(vals, dtype=float), np.asarray(wts, dtype=float)

    l_vals, l_w = _collect('direct', weight_names_left)
    r_vals, r_w = _collect('reflected', weight_names_right)
    l_counts, _ = np.histogram(l_vals, bins=bins, weights=l_w) if l_vals.size else (np.zeros(len(bins) - 1), bins)
    r_counts, _ = np.histogram(r_vals, bins=bins, weights=r_w) if r_vals.size else (np.zeros(len(bins) - 1), bins)

    centers = 0.5 * (bins[:-1] + bins[1:])
    width = (bins[1] - bins[0]) * 0.4

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.bar(centers - width / 2, np.maximum(l_counts, _positive_floor(l_counts)), width=width, label=left_label)
    ax2.bar(centers + width / 2, np.maximum(r_counts, _positive_floor(r_counts)), width=width, label=right_label, alpha=0.7)

    ax1.set_xlabel('Max station SNR (triggered)')
    ax1.set_ylabel(f'Rate [1/yr] ({left_label})')
    ax2.set_ylabel(f'Rate [1/yr] ({right_label})')
    ax1.set_title('Weighted SNR Amplitude (dual axis)')

    _apply_log_y(ax1, l_counts)
    _apply_log_y(ax2, r_counts)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    savename = os.path.join(outdir, 'weighted_station_snr_amplitude_hist_dual_axis.png')
    fig.savefig(savename)
    plt.close(fig)
    return savename


def weighted_reco_angle_hists(
    HRAeventList,
    outdir,
    weight_names,
    sigma=4.5,
    bad_stations=None,
    station_mode='all',
    zen_bins=None,
    az_bins=None,
    logy=True,
):
    reco_zen = []
    reco_az = []
    wts = []

    for ev in HRAeventList:
        w = get_net_event_weight(ev, weight_names, sigma=sigma)
        if w <= 0:
            continue
        trig = get_triggered_station_ids(ev, sigma=sigma, bad_stations=bad_stations, station_mode=station_mode)
        if len(trig) == 0:
            continue

        z_list = []
        a_list = []
        for st in trig:
            z = ev.recon_zenith.get(st)
            a = ev.recon_azimuth.get(st)
            if z is None or a is None:
                continue
            try:
                z = float(z)
                a = float(a)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(z) and np.isfinite(a)):
                continue
            z_list.append(z)
            a_list.append(a)
        if len(z_list) == 0:
            continue
        reco_zen.append(float(np.rad2deg(np.mean(z_list))))
        reco_az.append(float(np.rad2deg(np.mean(a_list))) % 360.0)
        wts.append(w)

    if len(wts) == 0:
        return []

    saved = []

    zen_bins = DEFAULT_ZEN_BINS_DEG if zen_bins is None else zen_bins
    az_bins = DEFAULT_AZ_BINS_DEG if az_bins is None else az_bins

    fig, ax = plt.subplots(figsize=(7, 5))
    counts, _ = np.histogram(reco_zen, bins=zen_bins, weights=wts)
    ax.hist(reco_zen, bins=zen_bins, weights=wts)
    ax.set_xlabel('Reco Zenith [deg] (event-avg across triggered stations)')
    ax.set_ylabel('Rate [1/yr]')
    ax.set_title(f"Weighted Reco Zenith ({station_mode})")
    if logy:
        _apply_log_y(ax, counts)
    fig.tight_layout()
    savename = os.path.join(outdir, f'weighted_reco_zenith_hist_{station_mode}.png')
    fig.savefig(savename)
    plt.close(fig)
    saved.append(savename)

    fig, ax = plt.subplots(figsize=(7, 5))
    counts, _ = np.histogram(reco_az, bins=az_bins, weights=wts)
    ax.hist(reco_az, bins=az_bins, weights=wts)
    ax.set_xlabel('Reco Azimuth [deg] (event-avg across triggered stations)')
    ax.set_ylabel('Rate [1/yr]')
    ax.set_title(f"Weighted Reco Azimuth ({station_mode})")
    if logy:
        _apply_log_y(ax, counts)
    fig.tight_layout()
    savename = os.path.join(outdir, f'weighted_reco_azimuth_hist_{station_mode}.png')
    fig.savefig(savename)
    plt.close(fig)
    saved.append(savename)

    return saved


def weighted_reco_angle_hists_dual_axis(
    HRAeventList,
    outdir,
    weight_names_left,
    weight_names_right,
    sigma=4.5,
    bad_stations=None,
    zen_bins=None,
    az_bins=None,
    left_label='Direct',
    right_label='Reflected',
):
    zen_bins = DEFAULT_ZEN_BINS_DEG if zen_bins is None else zen_bins
    az_bins = DEFAULT_AZ_BINS_DEG if az_bins is None else az_bins

    def _collect(mode, weight_names):
        rz, ra, wts = [], [], []
        for ev in HRAeventList:
            w = get_net_event_weight(ev, weight_names, sigma=sigma)
            if w <= 0:
                continue
            trig = get_triggered_station_ids(ev, sigma=sigma, bad_stations=bad_stations, station_mode=mode)
            if len(trig) == 0:
                continue
            z_list, a_list = [], []
            for st in trig:
                z = ev.recon_zenith.get(st)
                a = ev.recon_azimuth.get(st)
                if z is None or a is None:
                    continue
                try:
                    z = float(z)
                    a = float(a)
                except (TypeError, ValueError):
                    continue
                if not (np.isfinite(z) and np.isfinite(a)):
                    continue
                z_list.append(z)
                a_list.append(a)
            if len(z_list) == 0:
                continue
            rz.append(float(np.rad2deg(np.mean(z_list))))
            ra.append(float(np.rad2deg(np.mean(a_list))) % 360.0)
            wts.append(w)
        return np.asarray(rz, dtype=float), np.asarray(ra, dtype=float), np.asarray(wts, dtype=float)

    l_rz, l_ra, l_w = _collect('direct', weight_names_left)
    r_rz, r_ra, r_w = _collect('reflected', weight_names_right)

    saved = []

    # Zenith dual-axis
    l_counts, _ = np.histogram(l_rz, bins=zen_bins, weights=l_w) if l_rz.size else (np.zeros(len(zen_bins) - 1), zen_bins)
    r_counts, _ = np.histogram(r_rz, bins=zen_bins, weights=r_w) if r_rz.size else (np.zeros(len(zen_bins) - 1), zen_bins)
    centers = 0.5 * (zen_bins[:-1] + zen_bins[1:])
    width = (zen_bins[1] - zen_bins[0]) * 0.4
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.bar(centers - width / 2, np.maximum(l_counts, _positive_floor(l_counts)), width=width, label=left_label)
    ax2.bar(centers + width / 2, np.maximum(r_counts, _positive_floor(r_counts)), width=width, label=right_label, alpha=0.7)
    ax1.set_xlabel('Reco Zenith [deg]')
    ax1.set_ylabel(f'Rate [1/yr] ({left_label})')
    ax2.set_ylabel(f'Rate [1/yr] ({right_label})')
    ax1.set_title('Weighted Reco Zenith (dual axis)')
    _apply_log_y(ax1, l_counts)
    _apply_log_y(ax2, r_counts)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    savename = os.path.join(outdir, 'weighted_reco_zenith_hist_dual_axis.png')
    fig.savefig(savename)
    plt.close(fig)
    saved.append(savename)

    # Azimuth dual-axis
    l_counts, _ = np.histogram(l_ra, bins=az_bins, weights=l_w) if l_ra.size else (np.zeros(len(az_bins) - 1), az_bins)
    r_counts, _ = np.histogram(r_ra, bins=az_bins, weights=r_w) if r_ra.size else (np.zeros(len(az_bins) - 1), az_bins)
    centers = 0.5 * (az_bins[:-1] + az_bins[1:])
    width = (az_bins[1] - az_bins[0]) * 0.4
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.bar(centers - width / 2, np.maximum(l_counts, _positive_floor(l_counts)), width=width, label=left_label)
    ax2.bar(centers + width / 2, np.maximum(r_counts, _positive_floor(r_counts)), width=width, label=right_label, alpha=0.7)
    ax1.set_xlabel('Reco Azimuth [deg]')
    ax1.set_ylabel(f'Rate [1/yr] ({left_label})')
    ax2.set_ylabel(f'Rate [1/yr] ({right_label})')
    ax1.set_title('Weighted Reco Azimuth (dual axis)')
    _apply_log_y(ax1, l_counts)
    _apply_log_y(ax2, r_counts)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    savename = os.path.join(outdir, 'weighted_reco_azimuth_hist_dual_axis.png')
    fig.savefig(savename)
    plt.close(fig)
    saved.append(savename)

    return saved


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extra weighted plots: true angle distribution (avg station rate), "
            "n=2 pair rates (all + categorized + distance), weighted SNR amplitude histogram, "
            "and weighted reco az/zen hists for net individual rate."
        )
    )
    parser.add_argument('--config', default='HRASimulation/config.ini')
    parser.add_argument('--h5', default=None, help='Path to HRAeventList.h5 (defaults to config numpy_folder)')
    parser.add_argument('--outdir', default=None, help='Output directory (defaults to config save_folder)')
    parser.add_argument('--sigma', type=float, default=None)
    parser.add_argument(
        '--weight-names',
        nargs='+',
        default=None,
        help=(
            "One or more weight names to sum for the net event rate. "
            "Default: combined_direct combined_reflected"
        ),
    )
    parser.add_argument(
        '--weight-direct',
        nargs='+',
        default=None,
        help="Weight name(s) for direct-only plots (default: combined_direct)",
    )
    parser.add_argument(
        '--weight-reflected',
        nargs='+',
        default=None,
        help="Weight name(s) for reflected plots (default: combined_reflected)",
    )
    parser.add_argument(
        '--pair-weight-direct',
        nargs='+',
        default=None,
        help="Weight name(s) for n=2 direct-only pair plots (default: 2_coincidence_norefl)",
    )
    parser.add_argument(
        '--pair-weight-reflected',
        nargs='+',
        default=None,
        help="Weight name(s) for n=2 reflection-required pair plots (default: 2_coincidence_wrefl)",
    )
    parser.add_argument(
        '--base-stations',
        nargs='+',
        type=int,
        default=[13, 14, 15, 17, 18, 19, 30],
        help='Base station IDs to use for effective n=2 pairing.',
    )
    parser.add_argument(
        '--bad-stations',
        nargs='*',
        type=int,
        default=None,
        help='Triggered station IDs to exclude (optional).',
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    numpy_folder = _resolve_path(config['FOLDERS']['numpy_folder'])
    save_folder = _resolve_path(config['FOLDERS']['save_folder'])
    sigma = args.sigma
    if sigma is None:
        sigma = float(config['PLOTPARAMETERS']['trigger_sigma'])

    h5_path = args.h5
    if h5_path is None:
        h5_path = os.path.join(numpy_folder, 'HRAeventList.h5')
    h5_path = _resolve_path(h5_path)

    outdir = args.outdir
    if outdir is None:
        outdir = os.path.join(save_folder, 'extra_weighted_plots')
    outdir = _resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)

    # Subfolders for pair/coincidence-dependent plots
    outdir_n2only = os.path.join(outdir, 'n2only')
    outdir_alln = os.path.join(outdir, 'alln')
    os.makedirs(outdir_n2only, exist_ok=True)
    os.makedirs(outdir_alln, exist_ok=True)

    weight_names = args.weight_names
    if weight_names is None:
        weight_names = ['combined_direct', 'combined_reflected']

    weight_direct = args.weight_direct if args.weight_direct is not None else ['combined_direct']
    weight_reflected = args.weight_reflected if args.weight_reflected is not None else ['combined_reflected']
    pair_weight_direct = (
        args.pair_weight_direct if args.pair_weight_direct is not None else ['2_coincidence_norefl']
    )
    pair_weight_reflected = (
        args.pair_weight_reflected if args.pair_weight_reflected is not None else ['2_coincidence_wrefl']
    )

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Could not find HRAeventList.h5 at '{h5_path}'. "
            "Pass --h5 with the correct path, or generate it by running HRASimulation/S02_HRANurToNpy.py."
        )

    print(f"Loading HRA event list: {h5_path}")
    HRAeventList = load_hra_from_h5(h5_path)
    print(f"Loaded {len(HRAeventList)} events")
    print(f"Using sigma={sigma}")
    print(f"Net weights (combined): {weight_names}")
    print(f"Direct weights: {weight_direct}")
    print(f"Reflected weights: {weight_reflected}")
    print(f"Pair weights direct-only: {pair_weight_direct}")
    print(f"Pair weights reflection-required: {pair_weight_reflected}")

    # 1) weighted true angular distribution using average station rate (combined)
    weighted_true_angle_distribution(
        HRAeventList,
        outdir,
        weight_names,
        sigma=sigma,
        az_bins=DEFAULT_AZ_BINS_DEG,
        zen_bins=DEFAULT_ZEN_BINS_DEG,
    )

    # Requested: 2D angular distributions with 30deg az bins and 5deg zen bins
    # Make separately for direct-only vs reflection-required, and for both true and reconstructed angles.
    weighted_angle_distribution_2d_by_mode(
        HRAeventList,
        outdir,
        weight_direct,
        base_stations=args.base_stations,
        mode='direct_only',
        sigma=sigma,
        bad_stations=args.bad_stations,
        use_reco=False,
        az_bins=ANGLE2D_AZ_BINS_DEG,
        zen_bins=ANGLE2D_ZEN_BINS_DEG,
    )
    weighted_angle_distribution_2d_by_mode(
        HRAeventList,
        outdir,
        weight_reflected,
        base_stations=args.base_stations,
        mode='reflection_required',
        sigma=sigma,
        bad_stations=args.bad_stations,
        use_reco=False,
        az_bins=ANGLE2D_AZ_BINS_DEG,
        zen_bins=ANGLE2D_ZEN_BINS_DEG,
    )
    weighted_angle_distribution_2d_by_mode(
        HRAeventList,
        outdir,
        weight_direct,
        base_stations=args.base_stations,
        mode='direct_only',
        sigma=sigma,
        bad_stations=args.bad_stations,
        use_reco=True,
        az_bins=ANGLE2D_AZ_BINS_DEG,
        zen_bins=ANGLE2D_ZEN_BINS_DEG,
    )
    weighted_angle_distribution_2d_by_mode(
        HRAeventList,
        outdir,
        weight_reflected,
        base_stations=args.base_stations,
        mode='reflection_required',
        sigma=sigma,
        bad_stations=args.bad_stations,
        use_reco=True,
        az_bins=ANGLE2D_AZ_BINS_DEG,
        zen_bins=ANGLE2D_ZEN_BINS_DEG,
    )

    # Also provide 1D true angle hists (direct/reflected + dual-axis)
    weighted_true_angle_1d_hists(HRAeventList, outdir, weight_direct, tag='direct', sigma=sigma, logy=True)
    weighted_true_angle_1d_hists(HRAeventList, outdir, weight_reflected, tag='reflected', sigma=sigma, logy=True)
    # Dual-axis: zenith
    # (use bar-based dual-axis in the same style as other dual plots)
    # We reuse the existing helper by building per-bin arrays and plotting as dual-axis bars.
    # Keep output minimal: one zenith and one azimuth dual-axis plot.
    for angle_name, bins, getter in [
        ('zenith', DEFAULT_ZEN_BINS_DEG, lambda ev: float(np.rad2deg(ev.getAngles()[0]))),
        ('azimuth', DEFAULT_AZ_BINS_DEG, lambda ev: float(np.rad2deg(ev.getAngles()[1])) % 360.0),
    ]:
        l_vals, l_w = [], []
        r_vals, r_w = [], []
        for ev in HRAeventList:
            wl = get_net_event_weight(ev, weight_direct, sigma=sigma)
            if wl > 0:
                l_vals.append(getter(ev))
                l_w.append(wl)
            wr = get_net_event_weight(ev, weight_reflected, sigma=sigma)
            if wr > 0:
                r_vals.append(getter(ev))
                r_w.append(wr)
        l_counts, _ = np.histogram(l_vals, bins=bins, weights=l_w) if len(l_vals) else (np.zeros(len(bins) - 1), bins)
        r_counts, _ = np.histogram(r_vals, bins=bins, weights=r_w) if len(r_vals) else (np.zeros(len(bins) - 1), bins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        width = (bins[1] - bins[0]) * 0.4
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ax1.bar(centers - width / 2, np.maximum(l_counts, _positive_floor(l_counts)), width=width, label='Direct')
        ax2.bar(centers + width / 2, np.maximum(r_counts, _positive_floor(r_counts)), width=width, label='Reflected', alpha=0.7)
        ax1.set_xlabel(f'True {angle_name.title()} [deg]')
        ax1.set_ylabel('Rate [1/yr] (Direct)')
        ax2.set_ylabel('Rate [1/yr] (Reflected)')
        ax1.set_title(f'Weighted True {angle_name.title()} (dual axis)')
        _apply_log_y(ax1, l_counts)
        _apply_log_y(ax2, r_counts)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'weighted_true_{angle_name}_hist_dual_axis.png'))
        plt.close(fig)

    # 2) weighted rate of all effective n=2 pairs (direct-only vs reflection-required)
    # Save under n2only/
    pair_rates_direct = compute_n2_pair_rates(
        HRAeventList,
        base_stations=args.base_stations,
        weight_names=pair_weight_direct,
        sigma=sigma,
        bad_stations=args.bad_stations,
        pair_mode='direct_only',
    )
    pair_rates_reflreq = compute_n2_pair_rates(
        HRAeventList,
        base_stations=args.base_stations,
        weight_names=pair_weight_reflected,
        sigma=sigma,
        bad_stations=args.bad_stations,
        pair_mode='reflection_required',
    )
    plot_pair_rate_bars(
        pair_rates_direct,
        outdir_n2only,
        filename='weighted_n2_pair_rates_all_direct_only.png',
        title='Weighted effective n=2 pair rates (direct-only)',
        logy=True,
    )
    plot_pair_rate_bars(
        pair_rates_reflreq,
        outdir_n2only,
        filename='weighted_n2_pair_rates_all_reflection_required.png',
        title='Weighted effective n=2 pair rates (reflection-required)',
        logy=True,
    )
    plot_pair_rate_bars_dual_axis(
        pair_rates_direct,
        pair_rates_reflreq,
        outdir_n2only,
        filename='weighted_n2_pair_rates_all_dual_axis.png',
        title='Weighted effective n=2 pair rates (dual axis)',
        left_label='Direct-only',
        right_label='Reflection-required',
    )

    # 3-4) category lists (dummy placeholders for now)
    categories = {
        'horizontal': [
            [14, 19],
            [18, 30],
            [13, 18],
            [15, 17]
        ],
        'forward_diag': [
            [19, 30],
            [14, 18],
            [15, 18],
            [13, 17]
        ],
        'backward_diag': [
            [18, 19],
            [13, 14],
            [15, 30],
            [17, 18]
        ],
        'other': [
            [13, 15],
            [13, 19],
            [13, 30],
            [14, 15],
            [14, 17],
            [14, 30],
            [15, 19],
            [17, 19],
            [17, 30]
        ],
    }
    category_distances_km = {
        'horizontal': 1.0,
        'forward_diag': 1.0,
        'backward_diag': 1.0,
        'other': 2.0,
    }
    plot_pair_categories(pair_rates_direct, outdir_n2only, categories, category_distances_km)
    # Also separate plots for reflection-required categories
    plot_pair_categories(pair_rates_reflreq, outdir_n2only, categories, category_distances_km)
    # Dual-axis category plots
    plot_pair_categories_dual_axis(
        pair_rates_direct,
        pair_rates_reflreq,
        outdir_n2only,
        categories,
        category_distances_km,
        left_label='Direct-only',
        right_label='Reflection-required',
    )

    # alln/ versions: include events with n>=2 and count each event toward all pairs it contains
    max_n = max(2, len(args.base_stations))
    pair_weight_direct_alln = [f'{n}_coincidence_norefl' for n in range(max_n, 1, -1)]
    pair_weight_reflected_alln = [f'{n}_coincidence_wrefl' for n in range(max_n, 1, -1)]

    print(f"Pair weights direct-only (all n>=2, highest-n per event): {pair_weight_direct_alln}")
    print(f"Pair weights reflection-required (all n>=2, highest-n per event): {pair_weight_reflected_alln}")

    pair_rates_direct_alln = compute_alln_pair_rates(
        HRAeventList,
        base_stations=args.base_stations,
        weight_names_descending=pair_weight_direct_alln,
        sigma=sigma,
        bad_stations=args.bad_stations,
        pair_mode='direct_only',
        min_n=2,
    )
    pair_rates_reflreq_alln = compute_alln_pair_rates(
        HRAeventList,
        base_stations=args.base_stations,
        weight_names_descending=pair_weight_reflected_alln,
        sigma=sigma,
        bad_stations=args.bad_stations,
        pair_mode='reflection_required',
        min_n=2,
    )

    plot_pair_rate_bars(
        pair_rates_direct_alln,
        outdir_alln,
        filename='weighted_n2_pair_rates_all_direct_only.png',
        title='Weighted effective pair rates (all n>=2, direct-only)',
        logy=True,
    )
    plot_pair_rate_bars(
        pair_rates_reflreq_alln,
        outdir_alln,
        filename='weighted_n2_pair_rates_all_reflection_required.png',
        title='Weighted effective pair rates (all n>=2, reflection-required)',
        logy=True,
    )
    plot_pair_rate_bars_dual_axis(
        pair_rates_direct_alln,
        pair_rates_reflreq_alln,
        outdir_alln,
        filename='weighted_n2_pair_rates_all_dual_axis.png',
        title='Weighted effective pair rates (all n>=2, dual axis)',
        left_label='Direct-only',
        right_label='Reflection-required',
    )

    plot_pair_categories(pair_rates_direct_alln, outdir_alln, categories, category_distances_km)
    plot_pair_categories(pair_rates_reflreq_alln, outdir_alln, categories, category_distances_km)
    plot_pair_categories_dual_axis(
        pair_rates_direct_alln,
        pair_rates_reflreq_alln,
        outdir_alln,
        categories,
        category_distances_km,
        left_label='Direct-only',
        right_label='Reflection-required',
    )

    # 5) weighted amplitude (station max SNR) distribution
    # 5) weighted amplitude (station max SNR) distribution (separate + dual-axis)
    weighted_snr_amplitude_histogram(
        HRAeventList,
        outdir,
        weight_direct,
        sigma=sigma,
        bad_stations=args.bad_stations,
        station_mode='direct',
        logy=True,
    )
    weighted_snr_amplitude_histogram(
        HRAeventList,
        outdir,
        weight_reflected,
        sigma=sigma,
        bad_stations=args.bad_stations,
        station_mode='reflected',
        logy=True,
    )
    weighted_snr_amplitude_histogram_dual_axis(
        HRAeventList,
        outdir,
        weight_direct,
        weight_reflected,
        sigma=sigma,
        bad_stations=args.bad_stations,
        bins=DEFAULT_SNR_BINS,
        left_label='Direct',
        right_label='Reflected',
    )

    # 6) weighted reco az/zen hists for net individual rate
    # 6) weighted reco az/zen hists (separate + dual-axis)
    weighted_reco_angle_hists(
        HRAeventList,
        outdir,
        weight_direct,
        sigma=sigma,
        bad_stations=args.bad_stations,
        station_mode='direct',
        logy=True,
    )
    weighted_reco_angle_hists(
        HRAeventList,
        outdir,
        weight_reflected,
        sigma=sigma,
        bad_stations=args.bad_stations,
        station_mode='reflected',
        logy=True,
    )
    weighted_reco_angle_hists_dual_axis(
        HRAeventList,
        outdir,
        weight_direct,
        weight_reflected,
        sigma=sigma,
        bad_stations=args.bad_stations,
        left_label='Direct',
        right_label='Reflected',
    )

    print(f"Saved n-independent plots under: {outdir}")
    print(f"Saved n=2-only pair plots under: {outdir_n2only}")
    print(f"Saved all-n>=2 pair plots under: {outdir_alln}")


if __name__ == '__main__':
    main()
