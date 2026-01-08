import argparse
import configparser
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


def get_triggered_station_ids(event, sigma=4.5, bad_stations=None):
    """Return triggered station IDs for the given sigma (ints only).

    Filters out synthetic string triggers (e.g. 'combined_direct').
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
        triggered.append(st_id)
    return triggered


def get_effective_bases_for_event(event, base_stations, sigma=4.5, bad_stations=None):
    """Compute effective base-station set for an event.

    Uses the same notion as HRAAnalysis.categorize_events_by_coincidence:
    - direct triggers contribute base station ID
    - reflected triggers contribute (station_id - 100) base station ID
    """
    base_set = set(base_stations)
    triggered = get_triggered_station_ids(event, sigma=sigma, bad_stations=bad_stations)
    direct = {st for st in triggered if st in base_set}
    reflected = {st - 100 for st in triggered if (st - 100) in base_set and st >= 100}
    return direct.union(reflected)


def get_net_event_weight(event, weight_names, sigma=4.5):
    w = 0.0
    for name in weight_names:
        if not event.hasWeight(name, sigma=sigma):
            continue
        w += _safe_weight(event.getWeight(name, sigma=sigma))
    return w


def weighted_true_angle_distribution(HRAeventList, outdir, weight_names, sigma=4.5):
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
    bins_az = np.linspace(0, 360, 73)
    bins_zen = np.linspace(0, 90, 46)
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


def compute_n2_pair_rates(
    HRAeventList,
    base_stations,
    weight_names,
    sigma=4.5,
    bad_stations=None,
):
    """Return dict { (i,j): rate } for effective n=2 (two base stations) events."""
    rates = defaultdict(float)
    for ev in HRAeventList:
        w = get_net_event_weight(ev, weight_names, sigma=sigma)
        if w <= 0:
            continue
        bases = sorted(get_effective_bases_for_event(ev, base_stations, sigma=sigma, bad_stations=bad_stations))
        if len(bases) != 2:
            continue
        rates[(bases[0], bases[1])] += w
    return dict(rates)


def plot_pair_rate_bars(pair_rates, outdir, filename, title):
    if not pair_rates:
        return None
    items = sorted(pair_rates.items(), key=lambda kv: kv[1], reverse=True)
    labels = [f"{a}-{b}" for (a, b), _ in items]
    values = [v for _, v in items]

    fig = plt.figure(figsize=(max(10, 0.35 * len(labels)), 5))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=90)
    plt.ylabel('Rate [1/yr]')
    plt.title(title)
    plt.tight_layout()
    savename = os.path.join(outdir, filename)
    plt.savefig(savename)
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

    # 4) category totals vs distance
    cat_names = []
    cat_dist = []
    cat_total = []
    for cat_name, pairs in categories.items():
        tot = 0.0
        for a, b in pairs:
            key = tuple(sorted((int(a), int(b))))
            tot += float(pair_rates.get(key, 0.0))
        dist = _as_float(category_distances_km.get(cat_name), default=np.nan)
        if not np.isfinite(dist):
            continue
        cat_names.append(cat_name)
        cat_dist.append(dist)
        cat_total.append(tot)

    if len(cat_total) > 0:
        fig = plt.figure(figsize=(7, 5))
        plt.scatter(cat_dist, cat_total)
        for name, x, y in zip(cat_names, cat_dist, cat_total):
            plt.text(x, y, name)
        plt.xlabel('Pair separation [km] (category constant)')
        plt.ylabel('Total rate [1/yr]')
        plt.title('Weighted n=2 Category Rate vs Distance')
        plt.tight_layout()
        savename = os.path.join(outdir, 'weighted_n2_category_rate_vs_distance.png')
        plt.savefig(savename)
        plt.close(fig)
        saved.append(savename)

    return saved


def weighted_snr_amplitude_histogram(
    HRAeventList,
    outdir,
    weight_names,
    sigma=4.5,
    bad_stations=None,
    bins=None,
):
    snr_vals = []
    snr_w = []

    for ev in HRAeventList:
        w = get_net_event_weight(ev, weight_names, sigma=sigma)
        if w <= 0:
            continue
        trig = get_triggered_station_ids(ev, sigma=sigma, bad_stations=bad_stations)
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
        bins = np.linspace(0, 30, 61)

    fig = plt.figure(figsize=(7, 5))
    plt.hist(snr_vals, bins=bins, weights=snr_w)
    plt.xlabel('Max station SNR (triggered)')
    plt.ylabel('Rate [1/yr]')
    plt.title(f"Weighted SNR Amplitude Distribution ({', '.join(weight_names)})")
    plt.tight_layout()
    savename = os.path.join(outdir, 'weighted_station_snr_amplitude_hist.png')
    plt.savefig(savename)
    plt.close(fig)
    return savename


def weighted_reco_angle_hists(
    HRAeventList,
    outdir,
    weight_names,
    sigma=4.5,
    bad_stations=None,
):
    reco_zen = []
    reco_az = []
    wts = []

    for ev in HRAeventList:
        w = get_net_event_weight(ev, weight_names, sigma=sigma)
        if w <= 0:
            continue
        trig = get_triggered_station_ids(ev, sigma=sigma, bad_stations=bad_stations)
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

    fig = plt.figure(figsize=(7, 5))
    plt.hist(reco_zen, bins=np.linspace(0, 90, 46), weights=wts)
    plt.xlabel('Reco Zenith [deg] (event-avg across triggered stations)')
    plt.ylabel('Rate [1/yr]')
    plt.title(f"Weighted Reco Zenith (net individual rate)")
    plt.tight_layout()
    savename = os.path.join(outdir, 'weighted_reco_zenith_hist.png')
    plt.savefig(savename)
    plt.close(fig)
    saved.append(savename)

    fig = plt.figure(figsize=(7, 5))
    plt.hist(reco_az, bins=np.linspace(0, 360, 73), weights=wts)
    plt.xlabel('Reco Azimuth [deg] (event-avg across triggered stations)')
    plt.ylabel('Rate [1/yr]')
    plt.title(f"Weighted Reco Azimuth (net individual rate)")
    plt.tight_layout()
    savename = os.path.join(outdir, 'weighted_reco_azimuth_hist.png')
    plt.savefig(savename)
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

    weight_names = args.weight_names
    if weight_names is None:
        weight_names = ['combined_direct', 'combined_reflected']

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Could not find HRAeventList.h5 at '{h5_path}'. "
            "Pass --h5 with the correct path, or generate it by running HRASimulation/S02_HRANurToNpy.py."
        )

    print(f"Loading HRA event list: {h5_path}")
    HRAeventList = load_hra_from_h5(h5_path)
    print(f"Loaded {len(HRAeventList)} events")
    print(f"Using sigma={sigma} and weight_names={weight_names}")

    # 1) weighted true angular distribution using average station rate
    weighted_true_angle_distribution(HRAeventList, outdir, weight_names, sigma=sigma)

    # 2) weighted rate of all effective n=2 pairs
    pair_rates = compute_n2_pair_rates(
        HRAeventList,
        base_stations=args.base_stations,
        weight_names=weight_names,
        sigma=sigma,
        bad_stations=args.bad_stations,
    )
    plot_pair_rate_bars(
        pair_rates,
        outdir,
        filename='weighted_n2_pair_rates_all.png',
        title='Weighted effective n=2 pair rates (all pairs)',
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
    plot_pair_categories(pair_rates, outdir, categories, category_distances_km)

    # 5) weighted amplitude (station max SNR) distribution
    weighted_snr_amplitude_histogram(
        HRAeventList,
        outdir,
        weight_names=weight_names,
        sigma=sigma,
        bad_stations=args.bad_stations,
    )

    # 6) weighted reco az/zen hists for net individual rate
    weighted_reco_angle_hists(
        HRAeventList,
        outdir,
        weight_names=weight_names,
        sigma=sigma,
        bad_stations=args.bad_stations,
    )

    print(f"Saved plots under: {outdir}")


if __name__ == '__main__':
    main()
