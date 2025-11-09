import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import configparser
import h5py
import pickle
from icecream import ic
from mpl_toolkits.axes_grid1 import make_axes_locatable

from NuRadioReco.utilities import units
from HRASimulation.S02_HRANurToNpy import loadHRAfromH5
import HRASimulation.HRAAnalysis as HRAAnalysis


def ensure_coincidence_weight(event_list, coincidence_level, weight_name, sigma, sigma_52,
                              bad_stations, max_distance, force_stations=None):
    """Ensure the requested coincidence weight exists; returns True if updates were made."""
    if not event_list:
        return False
    if event_list[0].hasWeight(weight_name, sigma=sigma):
        return False

    ic(f"Weight '{weight_name}' not found. Calculating now...")
    trigger_rates = HRAAnalysis.getCoincidencesTriggerRates(
        event_list,
        bad_stations,
        sigma=sigma,
        sigma_52=sigma_52,
        force_stations=force_stations,
    )
    rate = trigger_rates.get(coincidence_level)
    if rate is None or not np.any(rate > 0):
        ic(f"No events found for {coincidence_level}-fold coincidence with weight '{weight_name}'.")
        return False

    HRAAnalysis.setNewTrigger(
        event_list,
        weight_name,
        bad_stations=bad_stations,
        sigma=sigma,
        sigma_52=sigma_52,
    )
    HRAAnalysis.setHRAeventListRateWeight(
        event_list,
        rate,
        weight_name=weight_name,
        max_distance=max_distance,
        sigma=sigma,
    )
    ic(f"Successfully calculated and added weights for '{weight_name}'.")
    return True


def compute_station_chi_delta(event, station_id):
    """Return ChiRCR - Chi2016 for a station, trying multiple key variants."""
    chi_2016 = event.getChi(station_id, "Chi2016")
    if chi_2016 is None or np.isnan(chi_2016):
        return None

    rcr_keys = ["ChiRCR", "ChiRCR100s", "ChiRCR200s", "RCR"]
    for key in rcr_keys:
        chi_rcr = event.getChi(station_id, key)
        if chi_rcr is not None and not np.isnan(chi_rcr):
            return chi_rcr - chi_2016
    return None


def collect_pair_metrics(
    event_list,
    weight_name,
    sigma,
    sigma_52,
    scenario,
    direct_exclusions=None,
    reflected_exclusions=None,
):
    """Collect average SNR (x), Chi-difference spread (y), and weights for each station pair."""
    direct_exclusions = set(direct_exclusions or [])
    reflected_exclusions = set(reflected_exclusions or [])
    xs, ys, ws = [], [], []

    for event in event_list:
        if not event.hasWeight(weight_name, sigma=sigma):
            continue
        event_weight = event.getWeight(weight_name, sigma=sigma)
        if event_weight is None or event_weight <= 0:
            continue

        direct_ids = [
            station
            for station in event.directTriggers(sigma=sigma, sigma_52=sigma_52)
            if isinstance(station, int) and station not in direct_exclusions
        ]
        reflected_ids = [
            station
            for station in event.reflectedTriggers(sigma=sigma, sigma_52=sigma_52)
            if isinstance(station, int) and station not in reflected_exclusions
        ]

        if scenario == "direct":
            candidate_pairs = list(itertools.combinations(direct_ids, 2))
        elif scenario == "direct_reflected":
            candidate_pairs = [(d, r) for d in direct_ids for r in reflected_ids]
        else:
            raise ValueError(f"Unknown scenario '{scenario}'")

        valid_pairs = []
        for station_a, station_b in candidate_pairs:
            snr_a = event.getSNR(station_a)
            snr_b = event.getSNR(station_b)
            if snr_a is None or snr_b is None:
                continue
            if np.isnan(snr_a) or np.isnan(snr_b):
                continue

            chi_delta_a = compute_station_chi_delta(event, station_a)
            chi_delta_b = compute_station_chi_delta(event, station_b)
            if chi_delta_a is None or chi_delta_b is None:
                continue
            if np.isnan(chi_delta_a) or np.isnan(chi_delta_b):
                continue

            avg_snr = 0.5 * (snr_a + snr_b)
            if avg_snr <= 0:
                continue
            spread = abs(chi_delta_a - chi_delta_b)
            valid_pairs.append((avg_snr, spread))

        if not valid_pairs:
            continue

        weight_per_pair = event_weight / len(valid_pairs)
        if weight_per_pair <= 0:
            continue

        for avg_snr, spread in valid_pairs:
            xs.append(avg_snr)
            ys.append(spread)
            ws.append(weight_per_pair)

    if xs:
        return np.array(xs), np.array(ys), np.array(ws)
    return np.array([]), np.array([]), np.array([])


def plot_single_scatter(data, cmap, marker, label, title, output_path):
    x_vals, y_vals, weights = data

    if x_vals.size == 0:
        ic(f"No coincidence pairs found for '{label}'; skipping {output_path}.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xscale('log')
    ax.set_xlim(3, 100)
    ax.set_xlabel('Average SNR')
    ax.set_ylabel('|Δ(ChiRCR - Chi2016)| between stations')

    if weights.size > 0:
        w_min = np.min(weights)
        w_max = np.max(weights)
        if w_max > w_min > 0:
            norm = colors.LogNorm(vmin=w_min, vmax=w_max)
        else:
            norm = colors.Normalize(vmin=w_min, vmax=w_max)
    else:
        norm = None

    order = np.argsort(weights) if weights.size > 0 else np.arange(x_vals.size)
    points = ax.scatter(
        x_vals[order],
        y_vals[order],
        c=weights[order] if weights.size > 0 else None,
        cmap=cmap,
        norm=norm,
        marker=marker,
        edgecolors='none',
        alpha=0.9,
        label=label,
    )

    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_title(title)

    if weights.size > 0:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.08)
        cbar = fig.colorbar(points, cax=cax)
        cbar.set_label('Weight (Evts/Yr)')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_event_list(event_list, path):
    """Persist the updated HRA event list back to HDF5."""
    with h5py.File(path, 'w') as hf:
        for idx, obj in enumerate(event_list):
            dataset_name = f'object_{idx}'
            if isinstance(obj, (np.ndarray, str, int, float)):
                hf.create_dataset(dataset_name, data=obj)
            else:
                obj_bytes = pickle.dumps(obj)
                dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                dset = hf.create_dataset(dataset_name, (1,), dtype=dt)
                dset[0] = np.frombuffer(obj_bytes, dtype='uint8')


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('HRASimulation/config.ini')

    numpy_folder = config['FOLDERS']['numpy_folder']
    save_folder = config['FOLDERS']['save_folder']
    diameter = float(config['SIMPARAMETERS']['diameter'])
    plot_sigma = float(config['PLOTPARAMETERS']['trigger_sigma'])
    sigma_52 = float(config['PLOTPARAMETERS']['trigger_simga_stn52'])

    max_distance = (diameter / 2.0) * units.km

    snr_plot_folder = os.path.join(save_folder, 'SNR_ChiScatter')
    os.makedirs(snr_plot_folder, exist_ok=True)

    hra_path = os.path.join(numpy_folder, 'HRAeventList.h5')
    ic("Loading HRA event list...")
    hra_events = loadHRAfromH5(hra_path)

    weights_added = False

    direct_weight = '2_coincidence_norefl'
    refl_weight = '2_coincidence_reflReq'

    direct_bad_stations = [32, 52, 113, 114, 115, 117, 118, 119, 130, 132, 152]
    refl_bad_stations = [32, 52, 132, 152]
    reflected_stations = [113, 114, 115, 117, 118, 119, 130]
    direct_exclusions = [32, 52]
    reflected_exclusions = [132, 152]

    if ensure_coincidence_weight(
        hra_events,
        coincidence_level=2,
        weight_name=direct_weight,
        sigma=plot_sigma,
        sigma_52=sigma_52,
        bad_stations=direct_bad_stations,
        max_distance=max_distance,
    ):
        weights_added = True

    if ensure_coincidence_weight(
        hra_events,
        coincidence_level=2,
        weight_name=refl_weight,
        sigma=plot_sigma,
        sigma_52=sigma_52,
        bad_stations=refl_bad_stations,
        max_distance=max_distance,
        force_stations=reflected_stations,
    ):
        weights_added = True

    ic("Collecting direct-only pair metrics...")
    direct_data = collect_pair_metrics(
        hra_events,
        weight_name=direct_weight,
        sigma=plot_sigma,
        sigma_52=sigma_52,
        scenario='direct',
        direct_exclusions=direct_exclusions,
        reflected_exclusions=reflected_exclusions,
    )

    ic("Collecting direct-reflected pair metrics...")
    refl_data = collect_pair_metrics(
        hra_events,
        weight_name=refl_weight,
        sigma=plot_sigma,
        sigma_52=sigma_52,
        scenario='direct_reflected',
        direct_exclusions=direct_exclusions,
        reflected_exclusions=reflected_exclusions,
    )

    direct_output = os.path.join(snr_plot_folder, 'snr_chi_diff_scatter_direct.png')
    plot_single_scatter(
        direct_data,
        cmap='Blues',
        marker='o',
        label='Direct-only pairs',
        title='Chi Difference Spread vs Average SNR (Direct Pairs)',
        output_path=direct_output,
    )
    ic(f"Saved direct pair scatter plot to {direct_output}")

    refl_output = os.path.join(snr_plot_folder, 'snr_chi_diff_scatter_direct_reflected.png')
    plot_single_scatter(
        refl_data,
        cmap='Oranges',
        marker='^',
        label='Direct-reflected pairs',
        title='Chi Difference Spread vs Average SNR (Direct-Reflected Pairs)',
        output_path=refl_output,
    )
    ic(f"Saved direct-reflected pair scatter plot to {refl_output}")

    if weights_added:
        ic("New weights added; resaving HRA event list...")
        save_event_list(hra_events, hra_path)
        ic("HRA event list updated.")
