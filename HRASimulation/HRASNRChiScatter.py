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


def collect_pair_metrics(event_list, weight_name, sigma, sigma_52, scenario):
    """Collect average SNR (x), Chi-difference spread (y), and weights for each station pair."""
    xs, ys, ws = [], [], []

    for event in event_list:
        if not event.hasWeight(weight_name, sigma=sigma):
            continue
        event_weight = event.getWeight(weight_name, sigma=sigma)
        if event_weight is None or event_weight <= 0:
            continue

        direct_ids = list(event.directTriggers(sigma=sigma, sigma_52=sigma_52))
        reflected_ids = list(event.reflectedTriggers(sigma=sigma, sigma_52=sigma_52))

        if scenario == "direct":
            candidate_pairs = list(itertools.combinations(direct_ids, 2))
        elif scenario == "refl":
            unique_ids = list(dict.fromkeys(direct_ids + reflected_ids))
            candidate_pairs = [
                combo for combo in itertools.combinations(unique_ids, 2)
                if (combo[0] in reflected_ids) or (combo[1] in reflected_ids)
            ]
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


def plot_scatter(direct_data, refl_data, output_path):
    x_direct, y_direct, w_direct = direct_data
    x_refl, y_refl, w_refl = refl_data

    if x_direct.size == 0 and x_refl.size == 0:
        ic("No coincidence pairs found; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xscale('log')
    ax.set_xlim(3, 100)
    ax.set_xlabel('Average SNR')
    ax.set_ylabel('|Δ(ChiRCR - Chi2016)| between stations')

    weight_concat = []
    if w_direct.size > 0:
        weight_concat.append(w_direct)
    if w_refl.size > 0:
        weight_concat.append(w_refl)
    weight_concat = np.concatenate(weight_concat) if weight_concat else np.array([])

    if weight_concat.size > 0:
        w_min = np.min(weight_concat)
        w_max = np.max(weight_concat)
        if w_max > w_min > 0:
            norm = colors.LogNorm(vmin=w_min, vmax=w_max)
        else:
            norm = colors.Normalize(vmin=w_min, vmax=w_max)
    else:
        norm = None

    divider = make_axes_locatable(ax)
    scatter_direct = None
    scatter_refl = None

    if x_refl.size > 0:
        order = np.argsort(w_refl)
        scatter_refl = ax.scatter(
            x_refl[order],
            y_refl[order],
            c=w_refl[order],
            cmap='Oranges',
            norm=norm,
            marker='^',
            edgecolors='none',
            alpha=0.9,
            label='Pairs with reflection',
        )

    if x_direct.size > 0:
        order = np.argsort(w_direct)
        scatter_direct = ax.scatter(
            x_direct[order],
            y_direct[order],
            c=w_direct[order],
            cmap='Blues',
            norm=norm,
            marker='o',
            edgecolors='none',
            alpha=0.9,
            label='Direct-only pairs',
        )

    if scatter_refl is not None:
        cax_refl = divider.append_axes('right', size='3%', pad=0.05)
        cbar_refl = fig.colorbar(scatter_refl, cax=cax_refl)
        cbar_refl.set_label('Weight (Evts/Yr) — Reflection')

    if scatter_direct is not None:
        pad_direct = 0.35 if scatter_refl is not None else 0.05
        cax_direct = divider.append_axes('right', size='3%', pad=pad_direct)
        cbar_direct = fig.colorbar(scatter_direct, cax=cax_direct)
        cbar_direct.set_label('Weight (Evts/Yr) — Direct')

    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_title('Chi Difference Spread vs Average SNR (2-Fold Coincidences)')

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
    )

    ic("Collecting reflection-required pair metrics...")
    refl_data = collect_pair_metrics(
        hra_events,
        weight_name=refl_weight,
        sigma=plot_sigma,
        sigma_52=sigma_52,
        scenario='refl',
    )

    output_file = os.path.join(snr_plot_folder, 'snr_chi_diff_scatter.png')
    plot_scatter(direct_data, refl_data, output_file)
    ic(f"Saved scatter plot to {output_file}")

    if weights_added:
        ic("New weights added; resaving HRA event list...")
        save_event_list(hra_events, hra_path)
        ic("HRA event list updated.")
