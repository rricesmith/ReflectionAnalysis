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


DEFAULT_VALIDATION_PASSING_EVENT_IDS = [
    3047,
    3432,
    10195,
    10231,
    10273,
    10284,
    10444,
    10449,
    10466,
    10471,
    10554,
    11197,
    11220,
    11230,
    11236,
    11243,
]
DEFAULT_VALIDATION_SPECIAL_EVENT_ID = 11230
DEFAULT_VALIDATION_SPECIAL_STATION_ID = 13
VALIDATION_PICKLE_NAME = "9.24.25_CoincidenceDatetimes_passing_cuts_with_all_params_recalcZenAzi_calcPol.pkl"
DELTA_CUT = 0.15

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


def apply_delta_cut(data_tuple, delta_cut):
    """Filter pair data by delta cut and return kept/total weights."""
    x_vals, y_vals, weights = data_tuple

    if x_vals.size == 0:
        return (x_vals, y_vals, weights), 0.0, 0.0

    mask = y_vals >= delta_cut
    if weights.size > 0:
        kept_weight = float(np.sum(weights[mask]))
        total_weight = float(np.sum(weights))
    else:
        kept_weight = float(np.count_nonzero(mask))
        total_weight = float(len(y_vals))

    filtered_weights = weights[mask] if weights.size > 0 else weights
    return (x_vals[mask], y_vals[mask], filtered_weights), kept_weight, total_weight


def find_file_recursive(filename, search_roots):
    """Search for a filename within the provided root directories."""
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if filename in filenames:
                return os.path.join(dirpath, filename)
    return None


def load_validation_events(pickle_name):
    """Attempt to locate and load the validation events pickle."""
    cwd = os.getcwd()
    search_roots = [
        cwd,
        os.path.join(cwd, 'HRAStationDataAnalysis'),
        os.path.join(cwd, 'HRAStationDataAnalysis', 'StationData'),
        os.path.join(cwd, 'HRAStationDataAnalysis', 'StationData', 'processedNumpyData'),
    ]
    located_path = find_file_recursive(pickle_name, search_roots)
    if located_path is None:
        ic(f"Validation events file '{pickle_name}' not found in search paths.")
        return None, None

    try:
        with open(located_path, 'rb') as f:
            try:
                data = pickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                data = pickle.load(f, encoding='latin1')
        ic(f"Loaded validation events from {located_path}")
        return data, located_path
    except Exception as exc:
        ic(f"Failed to load validation events file {located_path}: {exc}")
        return None, located_path


def extract_station_metrics_from_event(event_details, direct_exclusions, reflected_exclusions):
    """Extract per-station SNR and Chi delta metrics from an event record."""
    if not isinstance(event_details, dict):
        return []

    stations = event_details.get('stations', {})
    metrics = []
    for station_key, station_data in stations.items():
        try:
            station_id = int(station_key)
        except (TypeError, ValueError):
            continue

        if station_id in direct_exclusions or station_id in reflected_exclusions:
            continue

        snr_values = np.asarray(station_data.get('SNR', []), dtype=float)
        if snr_values.size == 0:
            continue
        snr_val = float(np.nanmax(snr_values))
        if not np.isfinite(snr_val):
            continue

        chi_rcr_values = np.asarray(station_data.get('ChiRCR', []), dtype=float)
        chi_2016_values = np.asarray(station_data.get('Chi2016', []), dtype=float)
        if chi_rcr_values.size == 0 or chi_2016_values.size == 0:
            continue

        max_len = min(chi_rcr_values.size, chi_2016_values.size)
        chi_delta_values = chi_rcr_values[:max_len] - chi_2016_values[:max_len]
        valid_mask = np.isfinite(chi_delta_values)
        if not np.any(valid_mask):
            continue
        chi_delta_values = chi_delta_values[valid_mask]
        # Take the value with the largest absolute difference
        idx = int(np.argmax(np.abs(chi_delta_values)))
        chi_delta = float(chi_delta_values[idx])

        metrics.append({
            'station_id': station_id,
            'snr': snr_val,
            'chi_delta': chi_delta,
            'is_reflected': station_id >= 100,
        })

    return metrics


def compute_event_pair_summary(event_id, event_details, direct_exclusions, reflected_exclusions):
    """Compute the station pair with the largest delta spread for an event."""
    station_metrics = extract_station_metrics_from_event(
        event_details,
        direct_exclusions,
        reflected_exclusions,
    )

    if len(station_metrics) < 2:
        return None

    pair_records = []
    for station_a, station_b in itertools.combinations(station_metrics, 2):
        delta_spread = abs(station_a['chi_delta'] - station_b['chi_delta'])
        avg_snr = 0.5 * (station_a['snr'] + station_b['snr'])
        pair_records.append({
            'station_a': station_a,
            'station_b': station_b,
            'avg_snr': avg_snr,
            'delta_spread': delta_spread,
        })

    if not pair_records:
        return None

    max_pair = max(pair_records, key=lambda rec: rec['delta_spread'])
    min_pair = min(pair_records, key=lambda rec: rec['delta_spread'])
    event_avg_snr = float(np.mean([s['snr'] for s in station_metrics])) if station_metrics else None

    summary = {
        'event_id': event_id,
        'station_a': max_pair['station_a'],
        'station_b': max_pair['station_b'],
        'avg_snr': max_pair['avg_snr'],
        'delta_spread': max_pair['delta_spread'],
        'stations': station_metrics,
        'max_pair': max_pair,
        'min_pair': min_pair,
        'event_avg_snr': event_avg_snr,
    }

    return summary


def resolve_event_anchor_snr(summary_record):
    """Resolve a representative SNR for plotting validation events with station counts > 2."""
    candidates = [
        summary_record.get('event_avg_snr'),
        summary_record.get('avg_snr'),
        summary_record.get('max_pair', {}).get('avg_snr') if isinstance(summary_record.get('max_pair'), dict) else None,
    ]

    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def build_validation_pairs(events_dict, event_ids, direct_exclusions, reflected_exclusions):
    """Gather pair summaries for the requested validation events."""
    summaries = []
    if not isinstance(events_dict, dict):
        return summaries

    for event_id in event_ids:
        event_details = None
        if event_id in events_dict:
            event_details = events_dict[event_id]
        elif str(event_id) in events_dict:
            event_details = events_dict[str(event_id)]

        if event_details is None:
            ic(f"Validation event {event_id} not found in loaded data.")
            continue

        summary = compute_event_pair_summary(
            event_id,
            event_details,
            direct_exclusions,
            reflected_exclusions,
        )
        if summary is None:
            ic(f"Validation event {event_id} does not have enough valid stations for pair analysis.")
            continue

        summaries.append(summary)
        max_text = (
            f"max pair {summary['station_a']['station_id']} & {summary['station_b']['station_id']} "
            f"avg SNR {summary['avg_snr']:.2f}, |Δ| {summary['delta_spread']:.3f}"
        )
        min_pair = summary.get('min_pair')
        if min_pair is not None:
            min_text = (
                f"min pair {min_pair['station_a']['station_id']} & {min_pair['station_b']['station_id']} "
                f"avg SNR {min_pair['avg_snr']:.2f}, |Δ| {min_pair['delta_spread']:.3f}"
            )
            ic(f"Validation event {event_id}: {max_text}; {min_text}")
        else:
            ic(f"Validation event {event_id}: {max_text}")

    return summaries


def plot_validation_pairs(pairs, special_event_id, output_path, delta_cut):
    """Plot validation pair metrics, highlighting the special event."""
    if not pairs:
        ic("No validation pairs available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xscale('log')
    ax.set_xlim(3, 100)
    ax.set_xlabel('Average SNR')
    ax.set_ylabel('|Δ(ChiRCR - Chi2016)| between stations')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.axhline(delta_cut, color='dimgray', linestyle='--', linewidth=1, label=f'Delta cut ({delta_cut})')

    category_plotted = set()

    for record in pairs:
        num_stations = len(record.get('stations', []))
        if num_stations > 2:
            min_pair = record.get('min_pair')
            max_pair = record.get('max_pair')
            if min_pair is None or max_pair is None:
                ic(f"Validation event {record['event_id']} missing min/max pair details; skipping range plot.")
                continue

            x_anchor = resolve_event_anchor_snr(record)
            if x_anchor is None:
                ic(f"Validation event {record['event_id']} lacks a valid SNR reference; skipping range plot.")
                continue

            y_min = float(min_pair.get('delta_spread', 0.0))
            y_max = float(max_pair.get('delta_spread', 0.0))
            if y_min > y_max:
                y_min, y_max = y_max, y_min
            category = 'Validation n>2 (range)'
            label = category if category not in category_plotted else None

            scatter_vals = ax.scatter(
                [x_anchor, x_anchor],
                [y_min, y_max],
                marker='o',
                c='grey',
                s=90,
                edgecolors='none',
                alpha=0.85,
                label=label,
            )
            ax.vlines(x_anchor, y_min, y_max, colors='grey', linewidth=1.5, alpha=0.75)
            ax.annotate(
                str(record['event_id']),
                (x_anchor, y_max),
                textcoords='offset points',
                xytext=(4, 6),
                fontsize=9,
            )

            if label is not None:
                category_plotted.add(category)
            continue
        elif record['event_id'] == special_event_id:
            category = 'Validation special event'
            marker = '*'
            color = 'crimson'
            size = 170
        else:
            category = 'Validation n≤2'
            marker = 'o'
            color = 'steelblue'
            size = 110

        label = category if category not in category_plotted else None
        ax.scatter(
            record['avg_snr'],
            record['delta_spread'],
            marker=marker,
            c=color,
            s=size,
            edgecolors='none',
            alpha=0.85,
            label=label,
        )

        ax.annotate(
            str(record['event_id']),
            (record['avg_snr'], record['delta_spread']),
            textcoords='offset points',
            xytext=(4, 6),
            fontsize=9,
        )

        if label is not None:
            category_plotted.add(category)

    ax.set_title('Validation Event Chi Difference Spread (range shown for n>2)')
    ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    ic(f"Saved validation pair scatter plot to {output_path}")


def plot_special_station_pairs(event_summary, special_station_id, output_path, delta_cut):
    """Plot the delta spread between the special station and other stations for its event."""
    stations = event_summary.get('stations', [])
    special_station = next((s for s in stations if s['station_id'] == special_station_id), None)
    if special_station is None:
        ic(f"Special station {special_station_id} not found in event {event_summary.get('event_id')}.")
        return

    comparison_points = []
    for station in stations:
        if station['station_id'] == special_station_id:
            continue
        delta_spread = abs(special_station['chi_delta'] - station['chi_delta'])
        avg_snr = 0.5 * (special_station['snr'] + station['snr'])
        comparison_points.append((avg_snr, delta_spread, station['station_id']))
        ic(
            f"Special event {event_summary.get('event_id')} station {special_station_id} vs station {station['station_id']}: "
            f"avg SNR {avg_snr:.2f}, |Δ| {delta_spread:.3f}"
        )

    if not comparison_points:
        ic(f"No comparison stations available for special station {special_station_id} in event {event_summary.get('event_id')}")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xscale('log')
    ax.set_xlim(3, 100)
    ax.set_xlabel('Average SNR (with station {})'.format(special_station_id))
    ax.set_ylabel('|Δ(ChiRCR - Chi2016)| to station {}'.format(special_station_id))
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.axhline(delta_cut, color='dimgray', linestyle='--', linewidth=1, label=f'Delta cut ({delta_cut})')

    for avg_snr, delta_spread, other_station in comparison_points:
        ax.scatter(avg_snr, delta_spread, marker='o', c='darkgreen', s=100, edgecolors='none', alpha=0.85)
        ax.annotate(
            f"St {other_station}",
            (avg_snr, delta_spread),
            textcoords='offset points',
            xytext=(4, 6),
            fontsize=9,
        )

    ax.set_title(
        f"Event {event_summary.get('event_id')} - Station {special_station_id} Comparisons"
    )
    ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    ic(f"Saved special station comparison plot to {output_path}")


def format_weight_label(base_label, kept_weight, total_weight, delta_cut):
    """Format the legend label with the percentage of weight above the delta cut."""
    if total_weight and total_weight > 0:
        percentage = 100.0 * kept_weight / total_weight
    else:
        percentage = 0.0
    return f"{base_label} (>Δ{delta_cut:.2f}: {percentage:.2f}% weight)"


def plot_simulation_with_validation(
    sim_data,
    legend_label,
    title,
    output_path,
    cmap,
    marker,
    delta_cut,
    validation_pairs,
    special_event_id,
):
    fig, ax, scatter_sim, cut_line = plot_single_scatter(
        sim_data,
        cmap=cmap,
        marker=marker,
        legend_label=legend_label,
        title=title,
        output_path=None,
        delta_cut=delta_cut,
        add_colorbar=True,
        add_legend=False,
    )

    if fig is None or ax is None:
        ic(f"Skipping validation overlay plot '{output_path}' due to empty simulation data.")
        return

    handles = []
    labels = []
    if scatter_sim is not None:
        handles.append(scatter_sim)
        labels.append(legend_label)
    if cut_line is not None:
        handles.append(cut_line)
        labels.append(cut_line.get_label())

    category_handles = {}
    for record in validation_pairs:
        num_stations = len(record.get('stations', []))
        if num_stations > 2:
            min_pair = record.get('min_pair')
            max_pair = record.get('max_pair')
            if min_pair is None or max_pair is None:
                ic(f"Validation event {record['event_id']} missing min/max pair details; skipping range overlay.")
                continue

            x_anchor = resolve_event_anchor_snr(record)
            if x_anchor is None:
                ic(f"Validation event {record['event_id']} lacks a valid SNR reference; skipping range overlay.")
                continue

            y_min = float(min_pair.get('delta_spread', 0.0))
            y_max = float(max_pair.get('delta_spread', 0.0))
            if y_min > y_max:
                y_min, y_max = y_max, y_min
            category = 'Validation n>2 (range)'
            label = category if category not in category_handles else None

            scatter_val = ax.scatter(
                [x_anchor, x_anchor],
                [y_min, y_max],
                marker='o',
                c='grey',
                s=80,
                edgecolors='none',
                alpha=0.9,
                label=label,
            )
            ax.vlines(x_anchor, y_min, y_max, colors='grey', linewidth=1.5, alpha=0.75)
            ax.annotate(
                str(record['event_id']),
                (x_anchor, y_max),
                textcoords='offset points',
                xytext=(4, 6),
                fontsize=9,
            )

            if category not in category_handles and label is not None:
                category_handles[category] = scatter_val
                handles.append(scatter_val)
                labels.append(category)
            continue
        elif record['event_id'] == special_event_id:
            category = 'Validation special event'
            color = 'crimson'
            marker_v = '*'
            size = 160
        else:
            category = 'Validation n≤2'
            color = 'steelblue'
            marker_v = 'o'
            size = 100

        label = category if category not in category_handles else None
        scatter_val = ax.scatter(
            record['avg_snr'],
            record['delta_spread'],
            marker=marker_v,
            c=color,
            s=size,
            edgecolors='none',
            alpha=0.9,
            label=label,
        )

        ax.annotate(
            str(record['event_id']),
            (record['avg_snr'], record['delta_spread']),
            textcoords='offset points',
            xytext=(4, 6),
            fontsize=9,
        )

        if category not in category_handles and label is not None:
            category_handles[category] = scatter_val
            handles.append(scatter_val)
            labels.append(category)

    ax.legend(handles, labels, loc='upper left')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    ic(f"Saved validation overlay scatter plot to {output_path}")

def plot_single_scatter(
    data,
    cmap,
    marker,
    legend_label,
    title,
    output_path,
    delta_cut,
    ax=None,
    add_colorbar=True,
    add_legend=True,
):
    """Plot weighted simulation scatter with optional reuse of an existing axis."""

    x_vals, y_vals, weights = data

    if x_vals.size == 0:
        ic(f"No coincidence pairs found for '{legend_label}'; skipping {output_path}.")
        return None, None, None, None

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    ax.set_xscale('log')
    ax.set_xlim(3, 100)
    ax.set_xlabel('Average SNR')
    ax.set_ylabel('|Δ(ChiRCR - Chi2016)| between stations')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_title(title)

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
        label=legend_label,
    )

    cut_line = None
    if delta_cut is not None:
        cut_line = ax.axhline(
            delta_cut,
            color='dimgray',
            linestyle='--',
            linewidth=1,
            label=f'Δ cut ({delta_cut})',
        )

    if add_colorbar and weights.size > 0:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.08)
        cbar = fig.colorbar(points, cax=cax)
        cbar.set_label('Weight (Evts/Yr)')

    if add_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper left')

    if own_fig:
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path)
            plt.close(fig)
            return None, None, None, None

    return fig, ax, points, cut_line


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
    direct_exclusions_set = set(direct_exclusions)
    reflected_exclusions_set = set(reflected_exclusions)

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
        direct_exclusions=direct_exclusions_set,
        reflected_exclusions=reflected_exclusions_set,
    )

    ic("Collecting direct-reflected pair metrics...")
    refl_data = collect_pair_metrics(
        hra_events,
        weight_name=refl_weight,
        sigma=plot_sigma,
        sigma_52=sigma_52,
        scenario='direct_reflected',
        direct_exclusions=direct_exclusions_set,
        reflected_exclusions=reflected_exclusions_set,
    )

    direct_filtered, direct_kept_weight, direct_total_weight = apply_delta_cut(direct_data, DELTA_CUT)
    refl_filtered, refl_kept_weight, refl_total_weight = apply_delta_cut(refl_data, DELTA_CUT)

    if direct_total_weight > 0:
        direct_fraction = direct_kept_weight / direct_total_weight if direct_total_weight else 0.0
        ic(
            f"Direct pairs above delta {DELTA_CUT:.2f}: {direct_fraction * 100:.2f}% of weight "
            f"({direct_kept_weight:.4e}/{direct_total_weight:.4e})"
        )
    else:
        ic("Direct pairs have zero total weight; skipping percentage report.")

    if refl_total_weight > 0:
        refl_fraction = refl_kept_weight / refl_total_weight if refl_total_weight else 0.0
        ic(
            f"Direct-reflected pairs above delta {DELTA_CUT:.2f}: {refl_fraction * 100:.2f}% of weight "
            f"({refl_kept_weight:.4e}/{refl_total_weight:.4e})"
        )
    else:
        ic("Direct-reflected pairs have zero total weight; skipping percentage report.")

    direct_output = os.path.join(snr_plot_folder, 'snr_chi_diff_scatter_direct.png')
    direct_label = format_weight_label('Direct-only pairs', direct_kept_weight, direct_total_weight, DELTA_CUT)
    if direct_filtered[0].size > 0:
        plot_single_scatter(
            direct_filtered,
            cmap='Blues',
            marker='o',
            legend_label=direct_label,
            title='Chi Difference Spread vs Average SNR (Direct Pairs)',
            output_path=direct_output,
            delta_cut=DELTA_CUT,
        )
        ic(f"Saved direct pair scatter plot to {direct_output}")
    else:
        ic("No direct pairs exceed the delta cut; skipping direct scatter plot.")

    refl_output = os.path.join(snr_plot_folder, 'snr_chi_diff_scatter_direct_reflected.png')
    refl_label = format_weight_label('Direct-reflected pairs', refl_kept_weight, refl_total_weight, DELTA_CUT)
    if refl_filtered[0].size > 0:
        plot_single_scatter(
            refl_filtered,
            cmap='Oranges',
            marker='^',
            legend_label=refl_label,
            title='Chi Difference Spread vs Average SNR (Direct-Reflected Pairs)',
            output_path=refl_output,
            delta_cut=DELTA_CUT,
        )
        ic(f"Saved direct-reflected pair scatter plot to {refl_output}")
    else:
        ic("No direct-reflected pairs exceed the delta cut; skipping reflected scatter plot.")

    validation_events, validation_path = load_validation_events(VALIDATION_PICKLE_NAME)
    if validation_events is not None:
        validation_pairs = build_validation_pairs(
            validation_events,
            DEFAULT_VALIDATION_PASSING_EVENT_IDS,
            direct_exclusions_set,
            reflected_exclusions_set,
        )

        if validation_pairs:
            validation_output = os.path.join(snr_plot_folder, 'validation_event_pair_spread.png')
            plot_validation_pairs(
                validation_pairs,
                DEFAULT_VALIDATION_SPECIAL_EVENT_ID,
                validation_output,
                DELTA_CUT,
            )

            if direct_filtered[0].size > 0:
                overlay_direct_output = os.path.join(
                    snr_plot_folder,
                    'snr_chi_diff_scatter_direct_with_validation.png',
                )
                plot_simulation_with_validation(
                    direct_filtered,
                    legend_label=direct_label,
                    title='Chi Difference Spread vs Average SNR (Direct Pairs) — With Validation',
                    output_path=overlay_direct_output,
                    cmap='Blues',
                    marker='o',
                    delta_cut=DELTA_CUT,
                    validation_pairs=validation_pairs,
                    special_event_id=DEFAULT_VALIDATION_SPECIAL_EVENT_ID,
                )

            if refl_filtered[0].size > 0:
                overlay_refl_output = os.path.join(
                    snr_plot_folder,
                    'snr_chi_diff_scatter_direct_reflected_with_validation.png',
                )
                plot_simulation_with_validation(
                    refl_filtered,
                    legend_label=refl_label,
                    title='Chi Difference Spread vs Average SNR (Direct-Reflected Pairs) — With Validation',
                    output_path=overlay_refl_output,
                    cmap='Oranges',
                    marker='^',
                    delta_cut=DELTA_CUT,
                    validation_pairs=validation_pairs,
                    special_event_id=DEFAULT_VALIDATION_SPECIAL_EVENT_ID,
                )

            special_summary = next(
                (rec for rec in validation_pairs if rec['event_id'] == DEFAULT_VALIDATION_SPECIAL_EVENT_ID),
                None,
            )
            if special_summary is not None:
                special_output = os.path.join(
                    snr_plot_folder,
                    f"validation_special_event_{DEFAULT_VALIDATION_SPECIAL_EVENT_ID}_station_{DEFAULT_VALIDATION_SPECIAL_STATION_ID}.png",
                )
                plot_special_station_pairs(
                    special_summary,
                    DEFAULT_VALIDATION_SPECIAL_STATION_ID,
                    special_output,
                    DELTA_CUT,
                )
            else:
                ic(
                    f"Special validation event {DEFAULT_VALIDATION_SPECIAL_EVENT_ID} not included in loaded summaries; "
                    "skipping dedicated plot."
                )
        else:
            ic("No validation pairs computed; skipping validation plots.")
    else:
        ic(
            f"Validation events file '{VALIDATION_PICKLE_NAME}' could not be loaded. "
            "Skipping validation plotting."
        )

    if weights_added:
        ic("New weights added; resaving HRA event list...")
        save_event_list(hra_events, hra_path)
        ic("HRA event list updated.")
