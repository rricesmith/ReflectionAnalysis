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
from HRASimulation.HRAEventObject import stns_100s, stns_200s

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
DEFAULT_VALIDATION_SPECIAL_EVENT_IDS = [11230, 11243]
VALIDATION_PICKLE_NAME = "9.24.25_CoincidenceDatetimes_passing_cuts_with_all_params_recalcZenAzi_calcPol.pkl"
DELTA_CUT = 0.11

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


def collect_delta_plane_metrics(
    event_list,
    weight_name,
    sigma,
    sigma_52,
    scenario,
    direct_exclusions=None,
    reflected_exclusions=None,
):
    """Collect ordered Chi delta pairs (max, min) with weights for each station pair."""
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

            delta_a = compute_station_chi_delta(event, station_a)
            delta_b = compute_station_chi_delta(event, station_b)
            if delta_a is None or delta_b is None:
                continue
            if np.isnan(delta_a) or np.isnan(delta_b):
                continue

            larger = float(max(delta_a, delta_b))
            smaller = float(min(delta_a, delta_b))
            valid_pairs.append((larger, smaller))

        if not valid_pairs:
            continue

        weight_per_pair = event_weight / len(valid_pairs)
        if weight_per_pair <= 0:
            continue

        for larger, smaller in valid_pairs:
            xs.append(larger)
            ys.append(smaller)
            ws.append(weight_per_pair)

    if xs:
        return np.array(xs), np.array(ys), np.array(ws)
    return np.array([]), np.array([]), np.array([])


def compute_station_chi_delta(event, station_id):
    """Return ChiRCR - Chi2016 for the requested station if available."""

    def _safe_float(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    station_chi = event.getChi(station_id)
    if isinstance(station_chi, dict):
        chi_dict = station_chi
    else:
        chi_dict = {}

    def _extract_value(candidates):
        # Exact key matches first
        for key in candidates:
            if key in chi_dict:
                value = _safe_float(chi_dict[key])
                if value is not None:
                    return value

        # Case-insensitive matches
        lowered = {str(k).lower(): k for k in chi_dict}
        for key in candidates:
            lower_key = key.lower()
            if lower_key in lowered:
                value = _safe_float(chi_dict[lowered[lower_key]])
                if value is not None:
                    return value

        # Substring matches for fallbacks
        for existing_key, value in chi_dict.items():
            key_lower = str(existing_key).lower()
            if any(candidate.lower() in key_lower for candidate in candidates):
                extracted = _safe_float(value)
                if extracted is not None:
                    return extracted

        # Direct lookup via event accessor (covers default return of 0 when absent)
        for key in candidates:
            value = event.getChi(station_id, key)
            extracted = _safe_float(value)
            if extracted is not None:
                return extracted

        return None

    if station_id in stns_100s:
        rcr_candidates = ['ChiRCR100s', 'ChiRCR', 'ChiRCR200s', 'RCR']
    elif station_id in stns_200s:
        rcr_candidates = ['ChiRCR200s', 'ChiRCR', 'ChiRCR100s', 'RCR']
    else:
        rcr_candidates = ['ChiRCR', 'ChiRCR100s', 'ChiRCR200s', 'RCR']

    chi_rcr = _extract_value(rcr_candidates)
    chi_2016 = _extract_value(['Chi2016', '2016'])

    if chi_rcr is None or chi_2016 is None:
        return None

    return chi_rcr - chi_2016


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


def apply_lower_left_cut(data_tuple):
    """Remove points lying strictly in the lower-left quadrant (x < 0 and y < 0)."""
    x_vals, y_vals, weights = data_tuple

    if x_vals.size == 0:
        return (x_vals, y_vals, weights), 0.0, 0.0

    mask = ~((x_vals < 0) & (y_vals < 0))
    if weights.size > 0:
        kept_weight = float(np.sum(weights[mask]))
        total_weight = float(np.sum(weights))
    else:
        kept_weight = float(np.count_nonzero(mask))
        total_weight = float(len(x_vals))

    filtered_weights = weights[mask] if weights.size > 0 else weights
    return (x_vals[mask], y_vals[mask], filtered_weights), kept_weight, total_weight


def format_plane_cut_label(base_label, kept_weight, total_weight):
    """Create a legend label reporting the retained weight after the quadrant cut."""
    if total_weight and total_weight > 0:
        percentage = 100.0 * kept_weight / total_weight
    else:
        percentage = 0.0
    return f"{base_label} (quadrant cut: {percentage:.2f}% weight)"


def normalize_marker_sizes(weights, min_size=40.0, max_size=150.0):
    """Scale marker sizes based on weights for combined scatter plots."""
    if weights.size == 0 or not np.any(weights > 0):
        return np.full_like(weights, fill_value=min_size, dtype=float)

    w_min = float(np.min(weights[weights > 0]))
    w_max = float(np.max(weights))
    if w_max == w_min:
        return np.full(weights.shape, (min_size + max_size) * 0.5)

    normalized = (weights - w_min) / (w_max - w_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    return min_size + normalized * (max_size - min_size)


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
        'pair_records': pair_records,
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


def get_validation_plane_points(summary_record, exclude_lower_left=False):
    """Return ordered Chi delta pairs and optional range line data for validation plotting."""
    stations = summary_record.get('stations', []) if isinstance(summary_record, dict) else []
    chi_values = [
        float(s.get('chi_delta'))
        for s in stations
        if isinstance(s, dict) and s.get('chi_delta') is not None and np.isfinite(s.get('chi_delta'))
    ]

    def _filter_point(point):
        return not (exclude_lower_left and point[0] < 0 and point[1] < 0)

    points = []
    line_points = None

    if len(chi_values) > 3:
        chi_values_sorted = sorted(chi_values)
        min_delta = chi_values_sorted[0]
        max_delta = chi_values_sorted[-1]
        mid_index = len(chi_values_sorted) // 2
        if len(chi_values_sorted) % 2 == 0:
            median_delta = chi_values_sorted[mid_index - 1]
        else:
            median_delta = chi_values_sorted[mid_index]

        candidate_points = [
            (float(max(max_delta, min_delta)), float(min(max_delta, min_delta))),
            (float(max(median_delta, min_delta)), float(min(median_delta, min_delta))),
        ]

        for pt in candidate_points:
            if _filter_point(pt) and pt not in points:
                points.append(pt)

        if len(points) >= 2:
            line_points = (points[0], points[1])
    else:
        pair_records = summary_record.get('pair_records') or []
        for pair in pair_records:
            station_a = pair.get('station_a', {}) if isinstance(pair, dict) else {}
            station_b = pair.get('station_b', {}) if isinstance(pair, dict) else {}
            delta_a = station_a.get('chi_delta') if isinstance(station_a, dict) else None
            delta_b = station_b.get('chi_delta') if isinstance(station_b, dict) else None
            if delta_a is None or delta_b is None:
                continue
            if not (np.isfinite(delta_a) and np.isfinite(delta_b)):
                continue
            larger = float(max(delta_a, delta_b))
            smaller = float(min(delta_a, delta_b))
            point = (larger, smaller)
            if _filter_point(point) and point not in points:
                points.append(point)

    annotation_point = max(points, key=lambda pt: pt[0]) if points else None

    return {
        'points': points,
        'line_points': line_points,
        'annotation_point': annotation_point,
    }


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


def add_validation_snr_points(
    ax,
    validation_pairs,
    special_event_ids,
    category_handles=None,
    handles=None,
    labels=None,
):
    """Overlay validation markers on SNR-delta scatter axes."""
    if category_handles is None:
        category_handles = {}
    if handles is None:
        handles = []
    if labels is None:
        labels = []

    for record in validation_pairs:
        event_id = record.get('event_id')
        num_stations = len(record.get('stations', []))

        if num_stations > 2:
            min_pair = record.get('min_pair')
            max_pair = record.get('max_pair')
            if min_pair is None or max_pair is None:
                ic(f"Validation event {event_id} missing min/max pair details; skipping range plot.")
                continue

            x_anchor = resolve_event_anchor_snr(record)
            if x_anchor is None:
                ic(f"Validation event {event_id} lacks a valid SNR reference; skipping range plot.")
                continue

            y_min = float(min_pair.get('delta_spread', 0.0))
            y_max = float(max_pair.get('delta_spread', 0.0))
            if y_min > y_max:
                y_min, y_max = y_max, y_min

            category = 'Coincidence n>2'
            label = category if category not in category_handles else None
            scatter_val = ax.scatter(
                [x_anchor, x_anchor],
                [y_min, y_max],
                marker='s',
                c='grey',
                s=110,
                edgecolors='none',
                alpha=0.9,
                label=label,
                zorder=4,
            )
            ax.vlines(x_anchor, y_min, y_max, colors='grey', linewidth=1.5, alpha=0.75, zorder=3)
            annotate_point = (x_anchor, y_max)
        elif event_id in special_event_ids:
            category = 'Passing coincidence events'
            label = category if category not in category_handles else None
            scatter_val = ax.scatter(
                record['avg_snr'],
                record['delta_spread'],
                marker='*',
                c='crimson',
                s=170,
                edgecolors='none',
                alpha=0.9,
                label=label,
                zorder=5,
            )
            annotate_point = (record['avg_snr'], record['delta_spread'])
        else:
            category = 'Coincidence n=2'
            label = category if category not in category_handles else None
            scatter_val = ax.scatter(
                record['avg_snr'],
                record['delta_spread'],
                marker='s',
                c='forestgreen',
                s=130,
                edgecolors='none',
                alpha=0.9,
                label=label,
                zorder=4,
            )
            annotate_point = (record['avg_snr'], record['delta_spread'])

        ax.annotate(
            str(event_id),
            annotate_point,
            textcoords='offset points',
            xytext=(4, 6),
            fontsize=9,
            zorder=6,
        )

        if category not in category_handles and label is not None:
            category_handles[category] = scatter_val
            handles.append(scatter_val)
            labels.append(category)

    return handles, labels, category_handles


def plot_validation_pairs(pairs, special_event_ids, output_path, delta_cut):
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

    handles, labels, _ = add_validation_snr_points(ax, pairs, special_event_ids)

    ax.set_title('Validation Event Chi Difference Spread (range shown for n>2)')
    ax.legend(handles, labels, loc='upper left')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    ic(f"Saved validation pair scatter plot to {output_path}")





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
    special_event_ids,
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

    handles, labels, _ = add_validation_snr_points(
        ax,
        validation_pairs,
        special_event_ids,
        handles=handles,
        labels=labels,
    )

    ax.legend(handles, labels, loc='upper left')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    ic(f"Saved validation overlay scatter plot to {output_path}")


def plot_delta_plane(
    data,
    cmap,
    marker,
    legend_label,
    title,
    output_path,
    ax=None,
    add_colorbar=True,
    add_legend=True,
    zorder=3,
):
    """Plot Chi delta pairs as ordered coordinates (max, min)."""
    x_vals, y_vals, weights = data

    if x_vals.size == 0:
        ic(f"No coincidence pairs found for '{legend_label}'; skipping {output_path}.")
        return None, None, None, None

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    ax.set_xlabel('Larger Δ(ChiRCR - Chi2016)')
    ax.set_ylabel('Smaller Δ(ChiRCR - Chi2016)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)

    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.05 * max(1.0, abs(x_min) + abs(x_max))
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05 * max(1.0, abs(y_min) + abs(y_max))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect('equal', adjustable='box')

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
        zorder=zorder,
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

    return fig, ax, points


def add_validation_delta_plane_points(
    ax,
    validation_pairs,
    special_event_ids,
    category_handles=None,
    handles=None,
    labels=None,
    exclude_lower_left=False,
):
    """Overlay validation data onto a Chi delta plane axis."""
    if category_handles is None:
        category_handles = {}
    if handles is None:
        handles = []
    if labels is None:
        labels = []

    for record in validation_pairs:
        plane_data = get_validation_plane_points(record, exclude_lower_left=exclude_lower_left)
        points = plane_data.get('points', [])
        if not points:
            continue

        event_id = record.get('event_id')
        num_stations = len(record.get('stations', []))
        if event_id in special_event_ids:
            category = 'Passing coincidence events'
            color = 'crimson'
            marker_v = '*'
            size = 160
            point_z = 5
        elif num_stations > 3:
            category = 'Coincidence n>3'
            color = 'forestgreen'
            marker_v = 's'
            size = 120
            point_z = 4
        else:
            category = 'Validation n≤3'
            color = 'forestgreen'
            marker_v = 's'
            size = 120
            point_z = 4

        xs, ys = zip(*points)
        label = category if category not in category_handles else None
        scatter_val = ax.scatter(
            xs,
            ys,
            marker=marker_v,
            c=color,
            s=size,
            edgecolors='none',
            alpha=0.95,
            label=label,
            zorder=point_z,
        )

        line_segment = plane_data.get('line_points')
        if line_segment is not None and len(line_segment) == 2:
            (x0, y0), (x1, y1) = line_segment
            ax.plot(
                [x0, x1],
                [y0, y1],
                color='grey',
                linewidth=1.5,
                alpha=0.75,
                zorder=point_z - 1,
            )

        annotate_point = plane_data.get('annotation_point')
        if annotate_point is None:
            annotate_point = max(points, key=lambda pt: pt[0])
        ax.annotate(
            str(event_id),
            annotate_point,
            textcoords='offset points',
            xytext=(4, 6),
            fontsize=9,
            zorder=point_z + 1,
        )

        if category not in category_handles and label is not None:
            category_handles[category] = scatter_val
            handles.append(scatter_val)
            labels.append(category)

    return handles, labels, category_handles


def plot_delta_plane_with_validation(
    sim_data,
    legend_label,
    title,
    output_path,
    cmap,
    marker,
    validation_pairs,
    special_event_ids,
    exclude_lower_left=False,
):
    """Plot simulation delta plane data with validation overlays."""
    fig, ax, scatter_sim = plot_delta_plane(
        sim_data,
        cmap=cmap,
        marker=marker,
        legend_label=legend_label,
        title=title,
        output_path=None,
        add_colorbar=True,
        add_legend=False,
        zorder=1,
    )

    if fig is None or ax is None:
        ic(f"Skipping validation overlay plane plot '{output_path}' due to empty simulation data.")
        return

    handles = []
    labels = []
    if scatter_sim is not None:
        handles.append(scatter_sim)
        labels.append(legend_label)

    handles, labels, _ = add_validation_delta_plane_points(
        ax,
        validation_pairs,
        special_event_ids,
        handles=handles,
        labels=labels,
        exclude_lower_left=exclude_lower_left,
    )

    ax.legend(handles, labels, loc='upper left')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    ic(f"Saved validation overlay delta plane plot to {output_path}")


def plot_combined_delta_plane_with_validation(
    direct_data,
    direct_label,
    refl_data,
    refl_label,
    title,
    output_path,
    validation_pairs,
    special_event_ids,
    exclude_lower_left=False,
):
    """Plot both direct and direct-reflected delta planes with validation overlays."""
    direct_x, direct_y, direct_w = direct_data
    refl_x, refl_y, refl_w = refl_data

    if direct_x.size == 0 and refl_x.size == 0:
        ic(f"No delta plane data available for combined plot '{output_path}'.")
        return

    combined_x = np.concatenate([arr for arr in [direct_x, refl_x] if arr.size > 0])
    combined_y = np.concatenate([arr for arr in [direct_y, refl_y] if arr.size > 0])

    x_min = float(np.min(combined_x))
    x_max = float(np.max(combined_x))
    y_min = float(np.min(combined_y))
    y_max = float(np.max(combined_y))

    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.05 * max(1.0, abs(x_min) + abs(x_max))
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05 * max(1.0, abs(y_min) + abs(y_max))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlabel('Larger Δ(ChiRCR - Chi2016)')
    ax.set_ylabel('Smaller Δ(ChiRCR - Chi2016)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    handles = []
    labels = []

    if direct_x.size > 0:
        sizes_direct = normalize_marker_sizes(direct_w)
        scatter_direct = ax.scatter(
            direct_x,
            direct_y,
            s=sizes_direct,
            c='tab:blue',
            alpha=0.6,
            marker='o',
            edgecolors='none',
            label=direct_label,
            zorder=1,
        )
        handles.append(scatter_direct)
        labels.append(direct_label)

    if refl_x.size > 0:
        sizes_refl = normalize_marker_sizes(refl_w)
        scatter_refl = ax.scatter(
            refl_x,
            refl_y,
            s=sizes_refl,
            c='tab:orange',
            alpha=0.75,
            marker='^',
            edgecolors='none',
            label=refl_label,
            zorder=2,
        )
        handles.append(scatter_refl)
        labels.append(refl_label)

    handles, labels, _ = add_validation_delta_plane_points(
        ax,
        validation_pairs,
        special_event_ids,
        handles=handles,
        labels=labels,
        exclude_lower_left=exclude_lower_left,
    )

    ax.legend(handles, labels, loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    ic(f"Saved combined delta plane plot to {output_path}")


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
    ax.set_ylabel('|ΔChi| between stations')
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


def plot_combined_snr_delta_with_validation(
    direct_data,
    direct_label,
    refl_data,
    refl_label,
    output_path,
    delta_cut,
    validation_pairs,
    special_event_ids,
):
    """Create a combined SNR vs Δ scatter with validation overlays."""
    direct_x, direct_y, direct_w = direct_data
    refl_x, refl_y, refl_w = refl_data

    if direct_x.size == 0 and refl_x.size == 0:
        ic(f"No SNR-delta data available for combined plot '{output_path}'.")
        return

    combined_x = np.concatenate([arr for arr in [direct_x, refl_x] if arr.size > 0])
    x_min = float(np.min(combined_x))
    x_max = float(np.max(combined_x))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xscale('log')
    ax.set_xlim(max(3.0, x_min * 0.95), max(100.0, x_max * 1.05))
    ax.set_xlabel('Average SNR')
    ax.set_ylabel('|ΔChi| between stations')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_title('Chi Difference Spread vs Average SNR — Combined Pairs with Validation')

    handles = []
    labels = []

    if direct_x.size > 0:
        sizes_direct = normalize_marker_sizes(direct_w)
        scatter_direct = ax.scatter(
            direct_x,
            direct_y,
            s=sizes_direct,
            c='tab:blue',
            alpha=0.6,
            marker='o',
            edgecolors='none',
            label=direct_label,
            zorder=1,
        )
        handles.append(scatter_direct)
        labels.append(direct_label)

    if refl_x.size > 0:
        sizes_refl = normalize_marker_sizes(refl_w)
        scatter_refl = ax.scatter(
            refl_x,
            refl_y,
            s=sizes_refl,
            c='tab:orange',
            alpha=0.75,
            marker='^',
            edgecolors='none',
            label=refl_label,
            zorder=2,
        )
        handles.append(scatter_refl)
        labels.append(refl_label)

    cut_line = ax.axhline(
        delta_cut,
        color='dimgray',
        linestyle='--',
        linewidth=1,
        label=f'Δ cut ({delta_cut})',
        zorder=3,
    )
    handles.append(cut_line)
    labels.append(cut_line.get_label())

    handles, labels, _ = add_validation_snr_points(
        ax,
        validation_pairs,
        special_event_ids,
        handles=handles,
        labels=labels,
    )

    ax.legend(handles, labels, loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    ic(f"Saved combined SNR-delta plot to {output_path}")


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

    ic("Collecting direct-only delta plane metrics...")
    direct_plane_data = collect_delta_plane_metrics(
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

    ic("Collecting direct-reflected delta plane metrics...")
    refl_plane_data = collect_delta_plane_metrics(
        hra_events,
        weight_name=refl_weight,
        sigma=plot_sigma,
        sigma_52=sigma_52,
        scenario='direct_reflected',
        direct_exclusions=direct_exclusions_set,
        reflected_exclusions=reflected_exclusions_set,
    )

    direct_plane_cut, direct_plane_kept_weight, direct_plane_total_weight = apply_lower_left_cut(direct_plane_data)
    refl_plane_cut, refl_plane_kept_weight, refl_plane_total_weight = apply_lower_left_cut(refl_plane_data)

    _, direct_kept_weight, direct_total_weight = apply_delta_cut(direct_data, DELTA_CUT)
    _, refl_kept_weight, refl_total_weight = apply_delta_cut(refl_data, DELTA_CUT)

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

    if direct_plane_total_weight > 0:
        direct_plane_fraction = direct_plane_kept_weight / direct_plane_total_weight if direct_plane_total_weight else 0.0
        ic(
            f"Direct delta-plane pairs outside lower-left quadrant: {direct_plane_fraction * 100:.2f}% of weight "
            f"({direct_plane_kept_weight:.4e}/{direct_plane_total_weight:.4e})"
        )
    else:
        ic("Direct delta-plane pairs have zero total weight; skipping quadrant cut report.")

    if refl_plane_total_weight > 0:
        refl_plane_fraction = refl_plane_kept_weight / refl_plane_total_weight if refl_plane_total_weight else 0.0
        ic(
            f"Direct-reflected delta-plane pairs outside lower-left quadrant: {refl_plane_fraction * 100:.2f}% of weight "
            f"({refl_plane_kept_weight:.4e}/{refl_plane_total_weight:.4e})"
        )
    else:
        ic("Direct-reflected delta-plane pairs have zero total weight; skipping quadrant cut report.")

    direct_plane_label = 'Direct-only pairs'
    refl_plane_label = 'Direct-reflected pairs'
    direct_plane_cut_label = format_plane_cut_label(direct_plane_label, direct_plane_kept_weight, direct_plane_total_weight)
    refl_plane_cut_label = format_plane_cut_label(refl_plane_label, refl_plane_kept_weight, refl_plane_total_weight)

    direct_output = os.path.join(snr_plot_folder, 'snr_chi_diff_scatter_direct.png')
    direct_label = f'BL-only pairs, eff {direct_kept_weight/direct_total_weight:.2%}'
    if direct_data[0].size > 0:
        plot_single_scatter(
            direct_data,
            cmap='Blues',
            marker='o',
            legend_label=direct_label,
            title='Chi Difference Spread vs Average SNR (Direct Pairs)',
            output_path=direct_output,
            delta_cut=DELTA_CUT,
        )
        ic(f"Saved direct pair scatter plot to {direct_output}")
    else:
        ic("No direct pairs available for scatter plotting; skipping direct scatter plot.")

    refl_output = os.path.join(snr_plot_folder, 'snr_chi_diff_scatter_direct_reflected.png')
    refl_label = f'BL-RCR pairs, eff {refl_kept_weight/refl_total_weight:.2%}'   
    if refl_data[0].size > 0:
        plot_single_scatter(
            refl_data,
            cmap='Oranges',
            marker='^',
            legend_label=refl_label,
            title='Chi Difference Spread vs Average SNR (Direct-Reflected Pairs)',
            output_path=refl_output,
            delta_cut=DELTA_CUT,
        )
        ic(f"Saved direct-reflected pair scatter plot to {refl_output}")
    else:
        ic("No direct-reflected pairs available for scatter plotting; skipping reflected scatter plot.")

    direct_plane_output = os.path.join(snr_plot_folder, 'chi_delta_plane_direct.png')
    if direct_plane_data[0].size > 0:
        plot_delta_plane(
            direct_plane_data,
            cmap='Blues',
            marker='o',
            legend_label=direct_plane_label,
            title='Chi Delta Plane (Direct Pairs)',
            output_path=direct_plane_output,
        )
        ic(f"Saved direct pair delta plane plot to {direct_plane_output}")
    else:
        ic("No direct pairs available for delta plane plotting; skipping direct delta plane plot.")

    refl_plane_output = os.path.join(snr_plot_folder, 'chi_delta_plane_direct_reflected.png')
    if refl_plane_data[0].size > 0:
        plot_delta_plane(
            refl_plane_data,
            cmap='Oranges',
            marker='^',
            legend_label=refl_plane_label,
            title='Chi Delta Plane (Direct-Reflected Pairs)',
            output_path=refl_plane_output,
        )
        ic(f"Saved direct-reflected pair delta plane plot to {refl_plane_output}")
    else:
        ic("No direct-reflected pairs available for delta plane plotting; skipping reflected delta plane plot.")

    direct_plane_cut_output = os.path.join(snr_plot_folder, 'chi_delta_plane_direct_cut.png')
    if direct_plane_cut[0].size > 0:
        plot_delta_plane(
            direct_plane_cut,
            cmap='Blues',
            marker='o',
            legend_label=direct_plane_cut_label,
            title='Chi Delta Plane (Direct Pairs) — Quadrant Cut',
            output_path=direct_plane_cut_output,
        )
        ic(f"Saved direct pair delta plane cut plot to {direct_plane_cut_output}")
    else:
        ic("No direct pairs remain after quadrant cut; skipping direct delta plane cut plot.")

    refl_plane_cut_output = os.path.join(snr_plot_folder, 'chi_delta_plane_direct_reflected_cut.png')
    if refl_plane_cut[0].size > 0:
        plot_delta_plane(
            refl_plane_cut,
            cmap='Oranges',
            marker='^',
            legend_label=refl_plane_cut_label,
            title='Chi Delta Plane (Direct-Reflected Pairs) — Quadrant Cut',
            output_path=refl_plane_cut_output,
        )
        ic(f"Saved direct-reflected delta plane cut plot to {refl_plane_cut_output}")
    else:
        ic("No direct-reflected pairs remain after quadrant cut; skipping reflected delta plane cut plot.")

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
                DEFAULT_VALIDATION_SPECIAL_EVENT_IDS,
                validation_output,
                DELTA_CUT,
            )

            if direct_data[0].size > 0:
                overlay_direct_output = os.path.join(
                    snr_plot_folder,
                    'snr_chi_diff_scatter_direct_with_validation.png',
                )
                plot_simulation_with_validation(
                    direct_data,
                    legend_label=direct_label,
                    title='Chi Difference Spread vs Average SNR (Direct Pairs) — With Validation',
                    output_path=overlay_direct_output,
                    cmap='Blues',
                    marker='o',
                    delta_cut=DELTA_CUT,
                    validation_pairs=validation_pairs,
                    special_event_ids=DEFAULT_VALIDATION_SPECIAL_EVENT_IDS,
                )

            if refl_data[0].size > 0:
                overlay_refl_output = os.path.join(
                    snr_plot_folder,
                    'snr_chi_diff_scatter_direct_reflected_with_validation.png',
                )
                plot_simulation_with_validation(
                    refl_data,
                    legend_label=refl_label,
                    title='Chi Difference Spread vs Average SNR (Direct-Reflected Pairs) — With Validation',
                    output_path=overlay_refl_output,
                    cmap='Oranges',
                    marker='^',
                    delta_cut=DELTA_CUT,
                    validation_pairs=validation_pairs,
                    special_event_ids=DEFAULT_VALIDATION_SPECIAL_EVENT_IDS,
                )

            if direct_plane_data[0].size > 0:
                overlay_direct_plane_output = os.path.join(
                    snr_plot_folder,
                    'chi_delta_plane_direct_with_validation.png',
                )
                plot_delta_plane_with_validation(
                    direct_plane_data,
                    legend_label=direct_plane_label,
                    title='Chi Delta Plane (Direct Pairs) — With Validation',
                    output_path=overlay_direct_plane_output,
                    cmap='Blues',
                    marker='o',
                    validation_pairs=validation_pairs,
                    special_event_ids=DEFAULT_VALIDATION_SPECIAL_EVENT_IDS,
                )

            if refl_plane_data[0].size > 0:
                overlay_refl_plane_output = os.path.join(
                    snr_plot_folder,
                    'chi_delta_plane_direct_reflected_with_validation.png',
                )
                plot_delta_plane_with_validation(
                    refl_plane_data,
                    legend_label=refl_plane_label,
                    title='Chi Delta Plane (Direct-Reflected Pairs) — With Validation',
                    output_path=overlay_refl_plane_output,
                    cmap='Oranges',
                    marker='^',
                    validation_pairs=validation_pairs,
                    special_event_ids=DEFAULT_VALIDATION_SPECIAL_EVENT_IDS,
                )

            if direct_plane_cut[0].size > 0:
                overlay_direct_plane_cut_output = os.path.join(
                    snr_plot_folder,
                    'chi_delta_plane_direct_with_validation_cut.png',
                )
                plot_delta_plane_with_validation(
                    direct_plane_cut,
                    legend_label=direct_plane_cut_label,
                    title='Chi Delta Plane (Direct Pairs) — With Validation (Quadrant Cut)',
                    output_path=overlay_direct_plane_cut_output,
                    cmap='Blues',
                    marker='o',
                    validation_pairs=validation_pairs,
                    special_event_ids=DEFAULT_VALIDATION_SPECIAL_EVENT_IDS,
                    exclude_lower_left=True,
                )

            if refl_plane_cut[0].size > 0:
                overlay_refl_plane_cut_output = os.path.join(
                    snr_plot_folder,
                    'chi_delta_plane_direct_reflected_with_validation_cut.png',
                )
                plot_delta_plane_with_validation(
                    refl_plane_cut,
                    legend_label=refl_plane_cut_label,
                    title='Chi Delta Plane (Direct-Reflected Pairs) — With Validation (Quadrant Cut)',
                    output_path=overlay_refl_plane_cut_output,
                    cmap='Oranges',
                    marker='^',
                    validation_pairs=validation_pairs,
                    special_event_ids=DEFAULT_VALIDATION_SPECIAL_EVENT_IDS,
                    exclude_lower_left=True,
                )

            if direct_plane_data[0].size > 0 or refl_plane_data[0].size > 0:
                combined_plane_output = os.path.join(
                    snr_plot_folder,
                    'chi_delta_plane_combined_with_validation.png',
                )
                plot_combined_delta_plane_with_validation(
                    direct_plane_data,
                    direct_plane_label,
                    refl_plane_data,
                    refl_plane_label,
                    title='Chi Delta Plane — Combined Pairs with Validation',
                    output_path=combined_plane_output,
                    validation_pairs=validation_pairs,
                    special_event_ids=DEFAULT_VALIDATION_SPECIAL_EVENT_IDS,
                    exclude_lower_left=False,
                )

            if direct_plane_cut[0].size > 0 or refl_plane_cut[0].size > 0:
                combined_plane_cut_output = os.path.join(
                    snr_plot_folder,
                    'chi_delta_plane_combined_with_validation_cut.png',
                )
                plot_combined_delta_plane_with_validation(
                    direct_plane_cut,
                    direct_plane_cut_label,
                    refl_plane_cut,
                    refl_plane_cut_label,
                    title='Chi Delta Plane — Combined Pairs with Validation (Quadrant Cut)',
                    output_path=combined_plane_cut_output,
                    validation_pairs=validation_pairs,
                    special_event_ids=DEFAULT_VALIDATION_SPECIAL_EVENT_IDS,
                    exclude_lower_left=True,
                )

            if direct_data[0].size > 0 or refl_data[0].size > 0:
                combined_snr_output = os.path.join(
                    snr_plot_folder,
                    'snr_chi_diff_scatter_combined_with_validation.png',
                )
                plot_combined_snr_delta_with_validation(
                    direct_data,
                    direct_label,
                    refl_data,
                    refl_label,
                    combined_snr_output,
                    delta_cut=DELTA_CUT,
                    validation_pairs=validation_pairs,
                    special_event_ids=DEFAULT_VALIDATION_SPECIAL_EVENT_IDS,
                )

            special_summary = next(
                (rec for rec in validation_pairs if rec['event_id'] == DEFAULT_VALIDATION_SPECIAL_EVENT_IDS),
                None,
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
