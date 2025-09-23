import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import configparser
from icecream import ic
import h5py
import pickle
import itertools

# Project imports
from HRASimulation.HRAEventObject import HRAevent, stns_100s, stns_200s
from HRASimulation.HRANurToNpy import loadHRAfromH5
import HRASimulation.HRAAnalysis as HRAAnalysis
from NuRadioReco.utilities import units


# ----------------------------
# Cut implementations
# ----------------------------

def _safe_deg(val):
    try:
        return np.degrees(float(val))
    except Exception:
        return None


def _wrap_delta_az_deg(az1_deg, az2_deg):
    d = abs(az1_deg - az2_deg) % 360.0
    return min(d, 360.0 - d)


def event_passes_angle_cut(event: HRAevent, station_ids, zenith_margin_deg=10.0, azimuth_margin_deg=20.0):
    """Angle cut using reconstructed per-station angles from a simulated event.

    - Uses event.recon_zenith[st] and event.recon_azimuth[st] (assumed radians).
    - Skips (passes) if fewer than 2 stations have valid angles.
    - Passes if any station pair agrees within the provided margins.
    """
    angles = []
    for st in station_ids:
        z = event.recon_zenith.get(st)
        a = event.recon_azimuth.get(st)
        if z is None or a is None:
            continue
        z_deg = _safe_deg(z)
        # azimuth stored in radians already; convert to degrees
        try:
            a_deg = np.degrees(float(a))
        except Exception:
            a_deg = None
        if z_deg is None or a_deg is None:
            continue
        angles.append((z_deg, a_deg))

    if len(angles) < 2:
        # Skip the cut if not enough info
        return True

    for (z1, a1), (z2, a2) in itertools.combinations(angles, 2):
        dz = abs(z1 - z2)
        da = _wrap_delta_az_deg(a1, a2)
        if (dz <= zenith_margin_deg) and (da <= azimuth_margin_deg):
            return True
    return False


def _get_station_chi_values(event: HRAevent, station_id):
    """Try multiple key names to fetch ChiRCR and Chi2016 values for robustness.
    Returns (chi_rcr, chi_2016) where missing values are returned as None.
    """
    # RCR key choice based on station group
    rcr_keys = []
    if station_id in stns_100s:
        rcr_keys.extend(["ChiRCR100s", "ChiRCR", "RCR"])  # try specific then fallbacks
    elif station_id in stns_200s:
        rcr_keys.extend(["ChiRCR200s", "ChiRCR", "RCR"])  # try specific then fallbacks
    else:
        rcr_keys.extend(["ChiRCR", "RCR"])  # generic

    chi_rcr = None
    for k in rcr_keys:
        v = event.getChi(station_id, k)
        if v is not None and not (isinstance(v, (int, float)) and v == 0):
            try:
                chi_rcr = float(v)
                break
            except Exception:
                pass

    chi2016 = None
    for k in ("Chi2016", "2016"):
        v = event.getChi(station_id, k)
        if v is not None and not (isinstance(v, (int, float)) and v == 0):
            try:
                chi2016 = float(v)
                break
            except Exception:
                pass

    return chi_rcr, chi2016


def event_passes_chi_cut(event: HRAevent, station_ids, high_chi_threshold=0.6, low_chi_threshold=0.5, min_triggers_passing=2):
    """Chi cut approximating C03 logic.

    - For each station in station_ids, consider max(ChiRCR*, Chi2016).
    - high pass: any station >= high_chi_threshold
    - low pass count: number of stations >= low_chi_threshold
    - final pass: high_pass and low_pass_count >= (min_triggers_passing - 1)
    """
    any_high = False
    low_count = 0

    for st in station_ids:
        chi_rcr, chi2016 = _get_station_chi_values(event, st)
        vals = [v for v in (chi_rcr, chi2016) if v is not None]
        if not vals:
            continue
        m = max(vals)
        if m >= high_chi_threshold:
            any_high = True
        if m >= low_chi_threshold:
            low_count += 1

    return bool(any_high and (low_count >= (min_triggers_passing - 1)))


# ----------------------------
# Data extraction helpers (weighted)
# ----------------------------

def _get_trigger_station_lists(event: HRAevent, sigma=4.5, bad_stations=None):
    if bad_stations is None:
        bad_stations = []
    direct = [st for st in event.directTriggers(sigma=sigma) if st not in bad_stations]
    reflected = [st for st in event.reflectedTriggers(sigma=sigma) if st not in bad_stations]
    both = direct + reflected
    return direct, reflected, both


def collect_1d_snr_weighted(HRAeventList, weight_name, direct_stations, reflected_stations, sigma, 
                            bad_stations, predicate=lambda ev, sts: True):
    """Return arrays for direct/reflected SNRs and their weights filtered by predicate.

    predicate(ev, station_ids) determines if the event contributes (event-level).
    """
    direct_snrs, refl_snrs = [], []
    direct_w, refl_w = [], []

    for ev in HRAeventList:
        if not ev.hasWeight(weight_name, sigma=sigma):
            continue
        ev_w = ev.getWeight(weight_name, sigma=sigma)
        if ev_w <= 0:
            continue

        d_list, r_list, both = _get_trigger_station_lists(ev, sigma=sigma, bad_stations=bad_stations)
        if not both:
            continue

        if not predicate(ev, both):
            continue

        # Build SNR lists present for configured direct/reflected station IDs
        ev_direct_snrs = [ev.getSNR(st) for st in direct_stations if st in d_list and ev.getSNR(st) is not None]
        ev_reflected_snrs = [ev.getSNR(st) for st in reflected_stations if st in r_list and ev.getSNR(st) is not None]

        total_triggers = len(ev_direct_snrs) + len(ev_reflected_snrs)
        if total_triggers == 0:
            continue

        split_w = ev_w / total_triggers
        for s in ev_direct_snrs:
            direct_snrs.append(s)
            direct_w.append(split_w)
            refl_snrs.append(0)
            refl_w.append(0)
        for s in ev_reflected_snrs:
            direct_snrs.append(0)
            direct_w.append(0)
            refl_snrs.append(s)
            refl_w.append(split_w)

    return (np.array(direct_snrs), np.array(refl_snrs), np.array(direct_w), np.array(refl_w))


def collect_2d_pairs_weighted(HRAeventList, weight_name, direct_stations, reflected_stations, sigma,
                              bad_stations, predicate=lambda ev, sts: True):
    d_vals, r_vals, w_vals = [], [], []
    for ev in HRAeventList:
        if not ev.hasWeight(weight_name, sigma=sigma):
            continue
        ev_w = ev.getWeight(weight_name, sigma=sigma)
        if ev_w <= 0:
            continue

        d_list, r_list, both = _get_trigger_station_lists(ev, sigma=sigma, bad_stations=bad_stations)
        if not both:
            continue
        if not predicate(ev, both):
            continue

        ev_d = [ev.getSNR(st) for st in direct_stations if st in d_list and ev.getSNR(st) is not None]
        ev_r = [ev.getSNR(st) for st in reflected_stations if st in r_list and ev.getSNR(st) is not None]
        if not ev_d or not ev_r:
            continue

        num_pairs = len(ev_d) * len(ev_r)
        if num_pairs == 0:
            continue
        split_w = ev_w / num_pairs
        for ds, rs in itertools.product(ev_d, ev_r):
            d_vals.append(ds)
            r_vals.append(rs)
            w_vals.append(split_w)

    return np.array(d_vals), np.array(r_vals), np.array(w_vals)


# ----------------------------
# Plotting helpers
# ----------------------------

def plot_1d_combined_rows(fig_title, save_path, bins, rows_data):
    """rows_data: list of 4 entries, each is tuple(direct_snrs, refl_snrs, direct_w, refl_w, row_title)"""
    fig, axs = plt.subplots(4, 2, figsize=(12, 18), sharey=True)
    for i, (d_snr, r_snr, d_w, r_w, title) in enumerate(rows_data):
        ax_d = axs[i, 0]
        ax_r = axs[i, 1]

        dm = d_w > 0
        if np.any(dm):
            ax_d.hist(d_snr[dm], bins=bins, weights=d_w[dm], histtype='step', linewidth=2)
        ax_d.set_xscale('log')
        ax_d.set_yscale('log')
        ax_d.set_ylabel('Weighted Counts (Evts/Yr)')
        ax_d.set_title(f'{title} — Direct')

        rm = r_w > 0
        if np.any(rm):
            ax_r.hist(r_snr[rm], bins=bins, weights=r_w[rm], histtype='step', linewidth=2, color='C1')
        ax_r.set_xscale('log')
        ax_r.set_title(f'{title} — Reflected')

        if i == 3:
            ax_d.set_xlabel('SNR')
            ax_r.set_xlabel('SNR')

    plt.suptitle(fig_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ic(f"Saving 1D combined plot: {save_path}")
    plt.savefig(save_path)
    plt.close(fig)


def plot_2d_combined_rows(fig_title, save_path, bins, rows_data):
    """rows_data: list of 4 entries, each is tuple(direct_snrs, reflected_snrs, weights, row_title)"""
    fig, axs = plt.subplots(4, 1, figsize=(8, 18))
    for i, (dx, rx, w, title) in enumerate(rows_data):
        ax = axs[i]
        if len(w) > 0:
            h, xedges, yedges, im = ax.hist2d(
                dx, rx, bins=bins, weights=w, norm=colors.LogNorm(), cmin=1e-5
            )
            plt.colorbar(im, ax=ax, label='Weighted Counts (Evts/Yr)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('SNR (Direct)')
        ax.set_ylabel('SNR (Reflected)')
        ax.set_title(title)

    plt.suptitle(fig_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ic(f"Saving 2D combined plot: {save_path}")
    plt.savefig(save_path)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('HRASimulation/config.ini')
    numpy_folder = config['FOLDERS']['numpy_folder']
    save_folder = config['FOLDERS']['save_folder']
    diameter = config['SIMPARAMETERS']['diameter']
    max_distance = float(diameter) / 2 * units.km
    plot_sigma = float(config['PLOTPARAMETERS']['trigger_sigma'])

    out_folder = os.path.join(save_folder, 'SNR_Plots_CutEffects')
    os.makedirs(out_folder, exist_ok=True)

    ic("Loading HRA event list...")
    HRAeventList_path = f'{numpy_folder}HRAeventList.h5'
    HRAeventList = loadHRAfromH5(HRAeventList_path)

    weights_were_added = False
    direct_stations = [13, 14, 15, 17, 18, 19, 30]
    reflected_stations = [113, 114, 115, 117, 118, 119, 130]
    bad_stations = [32, 52, 132, 152]

    log_bins = np.logspace(np.log10(3), np.log10(100), 21)

    # Scenarios: (label, weight_name_builder, trigger_rates_args)
    scenarios = [
        ("noRefl", lambda i: f"{i}_coincidence", dict(force_stations=None)),
        ("reflReq", lambda i: f"{i}_coincidence_reflReq", dict(force_stations=reflected_stations)),
    ]

    for scen_label, weight_name_fn, tr_kwargs in scenarios:
        ic(f"Processing scenario: {scen_label}")
        scen_folder = os.path.join(out_folder, scen_label)
        os.makedirs(scen_folder, exist_ok=True)

        for i in range(2, 8):
            weight_name = weight_name_fn(i)

            # Ensure weights exist
            if not HRAeventList[0].hasWeight(weight_name, sigma=plot_sigma):
                ic(f"Weight '{weight_name}' not found. Calculating now...")
                weights_were_added = True
                trigger_rate_coincidence = HRAAnalysis.getCoincidencesTriggerRates(
                    HRAeventList, bad_stations, sigma=plot_sigma, **tr_kwargs
                )
                if i in trigger_rate_coincidence and np.any(trigger_rate_coincidence[i] > 0):
                    HRAAnalysis.setNewTrigger(HRAeventList, weight_name, bad_stations=bad_stations, sigma=plot_sigma)
                    HRAAnalysis.setHRAeventListRateWeight(
                        HRAeventList, trigger_rate_coincidence[i], weight_name=weight_name,
                        max_distance=max_distance, sigma=plot_sigma
                    )
                    ic(f"Successfully calculated and added weights for '{weight_name}'.")
                else:
                    ic(f"No events found for {i}-fold coincidence in scenario {scen_label}. Skipping.")
                    continue

            # Predicates for the four rows
            def pred_none(ev, sts):
                return True

            def pred_angle(ev, sts):
                return event_passes_angle_cut(ev, sts)

            def pred_chi(ev, sts):
                return event_passes_chi_cut(ev, sts)

            def pred_both(ev, sts):
                return pred_angle(ev, sts) and pred_chi(ev, sts)

            row_preds = [
                (pred_none, 'No Cuts'),
                (pred_angle, 'Angle Cut'),
                (pred_chi, 'Chi Cut'),
                (pred_both, 'Angle + Chi'),
            ]

            # 1D rows
            rows_1d = []
            for pred, title in row_preds:
                d_snr, r_snr, d_w, r_w = collect_1d_snr_weighted(
                    HRAeventList, weight_name, direct_stations, reflected_stations,
                    sigma=plot_sigma, bad_stations=bad_stations, predicate=pred
                )
                rows_1d.append((d_snr, r_snr, d_w, r_w, title))

            fig_title_1d = f'SNR Distribution ({i}-Fold, {scen_label}) — Cut Effects'
            save_path_1d = os.path.join(scen_folder, f'snr_cuteffects_{i}coinc_{scen_label}_1d.png')
            plot_1d_combined_rows(fig_title_1d, save_path_1d, bins=log_bins, rows_data=rows_1d)

            # 2D (only for i == 2)
            if i == 2:
                rows_2d = []
                for pred, title in row_preds:
                    dx, rx, w = collect_2d_pairs_weighted(
                        HRAeventList, weight_name, direct_stations, reflected_stations,
                        sigma=plot_sigma, bad_stations=bad_stations, predicate=pred
                    )
                    rows_2d.append((dx, rx, w, title))

                fig_title_2d = f'2D SNR Histogram (2-Fold, {scen_label}) — Cut Effects'
                save_path_2d = os.path.join(scen_folder, f'snr_cuteffects_{i}coinc_{scen_label}_2d.png')
                plot_2d_combined_rows(fig_title_2d, save_path_2d, bins=log_bins, rows_data=rows_2d)

    if weights_were_added:
        ic("New weights were added, resaving HRAeventList to H5 file...")
        with h5py.File(HRAeventList_path, 'w') as hf:
            for i, obj in enumerate(HRAeventList):
                if not isinstance(obj, (np.ndarray, str, int, float)):
                    obj_bytes = pickle.dumps(obj)
                    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                    dset = hf.create_dataset(f'object_{i}', (1,), dtype=dt)
                    dset[0] = np.frombuffer(obj_bytes, dtype='uint8')
                else:
                    hf.create_dataset(f'object_{i}', data=obj)
        ic("HRAeventList successfully updated and saved.")

    ic("\nAll SNR cut-effects plots have been generated!")
