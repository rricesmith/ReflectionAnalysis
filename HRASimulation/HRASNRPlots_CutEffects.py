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
from HRASimulation.S02_HRANurToNpy import loadHRAfromH5
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


def event_passes_angle_cut(event: HRAevent, station_ids, zenith_margin_deg=20.0, azimuth_margin_deg=45.0):
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

def collect_snr_chi_weighted(
    HRAeventList,
    weight_name,
    allowed_stations,
    sigma,
    bad_stations,
    predicate=lambda ev, sts: True,
):
    """Collect weighted (SNR, Chi) points for both ChiRCR and Chi2016 columns.

    Splits each event's weight equally across the total number of points produced
    across both columns so that the combined sum of subplot weights equals the
    total event rate for the selected events.
    """
    snr_rcr, chi_rcr, w_rcr = [], [], []
    snr_2016, chi_2016, w_2016 = [], [], []

    allowed = set(allowed_stations)

    for ev in HRAeventList:
        if not ev.hasWeight(weight_name, sigma=sigma):
            continue
        ev_w = ev.getWeight(weight_name, sigma=sigma)
        if ev_w <= 0:
            continue

        d_list, r_list, both = _get_trigger_station_lists(ev, sigma=sigma, bad_stations=bad_stations)
        if not both:
            continue
        stations = [st for st in both if st in allowed]
        if not stations:
            continue
        if not predicate(ev, stations):
            continue

        # Build per-event points
        rcr_points = []  # (snr, chi)
        c2016_points = []
        for st in stations:
            snr_val = ev.getSNR(st)
            if snr_val is None:
                continue
            chi_rcr_val, chi2016_val = _get_station_chi_values(ev, st)
            if chi_rcr_val is not None:
                rcr_points.append((snr_val, chi_rcr_val))
            if chi2016_val is not None:
                c2016_points.append((snr_val, chi2016_val))

        total_points = len(rcr_points) + len(c2016_points)
        if total_points == 0:
            continue

        split_w = ev_w / total_points
        for s, c in rcr_points:
            snr_rcr.append(s)
            chi_rcr.append(c)
            w_rcr.append(split_w)
        for s, c in c2016_points:
            snr_2016.append(s)
            chi_2016.append(c)
            w_2016.append(split_w)

    return (
        np.array(snr_rcr), np.array(chi_rcr), np.array(w_rcr),
        np.array(snr_2016), np.array(chi_2016), np.array(w_2016),
    )

def plot_1d_combined_rows(fig_title, save_path, bins, rows_data):
    """rows_data: list of 4 entries, each is tuple(direct_snrs, refl_snrs, direct_w, refl_w, row_title)"""
    fig, axs = plt.subplots(4, 2, figsize=(12, 18), sharey=True)
    # Baseline totals from the 'No Cuts' row (row 0)
    if rows_data:
        base_dm = rows_data[0][2] > 0
        base_rm = rows_data[0][3] > 0
        base_direct_total = float(np.sum(rows_data[0][2][base_dm])) if np.any(base_dm) else 0.0
        base_reflected_total = float(np.sum(rows_data[0][3][base_rm])) if np.any(base_rm) else 0.0
    else:
        base_direct_total = 0.0
        base_reflected_total = 0.0
    for i, (d_snr, r_snr, d_w, r_w, title) in enumerate(rows_data):
        ax_d = axs[i, 0]
        ax_r = axs[i, 1]

        dm = d_w > 0
        direct_total = float(np.sum(d_w[dm])) if np.any(dm) else 0.0
        if np.any(dm):
            ax_d.hist(d_snr[dm], bins=bins, weights=d_w[dm], histtype='step', linewidth=2)
        ax_d.set_xscale('log')
        ax_d.set_yscale('log')
        ax_d.set_ylabel('Weighted Counts (Evts/Yr)')
        ax_d.set_title(f'{title} — Direct')
        # Legend with total weight and efficiency (for cut rows)
        ax_d.plot([], [], ' ', label=f'{direct_total:.2f} Evts/Yr')
        if i > 0:
            if base_direct_total > 0:
                eff = 100.0 * direct_total / base_direct_total
                ax_d.plot([], [], ' ', label=f'{eff:.1f}% Eff')
            else:
                ax_d.plot([], [], ' ', label='N/A Eff')
        ax_d.legend(loc='upper right')

        rm = r_w > 0
        reflected_total = float(np.sum(r_w[rm])) if np.any(rm) else 0.0
        if np.any(rm):
            ax_r.hist(r_snr[rm], bins=bins, weights=r_w[rm], histtype='step', linewidth=2, color='C1')
        ax_r.set_xscale('log')
        ax_r.set_title(f'{title} — Reflected')
        # Legend with total weight and efficiency (for cut rows)
        ax_r.plot([], [], ' ', label=f'{reflected_total:.2f} Evts/Yr')
        if i > 0:
            if base_reflected_total > 0:
                eff = 100.0 * reflected_total / base_reflected_total
                ax_r.plot([], [], ' ', label=f'{eff:.1f}% Eff')
            else:
                ax_r.plot([], [], ' ', label='N/A Eff')
        ax_r.legend(loc='upper right')

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
    # Baseline from 'No Cuts'
    base_total = float(np.sum(rows_data[0][2])) if rows_data and len(rows_data[0][2]) > 0 else 0.0
    for i, (dx, rx, w, title) in enumerate(rows_data):
        ax = axs[i]
        total_w = float(np.sum(w)) if len(w) > 0 else 0.0
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
        # Legend with total weight and efficiency (for cut rows)
        ax.plot([], [], ' ', label=f'{total_w:.2f} Evts/Yr')
        if i > 0:
            if base_total > 0:
                eff = 100.0 * total_w / base_total
                ax.plot([], [], ' ', label=f'{eff:.1f}% Eff')
            else:
                ax.plot([], [], ' ', label='N/A Eff')
        ax.legend(loc='upper right')

    plt.suptitle(fig_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ic(f"Saving 2D combined plot: {save_path}")
    plt.savefig(save_path)
    plt.close(fig)


def plot_snr_vs_chi_combined_rows(fig_title, save_path, rows_data, xlim=(3, 100), ylim=(0, 1)):
    """rows_data: list of 4 entries, each is tuple(
        snr_rcr, chi_rcr, w_rcr, snr_2016, chi_2016, w_2016, row_title
    )
    Produces a 4x2 grid: columns (ChiRCR, Chi2016) across cuts rows.
    """
    fig, axs = plt.subplots(4, 2, figsize=(12, 18), sharex=True, sharey=True)
    # Baselines from 'No Cuts' row for each column
    base_rcr = float(np.sum(rows_data[0][2])) if rows_data and len(rows_data[0][2]) > 0 else 0.0
    base_2016 = float(np.sum(rows_data[0][5])) if rows_data and len(rows_data[0][5]) > 0 else 0.0

    for i, (sx_rcr, cx_rcr, wx_rcr, sx_16, cx_16, wx_16, title) in enumerate(rows_data):
        # Left column: RCR
        ax_l = axs[i, 0]
        if len(wx_rcr) > 0:
            ax_l.scatter(sx_rcr, cx_rcr, s=8, alpha=0.6)
        ax_l.set_xscale('log')
        ax_l.set_xlim(xlim)
        ax_l.set_ylim(ylim)
        ax_l.set_ylabel('Chi')
        if i == 0:
            ax_l.set_title('ChiRCR')
        # Legend with total weight and efficiency (for cut rows)
        tot_l = float(np.sum(wx_rcr)) if len(wx_rcr) > 0 else 0.0
        ax_l.plot([], [], ' ', label=f'{tot_l:.2f} Evts/Yr')
        if i > 0:
            if base_rcr > 0:
                eff = 100.0 * tot_l / base_rcr
                ax_l.plot([], [], ' ', label=f'{eff:.1f}% Eff')
            else:
                ax_l.plot([], [], ' ', label='N/A Eff')
        ax_l.legend(loc='upper right')

        # Right column: 2016
        ax_r = axs[i, 1]
        if len(wx_16) > 0:
            ax_r.scatter(sx_16, cx_16, s=8, alpha=0.6, color='C1')
        ax_r.set_xscale('log')
        ax_r.set_xlim(xlim)
        ax_r.set_ylim(ylim)
        if i == 0:
            ax_r.set_title('Chi2016')
        # Legend with total weight and efficiency (for cut rows)
        tot_r = float(np.sum(wx_16)) if len(wx_16) > 0 else 0.0
        ax_r.plot([], [], ' ', label=f'{tot_r:.2f} Evts/Yr')
        if i > 0:
            if base_2016 > 0:
                eff = 100.0 * tot_r / base_2016
                ax_r.plot([], [], ' ', label=f'{eff:.1f}% Eff')
            else:
                ax_r.plot([], [], ' ', label='N/A Eff')
        ax_r.legend(loc='upper right')

        # Row titles on left side
        axs[i, 0].text(0.02, 0.92, title, transform=axs[i, 0].transAxes,
                        fontsize=11, fontweight='bold', va='top', ha='left')

        if i == 3:
            ax_l.set_xlabel('SNR')
            ax_r.set_xlabel('SNR')

    plt.suptitle(fig_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ic(f"Saving SNR vs Chi combined plot: {save_path}")
    plt.savefig(save_path)
    plt.close(fig)


def collect_1d_chi_weighted(HRAeventList, weight_name, direct_stations, reflected_stations, sigma,
                             bad_stations, which: str, predicate=lambda ev, sts: True):
    """Collect weighted chi values for 1D histograms (direct/reflected split).

    which: 'rcr' or '2016'
    """
    direct_vals, refl_vals = [], []
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

        # Extract chi per station
        ev_direct = []
        for st in direct_stations:
            if st in d_list:
                rcr, c16 = _get_station_chi_values(ev, st)
                val = rcr if which == 'rcr' else c16
                if val is not None:
                    ev_direct.append(val)

        ev_refl = []
        for st in reflected_stations:
            if st in r_list:
                rcr, c16 = _get_station_chi_values(ev, st)
                val = rcr if which == 'rcr' else c16
                if val is not None:
                    ev_refl.append(val)

        total = len(ev_direct) + len(ev_refl)
        if total == 0:
            continue
        split_w = ev_w / total
        for v in ev_direct:
            direct_vals.append(v)
            direct_w.append(split_w)
            refl_vals.append(0)
            refl_w.append(0)
        for v in ev_refl:
            direct_vals.append(0)
            direct_w.append(0)
            refl_vals.append(v)
            refl_w.append(split_w)

    return np.array(direct_vals), np.array(refl_vals), np.array(direct_w), np.array(refl_w)


def plot_chi_hist_combined_rows(fig_title, save_path, bins, rows_data):
    """rows_data: list of 4 entries, each is tuple(direct_vals, refl_vals, direct_w, refl_w, row_title)
    Mirrors plot_1d_combined_rows but for Chi histograms.
    """
    fig, axs = plt.subplots(4, 2, figsize=(12, 18), sharey=True)
    # Baseline totals from 'No Cuts'
    if rows_data:
        base_dm = rows_data[0][2] > 0
        base_rm = rows_data[0][3] > 0
        base_direct_total = float(np.sum(rows_data[0][2][base_dm])) if np.any(base_dm) else 0.0
        base_reflected_total = float(np.sum(rows_data[0][3][base_rm])) if np.any(base_rm) else 0.0
    else:
        base_direct_total = 0.0
        base_reflected_total = 0.0

    for i, (d_vals, r_vals, d_w, r_w, title) in enumerate(rows_data):
        ax_d = axs[i, 0]
        ax_r = axs[i, 1]

        dm = d_w > 0
        direct_total = float(np.sum(d_w[dm])) if np.any(dm) else 0.0
        if np.any(dm):
            ax_d.hist(d_vals[dm], bins=bins, weights=d_w[dm], histtype='step', linewidth=2)
        ax_d.set_xlim(0, 1)
        ax_d.set_ylabel('Weighted Counts (Evts/Yr)')
        ax_d.set_title(f'{title} — Direct')
        ax_d.set_yscale('log')
        # Legend with total and efficiency
        ax_d.plot([], [], ' ', label=f'{direct_total:.2f} Evts/Yr')
        if i > 0:
            if base_direct_total > 0:
                eff = 100.0 * direct_total / base_direct_total
                ax_d.plot([], [], ' ', label=f'{eff:.1f}% Eff')
            else:
                ax_d.plot([], [], ' ', label='N/A Eff')
        ax_d.legend(loc='upper right')

        rm = r_w > 0
        reflected_total = float(np.sum(r_w[rm])) if np.any(rm) else 0.0
        if np.any(rm):
            ax_r.hist(r_vals[rm], bins=bins, weights=r_w[rm], histtype='step', linewidth=2, color='C1')
        ax_r.set_xlim(0, 1)
        ax_r.set_title(f'{title} — Reflected')
        # Legend with total and efficiency
        ax_r.plot([], [], ' ', label=f'{reflected_total:.2f} Evts/Yr')
        if i > 0:
            if base_reflected_total > 0:
                eff = 100.0 * reflected_total / base_reflected_total
                ax_r.plot([], [], ' ', label=f'{eff:.1f}% Eff')
            else:
                ax_r.plot([], [], ' ', label='N/A Eff')
        ax_r.legend(loc='upper right')

        if i == 3:
            ax_d.set_xlabel('Chi')
            ax_r.set_xlabel('Chi')

    plt.suptitle(fig_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ic(f"Saving Chi histogram plot: {save_path}")
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

            # Diagnostics: print total weights used per row (sum of direct+reflected weights)
            row_totals_1d = [(title, float(np.sum(d_w) + np.sum(r_w))) for (_, _, d_w, r_w, title) in rows_1d]
            ic(f"1D totals for {i}-Fold {scen_label}: {row_totals_1d}")

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

                # Diagnostics: print total weights per 2D row
                row_totals_2d = [(title, float(np.sum(w))) for (_, _, w, title) in rows_2d]
                ic(f"2D totals for {i}-Fold {scen_label}: {row_totals_2d}")

                fig_title_2d = f'2D SNR Histogram (2-Fold, {scen_label}) — Cut Effects'
                save_path_2d = os.path.join(scen_folder, f'snr_cuteffects_{i}coinc_{scen_label}_2d.png')
                plot_2d_combined_rows(fig_title_2d, save_path_2d, bins=log_bins, rows_data=rows_2d)

            # --- New: SNR vs Chi (4x2) per coincidence level ---
            # Allowed stations depend on scenario: both direct and reflected contribute points
            if scen_label == "noRefl":
                allowed = direct_stations + reflected_stations  # plotting includes all stations
            else:
                allowed = direct_stations + reflected_stations

            rows_snr_chi = []
            for pred, title in row_preds:
                sx_rcr, cx_rcr, wx_rcr, sx_16, cx_16, wx_16 = collect_snr_chi_weighted(
                    HRAeventList,
                    weight_name,
                    allowed_stations=allowed,
                    sigma=plot_sigma,
                    bad_stations=bad_stations,
                    predicate=pred,
                )
                rows_snr_chi.append((sx_rcr, cx_rcr, wx_rcr, sx_16, cx_16, wx_16, title))

            # Diagnostics: totals per row/column (RCR and 2016)
            diag = [(title, float(np.sum(wx_rcr)), float(np.sum(wx_16))) for (sx_rcr, cx_rcr, wx_rcr, sx_16, cx_16, wx_16, title) in rows_snr_chi]
            ic(f"SNR-vs-Chi totals (RCR, 2016) for {i}-Fold {scen_label}: {diag}")

            fig_title_sc = f'SNR vs Chi ({i}-Fold, {scen_label}) — Cut Effects'
            save_path_sc = os.path.join(scen_folder, f'snr_vs_chi_{i}coinc_{scen_label}.png')
            plot_snr_vs_chi_combined_rows(fig_title_sc, save_path_sc, rows_snr_chi, xlim=(3, 100), ylim=(0, 1))

            # --- New: Chi histograms (RCR and 2016) ---
            # RCR
            rows_chi_rcr = []
            for pred, title in row_preds:
                d_vals, r_vals, d_w, r_w = collect_1d_chi_weighted(
                    HRAeventList, weight_name, direct_stations, reflected_stations, plot_sigma,
                    bad_stations, which='rcr', predicate=pred
                )
                rows_chi_rcr.append((d_vals, r_vals, d_w, r_w, title))
            fig_title_chircr = f'ChiRCR Histograms ({i}-Fold, {scen_label}) — Cut Effects'
            save_path_chircr = os.path.join(scen_folder, f'chiRCR_hists_{i}coinc_{scen_label}.png')
            # Use uniform bins 0..1
            chi_bins = np.linspace(0, 1, 21)
            plot_chi_hist_combined_rows(fig_title_chircr, save_path_chircr, bins=chi_bins, rows_data=rows_chi_rcr)

            # 2016
            rows_chi_2016 = []
            for pred, title in row_preds:
                d_vals, r_vals, d_w, r_w = collect_1d_chi_weighted(
                    HRAeventList, weight_name, direct_stations, reflected_stations, plot_sigma,
                    bad_stations, which='2016', predicate=pred
                )
                rows_chi_2016.append((d_vals, r_vals, d_w, r_w, title))
            fig_title_chi2016 = f'Chi2016 Histograms ({i}-Fold, {scen_label}) — Cut Effects'
            save_path_chi2016 = os.path.join(scen_folder, f'chi2016_hists_{i}coinc_{scen_label}.png')
            plot_chi_hist_combined_rows(fig_title_chi2016, save_path_chi2016, bins=chi_bins, rows_data=rows_chi_2016)

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
