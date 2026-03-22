"""
S04c_singleStationAngularRecalc.py
===================================
Two purposes:
  1. Recalculate zenith/azimuth for single-station RCR-passing events where
     the angle values are missing (0.0 or NaN), using raw .nur data files and
     NuRadioReco's correlationDirectionFitter.
  2. Plot the azimuth-zenith distribution of all events (polar + 2D scatter).

Input:
    HRAStationDataAnalysis/ErrorAnalysis/output/{date_processing}/rcr_passing_events.npz
    (date_processing read from HRAStationDataAnalysis/config.ini)

Output:
    HRAStationDataAnalysis/ErrorAnalysis/output/{date_processing}/rcr_passing_events_with_angles.npz
    HRAStationDataAnalysis/ErrorAnalysis/plots/{date_processing}/rcr_single_station_plots/zen_azi_distribution.png
    HRAStationDataAnalysis/ErrorAnalysis/plots/{date_processing}/rcr_single_station_plots/zen_azi_distribution_2d.png

Usage:
    python -m HRAStationDataAnalysis.ErrorAnalysis.S04c_singleStationAngularRecalc
"""

import os
import sys
import configparser
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC / headless runs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from icecream import ic

# ---------------------------------------------------------------------------
# NuRadioReco imports
# ---------------------------------------------------------------------------
import NuRadioReco.modules.correlationDirectionFitter
from NuRadioReco.modules.io import NuRadioRecoio
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.detector import detector

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from HRAStationDataAnalysis.ErrorAnalysis.S03b_saveRCRPassingEvents import (
    load_rcr_events,
    CATEGORY_ALWAYS,
    CATEGORY_NOMINAL,
    CATEGORY_ADDITIONAL,
)
from HRAStationDataAnalysis.batchHRADataConversion import loadStationNurFiles


# ---------------------------------------------------------------------------
# Paths (relative to ReflectionAnalysis/ working directory)
# Dates are read from config.ini at runtime; these are computed in main().
# ---------------------------------------------------------------------------
DETECTOR_JSON   = 'HRASimulation/HRAStationLayoutForCoREAS.json'

# Per-station color palette (consistent with other thesis plots)
COLOR_MAP = {
    13: 'tab:blue',
    14: 'tab:orange',
    15: 'tab:green',
    17: 'tab:red',
    18: 'tab:purple',
    19: 'sienna',
    30: 'tab:brown',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def needs_recalc(azi_val, zen_val):
    """
    Return True if this event is missing a valid azimuth/zenith.

    Both-zero and any-NaN are treated as missing — the convention used
    throughout the RCR analysis pipeline.
    """
    is_zero = (azi_val == 0.0 and zen_val == 0.0)
    is_nan  = np.isnan(azi_val) or np.isnan(zen_val)
    return is_zero or is_nan


def group_events_by_station(events):
    """
    Build a per-station dict of indices that need recalculation.

    Returns:
        dict mapping station_id (int) -> list of indices into the flat arrays
        where recalculation is needed.
    """
    station_ids = events['station_ids']
    azi         = events['azi']
    zen         = events['zen']

    targets = {}
    for i in range(len(station_ids)):
        sid = int(station_ids[i])
        if needs_recalc(azi[i], zen[i]):
            targets.setdefault(sid, []).append(i)

    return targets


# ---------------------------------------------------------------------------
# Angular recalculation
# ---------------------------------------------------------------------------

def recalc_angles_for_station(station_id, target_indices, events,
                               azi_out, zen_out, det, fitter):
    """
    For a single station, scan its .nur files and fill in azimuth/zenith
    for every index in target_indices.

    Matching criterion: event_id must equal raw file event_id AND station
    Unix time must agree to within 1 second.

    Parameters
    ----------
    station_id      : int
    target_indices  : list of int — indices into events flat arrays
    events          : dict from load_rcr_events()
    azi_out         : mutable float array (N,) — updated in-place
    zen_out         : mutable float array (N,) — updated in-place
    det             : NuRadioReco Detector object (shared, updated per event)
    fitter          : correlationDirectionFitter (shared, already begun)
    """
    ic(f'Station {station_id}: {len(target_indices)} events need recalculation.')

    # Build lookup: (event_id, approx_time) -> index in flat array
    # Sorting by time lets us break early once we pass all targets (minor
    # optimization — .nur files are roughly time-ordered).
    target_event_ids  = events['event_ids'][target_indices]
    target_times      = events['times'][target_indices]

    # Map raw_event_id -> list of (target_time, flat_index)
    # One raw event_id could theoretically match multiple targets if they
    # share an ID across different days, so we store lists.
    lookup = {}
    for k, flat_idx in enumerate(target_indices):
        eid = int(target_event_ids[k])
        lookup.setdefault(eid, []).append((target_times[k], flat_idx))

    # Remaining targets set (flat indices); shrinks as we find matches
    remaining = set(target_indices)

    nur_files = loadStationNurFiles(station_id)
    ic(f'  Station {station_id}: {len(nur_files)} .nur files loaded.')

    if not nur_files:
        ic(f'  Station {station_id}: No .nur files found — leaving angles as NaN.')
        for idx in target_indices:
            azi_out[idx] = np.nan
            zen_out[idx] = np.nan
        return

    try:
        reader = NuRadioRecoio.NuRadioRecoio(nur_files)
    except Exception as e:
        ic(f'  Station {station_id}: Failed to open .nur reader: {e}')
        for idx in target_indices:
            azi_out[idx] = np.nan
            zen_out[idx] = np.nan
        return

    scanned = 0
    for raw_evt in reader.get_events():
        if not remaining:
            # All targets for this station have been matched — stop early
            break

        scanned += 1
        if scanned % 50_000 == 0:
            ic(f'  Station {station_id}: scanned {scanned} raw events, '
               f'{len(remaining)} targets remaining...')

        raw_eid  = raw_evt.get_id()

        # Fast pre-filter: skip if this raw event_id is not a target at all
        if raw_eid not in lookup:
            continue

        raw_stn = raw_evt.get_station(station_id)
        if raw_stn is None:
            continue

        raw_time_unix = raw_stn.get_station_time().unix
        raw_time_dt   = raw_stn.get_station_time()

        # Check each candidate with this event_id
        for (tgt_time, flat_idx) in list(lookup[raw_eid]):
            if abs(raw_time_unix - tgt_time) >= 1.0:
                continue  # Time doesn't match within tolerance

            if flat_idx not in remaining:
                continue  # Already filled from an earlier match

            ic(f'  Match: station {station_id}, raw_eid={raw_eid}, '
               f'flat_idx={flat_idx}. Running fitter...')

            # Update detector calibration to the event's timestamp;
            # fall back to a known-good date if the timestamp is outside
            # the detector's validity range.
            try:
                det.update(raw_time_dt)
            except LookupError:
                ic(f'    LookupError for time {raw_time_dt} — '
                   f'using fallback 2018-12-31.')
                det.update(datetime.datetime(2018, 12, 31,
                                             tzinfo=datetime.timezone.utc))

            try:
                fitter.run(raw_evt, raw_stn, det, n_index=1.35)
                new_azi = raw_stn.get_parameter(stnp.azimuth)
                new_zen = raw_stn.get_parameter(stnp.zenith)
            except Exception as e:
                ic(f'    Fitter error for flat_idx={flat_idx}: {e} — NaN.')
                new_azi = np.nan
                new_zen = np.nan

            azi_out[flat_idx] = new_azi
            zen_out[flat_idx] = new_zen
            remaining.discard(flat_idx)

            ic(f'    Updated flat_idx={flat_idx}: '
               f'azi={new_azi:.4f} rad, zen={new_zen:.4f} rad')

    ic(f'  Station {station_id}: done. Scanned {scanned} events. '
       f'{len(remaining)} targets unmatched.')

    # Anything still unmatched gets NaN so the caller can identify them
    for idx in remaining:
        ic(f'    Unmatched flat_idx={idx}, event_id={int(events["event_ids"][idx])}, '
           f'time={events["times"][idx]:.1f}. Setting NaN.')
        azi_out[idx] = np.nan
        zen_out[idx] = np.nan


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_polar(azi, zen, station_ids, save_path,
               categories_chircr=None, categories_chidiff=None, categories_combined=None,
               in_chircr=None, in_chidiff=None):
    """
    Polar (compass-style) azimuth-zenith scatter plot.

    Saves three separate files (one per categorization system), derived from
    save_path by inserting _chircr, _chidiff, _combined before the extension.

    Axes: North up, clockwise convention (standard for arrival-direction plots).
    Radial axis: zenith in degrees (0 at centre = vertical).
    Angular axis: azimuth in radians (NuRadioReco convention).
    """
    def _render_and_save(cat_array, out_path, subtitle, membership=None):
        fig = plt.figure(figsize=(8, 8))
        ax  = fig.add_subplot(111, polar=True)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(22.5)
        ax.set_rlim(0, 90)
        ax.set_rticks([0, 30, 60, 90])
        ax.grid(True, linestyle='--', alpha=0.5)

        present_stations = sorted(set(int(s) for s in station_ids))
        _CAT_MARKERS = {CATEGORY_ALWAYS: 'o', CATEGORY_NOMINAL: '^', CATEGORY_ADDITIONAL: '*'}
        _CAT_SIZES   = {CATEGORY_ALWAYS: 60,  CATEGORY_NOMINAL: 70,  CATEGORY_ADDITIONAL: 120}

        for i in range(len(station_ids)):
            # Skip if this event doesn't belong in this system's window
            if membership is not None and not bool(membership[i]):
                continue
            a = azi[i]; z = zen[i]
            if not (np.isfinite(a) and np.isfinite(z)):
                continue
            if a == 0.0 and z == 0.0:
                continue
            color = COLOR_MAP.get(int(station_ids[i]), 'gray')
            cat = int(cat_array[i]) if cat_array is not None and len(cat_array) > i else CATEGORY_NOMINAL
            ax.scatter(a, np.degrees(z), c=color,
                       marker=_CAT_MARKERS.get(cat, 'o'),
                       s=_CAT_SIZES.get(cat, 60), alpha=0.9, zorder=3)

        handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=COLOR_MAP.get(sid, 'gray'),
                          markersize=10, label=f'St {sid}')
                   for sid in present_stations]
        ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.3, 1.1), title='Station')

        cat_handles = [
            Line2D([0], [0], marker='o', color='k', linestyle='None',
                   markerfacecolor='gray', markersize=8, label='Always (tighter cuts)'),
            Line2D([0], [0], marker='^', color='k', linestyle='None',
                   markerfacecolor='gray', markersize=8, label='Nominal (could fail)'),
            Line2D([0], [0], marker='*', color='k', linestyle='None',
                   markerfacecolor='gray', markersize=10, label='Additional (looser cuts)'),
        ]
        ax.legend(handles=cat_handles, loc='lower right', title='Category', fontsize=9)
        ax.set_title(f'Single-Station RCR Events: Azimuth-Zenith ({subtitle})', pad=20)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        ic(f'Saved polar plot: {out_path}')

    base, ext = os.path.splitext(save_path)
    _render_and_save(categories_chircr,   f"{base}_chircr{ext}",   "chi-RCR variation",  membership=in_chircr)
    _render_and_save(categories_chidiff,  f"{base}_chidiff{ext}",  "chi-diff variation", membership=in_chidiff)
    _render_and_save(categories_combined, f"{base}_combined{ext}", "combined")


def plot_2d(azi, zen, station_ids, save_path,
            categories_chircr=None, categories_chidiff=None, categories_combined=None,
            in_chircr=None, in_chidiff=None):
    """
    Flat 2D scatter: azimuth (degrees, x-axis) vs zenith (degrees, y-axis).

    Saves three separate files (one per categorization system), derived from
    save_path by inserting _chircr, _chidiff, _combined before the extension.

    Provides an easier read for precise angle values than the polar plot.
    """
    def _render_and_save_2d(cat_array, out_path, subtitle, membership=None):
        fig, ax = plt.subplots(figsize=(10, 7))

        present_stations = sorted(set(int(s) for s in station_ids))
        station_ids_int = np.array([int(s) for s in station_ids])

        for sid in present_stations:
            sid_mask = station_ids_int == sid
            a = np.degrees(azi[sid_mask])
            z = np.degrees(zen[sid_mask])
            # Apply membership filter for this system's window
            mem_for_sid = (membership[sid_mask].astype(bool)
                           if membership is not None
                           else np.ones(np.sum(sid_mask), dtype=bool))
            valid = np.isfinite(a) & np.isfinite(z) & ~((a == 0.0) & (z == 0.0)) & mem_for_sid
            color = COLOR_MAP.get(sid, 'gray')
            cats_for_sid = (cat_array[sid_mask] if cat_array is not None
                            else np.full(np.sum(sid_mask), CATEGORY_NOMINAL))
            # Plot each category separately to vary marker
            first_for_station = True
            for cat_val, marker, msize in [
                (CATEGORY_ALWAYS, 'o', 60),
                (CATEGORY_NOMINAL, '^', 70),
                (CATEGORY_ADDITIONAL, '*', 120),
            ]:
                cat_mask = valid & (cats_for_sid == cat_val)
                if not np.any(cat_mask):
                    continue
                label = f'St {sid}' if first_for_station else '_nolegend_'
                ax.scatter(a[cat_mask], z[cat_mask],
                           c=color, marker=marker, s=msize, alpha=0.85,
                           label=label, zorder=3)
                first_for_station = False

        ax.set_xlabel('Azimuth (degrees)', fontsize=13)
        ax.set_ylabel('Zenith (degrees)', fontsize=13)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 90)
        ax.set_xticks(range(0, 361, 45))
        ax.set_yticks(range(0, 91, 15))
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(title='Station', loc='upper right')

        # Category marker legend
        cat_handles_2d = [
            Line2D([0], [0], marker='o', color='k', linestyle='None',
                   markerfacecolor='gray', markersize=8, label='Always (tighter cuts)'),
            Line2D([0], [0], marker='^', color='k', linestyle='None',
                   markerfacecolor='gray', markersize=8, label='Nominal (could fail)'),
            Line2D([0], [0], marker='*', color='k', linestyle='None',
                   markerfacecolor='gray', markersize=10, label='Additional (looser cuts)'),
        ]
        ax.legend(handles=cat_handles_2d, title='Category', loc='lower right', fontsize=9)
        ax.set_title(f'Single-Station RCR Events: Azimuth-Zenith Distribution (2D) ({subtitle})',
                     fontsize=14)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        ic(f'Saved 2D scatter plot: {out_path}')

    base, ext = os.path.splitext(save_path)
    _render_and_save_2d(categories_chircr,   f"{base}_chircr{ext}",   "chi-RCR variation",  membership=in_chircr)
    _render_and_save_2d(categories_chidiff,  f"{base}_chidiff{ext}",  "chi-diff variation", membership=in_chidiff)
    _render_and_save_2d(categories_combined, f"{base}_combined{ext}", "combined")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ic.configureOutput(prefix='S04c | ')

    # -----------------------------------------------------------------------
    # 0. Read config for date_processing
    # -----------------------------------------------------------------------
    config = configparser.ConfigParser()
    config_path = os.path.join('HRAStationDataAnalysis', 'config.ini')
    if not os.path.exists(config_path):
        config_path = 'config.ini'
    config.read(config_path)
    date_processing = config['PARAMETERS']['date_processing']

    input_npz  = os.path.join(
        'HRAStationDataAnalysis', 'ErrorAnalysis', 'output',
        date_processing, 'rcr_passing_events.npz'
    )
    output_npz = os.path.join(
        'HRAStationDataAnalysis', 'ErrorAnalysis', 'output',
        date_processing, 'rcr_passing_events_with_angles.npz'
    )
    plots_dir  = os.path.join(
        'HRAStationDataAnalysis', 'ErrorAnalysis', 'plots',
        date_processing, 'rcr_single_station_plots'
    )

    ic(f'date_processing={date_processing}')

    # -----------------------------------------------------------------------
    # 1. Load events
    # -----------------------------------------------------------------------
    ic(f'Loading events from: {input_npz}')
    events = load_rcr_events(input_npz)
    N = len(events['station_ids'])
    ic(f'Loaded {N} events.')

    # Work on mutable copies of the angle arrays so we don't touch the
    # original dict values if something goes wrong mid-run.
    azi_out = events['azi'].copy().astype(float)
    zen_out = events['zen'].copy().astype(float)

    # -----------------------------------------------------------------------
    # 2. Identify events needing recalculation, grouped by station
    # -----------------------------------------------------------------------
    station_targets = group_events_by_station(events)

    n_missing = sum(len(v) for v in station_targets.values())
    ic(f'Events needing angle recalculation: {n_missing} / {N}')

    if n_missing == 0:
        ic('All events already have valid angles. Skipping .nur scan.')
    else:
        # -------------------------------------------------------------------
        # 3. Initialise detector and fitter (shared across stations)
        # -------------------------------------------------------------------
        ic(f'Initialising detector from: {DETECTOR_JSON}')
        det = detector.Detector(DETECTOR_JSON)

        fitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
        fitter.begin(debug=False)
        ic('Detector and correlationDirectionFitter initialised.')

        # -------------------------------------------------------------------
        # 4. Per-station recalculation loop
        # -------------------------------------------------------------------
        for sid, idxs in sorted(station_targets.items()):
            recalc_angles_for_station(
                station_id=sid,
                target_indices=idxs,
                events=events,
                azi_out=azi_out,
                zen_out=zen_out,
                det=det,
                fitter=fitter,
            )

    # -----------------------------------------------------------------------
    # 5. Save updated .npz (all original arrays, updated azi/zen)
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)

    save_dict = {key: events[key] for key in events}
    save_dict['azi'] = azi_out
    save_dict['zen'] = zen_out

    np.savez_compressed(output_npz, **save_dict)
    ic(f'Saved updated events to: {output_npz}')

    # Quick summary of how many now have valid angles
    valid_mask = np.isfinite(azi_out) & np.isfinite(zen_out) \
                 & ~((azi_out == 0.0) & (zen_out == 0.0))
    ic(f'Events with valid angles after update: {valid_mask.sum()} / {N}')

    # -----------------------------------------------------------------------
    # 6. Plots
    # -----------------------------------------------------------------------
    polar_path = os.path.join(plots_dir, 'zen_azi_distribution.png')
    scatter_path = os.path.join(plots_dir, 'zen_azi_distribution_2d.png')

    cat_chircr   = events.get('category_chircr',   None)
    cat_chidiff  = events.get('category_chidiff',  None)
    cat_combined = events.get('category_combined', None)
    in_chircr    = events.get('in_chircr',         None)
    in_chidiff   = events.get('in_chidiff',        None)
    plot_polar(azi_out, zen_out, events['station_ids'], polar_path,
               categories_chircr=cat_chircr, categories_chidiff=cat_chidiff,
               categories_combined=cat_combined,
               in_chircr=in_chircr, in_chidiff=in_chidiff)
    plot_2d(azi_out, zen_out, events['station_ids'], scatter_path,
            categories_chircr=cat_chircr, categories_chidiff=cat_chidiff,
            categories_combined=cat_combined,
            in_chircr=in_chircr, in_chidiff=in_chidiff)

    ic('Done.')


if __name__ == '__main__':
    main()
