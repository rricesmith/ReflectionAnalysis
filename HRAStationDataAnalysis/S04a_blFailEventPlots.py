"""
S04a_blFailEventPlots.py
------------------------
Plots coincidence events that pass all standard analysis cuts (chi, angle,
time, FFT) but fail the between-station self-match chi cut.

A low between-station self-match chi indicates the waveforms are NOT
correlated across stations, which is the expected signature of backlobe events
(as opposed to RCR events where all stations see similar waveforms). This
script therefore naturally identifies BL-like events without requiring a
hardcoded ID list.

Selection logic:
  1. PASS  check_chi_cut(event_details)          — per-station chi OK
  2. PASS  angle cut                              — always True (disabled)
  3. PASS  check_time_cut(events_dict)            — time isolation OK
  4. PASS  check_fft_cut(event_details, event_id) — FFT spectral check OK
  5. FAIL  between-station self-match chi cut:
           best_match_abs_chi < CHI_STATION_THRESHOLD

The between-station chi (best_match_abs_chi) is computed by
compute_event_self_similarity():
  - Template = the loudest trace across all stations in the event.
  - For every other station, cross-correlate its waveforms against the
    template and keep the maximum absolute cross-correlation coefficient.
  - High best_match_abs_chi (near 1) -> stations agree -> RCR-like.
  - Low best_match_abs_chi           -> stations disagree -> BL-like / noise
    coincidence.

The metric is stored at:
  event_details["derived_metrics"]["self_match"]["best_match_abs_chi"]

Run from: ReflectionAnalysis/
"""

import os
import sys
import configparser
import numpy as np
from icecream import ic

# ---------------------------------------------------------------------------
# Allow imports from both the repo root and the HRAStationDataAnalysis subdir
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath('HRAStationDataAnalysis'))

from C03b_thesisCoincidencePlotting import (
    plot_single_master_event_thesis,
    _load_pickle,
    check_chi_cut,
    check_fft_cut,
    check_time_cut,
    SectionTimer,
    compute_event_self_similarity,
)


# ---------------------------------------------------------------------------
# Threshold for the between-station self-match chi cut.
# Events with best_match_abs_chi below this value fail the cut (BL-like).
# Matches the high_chi_threshold used in check_chi_cut.
# ---------------------------------------------------------------------------
CHI_STATION_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_data_file(date_of_data, date_of_coincidence):
    """
    Return the first existing candidate pickle path (highest priority first).
    Tries the DFS absolute path first, then local processed-data paths.
    """
    dfs_root = (
        "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data"
    )
    local_root = os.path.join(
        "HRAStationDataAnalysis", "StationData", "processedNumpyData", date_of_data
    )

    basename_variants = [
        f"{date_of_coincidence}_CoincidenceDatetimes_passing_cuts_with_all_params_recalcZenAzi_calcPol.pkl",
        f"{date_of_coincidence}_CoincidenceDatetimes_passing_cuts_with_all_params_recalcZenAzi.pkl",
        f"{date_of_coincidence}_CoincidenceDatetimes_passing_cuts_with_all_params.pkl",
    ]

    candidates = (
        [os.path.join(dfs_root, bn) for bn in basename_variants] +
        [os.path.join(local_root, bn) for bn in basename_variants]
    )

    ic("Searching for data file in order:")
    for c in candidates:
        ic(f"  {c}")
        if os.path.exists(c):
            return c

    return None


def _apply_cuts(events_data_dict, is_passing_cuts_dataset):
    """
    Apply the same cut chain used in C03b.

    For a 'passing_cuts' pickle (already pre-filtered by chi/angle):
      - Trust existing chi/angle flags, re-apply time + FFT.

    For a raw pickle:
      - Compute chi cut, skip angle cut (disabled in C03b), then time, then
        FFT.

    Mutates each event dict in-place, adding 'cut_results' and
    'passes_analysis_cuts'.
    """
    if is_passing_cuts_dataset:
        with SectionTimer("Apply time cut (passing_cuts dataset)"):
            time_cut_results = check_time_cut(events_data_dict, time_threshold_hours=24.0)

        num_pass = num_fail = 0
        for event_id, event_details in events_data_dict.items():
            if not isinstance(event_details, dict):
                num_fail += 1
                continue
            existing = event_details.get('cut_results', {})
            chi_ok   = existing.get('chi_cut_passed', True)
            angle_ok = existing.get('angle_cut_passed', True)
            time_ok  = time_cut_results.get(event_id, False)
            fft_ok   = check_fft_cut(event_details, event_id)
            event_details['cut_results'] = {
                'chi_cut_passed':   chi_ok,
                'angle_cut_passed': angle_ok,
                'time_cut_passed':  time_ok,
                'fft_cut_passed':   fft_ok,
            }
            event_details['passes_analysis_cuts'] = all(event_details['cut_results'].values())
            if event_details['passes_analysis_cuts']:
                num_pass += 1
            else:
                num_fail += 1
    else:
        num_pass = num_fail = 0
        total = len(events_data_dict)

        with SectionTimer("Apply chi/angle cuts"):
            for loop_idx, (event_id, event_details) in enumerate(events_data_dict.items()):
                if loop_idx % 500 == 0:
                    ic(f"  chi/angle cut progress: {loop_idx}/{total}")
                if not isinstance(event_details, dict):
                    continue
                chi_ok   = check_chi_cut(event_details)
                angle_ok = True   # angle cut disabled (mirrors C03b)
                event_details['cut_results'] = {
                    'chi_cut_passed':   chi_ok,
                    'angle_cut_passed': angle_ok,
                    'time_cut_passed':  True,
                    'fft_cut_passed':   True,
                }
                event_details['passes_analysis_cuts'] = chi_ok and angle_ok

        # Only run time cut on events that passed chi + angle (saves time)
        events_chi_angle = {
            eid: ed for eid, ed in events_data_dict.items()
            if isinstance(ed, dict) and
               ed.get('cut_results', {}).get('chi_cut_passed', False) and
               ed.get('cut_results', {}).get('angle_cut_passed', False)
        }

        if events_chi_angle:
            with SectionTimer("Apply time cut"):
                time_cut_results = check_time_cut(events_chi_angle, time_threshold_hours=24.0)
        else:
            time_cut_results = {}

        for event_id, event_details in events_data_dict.items():
            if not isinstance(event_details, dict) or 'cut_results' not in event_details:
                num_fail += 1
                continue
            cr = event_details['cut_results']
            time_ok = time_cut_results.get(event_id, False)
            cr['time_cut_passed'] = time_ok
            # FFT cut only runs if chi + angle + time all passed
            if cr['chi_cut_passed'] and cr['angle_cut_passed'] and time_ok:
                cr['fft_cut_passed'] = check_fft_cut(event_details, event_id)
            else:
                cr['fft_cut_passed'] = False
            event_details['passes_analysis_cuts'] = all(cr.values())
            if event_details['passes_analysis_cuts']:
                num_pass += 1
            else:
                num_fail += 1

    ic(f"Cuts complete: {num_pass} passed, {num_fail} failed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ic.enable()

    # --- Config ---
    config = configparser.ConfigParser()
    config_path = os.path.join('HRAStationDataAnalysis', 'config.ini')
    if not os.path.exists(config_path):
        config_path = 'config.ini'
    if not os.path.exists(config_path):
        ic("CRITICAL: config.ini not found.")
        sys.exit(1)

    config.read(config_path)
    date_of_data        = config['PARAMETERS']['date']
    date_of_coincidence = config['PARAMETERS']['date_coincidence']
    date_of_process     = config['PARAMETERS']['date_processing']

    ic(f"date_of_data={date_of_data}, date_of_coincidence={date_of_coincidence}, "
       f"date_of_process={date_of_process}")

    # --- Locate data file ---
    chosen_path = _find_data_file(date_of_data, date_of_coincidence)
    if chosen_path is None:
        ic("Error: no candidate data file found. Cannot proceed.")
        sys.exit(1)
    ic(f"Using data file: {chosen_path}")

    # --- Load ---
    events_data_dict = _load_pickle(chosen_path)
    if events_data_dict is None:
        ic(f"Could not load data from: {chosen_path}")
        sys.exit(1)
    ic(f"Loaded {len(events_data_dict)} top-level entries from pickle.")

    # --- Apply the four standard analysis cuts (mirrors C03b logic) ---
    is_passing_cuts_dataset = "passing_cuts" in os.path.basename(chosen_path)
    _apply_cuts(events_data_dict, is_passing_cuts_dataset)

    # --- Collect events that passed all four standard cuts ---
    events_passing_all_cuts = {
        eid: ed for eid, ed in events_data_dict.items()
        if isinstance(ed, dict) and ed.get('passes_analysis_cuts', False)
    }
    ic(f"Events passing all 4 standard cuts: {len(events_passing_all_cuts)}")

    if not events_passing_all_cuts:
        ic("No events passed the standard cuts. Nothing to evaluate for station chi.")
        sys.exit(0)

    # --- Compute between-station self-match chi for all 4-cut-passing events ---
    # compute_event_self_similarity mutates each event dict in-place, writing:
    #   event_details["derived_metrics"]["self_match"]["best_match_abs_chi"]
    with SectionTimer("Compute between-station self-match chi"):
        compute_event_self_similarity(events_passing_all_cuts)

    # --- Select events where best_match_abs_chi < CHI_STATION_THRESHOLD ---
    # These events pass all standard cuts but fail the between-station
    # waveform similarity check — the BL-like signature.
    target_events = {}
    n_no_selfmatch = 0
    for eid, ed in events_passing_all_cuts.items():
        sm = ed.get("derived_metrics", {}).get("self_match", {})
        best_chi = sm.get("best_match_abs_chi")
        if best_chi is None or not np.isfinite(best_chi):
            # Self-match chi could not be computed (e.g. single-station event)
            n_no_selfmatch += 1
            continue
        if best_chi < CHI_STATION_THRESHOLD:
            target_events[eid] = ed

    ic(f"Events with self-match chi computed: "
       f"{len(events_passing_all_cuts) - n_no_selfmatch} / {len(events_passing_all_cuts)}")
    ic(f"Events without computable self-match chi (skipped): {n_no_selfmatch}")
    ic(f"Events failing station chi cut "
       f"(best_match_abs_chi < {CHI_STATION_THRESHOLD}): {len(target_events)}")

    if not target_events:
        ic("No events fail the between-station chi cut. Nothing to plot.")
        sys.exit(0)

    # --- Output directory ---
    output_dir = os.path.join(
        "HRAStationDataAnalysis", "plots", "3.9.26", "BL_station_chi_fail"
    )
    os.makedirs(output_dir, exist_ok=True)
    ic(f"Saving plots to: {output_dir}")

    dataset_name = "CoincidenceEvents"

    # --- Plot each event that fails the between-station chi cut ---
    with SectionTimer("BL station-chi-fail event plots"):
        for event_id, event_details in sorted(target_events.items()):
            best_chi = (
                event_details
                .get("derived_metrics", {})
                .get("self_match", {})
                .get("best_match_abs_chi", float("nan"))
            )
            title_suffix = f" [BL / Low Station \u03c7={best_chi:.2f}]"
            ic(f"Plotting event {event_id} (best_match_abs_chi={best_chi:.3f}) ...")
            path = plot_single_master_event_thesis(
                event_id,
                event_details,
                output_dir,
                dataset_name,
                title_suffix=title_suffix,
            )
            if path:
                ic(f"  Saved: {path}")
            else:
                ic(f"  WARNING: plot failed for event {event_id}")

    ic(f"Done. {len(target_events)} event plot(s) written to {output_dir}")
