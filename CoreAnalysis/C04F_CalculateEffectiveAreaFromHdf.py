#!/bin/env python

import argparse
import h5py
from icecream import ic
import numpy as np
import os
# from plotting_tools import qualitative_colors
from radiotools import coordinatesystems as cstrans
import radiotools.helper as hp
import tqdm

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

from NuRadioReco.utilities import units
from NuRadioMC.utilities.Veff import FC_limits
from NuRadioMC.utilities import medium
import hdf5AnalysisUtils as hdau

# from analysis_tools.ray_solutions import GetPathVariables
# from analysis_tools.stats import WilsonMean, WilsonError

ABS_PATH_HERE = str(os.path.dirname(os.path.realpath(__file__)))
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

parser = argparse.ArgumentParser(description="Parses the hdf5 files to tabulate the effective volume")
parser.add_argument("input", type=str, nargs="+")
parser.add_argument("--rnog", action="store_true", help="Set this flag if parsing the RNO-G files")
parser.add_argument(
    "--trigtype",
    type=str,
    nargs="+",
    default=[
        # "PA_8ch_100Hz",
        # "PA_8ch_1Hz",
        "PA_4ch_100Hz",
        "PA_4ch_1Hz",
        "LPDA_2of4_100Hz",
        "LPDA_2of4_1Hz",
    ],
)
args = parser.parse_args()

if not len(args.input):
    exit()

if args.rnog:
    depth_str = "rnog"
    THIS_DEPTH = None
    output_dir = os.path.join(ABS_PATH_HERE, "data", "effective_area", depth_str)
    rnog_ice_model = medium.get_ice_model("greenland_simple")
else:
    depth_str = os.path.basename(args.input[0]).split("-")[2].replace(".hdf5", "")
    THIS_DEPTH = float(depth_str.replace("depth_", ""))
    output_dir = os.path.join(ABS_PATH_HERE, "data", "effective_area", depth_str)

if not os.path.isdir(output_dir):
    print("Making directory:", output_dir)
    os.makedirs(output_dir)

# Plotting tools

class ColorSet(object):
    def __init__(self, colors=[]):
        self.colors = colors

    def __len__(self):
        return len(self.colors)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[ii] for ii in range(*idx.indices(len(self)))]
        return self.colors[idx % len(self.colors)]

def qualitative_colors(n):
    if n < 1:
        raise ValueError("Minimum number of qualitative colors is 1.")
    elif n > 12:
        raise ValueError("Maximum number of qualitative colors is 12.")
    cols = [
        "#4477AA",
        "#332288",
        "#6699CC",
        "#88CCEE",
        "#44AA99",
        "#117733",
        "#999933",
        "#DDCC77",
        "#661100",
        "#CC6677",
        "#AA4466",
        "#882255",
        "#AA4499",
    ]
    indices = [
        [0],
        [0, 9],
        [0, 7, 9],
        [0, 5, 7, 9],
        [1, 3, 5, 7, 9],
        [1, 3, 5, 7, 9, 12],
        [1, 3, 4, 5, 7, 9, 12],
        [1, 3, 4, 5, 6, 7, 9, 12],
        [1, 3, 4, 5, 6, 7, 9, 11, 12],
        [1, 3, 4, 5, 6, 7, 8, 9, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ]
    return ColorSet([cols[ix] for ix in indices[n - 1]])



##########################################
### Define the antenna numbering and ids
##########################################

if args.rnog:
    LPDA_ids = [4, 5, 6, 7, 8, 9, 11, 12, 13]
    LPDA_pos = [
        [1.5, 11, -2],
        [-1.5, 11, -2],
        [-10.276, -4.2, -2],
        [-8.776, -6.8, -2],
        [8.776, -6.8, -2],
        [10.276, -4.2, -2],
        [1.5, 11, -2],
        [-10.276, -4.2, -2],
        [8.776, -6.8, -2],
    ]
else:
    LPDA_ids = [0, 1, 2, 3, 13, 14]
    LPDA_pos = np.array(
        [
            [3, 0, -3],
            [0, 3, -3],
            [-3, 0, -3],
            [0, -3, -3],
            [3, 0, -3],
            [0, 3, -3],
        ]
    )
assert len(LPDA_ids) == len(LPDA_pos)


if args.rnog:
    PA_ids = [0, 1, 2, 3, 10]
    PA_pos = [[0, 0, -97], [0, 0, -96], [0, 0, -95], [0, 0, -94], [0, 0, -97]]
else:
    PA_ids = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    PA_pos = np.array(
        [
            [0, 0, -143],
            [0, 0, -144],
            [0, 0, -145],
            [0, 0, -146],
            [0, 0, -147],
            [0, 0, -148],
            [0, 0, -149],
            [0, 0, -150],
            [0, 0, -150],
        ]
    )
assert len(PA_ids) == len(PA_pos)


def ParseOneFile(filename, trigtype=None):
    out = dict()
    with h5py.File(filename, "r") as fin:

        if "Emin" not in fin.attrs.keys():
            ic(filename)

        Emin = fin.attrs["Emin"]
        Emax = fin.attrs["Emax"]
        assert Emin == Emax

        if len(fin["shower_energies"]):
            assert fin["shower_energies"][0] == Emin
            assert fin["shower_energies"][0] == fin["energies"][0]

        out["lgE"] = np.log10(Emin)

        thetamin = fin.attrs["thetamin"]
        thetamax = fin.attrs["thetamax"]
        out["zenith"] = np.arccos(0.5 * (np.cos(thetamin) + np.cos(thetamax)))

        out["area_projected"] = fin.attrs["area"] * np.cos(out["zenith"])

        out["n_events"] = fin.attrs["n_events"]

        out["triggers"] = dict()

        if "trigger_names" not in fin.attrs:
            return out

        reco_cache = np.ones((len(fin["xx"]), 3)) * np.nan

        interaction_pos = np.array([np.array(fin["xx"]), np.array(fin["yy"]), np.array(fin["zz"])]).T

        for trig in args.trigtype:
            if not trig in fin.attrs["trigger_names"]:  # Skip empty triggers
                continue

            out["triggers"][trig] = {}

            # Only consider relevant antennas (based on the used .json file)
            if "LPDA_" in trig:
                ants_to_check = LPDA_ids
                this_pos = LPDA_pos
            elif "PA_" in trig:
                ants_to_check = PA_ids
                this_pos = PA_pos
            # Cut away channels that do not contribute to the trigger. Assumes the later channels are not used
            if "4ch" in trig or "2of4" in trig:
                ants_to_check = ants_to_check[:4]
                this_pos = this_pos[:4]

            # Find which events passed the trigger
            itrig = np.where(fin.attrs["trigger_names"] == trig)[0][0]
            passed_triggers = fin["multiple_triggers"][:, itrig]

            # Get the number of reflections
            n_reflections = np.array(fin["station_1001"]["ray_tracing_reflection"])
            n_reflections = n_reflections[:, ants_to_check]

            # Get the contribution from reach ray solution
            amps = np.array(fin["station_1001"]["max_amp_shower_and_ray"])
            amps = amps[:, ants_to_check]

            launch = np.array(fin["station_1001"]["launch_vectors"])
            launch = launch[:, ants_to_check]

            receive = np.array(fin["station_1001"]["receive_vectors"])
            receive = receive[:, ants_to_check]

            polarization = np.array(fin["station_1001"]["polarization"])
            polarization = polarization[:, ants_to_check]

            radius = np.sqrt(interaction_pos[:, 0] ** 2 + interaction_pos[:, 1] ** 2)

            # Filter out only the solutions that contributed most to the trigger
            n_reflections_top = np.ones(len(amps), dtype=bool) * np.nan
            reco_dir_top = np.ones((len(amps), 3)) * np.nan
            maximum_amps = np.zeros(len(amps))
            for i in range(len(amps)):
                if not passed_triggers[i]:
                    continue

                vals = amps[i]
                ch_id, sol_id = np.unravel_index(np.argmax(vals), vals.shape)
                n_reflections_top[i] = n_reflections[i, ch_id, sol_id]
                # Only check maximum amplitude of the antennas that contributed to the trigger
                maximum_amps[i] = np.max(vals)

                if np.isnan(reco_cache[i, 0]):  # Don't repeat path simulations more than once

                    antenna_pos = this_pos[ch_id]

                    if args.rnog:
                        n_launch = launch[i, ch_id, sol_id]
                        cherenkov_angle = rnog_ice_model.get_index_of_refraction(antenna_pos)
                        if np.any(np.isnan(n_launch)):
                            continue
                    else:
                        continue
                        # Repeat the ray tracing to get the launch vector for "image" position
                        _, n_launch, _, cherenkov_angle = GetPathVariables(interaction_pos[i], antenna_pos, THIS_DEPTH, sol_id)
                        if n_launch is None:
                            continue

                    n_receive = receive[i, ch_id, sol_id]
                    n_pol = polarization[i, ch_id, sol_id]

                    # This block transforms the n_pol at the antenna to n_pol at the launch point
                    cs_at_antenna = cstrans.cstrafo(*hp.cartesian_to_spherical(*n_receive))
                    polarization_on_sky = cs_at_antenna.transform_from_ground_to_onsky(n_pol)
                    cstrafo2 = cstrans.cstrafo(*hp.cartesian_to_spherical(*n_launch))
                    n_pol = cstrafo2.transform_from_onsky_to_ground(polarization_on_sky)

                    dir_nu = np.sin(cherenkov_angle) * n_pol + np.cos(cherenkov_angle) * n_launch
                    dir_nu /= np.sqrt(sum(dir_nu**2))  # normalize

                    reco_dir_top[i] = -1 * dir_nu  # Direction the neutrino was coming from, not going

                    reco_cache[i] = np.array(reco_dir_top[i])

                else:
                    reco_dir_top[i] = reco_cache[i]

            # trig_mask, trig_index = hdau.trigger_mask(fin, trigger = trig, station = "station_1001")


            # mask = ~np.isnan(reco_dir_top[:, 0])
            mask = ~np.isnan(n_reflections_top)
            # mask = trig_mask

            out["triggers"][trig]["reflected"] = n_reflections_top[mask]
            out["triggers"][trig]["reco_dir"] = reco_dir_top[mask]
            out["triggers"][trig]["radius"] = radius[mask]

            # Adding code to get the SNR of each event for future masking
            Vrms = fin["station_1001"].attrs['Vrms'][ants_to_check[0]]  # Same antennas of a trigger have identical Vrms
            SNR = maximum_amps[mask] / Vrms

            # ic(len(SNR))
            # quit()

            out["triggers"][trig]["SNR"] = SNR

    return out


def AddNewTrig(x, trig, zen, lgE):
    if not trig in x.keys():
        x[trig] = dict()
    AddNewZenith(x[trig], zenith, lgE)


def AddNewZenith(x, zen, lgE):
    if not zen in x.keys():
        x[zen] = dict()
    AddNewLgE(x[zen], lgE)


def AddNewLgE(x, lgE):
    if not lgE in x.keys():
        x[lgE] = dict()
        AddFields(x[lgE])


def AddFields(x):
    x["area_projected"] = None
    x["n_events"] = 0
    x["n_trig_direct"] = 0
    x["n_trig_reflect"] = 0
    x["reco_dir_reflect"] = np.zeros((0, 3))
    x["reco_dir_direct"] = np.zeros((0, 3))
    x["radius_direct"] = []
    x["radius_reflect"] = []
    x["n_trig_reflect_SNR"] = 0
    x["n_trig_direct_SNR"] = 0


###############################################
## Read in all the files and dump into a dict
###############################################

file_outputs = dict()

print("Will process", len(args.input), "files")
for filename in tqdm.tqdm(args.input):
    out = ParseOneFile(filename, args.trigtype)

    lgE = out["lgE"]
    zenith = out["zenith"]
    area_projected = out["area_projected"]
    n_events = out["n_events"]

    station = "station_1001"
    ic(lgE, zenith)

    # loop over triggers and combine it with previous files
    for trig in out["triggers"].keys():        
        ic(trig)
        AddNewTrig(file_outputs, trig, zenith, lgE)

        this_view = file_outputs[trig][zenith][lgE]

        if this_view["area_projected"] is None:
            file_outputs[trig][zenith][lgE]["area_projected"] = out["area_projected"]
        elif this_view["area_projected"] != area_projected:
            ic(out["area_projected"])
            ic(file_outputs[trig][zenith][lgE]["area_projected"])
            raise ValueError("Sored area must match current area")

        this_view["n_events"] += n_events

        if trig not in out["triggers"]:
            continue

        selction_reflect = out["triggers"][trig]["reflected"] == True
        selection_direct = out["triggers"][trig]["reflected"] == False

        n_reflected = sum(selction_reflect)
        n_direct = sum(selection_direct)

        reco_dir_reflect = out["triggers"][trig]["reco_dir"][selction_reflect]
        reco_dir_direct = out["triggers"][trig]["reco_dir"][selection_direct]

        if "PA" in trig:
            cut_SNR = 1
        elif "LPDA" in trig:
            cut_SNR = 2.5
        selection_SNR = out["triggers"][trig]["SNR"] > cut_SNR

        selection_SNR_reflect = np.logical_and(selction_reflect, selection_SNR)
        selection_SNR_direct = np.logical_and(selection_direct, selection_SNR)
        n_reflected_SNR = sum(selection_SNR_reflect)
        n_direct_SNR = sum(selection_SNR_direct)

        this_view["n_trig_reflect"] += n_reflected
        this_view["n_trig_direct"] += n_direct
        this_view["reco_dir_reflect"] = np.append(this_view["reco_dir_reflect"], reco_dir_reflect, axis=0)
        this_view["reco_dir_direct"] = np.append(this_view["reco_dir_direct"], reco_dir_direct, axis=0)
        this_view["radius_reflect"] += list(out["triggers"][trig]["radius"][selction_reflect])
        this_view["radius_direct"] += list(out["triggers"][trig]["radius"][selection_direct])

        this_view["n_trig_reflect_SNR"] += n_reflected_SNR
        this_view["n_trig_direct_SNR"] += n_direct_SNR
        ic(n_reflected_SNR, n_direct_SNR, n_reflected, n_direct)
        # quit()

#########################
## Get union of all keys
#########################

all_lgEs = []
all_zens = []

max_radius = 0

for x in file_outputs.values():
    for zen_key in x.keys():
        all_zens = np.union1d(all_zens, [zen_key])

        for eng_key in x[zen_key].keys():
            all_lgEs = np.union1d(all_lgEs, [eng_key])

            if len(x[zen_key][eng_key]["radius_reflect"]):
                max_radius = max(max_radius, max(x[zen_key][eng_key]["radius_reflect"]))
            if len(x[zen_key][eng_key]["radius_direct"]):
                max_radius = max(max_radius, max(x[zen_key][eng_key]["radius_direct"]))

ic(all_lgEs)
ic(all_zens)
ic(max_radius)

#########################
## Make plots
#########################

colors = qualitative_colors(len(all_zens))

# ncols = min(2, len(file_outputs))
# nrows = int(len(file_outputs) / ncols + 0.999)
trigs_to_plot = ["PA_4ch_100Hz", "LPDA_2of4_100Hz"]
ncols = len(trigs_to_plot)
nrows = 1

# Figure and axis for the eff area
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
ax = ax.flatten()

# Figure and axis for the eff area of direct only
fig_dir, ax_dir = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
ax_dir = ax_dir.flatten()

# Figure and axis for the eff area of reflected only
fig_refl, ax_refl = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2})
ax_refl = ax_refl.flatten()

# Figure and axis for the raw trigger efficiency
fig_raw, ax_raw = plt.subplots(
    ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2}
)
ax_raw = ax_raw.flatten()

# Figure and axis for the raw trigger efficiency
fig_reco_dir, ax_reco_dir = plt.subplots(
    ncols=ncols, nrows=nrows, figsize=(ncols * 8, nrows * 5), gridspec_kw={"wspace": 0.2, "hspace": 0.2}
)
ax_reco_dir = ax_reco_dir.flatten()

area_units = units.km**2
area_units_str = r"km$^2$"
min_area = 1e100
max_area = 0

reco_cos_zen_bins = np.linspace(-1, 1, 41)

n_radial_bins = max(100, int(max_radius / (10 * units.m)) + 1)
radial_bin_edges = np.linspace(0, max_radius, n_radial_bins + 1)
radial_bin_centers = 0.5 * (radial_bin_edges[:-1] + radial_bin_edges[1:])

# for itrig, trig_key in enumerate(file_outputs.keys()):
for itrig, trig_key in enumerate(trigs_to_plot):

    trigger_view = file_outputs[trig_key]

    # indexes are (zen, lgE, dir/ref, avg/low/high)
    storage = np.ones((len(all_zens), len(all_lgEs), 2, 3)) * np.nan
    storage_raw = np.ones((len(all_zens), len(all_lgEs), 2, 2)) * np.nan

    # indexes are (zen, lgE, dir/ref, reco-bin)
    storage_reco_dir = np.zeros((len(all_zens), len(all_lgEs), 2, len(reco_cos_zen_bins) - 1))
    storage_radius = np.zeros((len(all_zens), len(all_lgEs), 2, n_radial_bins))

    for izen, zenith in enumerate(all_zens):

        if not zenith in trigger_view.keys():
            continue

        # label = f"Zen: {np.rad2deg(zenith):0.1f}" + r"$^{\rm o}$" + f" cos{np.cos(zenith):0.3}"
        # ax[itrig].fill_between([], [], [], color=colors[izen], label=label)
        # ax_raw[itrig].fill_between([], [], [], color=colors[izen], label=label)
        # ax_reco_dir[itrig].fill_between([], [], [], color=colors[izen], label=label)

        zenith_view = trigger_view[zenith]

        # Get the energies and sort them for easy plotting
        lgEs = [x for x in zenith_view.keys()]
        lgEs.sort()

        # Extract data in lgE order
        proj_areas = np.array([zenith_view[lgE]["area_projected"] for lgE in lgEs])
        n_events = np.array([zenith_view[lgE]["n_events"] for lgE in lgEs])
        n_trig_direct = np.array([zenith_view[lgE]["n_trig_direct"] for lgE in lgEs])
        n_trig_reflect = np.array([zenith_view[lgE]["n_trig_reflect"] for lgE in lgEs])
        n_trig_reflect_SNR = np.array([zenith_view[lgE]["n_trig_reflect_SNR"] for lgE in lgEs])
        n_trig_direct_SNR = np.array([zenith_view[lgE]["n_trig_direct_SNR"] for lgE in lgEs])
        norm = proj_areas / n_events

        # Calculate the effective area and err bars for standard prop
        eff_area_direct = n_trig_direct * norm
        eff_area_direct_SNR = n_trig_direct_SNR * norm
        # mean = WilsonMean(n_trig_direct, n_events - n_trig_direct)
        # err = WilsonError(n_trig_direct, n_events - n_trig_direct)
        # eff_area_direct_low = (mean - err) * proj_areas
        # eff_area_direct_high = (mean + err) * proj_areas

        ax[itrig].plot(lgEs, eff_area_direct / area_units, linestyle="-", color=colors[izen])
        ax[itrig].plot(lgEs, eff_area_direct_SNR / area_units, linestyle=":", color=colors[izen])
        # ax[itrig].fill_between(
        #     lgEs, eff_area_direct_low / area_units, eff_area_direct_high / area_units, color=colors[izen], alpha=0.2
        # )
        ax_dir[itrig].plot(lgEs, n_trig_direct / n_events, linestyle="-", color=colors[izen])
        ax_dir[itrig].plot(lgEs, n_trig_direct_SNR / n_events, linestyle="--", color=colors[izen])

        ax_raw[itrig].plot(lgEs, n_trig_direct / n_events, linestyle="-", color=colors[izen])
        ax_raw[itrig].plot(lgEs, n_trig_direct_SNR / n_events, linestyle=":", color=colors[izen])
        # ax_raw[itrig].fill_between(lgEs, mean - err, mean + err, color=colors[izen], alpha=0.2)

        # Calculate the effective area and err bars for reflected solution
        eff_area_reflect = n_trig_reflect * norm
        eff_area_refl_SNR = n_trig_reflect_SNR * norm
        # mean = WilsonMean(n_trig_reflect, n_events - n_trig_reflect)
        # err = WilsonError(n_trig_reflect, n_events - n_trig_reflect)
        # eff_area_reflect_low = (mean - err) * proj_areas
        # eff_area_reflect_high = (mean + err) * proj_areas

        ax[itrig].plot(lgEs, eff_area_reflect / area_units, linestyle="--", color=colors[izen])
        ax[itrig].plot(lgEs, eff_area_refl_SNR / area_units, linestyle="-.", color=colors[izen])
        # ax[itrig].fill_between(
        #     lgEs, eff_area_reflect_low / area_units, eff_area_reflect_high / area_units, color=colors[izen], alpha=0.2
        # )
        ax_refl[itrig].plot(lgEs, n_trig_reflect / n_events, linestyle="-", color=colors[izen])
        ax_refl[itrig].plot(lgEs, n_trig_reflect_SNR / n_events, linestyle="--", color=colors[izen])

        ax_raw[itrig].plot(lgEs, n_trig_reflect / n_events, linestyle="--", color=colors[izen])
        ax_raw[itrig].plot(lgEs, n_trig_reflect_SNR / n_events, linestyle="-.", color=colors[izen])
        # ax_raw[itrig].fill_between(lgEs, mean - err, mean + err, color=colors[izen], alpha=0.2)

        ax_raw[itrig].plot(lgEs, (n_trig_reflect + n_trig_direct) / n_events, linestyle=":", color=colors[izen])
        ax_raw[itrig].plot(lgEs, (n_trig_reflect_SNR + n_trig_direct_SNR) / n_events, linestyle=":", color=colors[izen])

        # Store limits for setting uniform plotting ranges
        max_area = max(max_area, max(eff_area_direct), max(eff_area_reflect))
        if sum(eff_area_direct[eff_area_direct > 0]):
            min_are = min(min_area, min(eff_area_direct[eff_area_direct > 0]))
        if sum(eff_area_reflect[eff_area_reflect > 0]):
            min_are = min(min_area, min(eff_area_reflect[eff_area_reflect > 0]))

        # Store data to make numpy files
        for ie, lgE in enumerate(lgEs):
            je = np.argwhere(all_lgEs == lgE)[0]  # Don't assume sorted input

            lgE_view = zenith_view[lgE]

            # Only store cos zen (z-component = index 2)
            storage_reco_dir[izen, je, 0], _ = np.histogram(lgE_view["reco_dir_direct"][:, 2], bins=reco_cos_zen_bins)
            storage_reco_dir[izen, je, 1], _ = np.histogram(lgE_view["reco_dir_reflect"][:, 2], bins=reco_cos_zen_bins)
            storage_radius[izen, je, 0], _ = np.histogram(lgE_view["radius_direct"], bins=radial_bin_edges)
            storage_radius[izen, je, 1], _ = np.histogram(lgE_view["radius_reflect"], bins=radial_bin_edges)

            storage[izen, je, 0, 0] = eff_area_direct[ie]
            # storage[izen, je, 0, 1] = eff_area_direct_low[ie]
            # storage[izen, je, 0, 2] = eff_area_direct_high[ie]

            storage[izen, je, 1, 0] = eff_area_reflect[ie]
            # storage[izen, je, 1, 1] = eff_area_reflect_low[ie]
            # storage[izen, je, 1, 2] = eff_area_reflect_high[ie]

            storage_raw[izen, je, 0, 0] = n_trig_direct[ie]
            storage_raw[izen, je, 0, 1] = n_events[ie]

            storage_raw[izen, je, 1, 0] = n_trig_reflect[ie]
            storage_raw[izen, je, 1, 1] = n_events[ie]

        reco_cos_zen_centers = 0.5 * (reco_cos_zen_bins[1:] + reco_cos_zen_bins[:-1])
        yy = np.array(storage_reco_dir[izen, :])
        for ie in range(len(lgEs)):
            if sum(yy[ie, 0]):
                yy[ie, 0] /= sum(yy[ie, 0])
            if sum(yy[ie, 1]):
                yy[ie, 1] /= sum(yy[ie, 1])

        ax_reco_dir[itrig].hist(
            reco_cos_zen_centers,
            bins=reco_cos_zen_bins,
            weights=np.sum(yy, axis=0)[0] / np.sum(yy),
            color=colors[izen],
            histtype="step",
            linestyle="-",
        )

        ax_reco_dir[itrig].hist(
            reco_cos_zen_centers,
            bins=reco_cos_zen_bins,
            weights=np.sum(yy, axis=0)[1] / np.sum(yy),
            color=colors[izen],
            histtype="step",
            linestyle="--",
        )

    # Set plot style stuff
    for this in [ax, ax_raw, ax_reco_dir]:
        this[itrig].plot([], [], linestyle="-", color="k", label="Standard")
        this[itrig].plot([], [], linestyle="--", color="k", label="Reflected")
        # this[itrig].plot([], [], linestyle=":", color="k", label="Standard, SNR Cut")
        # this[itrig].plot([], [], linestyle="-.", color="k", label="Reflected, SNR Cut")
    for this in [ax_dir]:
        this[itrig].plot([], [], linestyle="-", color="k", label="Direct")
        this[itrig].plot([], [], linestyle="--", color="k", label="Direct SNR Cut")
    for this in [ax_refl]:
        this[itrig].plot([], [], linestyle="-", color="k", label="Reflected")
        this[itrig].plot([], [], linestyle="--", color="k", label="Reflected SNR Cut")
    for this in [ax, ax_raw, ax_reco_dir, ax_dir, ax_refl]:
        this[itrig].set_xlim(min(all_lgEs), max(all_lgEs))

        this[itrig].tick_params(axis="both", which="both", direction="in")
        this[itrig].yaxis.set_ticks_position("both")
        this[itrig].xaxis.set_ticks_position("both")

        this[itrig].set_title(f"Trigger: {trig_key}")
        this[itrig].set_xlabel(r"lg($E_{\rm sh}$ / eV)")
        this[itrig].set_yscale("log")


    ax[itrig].set_ylabel(f"Effective Area / {area_units_str}")
    ax[itrig].legend()
    ax_refl[itrig].set_ylabel(f"Effective Area / {area_units_str}")
    ax_refl[itrig].legend()
    ax_dir[itrig].set_ylabel(f"Effective Area / {area_units_str}")
    ax_dir[itrig].legend()
    ax_raw[itrig].set_ylabel(f"Raw trigger Eff")
    ax_raw[itrig].legend()

    ax_reco_dir[itrig].set_ylabel("Normalized Counts")
    ax_reco_dir[itrig].set_xlabel(r"Zenith Angle cos($\theta_{\rm reco}$)")
    ax_reco_dir[itrig].set_xlim(reco_cos_zen_bins[0], reco_cos_zen_bins[-1])
    ax_reco_dir[itrig].xaxis.set_minor_locator(MultipleLocator(0.05))
    ax_reco_dir[itrig].set_yscale("linear")
    ax_reco_dir[itrig].legend(loc="upper left")

    output_path = os.path.join(output_dir, f"eff_area-{depth_str}-{trig_key}")
    print(f"Saving {output_path}.npz")
    np.savez(
        output_path,
        lgEs=all_lgEs,
        zens=all_zens,
        eff_area=storage,
        raw_counts=storage_raw,
        reco_dir=storage_reco_dir,
        reco_cos=reco_cos_zen_centers,
        radius=storage_radius,
        radius_bins=radial_bin_centers,
    )

# Force same y-scale for all plots
# for itrig, trig_key in enumerate(file_outputs.keys()):
for itrig, trig_key in enumerate(trigs_to_plot):
    ax[itrig].set_ylim(0.8 * min_are / area_units, 1.3 * max_area / area_units)
    ax_dir[itrig].set_ylim(0.8 * min_are / area_units, 1.3 * max_area / area_units)
    ax_refl[itrig].set_ylim(0.8 * min_are / area_units, 1.3 * max_area / area_units)

filename = f"{ABS_PATH_HERE}/plots/EffectiveArea_{depth_str}.pdf"
print("Saving", filename)
fig.savefig(filename, bbox_inches="tight")

filename = f"{ABS_PATH_HERE}/plots/EffectiveArea_{depth_str}_direct_SNR_cut.pdf"
print("Saving", filename)
fig_dir.savefig(filename, bbox_inches="tight")

filename = f"{ABS_PATH_HERE}/plots/EffectiveArea_{depth_str}_reflected_SNR_cut.pdf"
print("Saving", filename)
fig_refl.savefig(filename, bbox_inches="tight")

filename = f"{ABS_PATH_HERE}/plots/EffectiveArea_{depth_str}_raw.pdf"
print("Saving", filename)
fig_raw.savefig(filename, bbox_inches="tight")

filename = f"{ABS_PATH_HERE}/plots/EffectiveArea_{depth_str}_reco_dir.pdf"
print("Saving", filename)
fig_reco_dir.savefig(filename, bbox_inches="tight")
