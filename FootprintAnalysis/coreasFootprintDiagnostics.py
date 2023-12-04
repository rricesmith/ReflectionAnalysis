import os
import h5py
import argparse
from NuRadioReco.utilities import units
import numpy as np
from radiotools import helper as hp
import matplotlib.pyplot as plt





parser = argparse.ArgumentParser(description='Print out diagnostic information of footprint files')
parser.add_argument('folder', type=str, help='Folder containing footprints to analize')
parser.add_argument('saveName', type=str, help='Prefix files will be saved with')

args = parser.parse_args()

folder = args.folder
saveName = args.saveName

fp_energy = []
fp_zenith = []
fp_azimuth = []

input_files = []
max_file = 999
i = 0
while i < max_file:
    file = f'../MBFootprints/00{i:04d}.hdf5'
    if os.path.exists(file):
        input_files.append(file)
    i += 1

#for file in os.listdir(folder):
#    filepath = os.path.join(folder, file)
#    corsika = h5py.File(filepath, "r")
for file in input_files:
    corsika = h5py.File(file, "r")

    energy = corsika['inputs'].attrs["ERANGE"][0] * units.GeV
    zenith = np.deg2rad(corsika['inputs'].attrs["THETAP"][0])
    azimuth = hp.get_normalized_angle(3 * np.pi / 2. + np.deg2rad(corsika['inputs'].attrs["PHIP"][0]))

    fp_energy.append(np.log10(energy))
    fp_zenith.append(np.rad2deg(zenith))
    fp_azimuth.append(np.rad2deg(azimuth))


#Make a plot of energy vs zenith
plt.scatter(fp_energy, fp_zenith, label=f'Num {len(fp_energy)}', alpha = .1)
plt.xlabel('CR Energy (eV)')
plt.ylabel('Zenith (deg)')
plt.legend()
plt.savefig(f'plots/FootprintAnalysis/{saveName}_FP_EngZen_diagnostic.png')
plt.clf()


#Make plot of zenith vs azimuth
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(fp_azimuth, fp_zenith, label=f'Num {len(fp_energy)}', alpha=0.1)
plt.legend()
plt.savefig(f'plots/FootprintAnalysis/{saveName}_FP_AziZen_diagnostic.png')
plt.clf()
