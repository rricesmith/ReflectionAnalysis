from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
import argparse
import h5py
import numpy as np
import os

PathToReflectiveAnalysis = os.environ['ReflectiveAnalysis']

def generate_events(n_throws, max_rad, shower_energy, shower_energy_high = 0, seed=None, depositEnergy=False):
#shower energy is in power of EeV, ie 10 => 10*10^18 eV =10^19 eV = 10 EeV   

    eMin = 10 ** shower_energy * units.eV
    eMax = 10 ** shower_energy_high * units.eV
#    eMin = shower_energy * units.EeV
#    eMax = shower_energy_high * units.EeV
    if shower_energy_high <= shower_energy:
        eMax = shower_energy

    # define simulation volume
    zmin = -3 * units.m
    zmax = 0 * units.km
    rmin = 0 * units.km
    rmax = max_rad * units.km
#    ice = medium.southpole_reflection_800m
#    ice = medium.mooresbay_reflection_simple2

    volume = {"fiducial_rmin": rmin, "fiducial_rmax" : rmax, "fiducial_zmin" : zmin, "fiducial_zmax" : zmax}

    saveFile = PathToReflectiveAnalysis + f'/tempdata/CR_cyl_{max_rad}rad_{n_throws}cores_{shower_energy}to{shower_energy_high}eV.hdf5'

    generate_eventlist_cylinder(saveFile, n_throws, eMin, eMax, volume, 0.0 * units.rad, np.deg2rad(60) * units.rad, deposited=depositEnergy)
    return saveFile

def generate_events_square(n_throws, min_x, min_y, max_x, max_y, min_z, shower_energy, shower_energy_high = 0, zen_low = 0 * units.deg, zen_high = 60 * units.deg, seed=None, depositEnergy=False, part=0):
#shower energy is in log10eV, ie 18 -> 10^18eV

    eMin = 10 ** shower_energy * units.eV
    eMax = 10 ** shower_energy_high * units.eV
#    eMin = shower_energy * units.EeV
#    eMax = shower_energy * units.EeV
    if shower_energy_high <= shower_energy:
        eMax = eMin

    # define simulation volume
    zmin = -min_z * units.m
    zmax = 0 * units.km
    xmin = min_x * units.km
    xmax = max_x * units.km
    ymin = min_y * units.km
    ymax = max_y * units.km

    zen_low = zen_low / units.rad
    zen_high = zen_high / units.rad

#    ice = medium.southpole_reflection_800m
#    ice = medium.mooresbay_reflection_simple2

    volume = {"fiducial_xmin": xmin, "fiducial_xmax" : xmax, "fiducial_ymin": ymin, "fiducial_ymax": ymax, "fiducial_zmin" : zmin, "fiducial_zmax" : zmax}

    saveFile = PathToReflectiveAnalysis + f'/tempdata/CR_{n_throws}cores_{shower_energy:.2f}to{shower_energy_high:.2f}eV_zen{zen_low:.2f}-{zen_high:.2f}_{min_x*2:.2f}length_part{part}.hdf5'

    print(f'Parameters {zmin}zmin {zmax}zmax {eMin}eMin {eMax}eMax {zen_low}zen_low {zen_high}zen_high')

    generate_eventlist_cylinder(saveFile, n_throws, eMin, eMax, volume, zen_low, zen_high, deposited=depositEnergy)
    return saveFile




if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Check input configurations')
    parser.add_argument('--eMin', type=float, default='1e19',
                    help='minimum energy to produce in eV, default 1e19 eV')
    parser.add_argument('--eMax', type=float, default='1e19', help='maximum energy to produce in eV, default 1e19 eV')
    parser.add_argument('--throws', type=int, default='100000', help='number of rays to produce')
    args = parser.parse_args()

    throws = args.throws
    eMin = args.eMin * units.eV
    eMax = args.eMax * units.eV

    # define simulation volume
    zmin = -2 * units.m 
    zmax = 0 * units.km
    rmin = 0 * units.km
    rmax = 3 * units.km
#    ice = medium.southpole_reflection_800m 

    volume = {"fiducial_rmin": rmin, "fiducial_rmax" : rmax, "fiducial_zmin" : zmin, "fiducial_zmax" : zmax}
    # generate one event list at 1e19 eV with 100 neutrinos for now

    saveFile = PathToARIANNAanalysis + '/tempdata/SouthPoleCR_' + str(args.eMin) + '_' + str(args.eMax) + '.hdf5'
    print(saveFile)
#Example saveFile : /tempdata/SouthPoleCR_1e19_1e19.hdf5
    generate_eventlist_cylinder(saveFile, throws, eMin, eMax, volume, 0.0 * units.rad, 1.05 * units.rad, deposited=True)
#    generate_eventlist_cylinder(saveFile, 100000, 1e19 * units.eV, 1e19 * units.eV, volume, 0.0 * units.rad, 1.05 * units.rad, deposited=False)
