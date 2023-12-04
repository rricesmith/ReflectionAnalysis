from __future__ import absolute_import, division, print_function
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder

# define simulation volume
volume = {
'fiducial_zmin':-.570 * units.km,  # the ice sheet at MB is 576m deep
'fiducial_zmax': 0 * units.km,
'fiducial_rmin': 0 * units.km,
'fiducial_rmax': 4 * units.km}

# generate one event list from 1e17-1e19 eV with 10000 neutrinos
nums = [0, 1]
for i in nums:
    generate_eventlist_cylinder(f'GeneratedNeutrinoEvents/N01_1e5Nu_e18e19_part{i}.hdf5', 1e5, 1e18 * units.eV, 1e19 * units.eV, volume)
