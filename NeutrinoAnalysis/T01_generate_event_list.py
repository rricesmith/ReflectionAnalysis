from __future__ import absolute_import, division, print_function
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
import numpy as np


loc = 'SP'
max_rad = 6

if loc == 'SP':
    min = -2.7
elif loc == 'MB':
    min = -0.575
elif loc == 'GL':
    min = -3.0

# define simulation volume
volume = {
#'fiducial_zmin':-2.7 * units.km,  # the ice sheet at South Pole is 2.7km deep
#'fiducial_zmin':-0.575 * units.km,  # the ice sheet at South Pole is 576mkm deep
'fiducial_zmin':min*units.km,
'fiducial_zmax': 0 * units.km,
'fiducial_rmin': 0 * units.km,
'fiducial_rmax': max_rad * units.km}


energies = np.logspace(16, 20, num=20)

for energy in energies:
    if energy < 5*1e17:
        num = 1e6
    elif energy > 1e19:
        num = 1e4
    else:
        num = 1e5
    generate_eventlist_cylinder(f'NeutrinoAnalysis/GeneratedEvents/R{max_rad}km_{loc}_{energy:.4e}_n{num:.4e}.hdf5', num, energy * units.eV, energy * units.eV, volume, n_events_per_file=5*1e4)
#    generate_eventlist_cylinder(f'NeutrinoAnalysis/GeneratedEvents/AddedStats_{loc}_{energy:.4e}_n{num:.4e}.hdf5', num, energy * units.eV, energy * units.eV, volume, n_events_per_file=5*1e4)


quit()


generate_eventlist_cylinder(f'NeutrinoAnalysis/GeneratedEvents/{loc}_1e17_n5e6.hdf5', 5*1e6, 1e17 * units.eV, 1e17 * units.eV, volume, n_events_per_file=5*1e4)
quit()
generate_eventlist_cylinder(f'NeutrinoAnalysis/GeneratedEvents/{loc}_1e16_n1e7.hdf5', 1e7, 1e16 * units.eV, 1e16 * units.eV, volume, n_events_per_file=5*1e4)
generate_eventlist_cylinder(f'NeutrinoAnalysis/GeneratedEvents/{loc}_3e17_n1e6.hdf5', 1e6, 3*1e17 * units.eV, 3*1e17 * units.eV, volume, n_events_per_file=5*1e4)

generate_eventlist_cylinder(f'NeutrinoAnalysis/GeneratedEvents/{loc}_1e17_n1e6.hdf5', 1e6, 1e17 * units.eV, 1e17 * units.eV, volume, n_events_per_file=5*1e4)
generate_eventlist_cylinder(f'NeutrinoAnalysis/GeneratedEvents/{loc}_3e16_n1e6.hdf5', 1e6, 3*1e16 * units.eV, 3*1e16 * units.eV, volume, n_events_per_file=5*1e4)
generate_eventlist_cylinder(f'NeutrinoAnalysis/GeneratedEvents/{loc}_5e16_n1e6.hdf5', 1e6, 5*1e16 * units.eV, 5*1e16 * units.eV, volume, n_events_per_file=5*1e4)
quit()

# generate one event list at 1e20 eV with 1000 neutrinos
generate_eventlist_cylinder('NeutrinoAnalysis/GeneratedEvents/MB_1e20_n1e3.hdf5', 1e4, 1e20 * units.eV, 1e20 * units.eV, volume)

# generate one event list at 1e19 eV with 1000 neutrinos
generate_eventlist_cylinder('NeutrinoAnalysis/GeneratedEvents/MB_1e19_n1e3.hdf5', 1e4, 1e19 * units.eV, 1e19 * units.eV, volume)

# generate one event list at 1e18 eV with 10000 neutrinos
generate_eventlist_cylinder('NeutrinoAnalysis/GeneratedEvents/MB_1e18_n1e5.hdf5', 1e5, 1e18 * units.eV, 1e18 * units.eV, volume, n_events_per_file=1e4)

# generate one event list at 1e17 eV with 100000 neutrinos
generate_eventlist_cylinder('NeutrinoAnalysis/GeneratedEvents/MB_1e17_n1e5.hdf5', 1e5, 1e17 * units.eV, 1e17 * units.eV, volume, n_events_per_file=1e4)

#generate one event list at 1e17 eV with 100000 neutrinos
generate_eventlist_cylinder('NeutrinoAnalysis/GeneratedEvents/MB_1e16_n1e5.hdf5', 1e5, 1e16 * units.eV, 1e16 * units.eV, volume, n_events_per_file=1e4)




#generate_eventlist_cylinder('NeutrinoAnalysis/GeneratedEvents/GZK.hdf5', 1e6, 1e16 * units.eV, 1e20 * units.eV, volume, n_events_per_file=1e5, spectrum='GZK-1')
