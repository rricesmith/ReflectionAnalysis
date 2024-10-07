from __future__ import absolute_import, division, print_function
from NuRadioReco.utilities import units
from NuRadioMC.EvtGen.generator import generate_eventlist_cylinder
import numpy as np


# loc = 'SP'
loc = ['MB']
# loc = ['MB', 'SP']
max_rad = 3



# n_bins = 12
n_bins = 20
energies = np.logspace(17, 20, num=n_bins)

for l in loc:
    if l == 'SP':
        zmin = -2.7
    elif l == 'MB':
        zmin = -0.575
    elif l == 'GL':
        zmin = -3.0

    # define simulation volume
    volume = {
    #'fiducial_zmin':-2.7 * units.km,  # the ice sheet at South Pole is 2.7km deep
    #'fiducial_zmin':-0.575 * units.km,  # the ice sheet at South Pole is 576mkm deep
    'fiducial_zmin': zmin * units.km,
    'fiducial_zmax': 0 * units.km,
    'fiducial_rmin': 0 * units.km,
    'fiducial_rmax': max_rad * units.km}


    # for energy in energies:
    for iE in range(len(energies)-1):
        energy = energies[iE]
        # if energy < 5*1e17:
        #     num = 1e6
        if energy < 1e18:
            num = 5e6
        else:
            continue
        # elif energy > 1e19:
        #     num = 1e4
        # else:
        #     num = 1e5
        generate_eventlist_cylinder(f'NeutrinoAnalysis/GeneratedEvents/MJob_{l}_{energy:.4e}_n{num:.4e}.hdf5', num, energies[iE] * units.eV, energies[iE+1] * units.eV, volume, n_events_per_file=5*1e4)
        # generate_eventlist_cylinder(f'NeutrinoAnalysis/GeneratedEvents/MJob_{l}_{energy:.4e}_n{num:.4e}.hdf5', num, energy * units.eV, energy * units.eV, volume, n_events_per_file=5*1e4)
    #    generate_eventlist_cylinder(f'NeutrinoAnalysis/GeneratedEvents/AddedStats_{loc}_{energy:.4e}_n{num:.4e}.hdf5', num, energy * units.eV, energy * units.eV, volume, n_events_per_file=5*1e4)


quit()
