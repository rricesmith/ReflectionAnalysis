import matplotlib.pyplot as plt
import numpy as np
import time
from NuRadioMC.SignalProp import analyticraytracing as ray
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium, medium_base
import logging
from radiotools import helper as hp
from radiotools import plthelpers as php
import NuRadioReco.framework.electric_field
from NuRadioMC.SignalGen import askaryan
import math
from icecream import ic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('raytracing')
# ray.cpp_available=False

# x1 = np.array([478., 0., -149.]) * units.m
# x2 = np.array([635., 0., -5.]) * units.m  # direct ray solution
# x3 = np.array([1000., 0., -90.]) * units.m  # refracted/reflected ray solution
# x4 = np.array([700., 0., -149.]) * units.m  # refracted/reflected ray solution
# x5 = np.array([1000., 0., -5.]) * units.m  # no solution

# Ice model of 300m, 50dB at SP
class ice_model_reflection(medium_base.IceModelSimple):
    def __init__(self):
        super().__init__(
            n_ice = 1.78, 
            z_0 = 77. * units.m, 
            delta_n = 0.423,
            )
        self.add_reflective_bottom( 
            refl_z = -300*units.m, 
            refl_coef = 0.00316, 
            refl_phase_shift = 180*units.deg,
            )
ice = ice_model_reflection()

lss = ['-', '--', ':']
colors = ['b', 'g', 'r', 'y', 'c']
# for i, x in enumerate([x2, x3, x4, x5]):
def plotRayTrace(core_x, core_y, core_z, antenna_depth=-3, index_max=-1, fig=None, ax=None, alpha=1):
    if ax == None:
        fig, ax = plt.subplots(1, 1)

    receive_vectors = np.zeros((4, 4, 3)) * np.nan
    ray_tracing_C0 = np.zeros((4, 4)) * np.nan
    ray_tracing_C1 = np.zeros((4, 4)) * np.nan
    ray_tracing_solution_type = np.zeros((4, 4), dtype=int) * np.nan
    travel_times = np.zeros((4, 4)) * np.nan
    travel_distances = np.zeros((4, 4)) * np.nan

    x1 = np.array([core_x, core_y, core_z]) * units.m  # core location
    x2 = np.array([0, 0., antenna_depth]) * units.m # Station location
    ic(x1, x2)

    ax.plot(x2[0], x2[2], 'ko')
    print('finding solutions for ', x2)
    r = ray.ray_tracing(ice, n_reflections=1)
    r.set_start_and_end_point(x1, x2)
    r.find_solutions()
    if(r.has_solution()):
        for iS in range(r.get_number_of_solutions()):
            if not index_max == -1 and not iS == index_max:
                continue
            try:
                receive_vector = r.get_receive_vector(iS)
                receive_vectors[0, iS] = receive_vector
            except:
                continue
            # if iS >= 4:
            #     ic(iS, 'greater than 3')
            #     continue
            ray_tracing_C0[0, iS] = r.get_results()[iS]['C0']
            # ic(ray_tracing_C0[0, iS])
            # ray_tracing_C1[0, iS] = r.get_results()[iS]['C1']
            # ic(ray_tracing_C1[0, iS])
            ray_tracing_solution_type[0, iS] = r.get_solution_type(iS)
            print("     Solution %d, Type %d: " % (iS, ray_tracing_solution_type[0, iS]))
            R = r.get_path_length(iS)  # calculate path length
            T = r.get_travel_time(iS)  # calculate travel time
            print("     Ray Distance %.3f and Travel Time %.3f" % (R / units.m, T / units.ns))
            
            
            receive_vector = r.get_receive_vector(iS)
            receive_vectors[0, iS] = receive_vector
            zenith, azimuth = hp.cartesian_to_spherical(*receive_vector)
            print("     Receiving Zenith %.3f and Azimuth %.3f " % (zenith / units.deg, azimuth / units.deg))

            # to readout the actual trace, we have to flatten to 2D
            dX = x2 - x1
            dPhi = -np.arctan2(dX[1], dX[0])
            c, s = np.cos(dPhi), np.sin(dPhi)
            R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            X1r = x1
            X2r = np.dot(R, x2 - x1) + x1
            # x1_2d = np.array([X1r[0], X1r[2]])
            # x2_2d = np.array([X2r[0], X2r[2]])
            x1_2d = np.array([x1[0], x1[2]])
            x2_2d = np.array([x2[0], x2[2]])
            # r_2d = ray.ray_tracing_2D(ice)
            # yy, zz = r_2d.get_path(x1_2d, x2_2d, ray_tracing_C0[0, iS])
            rx = r.get_path(iS)
            xx, yy, zz = zip(*rx)


            # test attenuation
            n_samples = 256 #define number of sample in the trace
            dt = 0.5 * units.ns #definte time resolution (bin width) 0.5ns
            trace = askaryan.get_time_trace(1e17 * units.eV, 55 * units.deg, n_samples, dt, shower_type='had', n_index=1.78, R=10*units.m, model='Alvarez2009')
            raw_efield = NuRadioReco.framework.electric_field.ElectricField([0])
            raw_efield.set_trace(trace, sampling_rate=1/dt)
            max_raw = np.max(raw_efield.get_trace())
            # Attenuate the efield using ray_tracing.apply_propogation_effects
            efield = r.apply_propagation_effects(raw_efield, iS)
            max_att = np.max(efield.get_trace())
            # ic(raw_efield, efield)
            # ic(max_raw, max_att)

            # ic(rx)
            # ic(xx, yy, zz)
            # ax.plot(xx, zz, '{}'.format(php.get_color_linestyle(0)), label='{} C0 = {:.4f}, {:.1f} field reduction, {:.0f}ns time diff'.format(ray_tracing_solution_type[0, iS], ray_tracing_C0[0, iS], max_att/max_raw*100, T - r.get_travel_time(0)), color=colors[iS], alpha=alpha)
            # ax.plot(xx, zz, '{}'.format(php.get_color_linestyle(0)), color=colors[iS], alpha=alpha)
            ax.plot(xx, zz, '{}'.format(php.get_color_linestyle(0)), label='{} C0 = {:.4f}, {:.1f}deg arrival zenith, {:.0f}ns time diff, refl angle {:.0f}'.format(ray_tracing_solution_type[0, iS], ray_tracing_C0[0, iS], zenith / units.deg, T - r.get_travel_time(0), get_reflection_angle(xx, zz)), color=colors[iS], alpha=alpha)

            # yy, zz = r_2d.get_path(x1_2d, x2_2d, ray_tracing_C1[0, iS])
            # ic(yy, zz)
            # ax.plot(xx, zz, '{}'.format(php.get_color_linestyle(0)), label='{} C1 = {:.4f}'.format(ray_tracing_solution_type[0, iS], ray_tracing_C1[0, iS]))
            ax.plot(x2_2d[0], x2_2d[1], '{}{}-'.format('d', php.get_color(0)))
    return fig, ax

def get_reflection_angle(xx, zz):
    # This function will return the angle of the reflection at the bottom ice layer
    bottom_index = np.argmin(zz)
    bottom_x = xx[bottom_index] - xx[bottom_index-1]
    bottom_z = zz[bottom_index] - zz[bottom_index-1]
    # ic(bottom_index, zz[bottom_index], xx[bottom_index], bottom_z, bottom_x)
    # reflection_angle = np.arctan(np.abs(bottom_z), np.abs(bottom_x))
    reflection_angle = math.atan(np.abs(bottom_z)/np.abs(bottom_x))
    return np.rad2deg(reflection_angle)

def get_path_for_solution(core_location, station_location=np.array([0., 0., -3])*units.m, iS=0):
    # This function will return the path for a given solution
    # core_location - 3dim numpy array with units
    # station_location - 3dim numpy array with units
    # iS - solution index
    r = ray.ray_tracing(ice, n_reflections=1)


    x2 = station_location
    x1 = core_location

    # Turn dimensions into 2d for solving
    dX = x2 - x1
    dPhi = -np.arctan2(dX[1], dX[0])
    c, s = np.cos(dPhi), np.sin(dPhi)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    X1r = x1
    X2r = np.dot(R, x2 - x1) + x1

    for iA, (a, b) in enumerate(zip(X1r, X2r)):
        ic(iA, a, b)
        if math.isclose(a, b):
            X1r[iA] = 0
            X2r[iA] = 0
    ic(core_location, station_location, X1r, X2r)

    r.set_start_and_end_point(X1r, X2r)
    r.find_solutions()
    rx = r.get_path(iS)
    xx, yy, zz = zip(*rx)
    return xx, yy, zz

if __name__ == "__main__":
    dists = np.arange(50, 1001, 50)

    test = [[100, 100, -5], [0, 100, -5], [100, 0, -5], [75, 25, -5]]
    for t in test:
        get_path_for_solution(np.array(t)*units.m, station_location=np.array([0., 0., -150])*units.m, iS=6)
    quit()
    for d in dists:
        ic(d)
        fig, ax = plotRayTrace(d, 0., -5., antenna_depth=-3, index_max=-1)
        ax.legend()
        ax.set_xlabel("y [m]")
        ax.set_ylabel("z [m]")
        ax.set_ylim(-300, 100)
        fig.tight_layout()
        fig.savefig(f'plots/CoreAnalysis/CorePaper/rayTracing/{d}m.png')

        fig, ax = plotRayTrace(d, 0., -5., antenna_depth=-150, index_max=-1)
        ax.legend()
        ax.set_xlabel("y [m]")
        ax.set_ylabel("z [m]")
        ax.set_ylim(-300, 100)
        fig.tight_layout()
        fig.savefig(f'plots/CoreAnalysis/CorePaper/rayTracing/PA_{d}m.png')