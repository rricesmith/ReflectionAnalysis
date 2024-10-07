from __future__ import absolute_import, division, print_function
import numpy as np
from radiotools import helper as hp
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from NuRadioReco.utilities import units
from NuRadioMC.utilities import medium
from NuRadioMC.SignalProp import analyticraytracing as ray
import h5py
import argparse
import json
import time
import os
from icecream import ic
from pathlib import Path
import hdf5AnalysisUtils as hdau
from NuRadioMC.utilities import medium, medium_base
from CoreAnalysis.C04_plotRayTracingSolutions import plotRayTrace

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


# Read in hdf5 file and plot the ray tracing solutions from the file
# Taken from NuRadioMC/NuRadioMC/simulation/scripts/T07plot_ray_tracing_solutions.py

plot_folder = 'plots/CoreAnalysis/CorePaper/rayTracing/hdf5Plots/'
station = 'station_1001'
sim = [ ['LPDA_2of4_100Hz', [0, 1, 2, 3], [0, 0, -3]], ['PA_8ch_100Hz', [8, 9, 10, 11], [0, 0, -150]]]
# trigger_name = 'LPDA_2of4_100Hz'
# antennas = [0, 1, 2, 3]
# antenna_depth = [0, 0, -3]
# trigger_name = 'PA_8ch_100Hz'
# antennas = [8, 9, 10, 11]
# antenna_depth = [0, 0, -150]


# parser = argparse.ArgumentParser(description='Plot NuRadioMC event list output.')
# parser.add_argument('inputfilename', type=str,
#                     help='path to NuRadioMC hdf5 simulation output')
# parser.add_argument('outputfilename', type=str,
#                     help='name of output file storing the electric field traces at detector positions')
if __name__ == "__main__":
    # args = parser.parse_args()

    # fin = h5py.File(args.inputfilename, 'r')
    # filename = os.path.basename(args.inputfilename).replace('.hdf5', '')

    # energies = ['15.50', '15.60', '15.70', '15.80', '15.90', '16.00', '16.10', '16.20', '16.30', '16.40', '16.50', 
    #             '16.60', '16.70', '16.80', '16.90', '17.00', '17.10', '17.20', '17.30', '17.40', '17.50']
    # energies = ['16.60']
    energies = np.arange(15, 18.01, 0.1)
    cosbin = ['0.95']


    for s in sim:
        ray_hist = {}
        for energy in energies:
            energy = f'{energy:.2f}'
            for cos in cosbin:
                filename = f'../CorePaperhdf5/gen2/lgE_{energy}-cos_{cos}-depth_300.hdf5'
                fin = h5py.File(filename, 'r')
                filename = os.path.basename(filename).replace('.hdf5', '')


                weights = np.array(fin['weights'])
                triggered = np.array(fin['triggered'])
                n_events = fin.attrs['n_events']

                trigger_name = s[0]
                antennas = s[1]
                antenna_depth = s[2]
                Path(plot_folder + filename + '/' + trigger_name).mkdir(parents=True, exist_ok=True)

                # Get masks for specific trigger and reflected events only
                trig_mask, trig_index = hdau.trigger_mask(fin, trigger = trigger_name, station = station)
                ra_mask, rb_mask, da_mask, db_mask, arrival_z, launch_z = hdau.multi_array_direction_masks(fin, antennas, station, trig_index)
                refl_mask = ra_mask | rb_mask


                # plot vertex distribution
                fig, ax = plt.subplots(1, 1)
                xx = np.array(fin['xx'])

                if len(xx[trig_mask][refl_mask]) == 0:
                    # No triggers of this geometry in this file, go to next
                    continue

                yy = np.array(fin['yy'])
                rr = (xx ** 2 + yy ** 2) ** 0.5
                zz = np.array(fin['zz'])
                h = ax.hist2d(rr[trig_mask][refl_mask] / units.m, zz[trig_mask][refl_mask] / units.m, bins=[np.arange(0, 1501, 50), np.array([-300, 10])],
                            cmap=plt.get_cmap('Blues'), weights=weights[trig_mask][refl_mask])    # NOT doing area weighting, because that is already factored into n per bin. n throws scales w/r^2, adding area would double the scaling
                            # cmap=plt.get_cmap('Blues'), weights=weights[trig_mask][refl_mask] * rr[trig_mask][refl_mask]**2)    #use area as weighting of event, which scales as r^2
                # cb = plt.colorbar(h[3], ax=ax)
                # cb.set_label("weighted number of events")
                ax.set_aspect('equal')
                ax.set_xlabel("r [m]")
                ax.set_ylabel("z [m]")
                ax.set_title(f'{energy}lgE cos{cos}')
                fig.tight_layout()
                fig.savefig(f'{plot_folder}/{filename}/{trigger_name}_hist_{energy}lgE_cos{cos}.png')
                ic(f'saving {plot_folder}/{filename}/{trigger_name}_hist_{energy}lgE_cos{cos}.png')


                index_max_amp = {}
                for antenna in antennas:
                    index_max_amp[antenna] = hdau.index_max_amp_per_event(fin, antenna, station, trig_index)[refl_mask]
                # ic(index_max_amp) #Seems like its the same for every antenna, just going with that for now. Need to check TODO
                # quit()
                # mask = (zz < -2000 * units.m) & (rr < 5000 * units.m)
                # for i in np.array(range(len(xx)))[mask]:
                # ic(len(xx), len(xx[trig_mask]), len(xx[trig_mask][refl_mask]), len(index_max_amp[0]))
                # quit()
                
                for iX, i in enumerate(np.array(range(len(xx)))[trig_mask][refl_mask]):
                    ant_0 = antennas[0]
                    C0 = fin[station]['ray_tracing_C0'][i][0][index_max_amp[ant_0][iX]]    # m_showers, n_channels, n_ray_tracing_solutions
                    C1 = fin[station]['ray_tracing_C1'][i][0][index_max_amp[ant_0][iX]]
                    ic(C0, C1, index_max_amp[ant_0][iX])
                    ic(fin[station]['ray_tracing_C1'][i][ant_0])
                    ic(fin[station]['ray_tracing_C0'][i][ant_0])
                    ic(fin[station]['ray_tracing_reflection'][i][ant_0])
                    ic(fin[station]['ray_tracing_reflection_case'][i][ant_0])
                    ic(fin[station]['ray_tracing_solution_type'][i][ant_0]) # 1 direct, 2 refracted, 3 reflected
                    # continue
                    # if i > 100:
                    #     quit()

                    # My other plotting function
                    if True:
                        x1 = np.array([xx[i], yy[i], zz[i]])
                        fig, ax = plotRayTrace(abs(xx[i]), abs(yy[i]), zz[i], antenna_depth=antenna_depth[2], index_max=index_max_amp[ant_0][iX], fig=fig, ax=ax, alpha=0.1)
                        # ax.legend()
                        # fig.tight_layout()
                        # fig.savefig(f'{plot_folder}/{filename}/{trigger_name}/{i}.png')
                        # ic(f'saving {plot_folder}/{filename}/{trigger_name}/{i}.png')
                        continue

                #     C0 = fin['ray_tracing_C0'][i][0][0]
                #     C1 = fin['ray_tracing_C1'][i][0][0]
                    print('weight = {:.2f}'.format(weights[i]))
                    x1 = np.array([xx[i], yy[i], zz[i]])
                    ic(x1)
                    x2 = np.array(antenna_depth)
                    # r = ray.ray_tracing(x1, x2, ice, n_reflections=1)
                    r = ray.ray_tracing(ice, n_reflections=1)
                    r.set_start_and_end_point(x1, x2)
                    r.find_solutions()

                    # C0 = r.get_results()[0]['C0']
                    x1 = np.array([-rr[i], zz[i]])
                    x2 = np.array([0, antenna_depth[2]])
                    r2 = ray.ray_tracing_2D(ice)
                    # r2.find_solutions(x1, x2, plot=True, reflection=1)
                    r2.plot_result(x1, x2, C0, ax)
                    ic(x1, x2, C0)
                    ax.legend()
                    ax.set_xlim(-500, 500)
                    ax.set_ylim(-300, 10)
                    plt.savefig(f'{plot_folder}/{filename}/{trigger_name}/{i}.png')
                    # plt.clf()
                    continue


                    yyy, zzz = r2.get_path(x1, x2, C0)
                    ic(yyy, zzz)

                    launch_vector = fin[station]['launch_vectors'][i][0][0]
                    print(launch_vector)
                    zenith_nu = fin['zeniths'][i]
                    print(zenith_nu / units.deg)

                    fig, ax = plt.subplots(1, 1)
                    ax.plot(-rr[i], zz[i], 'o', label='o')
                    ax.plot([-rr[i], -rr[i] + 100 * np.cos(zenith_nu)], [zz[i], zz[i] + 100 * np.sin(zenith_nu)], '-C0', label='c0')
                    ax.plot(0, -3, 'd', label='d')
                    ax.plot(yyy, zzz, label='yyyzzz')
                    # ax.set_aspect('equal')
                    # plt.show()
                    ax.set_title(f'{energy}lgE cos{cos}')
                    ax.legend()
                    plt.savefig(f'{plot_folder}/{filename}/{trigger_name}/{i}.png')
                    plt.clf()


                colors = ['b', 'g', 'r', 'y', 'c']
                for iC, c in enumerate(colors):
                    ax.plot([], [], color=c, label=f'Soln {iC}, n={np.count_nonzero(index_max_amp[ant_0] == iC)}')
                ax.legend()
                ax.set_title(f'{energy}lgE cos{cos}')
                ax.set_xlabel("y [m]")
                ax.set_ylabel("z [m]")
                # ax.set_xlim(0, 500)
                ax.set_ylim(-300, 0)

                fig.tight_layout()
                fig.savefig(f'{plot_folder}/{filename}/{trigger_name}_all_{energy}lgE_cos{cos}.png')
                ic(f'saving {plot_folder}/{filename}/{trigger_name}_all_{energy}lgE_cos{cos}.png')
                plt.clf()

                # Make a histogram of ray tracing solutions
                fig, ax = plt.subplots(1, 1)
                ax.hist(index_max_amp[ant_0], bins=[0, 1, 2, 3, 4])
                ax.set_xlabel('Ray Tracing Solution #')
                ax.set_ylabel('N-trig ~ weighted distribution')
                ax.set_title(f'{energy}lgE cos{cos}')
                fig.tight_layout()
                fig.savefig(f'{plot_folder}/{filename}/{trigger_name}_RayHist_{energy}lgE_cos{cos}.png')
                ic(f'saving {plot_folder}/{filename}/{trigger_name}_RayHist_{energy}lgE_cos{cos}.png')
                plt.clf()

            if len(xx[trig_mask][refl_mask]) == 0:
                # No triggers of this geometry in this file, go to next
                ray_hist[energy] = []
                continue
            ray_hist[energy] = index_max_amp[ant_0]


        continue
        # Make a histogram of ray tracing solutions for all energies examined
        fig, ax = plt.subplots(1, 1)
        for energy in energies:
            ax.hist(ray_hist[energy], bins=[0, 1, 2, 3, 4], label=energy, alpha=1, histtype='step')
        ax.set_xlabel('Ray Tracing Solution #')
        ax.set_ylabel('N-trig ~ weighted distribution')
        ax.set_yscale('log')
        ax.legend(prop={'size': 8})
        fig.tight_layout()
        fig.savefig(f'{plot_folder}/{trigger_name}_All_RayHist.png')
        ic(f'saving {plot_folder}/{trigger_name}_All_RayHist.png')
        plt.clf()
