from NuRadioReco.utilities import units
import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
import time
import os


"""
This file takes input from W03CheckOutput.py from the NuRadioMC tutorial file
Look there for an in depth example of code analyzing output files
"""

#This function is a diagnostics function printing out all relavant information to a given group event id
def group_event_id_diagnostic(fin, event_group_id, trigger = 'LPDA_2of4_100Hz', station='station_1'):
    group_ids_mask = np.array(fin['event_group_ids']) == event_group_id

    print('xx ' + str(fin['xx'][group_ids_mask][0]))
    print('yy ' + str(fin['yy'][group_ids_mask][0]))
    print('zz ' + str(fin['zz'][group_ids_mask][0]))
    print('interaction type ' + str(fin['interaction_type'][group_ids_mask][0]))
    print('shower type ' + str(fin['shower_type'][group_ids_mask][0]))
    print('azimuths ' + str(fin['azimuths'][group_ids_mask][0]))
    print('energies ' + str(fin['energies'][group_ids_mask][0]))
    print('vertex times ' + str(fin['vertex_times'][group_ids_mask][0]))
    print('zeniths ' + str(fin['zeniths'][group_ids_mask][0]))

    return

#This function returns a mask of all events corresponding to those that triggered on the given trigger
#It also returns the index of the chosen trigger to be passed and used in other files
def trigger_mask(fin, trigger = 'LPDA_2of4_100Hz', station = 'station_1', DEBUG=False):
    trigger_names = np.array(fin.attrs['trigger_names'])
    trigger_index = np.squeeze(np.argwhere(trigger_names == trigger))

    """
    print(f'trigger index {trigger_index}')
    if not 'multiple_triggers' in fin:
        print(f'not in')
        print(f'trigger names {trigger_names}')
        for key in fin:
            print(f'{key}')
        #returning empty array for checking on other end
        return [], trigger_index
    """
    mask_coinc_trigger = np.array(fin[station]['multiple_triggers'])[:, trigger_index]
    if DEBUG:
        print(f'len mask coinc trigger {len(mask_coinc_trigger)}')

    return mask_coinc_trigger, trigger_index

#Trigge mask for when multiple triggers occured simultaneously
def dual_trigger_mask(fin, *args, station = 'station_1'):
    trigger_names = np.array(fin.attrs['trigger_names'])

    coinc_trigger = np.array(fin['multiple_triggers'])
    dual_mask = np.ones(len(coinc_trigger), dtype=bool)
    for trigger in args:
        trigger_index = np.squeeze(np.argwhere(trigger_names == trigger))
        dual_mask &= coinc_trigger[:, trigger_index]

    return dual_mask


#This function returns an array of reflection type per shower at a particular antenna
#Number corresponds to number of bottom reflections from the ray tracing solution (top reflections not counted)
def reflection_types(fin, ant_num = 0, station = 'station_1', trigger_index = 1, DEBUG=False):
    multi_trigger_mask = np.array(fin[station]['multiple_triggers'])[:, trigger_index]
    max_amp_per_ray = fin[station]['max_amp_shower_and_ray'][:, ant_num, :]
    index_of_max_amplitude_per_event = np.argmax(max_amp_per_ray, axis=1)[multi_trigger_mask]
    reflection_type_per_ray = np.array(fin[station]['ray_tracing_reflection'])[:, ant_num, :][multi_trigger_mask]
    ray_trace = np.array(fin[station]['ray_tracing_reflection'])

    reflection_type = np.zeros(len(index_of_max_amplitude_per_event))
    for iShower in range(len(index_of_max_amplitude_per_event)):
        reflection_type[iShower] = reflection_type_per_ray[iShower, index_of_max_amplitude_per_event[iShower]]

    if DEBUG:
        print(f'REFL TYPES')
        print(f'using ant {ant_num} station {station} and trig ind {trigger_index}')
        print(f'len multi trigger mask {len(multi_trigger_mask)}')
        print(f'len max amp per ray before mask {len(max_amp_per_ray)}')
        print(f'len index {len(index_of_max_amplitude_per_event)}')
        print(f'len refl type per ray before mask {len(ray_trace)}')
        print(f'len refl_type {len(reflection_type)}')


    return reflection_type

#Function takes in antenna number for a file and returns 4 arrays that act as masks for reflected/direct above & below triggers
#Last array return has the z_arrival direction for each triggered signal
def dipole_direction_masks(fin, ant_num = 5 , station = 'station_1', trigger_index = 1, DEBUG=False):
    print(f'DIPOLE MASK')
    print(f'using ant {ant_num} station {station} and trig ind {trigger_index}')
    multi_trigger_mask = np.array(fin[station]['multiple_triggers'])[:, trigger_index]
    print(f'len mult trig mask {len(multi_trigger_mask)}')
    max_amp_per_ray = fin[station]['max_amp_shower_and_ray'][:, ant_num, :]
    print(f'len max amp ray {len(max_amp_per_ray)}')
    index_of_max_amplitude_per_event = np.argmax(max_amp_per_ray, axis=1)[multi_trigger_mask]
    print(f'len index {len(index_of_max_amplitude_per_event)}')
    receive_vecs = fin[station]['receive_vectors'][:, ant_num, :][multi_trigger_mask]
    print(f'len recieve vec {len(receive_vecs)}')
    num_events = len(receive_vecs)
    

    refl_abv_mask = np.zeros(num_events, dtype = np.bool)
    refl_bel_mask = np.zeros(num_events, dtype = np.bool)
    dir_abv_mask = np.zeros(num_events, dtype = np.bool)
    dir_bel_mask = np.zeros(num_events, dtype = np.bool)

    z_arrival = np.zeros(num_events)

    refl_mask = reflection_types(fin, ant_num, station, trigger_index).astype(np.int) == 1
    print(f'len refl mask {len(refl_mask)}')

#    print(f'Check : len index_mask_amp {len(index_of_max_amplitude_per_event)} len rec_vecs {len(receive_vecs)} len refl_mask {len(refl_mask)}')
    for iShower in range(num_events):
        z_arrival[iShower] = receive_vecs[iShower][index_of_max_amplitude_per_event][iShower][2]
        if refl_mask[iShower]:
            if z_arrival[iShower] > 0:
                refl_abv_mask[iShower] = True
            else:
                refl_bel_mask[iShower] = True
        else:
            if z_arrival[iShower] > 0:
                dir_abv_mask[iShower] = True
            else:
                dir_bel_mask[iShower] = True

    return refl_abv_mask, refl_bel_mask, dir_abv_mask, dir_bel_mask, z_arrival


#This function returns the launch vectors of reflected triggers for use to get viewing angle
def refl_launch_vectors(fin, ant_num = 5, station = 'station_1', trigger_index = 1):
    multi_trigger_mask = np.array(fin[station]['multiple_triggers'])[:, trigger_index]
    max_amp_per_ray = fin[station]['max_amp_shower_and_ray'][:, ant_num, :]
    index_of_max_amplitude_per_event = np.argmax(max_amp_per_ray, axis=1)[multi_trigger_mask]
    launch_vecs = fin[station]['launch_vectors'][:, ant_num, :][multi_trigger_mask]
    num_events = len(launch_vecs)
    

#    z_arrival = np.zeros(num_events)

#    refl_mask = reflection_types(fin, ant_num, station, trigger_index).astype(np.int) == 1
#    if refl == False:
#        refl_mask = ~refl_mask

    return_vecs = []
    for iShower in range(num_events):
        return_vecs.append(launch_vecs[iShower][index_of_max_amplitude_per_event[iShower]])
#    print(f'return vecs {return_vecs}')
#    print(f'shape return vecs {np.shape(return_vecs)}')
    return np.array(return_vecs)

    print(f'pre return shape launch vecs {np.shape(launch_vecs)}')
    print(f'max amp index array {np.shape(index_of_max_amplitude_per_event)}')
    print(f'max amp index array {index_of_max_amplitude_per_event}')
#    print(f'shape refl mask in launch vecs {np.shape(refl_mask)}')
    launch_vecs = np.array(launch_vecs[:,index_of_max_amplitude_per_event])
    print(f'changed launch vecs shape {np.shape(launch_vecs)}')
#    launch_vecs = np.array(launch_vecs[:, index_of_max_amplitude_per_event])[refl_mask]
#    launch_vecs = launch_vecs[:,index_of_max_amplitude_per_event][refl_mask]
    return launch_vecs
"""
#    print(f'Check : len index_mask_amp {len(index_of_max_amplitude_per_event)} len rec_vecs {len(receive_vecs)} len refl_mask {len(refl_mask)}')
    for iShower in range(num_events):
        z_arrival[iShower] = receive_vecs[iShower][index_of_max_amplitude_per_event][iShower][2]
        if refl_mask[iShower]:
            if z_arrival[iShower] > 0:
                refl_abv_mask[iShower] = True
            else:
                refl_bel_mask[iShower] = True
        else:
            if z_arrival[iShower] > 0:
                dir_abv_mask[iShower] = True
            else:
                dir_bel_mask[iShower] = True

    return refl_abv_mask, refl_bel_mask, dir_abv_mask, dir_bel_mask, z_arrival
"""

#This function returns the zenith as calculated from the triggered recieve vector
def receive_zenith(fin, ant_num = 5, station = 'station_1', trigger_index = 1):
    multi_trigger_mask = np.array(fin[station]['multiple_triggers'])[:, trigger_index]
    max_amp_per_ray = fin[station]['max_amp_shower_and_ray'][:, ant_num, :]
    index_of_max_amplitude_per_event = np.argmax(max_amp_per_ray, axis=1)[multi_trigger_mask]
    receive = fin[station]['receive_vectors'][:, ant_num, :][multi_trigger_mask]
    num_events = len(receive)
    
    return_zens = []
    for iShower in range(num_events):
        vec = receive[iShower][index_of_max_amplitude_per_event[iShower]]
#        return_zens.append(receive[iShower][index_of_max_amplitude_per_event[iShower]])
#        print(f'z of vec is {vec[2]}')
        return_zens.append(np.arctan(np.sqrt(vec[0]**2 + vec[1]**2)/vec[2]))
    return np.array(return_zens)


#This function returns the maximum amplitude of signal at triggered antenna averaged over antennas given
def triggered_max_amps(fin, ant_num = [5], station = 'station_1', trigger_index = 1):
    multi_trigger_mask = np.array(fin[station]['multiple_triggers'])[:, trigger_index]
    max_amp_per_ray = fin[station]['max_amp_shower_and_ray'][:, ant_num, :]
    index_of_max_amplitude_per_event = np.argmax(max_amp_per_ray, axis=1)[multi_trigger_mask]


    return_amps = []
    for ant in ant_num:
#        max_amps = fin[station]['maximum_amplitudes'][:, ant, :][multi_trigger_mask]
        max_amps = fin[station]['maximum_amplitudes'][:, ant][multi_trigger_mask]
        num_events = len(max_amps)

        """
        max_amps = np.array(max_amps)
        print(f'shape max amps {np.shape(max_amps)}')    
        add_amps = []
        for iShower in range(num_events):
            amp = max_amps[iShower][index_of_max_amplitude_per_event[iShower]]
            add_amps.append(amp)
        """
        if len(return_amps) == 0:
            return_amps = max_amps
        else:
            return_amps += max_amps

    return_amps = np.array(return_amps)
    return_amps = return_amps / len(ant_num)

#    print(f'amp found is {amp}, with 10Vrms have SNR {amp/10**-5}')
    return return_amps



#This function returns the mask for an array such as an LPDA
#It runs through each antenna and returns masks for directions as well as the average z arrival between all antennas
def multi_array_direction_masks(fin, ant_num, station = 'station_1', trigger_index = 1):
    refl_abv_dic = {}
    refl_bel_dic = {}
    dir_abv_dic = {}
    dir_bel_dic = {}
    z_vals = {}
    length = 0

    for ant in ant_num:
        refl_abv_dic[ant], refl_bel_dic[ant], dir_abv_dic[ant], dir_bel_dic[ant], z_vals[ant] = dipole_direction_masks(fin, ant, station, trigger_index)
        length = len(refl_abv_dic[ant])
        print(f'length for ant {ant} is {length}')
#        print(length)
    """
    refl_abv_mask = np.zeros(length, dtype=bool)
    refl_bel_mask = np.zeros(length, dtype=bool)
    dir_abv_mask = np.zeros(length, dtype=bool)
    dir_bel_mask = np.zeros(length, dtype=bool)
    """
    refl_abv_mask = np.zeros(length)
    refl_bel_mask = np.zeros(length)
    dir_abv_mask = np.zeros(length)
    dir_bel_mask = np.zeros(length)

    for ant in ant_num:
        refl_abv_mask += refl_abv_dic[ant].astype(int)
        refl_bel_mask += refl_bel_dic[ant].astype(int)
        dir_abv_mask += dir_abv_dic[ant].astype(int)
        dir_bel_mask += dir_bel_dic[ant].astype(int)
        """
        refl_abv_mask = refl_abv_mask | refl_abv_dic[ant]
        refl_bel_mask = refl_bel_mask | refl_bel_dic[ant]
        dir_abv_mask = dir_abv_mask | dir_abv_dic[ant]
        dir_bel_mask = dir_bel_mask | dir_bel_dic[ant]
        """
    refl_abv_mask = refl_abv_mask > (len(ant_num)/2)
    refl_bel_mask = refl_bel_mask > (len(ant_num)/2)
    dir_abv_mask = dir_abv_mask > (len(ant_num)/2)
    dir_bel_mask = dir_bel_mask > (len(ant_num)/2)

    z_array_vals = np.zeros(length)
    for zz in range(len(z_array_vals)):
        iZZ = np.empty(len(ant_num))
        for iAnt, ant in enumerate(ant_num):
            iZZ[iAnt] = z_vals[ant][zz]
        z_array_vals[zz] = np.nanmean(iZZ)

    return refl_abv_mask, refl_bel_mask, dir_abv_mask, dir_bel_mask, z_array_vals

#plots Radius vs Zenith distribution after being masked by reflected/direct and above/below
def plot_rad_zenith(rr, zen, refl_abv, refl_bel, dir_abv, dir_bel):
    rr_refl_abv = rr[refl_abv]
    rr_refl_bel = rr[refl_bel]
    rr_dir_abv = rr[dir_abv]
    rr_dir_bel = rr[dir_bel]

    zen_refl_abv = zen[refl_abv]
    zen_refl_bel = zen[refl_bel]
    zen_dir_abv = zen[dir_abv]
    zen_dir_bel = zen[dir_bel]

    RAplt = plt.scatter(rr_refl_abv, zen_refl_abv)
    RBplt = plt.scatter(rr_refl_bel, zen_refl_bel)
    DAplt = plt.scatter(rr_dir_abv, zen_dir_abv)
    DBplt = plt.scatter(rr_dir_bel, zen_dir_bel)

    plt.xlabel('Radius (m)')
    plt.ylabel('Zenith (rad)')
    plt.title('Radius vs Zenith for Reflected/Direct Signals from Above & Below')
    plt.legend((RAplt, RBplt, DAplt, DBplt), (
                        'ReflAbv ' + str(sum(refl_abv)), 
                        'ReflBel '+ str(sum(refl_bel)),
                        'DirAbv ' + str(sum(dir_abv)),
                        'DirBel' + str(sum(dir_bel))))

    plt.show()
    return

def plot_LPDA_PA_Amps(fin, depth, dB, energy, LPDA_ant=[0,1,2,3], PA_ant=9, station='station_1', trigger_index=1):
    refl_mask = reflection_types(fin, LPDA_ant[0], station, trigger_index).astype(np.int) == 1

    multi_trigger_mask = np.array(fin[station]['multiple_triggers'])[:, trigger_index]
    print(multi_trigger_mask)
    print('that was multi trig mask')
    
    PA_max_amp_per_ray = fin[station]['max_amp_shower_and_ray'][:, PA_ant, :]
    PA_index_of_max_amplitude_per_event = np.argmax(PA_max_amp_per_ray, axis=1)[multi_trigger_mask]
    PA_max_amps = np.zeros(len(PA_index_of_max_amplitude_per_event))
    for ray in range(len(PA_index_of_max_amplitude_per_event)):
        PA_max_amps[ray] = PA_max_amp_per_ray[ray][PA_index_of_max_amplitude_per_event[ray]]
    

    LPDA_max_amps = np.zeros(len(PA_max_amps))
    for i_ant in range(len(LPDA_ant)):
        LPDA_max_amp_per_ray = fin[station]['max_amp_shower_and_ray'][:, LPDA_ant[i_ant],:][multi_trigger_mask]
        LPDA_index_of_max_amplitude_per_event = np.argmax(LPDA_max_amp_per_ray, axis=1)
        print(f'{len(LPDA_max_amps)} {len(LPDA_max_amp_per_ray)} {len(LPDA_index_of_max_amplitude_per_event)}')
        for ray in range(len(LPDA_max_amp_per_ray)):
            LPDA_max_amps[ray] += LPDA_max_amp_per_ray[ray][LPDA_index_of_max_amplitude_per_event[ray]]
    
    LPDA_max_amps = LPDA_max_amps/len(LPDA_ant)
#    refl_mask = reflection_types(fin, PA_ant, station, trigger_index).astype(np.int) == 1
    print(f'Check: len PA amps {len(PA_max_amps)} len LPDA amps {len(LPDA_max_amps)} len refl_mask {len(refl_mask)}')
#    shower_type = fin['shower_type'][:]
#    print(f'len shower type {len(shower_type)}')
#    print(shower_type)
#    print(f'len masked shower type {len(shower_type[multi_trigger_mask])}')
#    LPDA_x = np.array(LPDA_max_amps[refl_mask])
#    PA_y = np.array(PA_max_amps[refl_mask])
#    print(f'{len(LPDA_x)} {len(PA_y)}')
#    print(LPDA_x.shape)
#    print(PA_y.shape)
    evnt_group_ids = np.array(fin[station]['event_group_ids'][multi_trigger_mask])
    print(evnt_group_ids)
    
    print(f'len max_amps {len(evnt_group_ids)} len max_amps {len(LPDA_max_amps)}')

    group_event_id_diagnostic(fin, evnt_group_ids[np.argmax(LPDA_max_amps)], trigger = 'LPDA_2of4_100Hz', station='station_1')

    PA_LPDA_amp_upward_scatter = plt.scatter(LPDA_max_amps[refl_mask], PA_max_amps[refl_mask])
    plt.xlabel('LPDA Amplitude (V)')
    plt.ylabel('Dipole Amplitude (V)')
    plt.yscale('log')
    plt.xscale('log')
#    plt.xlim(0.5*min(LPDA_max_amps[refl_mask]), 2*max(LPDA_max_amps[refl_mask]))    
#    plt.ylim(0.5*min(PA_max_amps[refl_mask]), 2*max(PA_max_amps[refl_mask]))
    plt.xlim(10**-8, 10**-2)    
    plt.ylim(10**-8, 10**-2)
    plt.title('Amplitude of reflected signals at LPDA vs -18m Dipole for -' + depth + 'm -' + dB + 'dB ' + energy + '*10^18 eV')
    n_events = sum(refl_mask)
    plt.legend([str(n_events) + ' events'])
    return PA_LPDA_amp_upward_scatter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze hdf5 NuRadioMC output')
    parser.add_argument('--filename', type=str, default='results/NuMC_output.hdf5',
                        help='path to NuRadioMC simulation output')
#    parser.add_argument('--antenna_num', type=str, default='5', help='Antenna number to look at, defaults to 5')
#    parser.add_argument('--trigger', type=str, default='dipole', help='Trigger type to run on, dipole, LPDA, or both')
    parser.add_argument('--depth', type=str, default=None, help='Layer depth for plot names')
    parser.add_argument('--dB', type=str, default=None, help='dB of layer for plot names')
    parser.add_argument('--energy', type=str, default=None, help='Energy of sim for plot names')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode on/off, True or False, default False')
    args = parser.parse_args()

    filename = args.filename
#    trigger_type = args.trigger
    depth = args.depth
    dB = args.dB   
    energy = args.energy
    debug = args.debug

    fin = h5py.File(filename, 'r')

    dual_mask = dual_trigger_mask(fin, 'single_dipole_trigger_2sig', 'LPDA_2of4_100Hz')
    dipole_trigger_mask, dipole_index = trigger_mask(fin, trigger = 'single_dipole_trigger_2sig')
    LPDA_trigger_mask, LPDA_index = trigger_mask(fin, trigger = 'LPDA_2of4_100Hz')

    print(f'Fraction of dual reflection triggers {np.sum(dual_mask)/len(dual_mask):.4f}')

    xx = np.array(fin['xx'])
    yy = np.array(fin['yy'])
    zz = np.array(fin['zz'])
    zeniths = np.array(fin['zeniths'])
    weights = np.array(fin['weights'])

    print(weights)
    print(f'weights above, len {len(weights)} and xx {len(xx)}')

    xx_dip = xx[dipole_trigger_mask]
    yy_dip = yy[dipole_trigger_mask]
    zz_dip = zz[dipole_trigger_mask]
    zen_dip = zeniths[dipole_trigger_mask]
    rr_dip = (xx_dip **2 + yy_dip ** 2) ** 0.5

    xx_lpda = xx[LPDA_trigger_mask]
    yy_lpda = yy[LPDA_trigger_mask]
    zz_lpda = zz[LPDA_trigger_mask]
    zen_lpda = zeniths[LPDA_trigger_mask]
    rr_lpda = (xx_lpda ** 2 + yy_lpda ** 2) ** 0.5

    dip_refl_abv, dip_refl_bel, dip_dir_abv, dip_dir_bel, dip_z_arrival = dipole_direction_masks(fin, ant_num=13, trigger_index=1)
    LPDA_refl_abv, LPDA_refl_bel, LPDA_dir_abv, LPDA_dir_bel, LPDA_z_arrival = multi_array_direction_masks(fin, ant_num=[0,1,2,3],trigger_index=LPDA_index)

    """
    #Weighted 2d Histogram
#    plt.hist2d(rr_lpda[LPDA_refl_bel], zz_lpda[LPDA_refl_bel], bins=50, weights=weights[LPDA_trigger_mask][LPDA_refl_bel])
    plt.hist2d(rr_dip, zz_dip, bins=50, weights=weights[dipole_trigger_mask])
#    plt.title('Radius and Depths for LPDA Reflected Below Signals of -' + depth + 'm -' + dB + 'dB ' + energy + 'EeV')
    plt.title('Radius and Depths for Dipole Direct Signals with no Reflection Layer for ' + energy + 'EeV and max radius 11km')
    plt.ylabel('Depth (m)')
    plt.xlabel('Radius (m)')
    plt.show()

    plt.hist2d(rr_lpda[LPDA_refl_abv], zz_lpda[LPDA_refl_abv], bins=50, weights=weights[LPDA_trigger_mask][LPDA_refl_abv])
    plt.title('Radius and Depths for LPDA Reflected Above Signals of -' + depth + 'm -' + dB + 'dB ' + energy + 'EeV')
    plt.ylabel('Depth (m)')
    plt.xlabel('Radius (m)')
    plt.show()

    quit()
    """

    #First going to plot upward detected signals for LPDA and Dipole
    LPDA_refl_bel_plt = plt.scatter(rr_lpda[LPDA_refl_bel], zen_lpda[LPDA_refl_bel])
    dipole_refl_bel_plt = plt.scatter(rr_dip[dip_refl_bel], zen_dip[dip_refl_bel])
    dipole_dir_bel_plt = plt.scatter(rr_dip[dip_dir_bel], zen_dip[dip_dir_bel])
    LPDA_dir_bel_plt = plt.scatter(rr_lpda[LPDA_dir_bel], zen_lpda[LPDA_dir_bel])

    plt.xlabel('Radius (m)')
    plt.ylabel('Zenith (rad)')
    plt.title('Radius vs Zenith of Upward Signals for Dipole & LPDA for -' + depth + 'm -' + dB + 'dB ' + energy + 'EeV')
    plt.legend((dipole_refl_bel_plt, dipole_dir_bel_plt, LPDA_refl_bel_plt, LPDA_dir_bel_plt), ('Dipole Reflected ' + str(sum(dip_refl_bel)), 'Dipole Direct ' + str(sum(dip_dir_bel)), 'LPDA Reflected ' + str(sum(LPDA_refl_bel)), 'LPDA Direct ' + str(sum(LPDA_dir_bel))))
    plt.show()

    #Now going to plot the shower vs signal arrival zeniths for LPDA/dipole

    sig_shower_lpda_refl_bel = plt.scatter(zen_lpda[LPDA_refl_bel], np.arccos(LPDA_z_arrival[LPDA_refl_bel]))
    sig_shower_dip_relf_bel = plt.scatter(zen_dip[dip_refl_bel], np.arccos(dip_z_arrival[dip_refl_bel]))
    sig_shower_dip_dir_bel = plt.scatter(zen_dip[dip_dir_bel], np.arccos(dip_z_arrival[dip_dir_bel]))
    sig_shower_lpda_dir_bel = plt.scatter(zen_lpda[LPDA_dir_bel], np.arccos(LPDA_z_arrival[LPDA_dir_bel]))

    plt.xlabel('CR Shower Arrival Zenith (rad)')
    plt.ylabel('Signal Arrival Angle (rad')
    plt.title('CR Arrival direction and station signal direction at -' + depth + 'm -' + dB + 'dB ' + energy + 'EeV')
    plt.legend((sig_shower_dip_relf_bel, sig_shower_dip_dir_bel, sig_shower_lpda_refl_bel, sig_shower_lpda_dir_bel), ('Dipole Reflected ' + str(sum(dip_refl_bel)), 'Dipole Direct ' + str(sum(dip_dir_bel)), 'LPDA Reflected ' + str(sum(LPDA_refl_bel)), 'LPDA Direct ' + str(sum(LPDA_dir_bel))))
    plt.show()

    #plotting fraction detected vs Cos(theta) for both now

    dip_fractions = np.zeros(10)
    lpda_fractions = np.zeros(10)
    cosVals = np.zeros(10)
    thetaVals = np.radians(np.array([60, 56.63298703, 53.13010235, 49.45839813, 45.572996 , 41.40962211, 36.86989765, 31.78833062, 25.84193276, 18.19487234, 0.]))

    print('Cos|Dip|LPDA')
    for i in range(10):
        dip_fractions[i] = ((thetaVals[i] > zen_dip[dip_refl_bel]) & (zen_dip[dip_refl_bel] > thetaVals[i+1])).sum()
        lpda_fractions[i] = ((thetaVals[i] > zen_lpda[LPDA_refl_bel]) & (zen_lpda[LPDA_refl_bel] > thetaVals[i+1])).sum()        
        cosVals[i] = (np.cos(thetaVals[i]) + np.cos(thetaVals[i+1]))/2
        print(cosVals[i], dip_fractions[i], lpda_fractions[i])

#    dip_frac_plt = plt.scatter(cosVals, dip_fractions/10000, label='Dipole')
    dip_err = plt.errorbar(cosVals, dip_fractions/10000, yerr=(np.sqrt(dip_fractions)/10000), label='Dipole')
#    lpda_frac_plt = plt.scatter(cosVals, lpda_fractions/10000, label='LPDA')
    lpda_err = plt.errorbar(cosVals, lpda_fractions/10000, yerr=(np.sqrt(lpda_fractions)/10000), label='LPDA')
    
    plt.xlabel('Cos[theta]')
    plt.ylabel('fraction of upward reflected detections per angular bin')
    plt.legend((dip_err, lpda_err), ('Dipole', 'LPDA'))
    plt.title('Hit fractions per bin of Dipole and LPDA for -' + depth + 'm -' + dB + 'dB ' + energy + 'EeV')
    plt.show()

    LPDA_PA_plt = plot_LPDA_PA_Amps(fin, depth, dB, energy, [0,1,2,3], 9)
    plt.show()

    print('random reflection test')
    refl_mask_0 = reflection_types(fin, 0, 'station_1', LPDA_index).astype(np.int) == 1                                                                           
    refl_mask_1 = reflection_types(fin, 1, 'station_1', LPDA_index).astype(np.int) == 1                                                                           
    refl_mask_2 = reflection_types(fin, 2, 'station_1', LPDA_index).astype(np.int) == 1                                                                           
    refl_mask_3 = reflection_types(fin, 3, 'station_1', LPDA_index).astype(np.int) == 1                                                                           
    
    print(f'0 {len(refl_mask_0)} 1 {len(refl_mask_1)} 2 {len(refl_mask_2)} 3 {len(refl_mask_3)}')
