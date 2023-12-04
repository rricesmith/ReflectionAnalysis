from NuRadioMC.utilities.plotting import plot_vertex_distribution
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


#This function returns a mask of all events corresponding to those that triggered on the given trigger
def trigger_mask(fin, trigger = 'LPDA_2of4_100Hz', station = 'station_1'):
        trigger_names = np.array(fin.attrs['trigger_names'])
        trigger_index = np.squeeze(np.argwhere(trigger_names == trigger))

        mask_coinc_trigger = np.array(fin['multiple_triggers'])[:, trigger_index]
#        print(f'mask coinc trigger len {len(mask_coinc_trigger)}')
        return mask_coinc_trigger


#This function returns an array of reflection type per shower at a particular antenna
#Number corresponds to number of bottom reflections from the ray tracing solution (top reflections not counted)
def reflection_types(fin, ant_num = 5, station = 'station_1', trigger_index = 1):
#        multi_trigger_mask = trigger_mask(fin)
        multi_trigger_mask = np.array(fin[station]['multiple_triggers'])[:, trigger_index]
#        print(f'len multi trigger mask in reflection_types {len(multi_trigger_mask)}')
        max_amp_per_ray = fin[station]['max_amp_shower_and_ray'][:, ant_num, :]
#        launch_vecs = np.array(fin[station]['launch_vectors'])
        print(f'len max amp per ray before mask {len(max_amp_per_ray)}')
        index_of_max_amplitude_per_event = np.argmax(max_amp_per_ray, axis=1)[multi_trigger_mask]
        reflection_type_per_ray = np.array(fin[station]['ray_tracing_reflection'])[:, ant_num, :][multi_trigger_mask]


#        triggered = np.array(fin[station]['multiple_triggers'])
#        print(triggered[:,0])
#        print(triggered[:,1])
#        print(f'len triggered {len(triggered)}')
#        print(f'sum triggered 0 {np.sum(triggered[:,0])}')
#        print(f'sum triggered 1 {np.sum(triggered[:,1])}')

#        trig_mask = trigger_mask(fin)
#        print(f'len trigger mask in refl_types {len(trig_mask)}')
#        print(f'sum trig mask in refl_types {np.sum(trig_mask)}')


#        multi_trigger_mask = np.array(fin[station]['multiple_triggers'])[:, 1]

        reflection_type = np.zeros(len(index_of_max_amplitude_per_event))
        for iShower in range(len(index_of_max_amplitude_per_event)):
                reflection_type[iShower] = reflection_type_per_ray[iShower, index_of_max_amplitude_per_event[iShower]]

#        return reflection_type[multi_trigger_mask]
        return reflection_type

#Function takes in antenna number for a file and returns 4 arrays that act as masks for reflected/direct above & below triggers
#In this case, reflected means bottom-reflections are >0
#Last array return has the z_arrival direction for each triggered signal
def dipole_direction_masks(fin, ant_num = 5 , station = 'station_1', trigger_index = 1):
#        multi_trigger_mask = trigger_mask(fin)
        multi_trigger_mask = np.array(fin[station]['multiple_triggers'])[:, trigger_index]
        max_amp_per_ray = fin[station]['max_amp_shower_and_ray'][:, ant_num, :]
        index_of_max_amplitude_per_event = np.argmax(max_amp_per_ray, axis=1)[multi_trigger_mask]
        receive_vecs = fin[station]['receive_vectors'][:, ant_num, :][multi_trigger_mask]
        num_events = len(receive_vecs)
        
        refl_abv_mask = np.zeros(num_events, dtype = np.bool)
        refl_bel_mask = np.zeros(num_events, dtype = np.bool)
        dir_abv_mask = np.zeros(num_events, dtype = np.bool)
        dir_bel_mask = np.zeros(num_events, dtype = np.bool)

        z_arrival = np.zeros(num_events)

        refl_mask = reflection_types(fin, ant_num, station, trigger_index).astype(np.int) == 1

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
                print(length)

        refl_abv_mask = np.zeros(length, dtype=bool)
        refl_bel_mask = np.zeros(length, dtype=bool)
        dir_abv_mask = np.zeros(length, dtype=bool)
        dir_bel_mask = np.zeros(length, dtype=bool)
        for ant in ant_num:
                refl_abv_mask = refl_abv_mask | refl_abv_dic[ant]
                refl_bel_mask = refl_bel_mask | refl_bel_dic[ant]
                dir_abv_mask = dir_abv_mask | dir_abv_dic[ant]
                dir_bel_mask = dir_bel_mask | dir_bel_dic[ant]
        
        z_array_vals = np.zeros(length)
        for zz in range(len(z_array_vals)):
                iZZ = np.empty(len(ant_num))
                for ant in ant_num:
                        iZZ[ant] = z_vals[ant][zz]
                z_array_vals[zz] = np.nanmean(iZZ)

        return refl_abv_mask, refl_bel_mask, dir_abv_mask, dir_bel_mask, z_array_vals


"""
        receive_vecs = {}
        max_amp_per_ray = {}
        index_of_max_amplitude_per_event = {}
        leng = 0
        for ant in ant_num:
                receive_vecs[ant] = fin[station]['recieve_vectors'][:, ant, :]
                max_amp_per_ray[ant] = fin[station]['max_amp_shower_and_ray'][:, ant, :]
                index_of_max_amplitude_per_event[ant] = np.argmax(max_amp_per_ray, axis=1)
                leng = len(index_of_max_amplitude_per_event[ant])

        refl_abv_mask = np.zeros(leng, dtype = np.bool)
        refl_bel_mask = np.zeros(leng, dtype = np.bool)
        dir_abv_mask = np.zeros(leng, dtype = np.bool)
        dir_bel_mask = np.zeros(leng, dtype = np.bool)

        z_arrival = np.zeros(leng)

        for iShower in range(leng):
                z_vals = np.empty(len(ant_num))
                for ant in ant_num:
                        zz[ant] = receive_vecs[ant][iShower][index_of_max_amplitude_per_event[ant]][iShower][2]

                z_arrival[iShower] = np.nanmean(z_vals)
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
"""
        
#Arguments Parsing

parser = argparse.ArgumentParser(description='Check NuRadioMC output')
parser.add_argument('--filename', type=str, default='results/NuMC_output.hdf5',
                    help='path to NuRadioMC simulation output')
parser.add_argument('--antenna_num', type=str, default='5', help='Antenna number to look at, defaults to 5')
parser.add_argument('--trigger', type=str, default='dipole', help='Trigger type to run on, dipole or LPDA')
parser.add_argument('--debug', type=bool, default=False, help='Debug mode on/off, True or False, default False')
args = parser.parse_args()

filename = args.filename
i_antenna = int(args.antenna_num)
trigger_type = args.trigger
debug = args.debug

trig_val = 0
if trigger_type == 'dipole':
        trig_val = 1

#Open the HDF5 file
fin = h5py.File(filename, 'r')


########################################
#Going to get number of reflections 
reflection_type = reflection_types(fin, ant_num = i_antenna, trigger_index=trig_val)

#Now we have array that has one number per event, 1 if reflection 0 if direct
print(f"fraction of reflected triggers{np.sum(reflection_type==1)/len(reflection_type):.4f}")
print(f"fraction of direct triggers{np.sum(reflection_type==0.)/len(reflection_type):.4f}")

#We can print the list of triggers for this detector
if debug:
        print('This is the list of used triggers for this detector')
        trigger_names = np.array(fin.attrs['trigger_names'])
        print(trigger_names)


#Now we get the index corresponding to the trigger we want

if trig_val == 1:
        chosen_trigger = 'single_dipole_trigger_2sig'
        refl_abv_mask, refl_bel_mask, dir_abv_mask, dir_bel_mask, z_arrival = dipole_direction_masks(fin, ant_num=13, trigger_index=trig_val)
else:
        chosen_trigger = 'LPDA_2of4_100Hz'
        refl_abv_mask, refl_bel_mask, dir_abv_mask, dir_bel_mask, z_arrival = multi_array_direction_masks(fin, [0,1,2,3], trigger_index=trig_val)

        
mask_coinc_trigger = trigger_mask(fin, trigger = chosen_trigger)

#Sanity check on having the number of reflection types equaling the sum of our mask, if it fails something is wrong
#Sanity check is wrong. refl_type length is for all events, including those not on given triger, mask_coinc sum is just for number on that trigger
#if len(reflection_type) != sum(mask_coinc_trigger):
#        print('ERROR: Reflection type length not equal to the number of triggers')
#        print(len(reflection_type))
#        print(sum(mask_coinc_trigger))
#        quit()
if debug:
        print(f'len refl type {len(reflection_type)}')
        print(f'len mask coinc trigger {len(mask_coinc_trigger)}')
        print(f'sum mask coinc trigger {np.sum(mask_coinc_trigger)}')
#mask_coinc_trigger2 = fin['shower_type'][:] == 'had'  
#print(len(mask_coinc_trigger2))

        test = np.array(fin['multiple_triggers'])
#print(test)
        print(f'len fin multiple triggers {len(test)}')
#print(list(np.array(fin['station_1']['event_group_ids'])))
#print(len(fin['station_1']['event_group_ids']))

#Get x, y, and z's of our triggered events
xx = np.array(fin['xx'])[mask_coinc_trigger]
yy = np.array(fin['yy'])[mask_coinc_trigger]
zz = np.array(fin['zz'])[mask_coinc_trigger]

weights = np.array(fin['weights'])[mask_coinc_trigger]

if debug:
        xx_norm = np.array(fin['xx'])
        print(f'len xx without mask {len(xx_norm)}')

#Get Radii and zenith angles
rr = (xx ** 2 + yy ** 2) ** 0.5
zeniths = np.array(fin['zeniths'])[mask_coinc_trigger]

if debug:
        print(f'len rr {len(rr)}')

#Turn reflection types into masks for reflected vs direct signals, then use them
refl_mask = reflection_type.astype(np.int) == 1
direct_mask = reflection_type.astype(np.int) == 0

"""
rr_refl = rr[refl_mask]
rr_direct = rr[direct_mask]
zen_refl = zeniths[refl_mask]
zen_direct = zeniths[direct_mask]
"""

#Getting arrival masks and vectors

#method for dipole/single antenna (change antenna num appropriately)
#refl_abv_mask, refl_bel_mask, dir_abv_mask, dir_bel_mask, z_arrival = dipole_direction_masks(fin, ant_num=13)
#method for array such as LPDA
#refl_abv_mask, refl_bel_mask, dir_abv_mask, dir_bel_mask, z_arrival = multi_array_direction_masks(fin, [0,1,2,3])


print('Reflected above ' + str(sum(refl_abv_mask)))
print('Reflected below ' + str(sum(refl_bel_mask)))
print('Direct above ' + str(sum(dir_abv_mask)))
print('Direct below ' + str(sum(dir_bel_mask)))


#plot of the above vs below and direct vs refl rad and zeniths

rr_refl_abv = rr[refl_abv_mask]
rr_refl_bel = rr[refl_bel_mask]
rr_dir_abv = rr[dir_abv_mask]
rr_dir_bel = rr[dir_bel_mask]
zen_refl_abv = zeniths[refl_abv_mask]
zen_refl_bel = zeniths[refl_bel_mask]
zen_dir_abv = zeniths[dir_abv_mask]
zen_dir_bel = zeniths[dir_bel_mask]

refl_abv = plt.scatter(rr_refl_abv, zen_refl_abv)
refl_bel = plt.scatter(rr_refl_bel, zen_refl_bel)
dir_abv = plt.scatter(rr_dir_abv, zen_dir_abv)
dir_bel = plt.scatter(rr_dir_bel, zen_dir_bel)

plt.xlabel('Radius (m)')
plt.ylabel('Zenith (rad)')
plt.title('Radius vs Zenith Distribution Separated')
plt.legend((refl_abv, refl_bel, dir_abv, dir_bel), ('N Refl Abv ' + str(sum(refl_abv_mask)), 'N Refl Bel ' + str(sum(refl_bel_mask)), 'N Dir Abv ' + str(sum(dir_abv_mask)), 'N Dir Bel ' +str(sum(dir_bel_mask))), loc='lower right')
plt.show()

sum_refl_bel = 0
sum_refl_abv = 0
sum_dir_abv = 0
for i in range(len(zen_refl_bel)):
    if zen_refl_bel[i] < 0.52:
        sum_refl_bel += 1

for i in range(len(zen_refl_abv)):
    if zen_refl_abv[i] < 0.52:
        sum_refl_abv += 1

for i in range(len(zen_dir_abv)):
    if zen_dir_abv[i] < 0.52:
        sum_dir_abv += 1

print('Num refl above ' + str(sum_refl_abv))
print('Num refl below ' + str(sum_refl_bel))
print('Num dir abvove ' + str(sum_dir_abv))


###
#plot of max amps
###


"""
max_amps = np.zeros(len(index_of_max_amplitude_per_event))
for num in range(len(max_amps)):
    max_amps[num] = max_amp_per_ray[num][index_of_max_amplitude_per_event[num]]

amps_refl_abv = max_amps[refl_abv_mask]
amps_refl_bel = max_amps[refl_bel_mask]
amps_dir_abv = max_amps[dir_abv_mask] 

#plt.scatter(rr[dir_abv_mask], amps_dir_abv)
#ax.set_yscale('log')
#plt.set_yscale('log')
plt.hist(amps_dir_abv, bins = [0,0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.01, 0.02, 0.05, 0.1, 1])
num_high = 0
for i in range(len(amps_dir_abv)):
    if amps_dir_abv[i] > 0.001:
        num_high += 1
plt.legend(['Num above 0.001: ' + str(num_high)])
plt.xlabel('Voltage (V)')
plt.ylabel('Num Counts')
plt.title('Voltage of Direct Above Signals')
plt.show()

avg_amps = 0
for i in range(len(amps_refl_bel)):
    avg_amps += amps_refl_bel[i]

avg_amps = avg_amps/len(amps_refl_bel)

ara = plt.scatter(rr[refl_abv_mask], amps_refl_abv)
arb = plt.scatter(rr[refl_bel_mask], amps_refl_bel)

plt.xlabel('Radius (m)')
plt.ylabel('Voltage (V)')
plt.title('Voltage of Reflected Detections from Above and Below, Avg Relf Bel Amps: ' + str(avg_amps))
plt.legend((ara, arb), ('Above Reflected Detections', 'Below Directed Detections'), loc='upper right')
plt.show()
"""


###
#Used to do below, now radius of shower vs arrival zenith
#Plot distance to vertex vs arrival direction zenith angle
###


z_refl_abv = z_arrival[refl_abv_mask]
z_refl_bel = z_arrival[refl_bel_mask]
z_dir_abv = z_arrival[dir_abv_mask]


#refl_abv_vert_arriv = plt.scatter(200 + rr_refl_abv, np.arccos(z_refl_abv))
refl_abv_vert_arriv = plt.scatter(rr_refl_abv, np.arccos(z_refl_abv))
#refl_bel_vert_arriv = plt.scatter(1600 - 200 + rr_refl_bel, np.arccos(z_refl_bel))
refl_bel_vert_arriv = plt.scatter(rr_refl_bel, np.arccos(z_refl_bel))
#dir_abv_vert_arriv = plt.scatter(200 + rr_dir_abv, np.arccos(z_dir_abv))
dir_abv_vert_arriv = plt.scatter(rr_dir_abv, np.arccos(z_dir_abv))

plt.xlabel('Shower Radius (m)')
plt.ylabel('Arrival Zenith Angle (rad)')
plt.title('Shower Radius versus Arrive Zenith of Detected Events')
plt.legend((refl_abv_vert_arriv, refl_bel_vert_arriv, dir_abv_vert_arriv), ('N Refl Abv ' + str(sum(refl_abv_mask)), 'N Refl Bel ' + str(sum(refl_bel_mask)), 'N Dir Abv ' + str(sum(dir_abv_mask))), loc='lower right')
plt.show()

###
#Now a plot of CR arrival direction vs signal detection direction
###

#Use above plot as example of signal arrival directions

refl_abv_CR_station = plt.scatter(zen_refl_abv, np.arccos(z_refl_abv))
refl_bel_CR_station = plt.scatter(zen_refl_bel, np.arccos(z_refl_bel))
dir_abv_CR_station = plt.scatter(zen_dir_abv, np.arccos(z_dir_abv))

plt.xlabel('CR Shower Arrival Zenith (rad)')
plt.ylabel('Signal Arrival Angle (rad)')
plt.title('CR Arrival Direction vs Station Signal Direction')
plt.legend((refl_abv_CR_station, refl_bel_CR_station, dir_abv_CR_station), ('N Refl Abv ' + str(sum(refl_abv_mask)), 'N Refl Bel ' + str(sum(refl_bel_mask)), 'N Dir Abv ' + str(sum(dir_abv_mask))), loc='lower right') 
plt.show()

###
#Plot fraction detected vs Cos(theta) below, 10 bins equally spaces in cos(theta)
###

fractions = np.zeros(10)
thetaVals = np.radians(np.array([60, 56.63298703, 53.13010235, 49.45839813, 45.572996 , 41.40962211, 36.86989765, 31.78833062, 25.84193276, 18.19487234, 0.]))
cosVals = np.zeros(10)

for i in range(len(fractions)):
    fractions[i] = ((thetaVals[i] > zen_refl_bel) & (zen_refl_bel > thetaVals[i+1])).sum()
    cosVals[i] = (np.cos(thetaVals[i]) + np.cos(thetaVals[i+1]))/2
    print(cosVals[i], fractions[i])

#fractions = fractions/10000

plt.scatter(cosVals, fractions/10000)
plt.errorbar(cosVals, fractions/10000, yerr=np.sqrt(fractions)/10000)
plt.xlabel('Cos[theta]')
plt.ylabel('fraction: upward reflected detections/throws per bin')
plt.show()

#plt.hist(np.cos(zen_refl_bel), np.cos(thetaVals))
#plt.show()

