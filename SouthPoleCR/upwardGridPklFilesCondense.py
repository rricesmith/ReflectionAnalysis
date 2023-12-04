import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

type = 'CR'
core = 100
spacing = 1

low = 0
high = 100

n = np.array([])
n_trig_up = np.array([])
n_trig = np.array([])
n_trig_att = np.array([])
energy = np.array([])
ss_energy = np.array([])
zenith = np.array([])
azimuth = np.array([])

condensed_output = {}

while high < 2801:
    with open(f'../output/detection_efficiency_{type}_{core}_{spacing:.2f}km_26muV_id{low}_{high}.pkl', 'rb') as fin:
        output = pickle.load(fin)

    for runid in output:
        print(runid)
        
        if (runid not in condensed_output):
            condensed_output[runid] = {}
            condensed_output[runid]['n'] = output[runid]['n']
            condensed_output[runid]['n_triggered'] = output[runid]['n_triggered']
            condensed_output[runid]['n_triggered_upward'] = output[runid]['n_triggered_upward']
            condensed_output[runid]['n_att_triggered'] = output[runid]['n_att_triggered']
            condensed_output[runid]['energy'] = output[runid]['energy']
            condensed_output[runid]['ss_energy'] = output[runid]['ss_energy']
            condensed_output[runid]['zenith'] = output[runid]['zenith']
            condensed_output[runid]['azimuth'] = output[runid]['azimuth']
        

#        for run in runid:
        n = np.append(n, [output[runid]['n']])
        n_trig_up = np.append(n_trig_up, [output[runid]['n_triggered_upward']])
        n_trig = np.append(n_trig, [output[runid]['n_triggered']])
        n_trig_att = np.append(n_trig_att, [output[runid]['n_att_triggered']])
        energy = np.append(energy, [output[runid]['energy']])
        ss_energy = np.append(ss_energy, [output[runid]['ss_energy']])
        zenith = np.append(zenith, [output[runid]['zenith']])
        azimuth = np.append(azimuth, [output[runid]['azimuth']])

    high += 100
    low += 100
    if low == 100:
        high += 100
        low += 100

plt.scatter(np.log10(energy), np.log10(ss_energy))
plt.show()



with open(f'data/upwardGridData_{type}_cores{core}_{spacing:.2f}km.pkl', 'wb') as fout:
    pickle.dump(condensed_output, fout)
