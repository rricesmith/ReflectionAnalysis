import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

type = 'CR'
core = 100
spacing = 1

n = []
n_trig_up = []
n_trig = []
n_trig_att = []
energy = []
ss_energy = []
zenith = []
azimuth = []

condensed_output = {}

#with open(f'data/upwardGridData_{type}_cores{core}_{spacing:.2f}km.pkl', 'rb') as fin:
with open(f'../output/detection_efficiency_noHighEM_CR_100_1.00km_26muV_id0_0.pkl', 'rb') as fin:
    output = pickle.load(fin)

for runid in output:
    """
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
    """

#        for run in runid:
    n.append(output[runid]['n'])
    n_trig_up.append(output[runid]['n_triggered_upward'])
    n_trig.append(output[runid]['n_triggered'])
    n_trig_att.append(output[runid]['n_att_triggered'])
    energy.append(output[runid]['energy'])
    ss_energy.append(output[runid]['ss_energy'])
    zenith.append(output[runid]['zenith'])
    azimuth.append(output[runid]['azimuth'])


logE = np.log10(energy)
degree=1

"""
plt.scatter(logE, n_trig_up, label=('Trig up'))
plt.plot(np.unique(logE), np.poly1d(np.polyfit(logE, n_trig_up, degree))(np.unique(logE)), label='Up fit')
#plt.scatter(logE, n_trig, label=('Trig down'))
#plt.plot(np.unique(logE), np.poly1d(np.polyfit(logE, n_trig, degree))(np.unique(logE)), label='Down fit')
#plt.scatter(logE, n_trig_att, label=('Trig att'))
#plt.plot(np.unique(logE), np.poly1d(np.polyfit(logE, n_trig_att, degree))(np.unique(logE)), label='Att fit')
plt.legend()
plt.show()
"""

N_E, eng_edges = np.histogram(logE, 20)
bin_cor = np.digitize(logE, eng_edges)
eng_mean = [0.5 * (eng_edges[i] + eng_edges[i+1]) for i in range(len(N_E))]


dCos = 0.1
coszen_bin_edges = np.arange(np.cos(np.deg2rad(90)), max(zenith), dCos)
coszens = 0.5 * (coszen_bin_edges[1:] + coszen_bin_edges[:-1])

zen_n = np.zeros((len(coszens),len(N_E)))
zen_trig = np.zeros_like(zen_n)
zen_trig_up = np.zeros_like(zen_n)
zen_trig_att = np.zeros_like(zen_n)

zen_bin = np.digitize(np.cos(zenith), coszen_bin_edges)

for iZ in range(len(zen_bin)):
    zbin = zen_bin[iZ] - 1
    ebin = bin_cor[iZ] - 1
    if ebin > 19:
        continue
    zen_n[zbin][ebin] += n[iZ]
    zen_trig[zbin][ebin] += n_trig[iZ]
    zen_trig_up[zbin][ebin] += n_trig_up[iZ]
    zen_trig_att[zbin][ebin] += n_trig_att[iZ]

"""
zen_n = np.zeros((3,len(N_E)))
zen_trig = np.zeros_like(zen_n)
zen_trig_up = np.zeros_like(zen_n)
zen_trig_att = np.zeros_like(zen_n)

for iZ, zen in enumerate(zenith):
    bin = bin_cor[iZ] - 1
    if bin > 19:
        continue
    if zen <= np.deg2rad(30):
        zen_n[0][bin] += n[iZ]
        zen_trig[0][bin] += n_trig[iZ]
        zen_trig_up[0][bin] += n_trig_up[iZ]
        zen_trig_att[0][bin] += n_trig_att[iZ]
    elif zen <= np.deg2rad(60):
        zen_n[1][bin] += n[iZ]
        zen_trig[1][bin] += n_trig[iZ]
        zen_trig_up[1][bin] += n_trig_up[iZ]
        zen_trig_att[1][bin] += n_trig_att[iZ]
    else:
        zen_n[2][bin] += n[iZ]
        zen_trig[2][bin] += n_trig[iZ]
        zen_trig_up[2][bin] += n_trig_up[iZ]
        zen_trig_att[2][bin] += n_trig_att[iZ]
"""

zen_frac = zen_trig/zen_n
zen_up_frac = zen_trig_up/zen_n
zen_att_frac = zen_trig_att/zen_n

zen_frac[np.isnan(zen_frac)] = 0
zen_up_frac[np.isnan(zen_up_frac)] = 0
zen_att_frac[np.isnan(zen_att_frac)] = 0

"""
for i in range(3):
#    plt.plot(eng_mean, zen_frac[i], label='Down')
    plt.plot(eng_mean, zen_up_frac[i], label='Up ' + str((i+1)*30) + 'deg')
#    plt.plot(eng_mean, zen_att_frac[i], label='Att')
#    plt.title('Trigger rates for between ' + str(i * 30) + '-' + str((i+1)*30) + 'deg')
"""

for i in range(len(coszens)):
    plt.plot(eng_mean, zen_up_frac[i], label='cos(theta) ' + str(coszen_bin_edges[i]) + '-' + str(coszen_bin_edges[i+1]))

plt.legend()
plt.title('Trigger Rates of ' + str(spacing) + 'km spacing Moores Bay CRs')
plt.xlabel('Energy (log10eV)')
plt.ylabel('#Trig/#Throws')
plt.show()

