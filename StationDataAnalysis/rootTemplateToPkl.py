import numpy as np
import h5py
import sys, os
import json
import scipy.signal as sig
import ROOT
import root_numpy

import argparse
import matplotlib.pyplot as plt

def getRefTemp(tempFn,nChan=4,nSamp=256,Eang=30.0,Hang=30.0,Cang=0.0):
    treenm="Templates"
    tree=ROOT.TChain(treenm)
    print("loading the Tree")
    tree.Add(tempFn)
    NEntries = tree.GetEntries()
    MetaArray = root_numpy.tree2array(tree,['EAng','HAng','coneAng'])

    DataArray = root_numpy.tree2array(tree,'wave.fData')
    DataArray = np.concatenate(DataArray)
    DataArray = np.reshape(DataArray,[NEntries,nChan,nSamp])
    DataArray = DataArray[:,1,:]

    iRef=None
    for i, angs in enumerate(MetaArray):
        #print('angs',float(angs[0]),float(angs[1]),float(angs[2]))
        #print('ref',float(Eang),float(Hang),float(Cang))
        if ((float(angs[0]) == float(Eang)) and (float(angs[1]) == float(Hang)) and (float(angs[2]) == float(Cang))):
            iRef = i

    if iRef:
        return DataArray[iRef]
    else:
        print('Error. Did not find reference template')
        sys.exit()






parser = argparse.ArgumentParser(description='Convert a root template into pickle format')
parser.add_argument('files', type=str, default='', help='File to run on')

args = parser.parse_args()
filesToRead = args.files




refTemp = getRefTemp(filesToRead)

plt.plot(refTemp)
plt.show()
