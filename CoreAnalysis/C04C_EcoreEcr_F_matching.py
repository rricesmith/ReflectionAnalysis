from NuRadioReco.utilities import units
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import coreDataObjects as CDO
import pickle
import CoreAnalysis.C00_coreAnalysisUtils as CDO_util
import itertools
plt.style.use('plotsStyle.mplstyle')
import CoreAnalysis.C00_plotCoreSpectra as C00_plotCoreSpectra



#Goal of this code is to make a plot
#x-axis is Ecore/Ecr
#y-axis is f
#For each sim, then for each Ecore/Ecr data point
#Find the f value that comes from the simulation result
#Plot will create a line, and we can plot Simon's f best fit line
#Then where those lines intersect tells us the f for that simulation
