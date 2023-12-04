import numpy as np
import matplotlib.pyplot as plt



###
#	Note : Standard is for Nu-Chi to be x-axis, R-CR-Chi to be y-axis
###

#Cut for for Chi-Chi plot of Neutrino template and R-CR
def nuCut(nuChi):
    x1 = 0.34
    x2 = 0.6
    y1 = 0.72
    y2 = 0.85
    if nuChi < x1:
        return y1
    elif nuChi < x2:
        return (nuChi - x2) * (y2-y1)/(x2-x1) + y2
    else:
        return y2 - 3.5*(nuChi - x2)**2


#Returns an array of corresponding fit values for plotting
def cutArray(nuArray):
    plot = np.zeros_like(nuArray)
    for iN, nu in enumerate(nuArray):
        plot[iN] = nuCut(nu)
    return plot


#Returns efficiency of how many points pass cut
def cutEfficiency(nuChi, crChi):
    eff = 0
    for iN, nu in enumerate(nuChi):
        if crChi[iN] >= nuCut(nu):
            eff += 1
    eff = 100 * eff / len(nuChi)
    return eff

#Boolean for passing cut check
def passesCut(nuChi, crChi):
    return crChi >= nuCut(nuChi)


#Boolean mask for full array of data
def cutMask(nuChi, crChi):
    cutVals = cutArray(nuChi)
    return crChi >= cutVals

#Plots cut line
def plotCut():
    x = np.linspace(0, 1, 100)
    plt.plot(x, cutArray(x), color='red', linestyle='-')
    return
