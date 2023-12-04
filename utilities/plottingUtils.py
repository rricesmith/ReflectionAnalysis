import pickle
import matplotlib.pyplot as plt


#Global parameter to save
#Set to true when plots being made want to be saved
#Otherwise they won't save until turned on
SET_SAVE = True

def savePlot(pltObj, saveName, location='plots/plotFigures/CoresPresentation_Spring23'):
    if not SET_SAVE:
        return False
    #To pass in pltObj
    #Call function as below after all plot changes have been made
    ### savePlot(plt.gca(), saveName, location = 'save/here')
    with open(f'{location}/{saveName}.pkl', 'wb') as fout:
        pickle.dump(pltObj, fout)
    print(f'Saved to {location}/{saveName}.pkl')
    return True

def openPlot(fileName):
    #To work with object after returned
    #Just do ax.draw()
    #Then make any changes as necessary, resave or reprint, etc
    with open(fileName, 'rb') as fin:
        ax = pickle.load(fin)
    return ax