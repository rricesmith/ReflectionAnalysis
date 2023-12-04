#Utility functions for a triangular grid of stations setup to work with triangle shape of readCoREASStationGrid.py
#Default setup for triangular grid:
#Stations 1 and 2 at y=0, spacing apart
#Only work to do it determine the y value of the third station given its an equilateral triangle

import numpy as np

def triangleGridHeight(grid_spacing):
    return np.sqrt(3/2) * grid_spacing

