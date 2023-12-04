import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt

from NuRadioReco.utilities import units







def area_equilateral_triangle(length):
    return 3**0.5 / 4 * length**2

def circle_circle_intersection_area(r1, r2, d):
    #r1 is radius first circle
    #r2 is radius second circle
    #d is distacnce between centers of circle
    #Method documentation : https://mathworld.wolfram.com/Circle-CircleIntersection.html
    r = r1
    R = r2
    if R < r:       #Swap for right math
        r = r2
        R = r1

    part1 = r ** 2 * np.arccos((d ** 2 + r ** 2 - R ** 2) / (2 * d * r))
    part2 = R ** 2 * np.arccos((d ** 2 + R ** 2 - r ** 2) / (2 * d * R))
    part3 = 0.5 * np.sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R))
    return part1 + part2 - part3

def area_covered(tri_length, radius):
    tri_area = area_equilateral_triangle(tri_length)
    tri_middle = tri_length * 3 ** -0.5
    if radius <= tri_length/2:
        circ_area = np.pi * radius**2 * 1/6         #60deg angle covered in circle, 1/6 total area
        return 3 * circ_area / tri_area
    elif radius >= tri_middle:
        return 1.
    else:
        area_intersected = circle_circle_intersection_area(radius, radius, tri_length) / 2      #1/2 because only have of intersected area inside triangle
        circ_area = np.pi * radius ** 2 * 1/6
        return 3 * (circ_area - area_intersected) / tri_area

    

"""
#spacing = [0.5*units.km, 1*units.km, 1.5*units.km]
spacing = [500, 1000, 1500]
rad = np.arange(0, 1000, 50)

frac_covered = np.zeros((len(spacing), len(rad)))

for r in range(len(rad)):
    for dist in range(len(spacing)):
        frac_covered[dist][r] = area_covered(spacing[dist] / units.m, rad[r])

plt.figure()
for dist in range(len(spacing)):
    plt.scatter(rad, frac_covered[dist], label='Spacing ' + str(spacing[dist]))

plt.title('Fraction of Triangle Covered for Radius Signal and Spacing')
plt.ylabel('% Triangle Covered')
plt.xlabel('Radius (m)')
plt.legend()
plt.show()
"""
