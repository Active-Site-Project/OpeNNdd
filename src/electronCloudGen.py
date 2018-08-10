"""
    Creates a .txt file containg the x,y,z coordinates of electron values that
    have been distributed around a centered aroud the point (0,0,0). The user
    must include the Van Der Waals radius in Angstroms and the element symbol
    as arguments. Text file saves in ElectronClouds directoryself.

    SYSTEM ARGUMENTS:
    1) Desired radius in Angstrom
    2) Desired output file name (do not include .txt)
    3) Path to DIRECTORY containing the other electron clouds
"""
from numpy import pi, cos, sin, arccos, arange
from math import ceil
import os
import sys


#Initializes all values that use systems args
radius = float(sys.argv[1])         #Van Der Waals radius in Angstroms
filename = str(sys.argv[2])+".txt"  #Output file name
cloudPath = str(sys.argv[3])        #Path to DIRECTORY with other cached clouds (output DIR)


num_pts = int(radius ** 2 * 4 * pi * 4) #makes a consistant number of electrons per radius
indices = arange(0, num_pts, dtype=float) + 0.5
os.chdir(cloudPath)
outFile = open(filename, 'w+')
xv = []
yv = []
zv = []

"""
    Generates a unit sphere or electrons that have been evenly distributed
    across the surphace of the spere.
"""
phi = arccos(1 - 2*indices/num_pts)
theta = pi * (1 + 5**0.5) * indices
x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi);

"""
    Takes the unit sphere and creates a scaled version of it at every
    hundreths of an Angstrom.
"""
for i in range(len(x)):
    xr = ceil(x[i] * 100) / 100.0
    yr = ceil(y[i] * 100) / 100.0
    zr = ceil(z[i] * 100) / 100.0
    for j in range(1, int(radius*10)):
        xf = ceil((1/10 * j * xr)*100) / 100.0
        yf = ceil((1/10 * j * yr)*100) / 100.0
        zf = ceil((1/10 * j * zr)*100) / 100.0
        xv.append(xf)
        yv.append(yf)
        zv.append(zf)

for i in range(len(xv)):
    outFile.write(str(xv[i])+", "+str(yv[i])+", "+str(zv[i]))
    outFile.write("\n")

outFile.close()
