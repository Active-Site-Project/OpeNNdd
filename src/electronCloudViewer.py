"""
    A program to take an SDF file containing only H,C,N,O,F,S and
    visualizing it in a point cloud format. Visualize the nuclei and the
    electron clouds.

    !!!NOTE!!! currently only works on sdf files with one molecule per file

    Inputs:
    1) Path to sdf to view in cloud form
    2) Path to electron cloud text files dir
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pybel
from math import ceil
import os
import sys

sdfFilePath = str(sys.argv[1])  #Path to FILE of sdf ligand to be voxelized
cloudPath = str(sys.argv[2])    #Path to DIRECTORY containing ElectronClouds
voxelRes = .5 #cubic width of voxels in Angstroms
voxelLWH = 72 #width lenght and height of the voxel grid (Angstroms/voxelRes)



def main():
    coords = [] #nucleus xyz location
    aNum = []   #elements atomic number
    ePos = []   #electron coordinateds in relation to nuclei

    """
        Give the user the ability to add custom transformations in the form of
        x, y , z shifts. These same shifts will automatically be applied to the
        ligands in pyVox.py.
    """
    customXTransformation = 0   # Angstroms
    customYTransformation = 0  # Angstroms
    customZTransformation = 7.5 # Angstroms

    """
        Reads in the active site pdb file and retrieves all atom (x,y,z)
        coordinates and the atomic numbers of each atom.
    """
    active = pybel.readfile("sdf", sdfFilePath)
    for mol in active:
        for atom in mol:
            coords.append(atom.coords)
            aNum.append(atom.atomicnum)
    """
        Utilizes the ElectronClouds directory information to create electron
        clouds centerd around each atoms coordinate in the active-site.
    """
    os.chdir(cloudPath)
    for i in range(len(coords)):
        cloudFile = open(getAtomType(aNum[i]) + ".txt", 'r')
        for line in cloudFile:
            split = [x.strip() for x in line.split(',')]
            tempx = ceil((coords[i][0] + float(split[0])) * 100) / 100.0
            tempy = ceil((coords[i][1] + float(split[1])) * 100) / 100.0
            tempz = ceil((coords[i][2] + float(split[2])) * 100) / 100.0
            ePos.append(tuple([tempx,tempy,tempz,aNum[i]]))

    """
        Finds the global minimum x, y, z values and then uses them to transform
        the electron clouds and the nuclei into octant 1. (make all values pos)
        Also applies the users custom x, y, z transformations
    """
    minX = min(ePos, key = lambda t: t[0])[0]
    minY = min(ePos, key = lambda t: t[1])[1]
    minZ = min(ePos, key = lambda t: t[2])[2]
    transformedElectrons = []


    """
        Transforms the electron clouds and assigns a unique color to each
        element type.
    """
    for i in range(len(ePos)):
        tempx = ceil((ePos[i][0] - minX + customXTransformation) * 100) / 100.0
        tempy = ceil((ePos[i][1] - minY + customYTransformation) * 100) / 100.0
        tempz = ceil((ePos[i][2] - minZ + customZTransformation) * 100) / 100.0
        if ePos[i][3] == 1:
            color = [255/255, 102/255, 102/255, .25]
        elif ePos[i][3] == 6:
            color = [255/255, 255/255, 102/255, .25]
        elif ePos[i][3] == 7:
            color = [102/255, 255/255, 102/255, .25]
        elif ePos[i][3] == 8:
            color = [102/255, 255/255, 255/255, .25]
        elif ePos[i][3] == 9:
            color = [1,0,1,.15]
        elif ePos[i][3] == 16:
            color = [255/255, 102/255, 255/255, .25]
        transformedElectrons.append(tuple([tempx,tempy,tempz, color]))
    transformedNuclei = []


    """
        Sets colors for all the nuclei
    """
    for i in range(len(coords)):
        if aNum[i] == 1:
            color = [255/255, 102/255, 102/255, .6]
        elif aNum[i] == 6:
            color = [255/255, 255/255, 102/255, .6]
        elif aNum[i] == 7:
            color = [102/255, 255/255, 102/255, .6]
        elif aNum[i] == 8:
            color = [102/255, 255/255, 255/255, .6]
        elif aNum[i] == 9:
            color = [1,0,1,.15]
        elif aNum[i] == 16:
            color = [255/255, 102/255, 255/255, .6]
        tempx = ceil((coords[i][0] - minX + customXTransformation) * 100) / 100.0
        tempy = ceil((coords[i][1] - minY + customYTransformation) * 100) / 100.0
        tempz = ceil((coords[i][2] - minZ + customZTransformation) * 100) / 100.0
        transformedNuclei.append(tuple([tempx,tempy,tempz,color]))


    """
        Configures 3D grid and plots the electrons and nuclei.
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(0, 36)
    ax.set_ylim3d(0, 36)
    ax.set_zlim3d(0, 36)
    electronX = []
    electronY = []
    electronZ = []
    electronC = []
    nucX = []
    nucY = []
    nucZ = []
    nucC = []
    for i in range(len(transformedElectrons)):
        electronX.append(transformedElectrons[i][0])
        electronY.append(transformedElectrons[i][1])
        electronZ.append(transformedElectrons[i][2])
        electronC.append(transformedElectrons[i][3])
    for i in range(len(transformedNuclei)):
        nucX.append(transformedNuclei[i][0])
        nucY.append(transformedNuclei[i][1])
        nucZ.append(transformedNuclei[i][2])
        nucC.append(transformedNuclei[i][3])
    ax.scatter(electronX, electronY, electronZ, c = electronC, s=.07)
    ax.scatter(nucX, nucY, nucZ, c = nucC, s=40, alpha = 1)
    plt.show()

def getAtomType(num):
    typeTuple = [(1,'H'),(6,'C'),(7,'N'),(8,'O'),(9,'F'),(16,'S')]
    dic = dict(typeTuple)
    return dic[num]

#Run the main fuction
if __name__ == "__main__":
    main()
