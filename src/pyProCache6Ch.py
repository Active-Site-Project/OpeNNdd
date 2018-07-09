import pybel
import numpy as np
import h5py
import os
from math import ceil
import sys

def main():
    coords = [] #nucleus xyz location
    aNum = []  #elements atomic number
    ePos = [] #electron coordinateds in relation to nuclei
    customXTransformation = 0 #in Angstroms
    customYTransformation = 0 #in Angstroms
    customZTransformation = 7.5 #in Angstroms

    file = str(sys.argv[1]) #file path to active site pdb
    clouds = str(sys.argv[2]) #path to electron clouds directory
    voxelizedDataPath = str(sys.argv[3]) #directory for where to save voxelized data
    """
        Reads in the active site pdb file and retrieves all atom (x,y,z)
        coordinates, all atomic numbers and the interation energy of the
        ligand/protien interaction.
    """

    active = pybel.readfile("pdb", file)
    for mol in active:
        for atom in mol:
            coords.append(atom.coords)
            aNum.append(atom.atomicnum)

    """
        Places electron cloud around each ligand atom.
    """
    os.chdir(clouds)
    for i in range(len(coords)):
        cloudFile = open(getAtomType(aNum[i]) + ".txt", 'r')
        for line in cloudFile:
            split = [x.strip() for x in line.split(',')]
            tempx = ceil((coords[i][0] + float(split[0])) * 100) / 100.0
            tempy = ceil((coords[i][1] + float(split[1])) * 100) / 100.0
            tempz = ceil((coords[i][2] + float(split[2])) * 100) / 100.0
            chan = gNum(aNum[i])
            ePos.append(tuple([tempx,tempy,tempz,chan]))

    """
        Finds the minimum values of the particles in order to transform the
        ligand/molecule to octant 1
    """

    minX = min(ePos, key = lambda t: t[0])[0]
    minY = min(ePos, key = lambda t: t[1])[1]
    minZ = min(ePos, key = lambda t: t[2])[2]


    """
        Transforms all electrons into octant 1
    """

    transformedElectrons = []
    for i in range(len(ePos)):
        tempx = ceil((ePos[i][0] - minX + customXTransformation) * 100) / 100.0
        tempy = ceil((ePos[i][1] - minY + customYTransformation) * 100) / 100.0
        tempz = ceil((ePos[i][2] - minZ + customZTransformation) * 100) / 100.0
        transformedElectrons.append(tuple([tempx,tempy,tempz,ePos[i][3]]))

    activeMinX = minX - customXTransformation
    activeMinY = minY - customYTransformation
    activeMinZ = minZ - customZTransformation
    activeMinTuple = [activeMinX,activeMinY,activeMinZ]
    a = np.zeros((voxelLWH, voxelLWH, voxelLWH, 6))
    voxelizedActive = voxData(a, transformedElectrons)

    os.chdir(voxelizedDataPath)
    hf = h5py.File('activeCache6Channel.h5', 'w')
    hf.create_dataset('activeCacheMatrix', data=voxelizedActive)
    hf.create_dataset('activeCacheTransformations', data=activeMinTuple)
    hf.close()



    """
    Take the scatter plot of electron points and nuclei coordinates and
    voxelizes the data into a 3d grid based system
    """

def upResCalculation(value):
    return int((value-(value % voxelRes)) * (1/voxelRes))

def voxData(matrix, eList):
    for i in range(len(eList)):
        vx = upResCalculation(eList[i][0])
        vy = upResCalculation(eList[i][1])
        vz = upResCalculation(eList[i][2])
        vt = eList[i][3]
        matrix[vx,vy,vz,vt] += 1
    return matrix


"""
    return the elements symbol based on the atomic number
"""

def getAtomType(num):
    typeTuple = [(1,'H'),(6,'C'),(7,'N'),(8,'O'),(9,'F'),(16,'S')]
    dic = dict(typeTuple)
    return dic[num]

def gNum(thing):
    typeTuple = [(1,0),(6,1),(7,2),(8,3),(9,4),(16,5)]
    dic = dict(typeTuple)
    return dic[thing]

if __name__ == "__main__":
    voxelRes = .5 #cubic width of voxels
    voxelLWH = 72 #width lenght and height of the voxel grid
    angDim = int(voxelRes * voxelLWH)
    main()
