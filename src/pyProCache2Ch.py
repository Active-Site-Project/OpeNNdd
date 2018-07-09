"""
    This program will generate a .hdf5 file based on a .pbd file that contains
    the active-site. The .hdf5 file serves as cached data for pyVox.py and
    contains the voxelized active-stie information as well as all transformations
    applied to the active-site.
"""
import pybel
import numpy as np
import h5py
import os
from math import ceil
import sys

activePath = str(sys.argv[1]) #replace path with location of active site pbd file
voxelizedDataPath = str(sys.argv[2]) #dir to store active cache
cloudPath = str(sys.argv[3]) #path to dir containing electron cloud
voxelRes = .5 #cubic width of voxels
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
    customYTransformation = 0   # Angstroms
    customZTransformation = 7.5 # Angstroms

    """
        Reads in the active site pdb file and retrieves all atom (x,y,z)
        coordinates and the atomic numbers of each atom.
    """
    os.chdir(activePath)
    active = pybel.readfile("pdb", "SARS_3CLpro_5c5o_ActiveSiteAtomsOnly.pdb")
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
            ePos.append(tuple([tempx,tempy,tempz]))

    """
        Finds the global minimum x, y, z values and then uses them to transform
        the electron clouds and the nuclei into octant 1. (make all values pos)
        Also applies the users custom x, y, z transformations
    """
    minX = min(ePos, key = lambda t: t[0])[0]
    minY = min(ePos, key = lambda t: t[1])[1]
    minZ = min(ePos, key = lambda t: t[2])[2]
    transformedElectrons = []
    for i in range(len(ePos)):
        tempx = ceil((ePos[i][0] - minX + customXTransformation) * 100) / 100.0
        tempy = ceil((ePos[i][1] - minY + customYTransformation) * 100) / 100.0
        tempz = ceil((ePos[i][2] - minZ + customZTransformation) * 100) / 100.0
        transformedElectrons.append(tuple([tempx,tempy,tempz]))
    transformedNuclei = []
    for i in range(len(coords)):
        tempx = ceil((coords[i][0] - minX + customXTransformation) * 100) / 100.0
        tempy = ceil((coords[i][1] - minY + customYTransformation) * 100) / 100.0
        tempz = ceil((coords[i][2] - minZ + customZTransformation) * 100) / 100.0
        transformedNuclei.append(tuple([tempx,tempy,tempz]))

    """
        Finds the net x, y, z transformations that were applied to the active-site
        elements. The transformations are stored into the .hdf5 file that pyVox.py
        reads. pyVox.py will then apply the same transformations to the ligands
        that it voxelizes.
    """
    activeMinX = minX - customXTransformation
    activeMinY = minY - customYTransformation
    activeMinZ = minZ - customZTransformation
    activeMinTuple = [activeMinX,activeMinY,activeMinZ]

    """
        voxelizes the active site and stores the data in a 4D numpy array.
        form [x,y,z,channels]. channel[0] = electron data. channel[1] = nuclei data.
    """
    a = np.zeros((voxelLWH, voxelLWH, voxelLWH, 2))
    voxelizedActive = voxData(a, transformedElectrons, transformedNuclei)

    """
        creates a .hdf5 file called activeCache and stores the voxelized active-site
        and the x, y, z transformations that were applied
    """
    os.chdir(voxelizedDataPath)
    hf = h5py.File('activeCache.h5', 'w')
    hf.create_dataset('activeCacheMatrix', data=voxelizedActive)
    hf.create_dataset('activeCacheTransformations', data=activeMinTuple)
    hf.close()




"""
    Converts the Angstrom value into the proper value based on desired voxel resolution
"""
def upResCalculation(value):
    return int((value-(value % voxelRes)) * (1/voxelRes))




"""
    Voxelizes the active site data. Counts the number of nuclei and number of
    electrons per voxel and places the data into the 4D matrix.
    form [x,y,z,channels]. channel[0] = electron data. channel[1] = nuclei data.
"""
def voxData(matrix, eList, nList):
    for i in range(len(eList)):
        vx = upResCalculation(eList[i][0])
        vy = upResCalculation(eList[i][1])
        vz = upResCalculation(eList[i][2])
        matrix[vx,vy,vz,0] += 1
    for i in range(len(nList)):
        vx = upResCalculation(nList[i][0])
        vy = upResCalculation(nList[i][1])
        vz = upResCalculation(nList[i][2])
        matrix[vx,vy,vz,1] += 1
    return matrix




"""
    Returns the proper element symbole based on the inputed atomic number.
"""
def getAtomType(num):
    typeTuple = [(1,'H'),(6,'C'),(7,'N'),(8,'O'),(9,'F'),(16,'S')]
    dic = dict(typeTuple)
    return dic[num]




if __name__ == "__main__":
    main()
