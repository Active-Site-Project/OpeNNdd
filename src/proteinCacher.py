"""
    Voxelizes and caches the protein active sites sdf file.
    Stores data in hdf5 file.

    SYSTEM ARGUMENTS
    1) Path to DIRECTORY to store the voxelized active-site hdf5 file
    2) Path to DIRECTORY containing the electron electron clouds
    3) Complete path to active-site pdb FILE
    4) Output file name for hdf5 file (Do not include .h5)
"""
import pybel
import numpy as np
import h5py
import os
import sys
from math import ceil


#Initialize variables that use the system arguments
outDirectory = str(sys.argv[1]) #Output DIRECTORY
eDirectory = str(sys.argv[2])   #ElectronClouds DIRECTORY
poseFile = str(sys.argv[3])     #Protein PDB FILE
outName = str(sys.argv[4])      #Name of output hdf5 file

def main():
    coords = [] #nucleus xyz location
    aNum = []  #elements atomic number
    ePos = [] #electron coordinateds in relation to nuclei
    customXTransformation = 0 #in Angstroms
    customYTransformation = 0 #in Angstroms
    customZTransformation = 7.5 #in Angstroms


    """
        Reads in the active site pdb file and retrieves all atom (x,y,z)
        coordinates, all atomic numbers and the interation energy of the
        ligand/protien interaction.
    """
    active = pybel.readfile("pdb", poseFile)
    for mol in active:
        for atom in mol:
            coords.append(atom.coords)
            aNum.append(atom.atomicnum)


    """
        Places electron cloud around each ligand atom.
    """
    os.chdir(eDirectory)
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


    """
    Saves all transformations applied to the active-site. Transformations are
    stored inside hdf5 file and applied to each ligand to keep keep Transformations
    consistant.
    """
    activeMinX = minX - customXTransformation
    activeMinY = minY - customYTransformation
    activeMinZ = minZ - customZTransformation
    activeMinTuple = [activeMinX,activeMinY,activeMinZ]

    #Initializes a matrix of proper dimensions and populates with voxelized protein data
    a = np.zeros((voxelLWH, voxelLWH, voxelLWH, 6))
    voxelizedActive = voxData(a, transformedElectrons)

    #Stores transforms and voxelized protein data to outputted hdf5 file
    os.chdir(outDirectory)
    hf = h5py.File(outName+'.h5', 'w')
    hf.create_dataset('activeCacheMatrix', data=voxelizedActive)
    hf.create_dataset('activeCacheTransformations', data=activeMinTuple)
    hf.close()




"""
    Scales the coords to the desired resolution (Angstroms/Voxel)
"""
def upResCalculation(value):
    return int((value-(value % voxelRes)) * (1/voxelRes))


"""
    Adds electron counts of the ligand into protein matrix
"""
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

"""
    Assigns each element type a channel value for the output matrix
"""
def gNum(anum):
    typeTuple = [(1,0),(6,1),(7,2),(8,3),(9,4),(16,5)]
    dic = dict(typeTuple)
    return dic[anum]

if __name__ == "__main__":
    voxelRes = .5 #cubic width of voxels
    voxelLWH = 72 #width lenght and height of the voxel grid
    main()
