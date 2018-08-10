"""
    Program displays statistics of the voxelized dataset based on a file path
    inputted by user

    SYSTEM ARGUMENTS:
    1)Path to hdf5 FILE containing dataset
"""
import numpy as np
import h5py
import os
import sys
from math import ceil


filePath = str(sys.argv[1]) #Complete FILEPATH to hdf5 file

def main():
    #Open and retrieve voxelized data from .hdf5 folder
    h5f = h5py.File(filePath,'r')
    ligands = h5f['ligands'][:]   #Voxelized matricies
    labels = h5f['labels'][:]     #Interaction energies
    files = h5f['files'][:]       #Sdf file names
    h5f.close()


    #Calculates the various statistcs
    sizeOfData = len(ligands)                #How many poses are in dataset
    dataShape = ligands.shape                #Shape of matricies in dataset
    high = np.max(labels)                    #Highest interaction energy in dataset
    highIndex = np.where(labels == high)[0]  #Index of highest within hdf5
    low = np.min(labels)                     #Lowest interaction energy in dataset
    lowIndex = np.where(labels == low)[0]    #Index of lowest within hdf5


    print("\n" * 5)
    print('---------------------------------------------------------')
    print('                 Dataset Information:')
    print('---------------------------------------------------------')
    print('Total size of data      : '+str(sizeOfData))
    print('Shape data              : '+str(dataShape))
    print('Highest energy in data  : '+str(high))
    print('Index of high in hdf5   : Index: '+ str(highIndex[0]))
    print('Sdf file containing High: '+ str(files[int(highIndex[0])]))
    print('Lowest energy in data   : '+str(low))
    print('Index of low in hdf5    : Index: '+ str(lowIndex[0]))
    print('Sdf file containing low : '+ str(files[int(lowIndex[0])]))
    print('---------------------------------------------------------')
    print("\n" * 2)



#Run the main fuction
if __name__ == "__main__":
    main()
