"""
    Program displays statistics of the voxelized dataset from .hdf5 format
"""
import numpy as np
import h5py
import os
from math import ceil
import sys

voxelizedDataPath = str(sys.argv[1]) #path with voxelized ligand protein dataset

def main():
    #Open and retrieve voxelized data from .hdf5 folder
    h5f = h5py.File(voxelizedDataPath,'r')
    trainData = h5f['train_ligands'][:]
    trainLabels = h5f['train_labels'][:]
    valData = h5f['val_ligands'][:]
    valLabels = h5f['val_labels'][:]
    testData = h5f['test_ligands'][:]
    testLabels = h5f['test_labels'][:]
    h5f.close()


    #Calculates the various statistcs
    sizeOfTrainingData = len(trainLabels)
    sizeOfValData = len(valLabels)
    sizeOfTestData = len(testLabels)
    totalSize = sizeOfTrainingData + sizeOfValData + sizeOfTestData
    percTraining = 100 * (ceil((sizeOfTrainingData / totalSize) * 10000) / 10000.0)
    percVal = 100 * (ceil((sizeOfValData / totalSize) * 10000) / 10000.0)
    percTest = 100 * (ceil((sizeOfTestData / totalSize) * 10000) / 10000.0)
    trainDataShape = trainData.shape
    valDataShape = valData.shape
    testDataShape = testData.shape

    highEnergy = []
    highEnergy.append(np.max(trainLabels))
    highEnergy.append(np.max(valLabels))
    highEnergy.append(np.max(testLabels))
    trueHighest = max(highEnergy)
    if trueHighest in trainLabels:
        highIn = 'train_labels'
        highIndex = np.where(trainLabels == trueHighest)[0]
    elif trueHighest in valLabels:
        highIn = 'val_labels'
        highIndex = np.where(valLabels == trueHighest)[0]
    elif trueHighest in testLabels:
        highIn = 'test_labels'
        highIndex = np.where(testLabels == trueHighest)[0]
    lowEnergy = []
    lowEnergy.append(np.min(trainLabels))
    lowEnergy.append(np.min(valLabels))
    lowEnergy.append(np.min(testLabels))
    trueLowest = min(lowEnergy)
    if trueLowest in trainLabels:
        lowIn = 'train_labels'
        lowIndex = np.where(trainLabels == trueLowest)[0]
    elif trueLowest in valLabels:
        lowIn = 'val_labels'
        lowIndex = np.where(valLabels == trueLowest)[0]
    elif trueHighest in testLabels:
        lowIn = 'test_labels'
        lowIndex = np.where(testLabels == trueLowest)[0]


    print("\n" * 5)
    print('---------------------------------------------------------')
    print('                 Dataset Information:')
    print('---------------------------------------------------------')
    print('Total size of data      : '+str(totalSize))
    print('Shape of training set   : '+str(trainDataShape))
    print('Shape of validation set : '+str(valDataShape))
    print('Shape of test set       : '+str(testDataShape))
    print('Percent used training   : '+str(percTraining)+'%')
    print('Percent used validating : '+str(percVal)+'%')
    print('Percent used testing    : '+str(percTest)+'%')
    print('Highest energy in data  : '+str(trueHighest))
    print('Location of high        : Set: '+highIn +" Index: "+ str(highIndex[0]))
    print('Lowest energy in data   : '+str(trueLowest))
    print('Location of low         : Set: '+lowIn +" Index: "+ str(lowIndex[0]))
    print('---------------------------------------------------------')
    print("\n" * 2)



#Run the main fuction
if __name__ == "__main__":
    main()
