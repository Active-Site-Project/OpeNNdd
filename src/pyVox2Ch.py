"""
    Program reads in an entire directory of .sdf files of ligands.
    From .sdf gets the ligand atom locations, atom types, and the active-site
    /ligands associated interation energy. The program voxelizes the ligand and
    combines the data with cached active-site voxelized data. All data is saved
    into a .hdf5 file. Output file is organized into training data, training
    labels, validation data, validation labels, test data, and test labels.
"""
import pybel
import numpy as np
import h5py
from tqdm import tqdm
from random import shuffle
import os
from math import ceil
import sys

voxelizedDataPath = str(sys.argv[1]) #path of directory containing activeCache.hdf5 and where to store voxelized data
posesPath = str(sys.argv[2]) #path of directory containg all ligand poses
cloudPath = str(sys.argv[3]) #path to dir with electron cloud data
voxelRes = .5 #cubic width of voxels
voxelLWH = 72 #width lenght and height of the voxel grid

def main():
    training = .70    #percent of the dataset reserved for training
    validation = .20  #percent of the dataset reserved for validation
    test = .10        #percent of the dataset reserved for testing

    """
        Reads in cached active site HDF5 file and retrieves cached voxelized
        active-site information as well as all x, y, z transformations that
        were applied to the active-site. Then opens new hdf5 file to write
        outputs to.
    """
    os.chdir(voxelizedDataPath)
    h5f = h5py.File('activeCache.h5','r')
    siteMatrix = h5f['activeCacheMatrix'][:]
    xTrans = h5f['activeCacheTransformations'][0]
    yTrans = h5f['activeCacheTransformations'][1]
    zTrans = h5f['activeCacheTransformations'][2]
    h5f.close()
    hf = h5py.File('new.h5', 'w')

    """
        Lists all files with in the directory of ligand sdf files and shuffles them
    """
    fileNames = [] #list of all file names in the poses folder
    os.chdir(posesPath)
    for filename in os.listdir(os.getcwd()):
        if not filename.startswith('.'):
            fileNames.append(filename)
    shuffle(fileNames)

    desiredFiles = []
    desiredSize = 4000
    for i in range(desiredSize):
        desiredFiles.append(fileNames[i])


    """
        Calculates the training size, validation size, and test size based on
        the desired percentages. Then creates
    """
    train_size = int(training * len(desiredFiles))
    validation_size = int(validation * len(desiredFiles))
    test_size = int(test * len(desiredFiles))
    train_data_shape = (train_size, voxelLWH, voxelLWH, voxelLWH, 2)
    train_label_shape = (train_size,)
    val_data_shape = (validation_size, voxelLWH, voxelLWH, voxelLWH, 2)
    val_label_shape = (validation_size,)
    test_data_shape = (test_size, voxelLWH, voxelLWH, voxelLWH, 2)
    test_label_shape = (test_size,)

    data = []
    labels = []
    for i in tqdm(range(len(desiredFiles))):
        sdfVox(desiredFiles[i], siteMatrix, xTrans, yTrans, zTrans, data, labels)
        os.chdir(posesPath)

    trainData = []
    trainLabels = []
    for i in range(0, train_size, 1):
        trainData.append(data[i])
        trainLabels.append(labels[i])

    valData = []
    valLabels = []
    for i in range(train_size, (train_size+validation_size), 1):
        valData.append(data[i])
        valLabels.append(labels[i])

    testData = []
    testLabels = []
    for i in range((train_size+validation_size), (train_size+validation_size+test_size), 1):
        testData.append(data[i])
        testLabels.append(labels[i])

    hf.create_dataset('train_ligands', train_data_shape, np.int8, trainData)
    hf.create_dataset('val_ligands', val_data_shape, np.int8, valData)
    hf.create_dataset('test_ligands', test_data_shape, np.int8, testData)
    hf.create_dataset('train_labels', train_label_shape, np.float32, trainLabels)
    hf.create_dataset('val_labels', val_label_shape, np.float32, valLabels)
    hf.create_dataset('test_labels', test_label_shape, np.float32, testLabels)
    hf.close()



#Scales up the number of voxels based on the desired resolution
def upResCalculation(value):
    return int((value-(value % voxelRes)) * (1/voxelRes))




#Transforms the electrons and nuclei into a simplified voxelized form
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




#Returns atom symbol based on the atomic number
def getAtomType(num):
    typeTuple = [(1,'H'),(6,'C'),(7,'N'),(8,'O'),(9,'F'),(16,'S')]
    dic = dict(typeTuple)
    return dic[num]




#Returns a sum that has been rounded to the hundreths place
def addRoundHundredth(num1,num2):
    sum = ceil((num1 + num2) * 100) / 100.0
    return sum




#Initializes the graph enviroment for visualization of voxels
def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax




#Voxelizes sdf ligand
def sdfVox(name, activeMatrix, xTrans, yTrans, zTrans, data, labels):
    coords = [] #nucleus xyz location
    aNum = []  #elements atomic number
    molEnergy = 0
    molCount = 0
    for mol in pybel.readfile('sdf', name):
        if molCount > 0:
            raise RuntimeError('Only takes one molecule per sdf file. Use pySplit.py') from error
        molEnergy = mol.data['minimizedAffinity']
        molCount += 1
        for atom in mol:
            aNum.append(atom.atomicnum)
            coords.append(atom.coords)

    """
        Transforms the nuclei by the same transformations of the activesite
    """
    transformedNuclei = []
    for i in range(len(coords)):
        transformedNuclei.append(tuple([
        addRoundHundredth(coords[i][0], -xTrans),
        addRoundHundredth(coords[i][1], -yTrans),
        addRoundHundredth(coords[i][2], -zTrans)]))

    """
        Places electron cloud around each ligand atom.
    """
    os.chdir(cloudPath)
    transformedElectrons = []
    for i in range(len(transformedNuclei)):
        cloudFile = open(getAtomType(aNum[i]) + ".txt", 'r')
        for line in cloudFile:
            split = [x.strip() for x in line.split(',')]
            transformedElectrons.append(tuple([
            addRoundHundredth(transformedNuclei[i][0],float(split[0])),
            addRoundHundredth(transformedNuclei[i][1],float(split[1])),
            addRoundHundredth(transformedNuclei[i][2],float(split[2]))]))

    """
        Initializes and populates the matrix of voxelized ligand data
    """

    ligandMatrix = np.zeros((voxelLWH, voxelLWH, voxelLWH, 2))
    ligandMatrix = voxData(ligandMatrix, transformedElectrons, transformedNuclei)

    """
        Initializes and populates the matrix that combines ligand and protien
        data
    """

    dockedLigandMatrix = np.zeros((voxelLWH, voxelLWH, voxelLWH, 2))
    dockedLigandMatrix[:,:,:,0] = ligandMatrix[:,:,:,0] + activeMatrix[:,:,:,0]
    dockedLigandMatrix[:,:,:,1] = ligandMatrix[:,:,:,1] + activeMatrix[:,:,:,1]

    outEnergy = np.asarray(molEnergy, dtype =np.float32)
    data.append(dockedLigandMatrix)
    labels.append(outEnergy)




#Run the main fuction
if __name__ == "__main__":
    main()
