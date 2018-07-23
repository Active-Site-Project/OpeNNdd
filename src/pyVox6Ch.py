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
import tables
from tqdm import tqdm
from random import shuffle
import os
from math import ceil
import sys

voxelizedDataPath = str(sys.argv[1]) #path of directory containing activeCache6Channel.hdf5 and where to store newly voxelized
posesPath = str(sys.argv[2]) #path of directory containg all ligand poses
cloudPath = str(sys.argv[3]) #path to electron clouds
voxelRes = .5 #cubic width of voxels
voxelLWH = 72 #width lenght and height of the voxel grid


class dataInfo:
    def __init__(self, dataset, shape):
        self.dataset = dataset
        self.shape = shape
        self.i = 0

    def appendVal(self, values):
        os.chdir(voxelizedDataPath)
        with h5py.File('new.h5', mode='a') as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + 1, ) + self.shape)
            dset[self.i] = [values]
            self.i += 1
            h5f.flush()


def main():
    training = .70    #percent of the dataset reserved for training
    validation = .10  #percent of the dataset reserved for validation
    test = .20        #percent of the dataset reserved for testing

    """
        Reads in cached active site HDF5 file and retrieves cached voxelized
        active-site information as well as all x, y, z transformations that
        were applied to the active-site. Then opens new hdf5 file to write
        outputs to.
    """
    os.chdir(voxelizedDataPath)
    hf = h5py.File('activeCache6Channel.h5','r')
    siteMatrix = hf['activeCacheMatrix'][:]
    trans = hf['activeCacheTransformations'][:]
    hf.close()



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
    desiredSize = 25000
    for i in range(desiredSize):
        desiredFiles.append(fileNames[i])

    """
        Calculates the training size, validation size, and test size based on
        the desired percentages. Then creates
    """
    train_size = int(training * len(desiredFiles))
    validation_size = int(validation * len(desiredFiles))
    test_size = int(test * len(desiredFiles))
    train_data_shape = (voxelLWH, voxelLWH, voxelLWH, 6)
    train_label_shape = ()
    val_data_shape = (voxelLWH, voxelLWH, voxelLWH, 6)
    val_label_shape = ()
    test_data_shape = (voxelLWH, voxelLWH, voxelLWH, 6)
    test_label_shape = ()




    dtype = np.float32
    compression="gzip"
    chunk_len=1

    os.chdir(voxelizedDataPath)


    with h5py.File('new.h5', mode='w') as h5f:
        shape = train_data_shape
        h5f.create_dataset('train_ligands', shape=(0, ) + shape, maxshape=(None, ) + shape, dtype=dtype, compression=compression, chunks=(chunk_len, ) + shape, compression_opts=9)
        shape = train_label_shape
        h5f.create_dataset('train_labels', shape=(0, ) + shape, maxshape=(None, ) + shape, dtype=dtype, compression=compression, chunks=(chunk_len, ) + shape, compression_opts=9)
        shape = val_data_shape
        h5f.create_dataset('val_ligands', shape=(0, ) + shape, maxshape=(None, ) + shape, dtype=dtype, compression=compression, chunks=(chunk_len, ) + shape, compression_opts=9)
        shape = val_label_shape
        h5f.create_dataset('val_labels', shape=(0, ) + shape, maxshape=(None, ) + shape, dtype=dtype, compression=compression, chunks=(chunk_len, ) + shape, compression_opts=9)
        shape = test_data_shape
        h5f.create_dataset('test_ligands', shape=(0, ) + shape, maxshape=(None, ) + shape, dtype=dtype, compression=compression, chunks=(chunk_len, ) + shape, compression_opts=9)
        shape = test_label_shape
        h5f.create_dataset('test_labels', shape=(0, ) + shape, maxshape=(None, ) + shape, dtype=dtype, compression=compression, chunks=(chunk_len, ) + shape, compression_opts=9)

    trainD = dataInfo('train_ligands', train_data_shape)
    trainL = dataInfo('train_labels', train_label_shape)
    valD = dataInfo('val_ligands', val_data_shape)
    valL = dataInfo('val_labels', val_label_shape)
    testD = dataInfo('test_ligands', test_data_shape)
    testL = dataInfo('test_labels', test_label_shape)


    for i in tqdm(range(train_size)):
        sdfVox(desiredFiles[i], siteMatrix, trans, trainD, trainL)


    for i in tqdm(range(train_size, train_size + validation_size)):
        sdfVox(desiredFiles[i], siteMatrix, trans, valD, valL)


    for i in tqdm(range(train_size + validation_size, train_size + validation_size + test_size)):
        sdfVox(desiredFiles[i], siteMatrix, trans, testD, testL)




#Scales up the number of voxels based on the desired resolution
def upResCalculation(value):
    return int((value-(value % voxelRes)) * (1/voxelRes))




#Transforms the electrons and nuclei into a simplified voxelized form
def voxData(matrix, eList):
    for i in range(len(eList)):
        vx = upResCalculation(eList[i][0])
        vy = upResCalculation(eList[i][1])
        vz = upResCalculation(eList[i][2])
        vt = eList[i][3]
        matrix[vx,vy,vz,vt] += 1
    return matrix


#Returns atom symbol based on the atomic number
def getAtomType(num):
    typeTuple = [(1,'H'),(6,'C'),(7,'N'),(8,'O'),(9,'F'),(16,'S')]
    dic = dict(typeTuple)
    return dic[num]

def gNum(thing):
    typeTuple = [(1,0),(6,1),(7,2),(8,3),(9,4),(16,5)]
    dic = dict(typeTuple)
    return dic[thing]

#Returns a sum that has been rounded to the hundreths place
def addRoundHundredth(num1,num2):
    sum = ceil((num1 + num2) * 100) / 100.0
    return sum


#Voxelizes sdf ligand
def sdfVox(name, activeMatrix, trans, d, l):
    coords = [] #nucleus xyz location
    aNum = []  #elements atomic number
    molEnergy = 0
    molCount = 0
    os.chdir(posesPath)
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
        addRoundHundredth(coords[i][0], -trans[0]),
        addRoundHundredth(coords[i][1], -trans[1]),
        addRoundHundredth(coords[i][2], -trans[2])]))

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
            addRoundHundredth(transformedNuclei[i][2],float(split[2])),
            gNum(aNum[i])]))

    """
        Initializes and populates the matrix of voxelized ligand data
    """

    dockedLigandMatrix = voxData(activeMatrix, transformedElectrons)

    """
        Initializes and populates the matrix that combines ligand and protien
        data
    """



    outEnergy = np.asarray(molEnergy, dtype = np.float32)
    d.appendVal(dockedLigandMatrix)
    l.appendVal(outEnergy)

    os.chdir(posesPath)


#Run the main fuction
if __name__ == "__main__":
    main()
