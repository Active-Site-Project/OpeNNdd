"""
    Program reads in an entire directory of .sdf files containing ligand poses.
    From .sdf extracts: ligand atom locations, atom types, and the active-site
    /ligands associated interation energy. The program voxelizes the ligand and
    combines the data with cached active-site voxelized data. All data is saved
    into a .hdf5 file. Program can be run in parallel using MPI Exex.
"""
import pybel
import numpy as np
import h5py
from random import shuffle
import os
from math import ceil
import sys
from mpi4py import MPI


"""
    SYSTEM ARGUMENTS:
    1) Path to DIRECTORY containing all poses
    2) Path to FILE containing the voxelized protein active-site matrix
    3) Path to directory in which outputed hdf5s are to be stored
    4) Path to directory containing the electron cloud files
    5) The number of sdf files to be voxelized in each MPI instance
"""
posesPath = sys.argv[1]          # path to DIR w/ ligand poses
activeSiteFilePath = sys.argv[2] # path to FILE w/ voxelized active-site
outputPath = sys.argv[3]         # path to DIR for desired outputs
cloudPath = sys.argv[4]          # path to DIR containing cached electron clouds
mult = sys.argv[5]               # number of files to vox per MPI


#Initializes variables used for parralelization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


#Set Globals
voxRes = .5                              #length of each voxel in Angstroms
voxDim = 72                              #number of voxels in X,Y,Z directons
data_shape = (voxDim, voxDim, voxDim, 6) #shape for voxelized matrix
label_shape = ()                         #shape for labels (interation energies)
outName = str(rank)+'.h5'                #sets output name for hdf5 as MPI rank


"""
    Class used to store dataset properties and provides functionality to append
    values to each dataset in hdf5 file.
"""
class dataInfo:
    def __init__(self, dataset, shape):
        self.dataset = dataset
        self.shape = shape
        self.i = 0

    def appendVal(self, values):
        os.chdir(outputPath)
        with h5py.File(outName, mode='a') as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + 1, ) + self.shape)
            dset[self.i] = [values]
            self.i += 1
            h5f.flush()


def main():
    """
        Reads in cached active site HDF5 file and retrieves cached voxelized
        active-site information as well as all x, y, z transformations that
        were applied to the active-site. Then opens new hdf5 file to write
        outputs to.
    """
    hf = h5py.File(activeSiteFilePath,'r')
    siteMatrix = hf['activeCacheMatrix'][:]
    trans = hf['activeCacheTransformations'][:]
    hf.close()

    """
        Creates the output hdf5 file and initializes the labels, ligands, and file
        datasets in such a way that allows parralelization and compression. This
        also is helpful for allowing appending of values to the dataset.

        Hdf5 datasets:
        ligands - stores matrix of voxelized docked poses
        labels  - stores the cooresponding interaction energy to the ligand poses
        file    - stores the original sdf filename that the ligand was pulled from
    """
    os.chdir(outputPath)
    dtype = np.float32
    chunk_len=1
    compMethod = "gzip" #Type of compression used
    compLevel = 9       #Level of compression 1 - 9, 9 being most compression
    dt = h5py.special_dtype(vlen=str) #Datatype used to store strings in hdf5 file
    with h5py.File(outName, mode='w') as h5f:
        shape = data_shape
        h5f.create_dataset('ligands', shape=(0, ) + shape, maxshape=(None, ) + shape, dtype=dtype, chunks=(chunk_len, ) + shape, compression = compMethod, compression_opts = compLevel)
        shape = label_shape
        h5f.create_dataset('labels', shape=(0, ) + shape, maxshape=(None, ) + shape, dtype=dtype, chunks=(chunk_len, ) + shape, compression = compMethod, compression_opts = compLevel)
        h5f.create_dataset('file', shape=(0, ) + shape, maxshape=(None, ) + shape, dtype= dt, chunks=(chunk_len, ) + shape, compression = compMethod, compression_opts = compLevel)

    #Creates data info classes for each dataset
    ligands = dataInfo('ligands', data_shape)
    labels = dataInfo('labels', label_shape)
    file = dataInfo('file', label_shape)

    """
        Reads in a directory and saves the names of all files within two indicies.
        These files are the sdfs that are going to be voxelized by the current MPI.
    """
    os.chdir(posesPath)
    fromSdf = getFrom()  #Uses mpi to get an index number of what file to begin reading from (based on mult var)
    toSdf = getTo()      #Uses mpi to get an index number of what file to stop reading at (based on mult var)
    fileList = []
    for f in os.listdir(posesPath):
        if not f.startswith('.'):
            fileList.append(str(f))
    size = len(fileList)
    desiredFiles = []
    for i in range(fromSdf,toSdf):
        if i <= size:
            if i in range(len(fileList)):
                if os.path.isfile(fileList[i]):
                    desiredFiles.append(fileList[i])


    """
        For all files that were determined to be within the desired indicies,
        the file info and active-site data are sent to the voxelize funtion.
        The fuction will read the sdf file, voxelize data, added with the active-site
        data, and saved to an hdf5 file.
    """
    for i in desiredFiles:
        sdfVox(i, siteMatrix, trans, ligands, labels, file)



#Scales values to fit desired resolution
def upResCalculation(value):
    return int((value-(value % voxRes)) * (1/voxRes))





#Electron clouds are simplified into voxel form
def voxData(matrix, eList):
    for i in range(len(eList)):
        vx = upResCalculation(eList[i][0]) #X pos of e-
        vy = upResCalculation(eList[i][1]) #Y pos of e-
        vz = upResCalculation(eList[i][2]) #Z pos of e-
        vt = eList[i][3]                   #Element type of e-
        matrix[vx,vy,vz,vt] += 1           #Increment voxel value
    return matrix


#Returns atom symbol based on the atomic number
def getAtomType(num):
    typeTuple = [(1,'H'),(6,'C'),(7,'N'),(8,'O'),(9,'F'),(16,'S')]
    dic = dict(typeTuple)
    return dic[num]


#Returns channel number for output matrix based on atomic number
def gNum(inputANum):
    typeTuple = [(1,0),(6,1),(7,2),(8,3),(9,4),(16,5)]
    dic = dict(typeTuple)
    return dic[inputANum]


#Returns a sum rounded to the hundreths place
def addRoundHundredth(num1,num2):
    sum = ceil((num1 + num2) * 100) / 100.0
    return sum


#Voxelizes sdf ligand
def sdfVox(name, activeMatrix, trans, d, l, f):
    """
        Read in all molecules form the sdf files and save to list
    """
    molList = []
    molEnergy = 0
    molCount = 0
    os.chdir(posesPath)
    for mol in pybel.readfile('sdf', name):
        molList.append(mol)
        molCount += 1
    print(molCount)

    """
        For every molecule from the sdf file, go through voxelization process
    """
    for mol in molList:
        """
            Transforms the nuclei by the same transformations of the activesite
        """
        coords = [] #nucleus xyz location
        aNum = []   #elements atomic number
        for atom in mol:
            aNum.append(atom.atomicnum)
            coords.append(atom.coords)

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
            Adds the ligand information into the protein active site matrix in
            a manner that simulates the docked pose
        """
        tempMat = activeMatrix
        dockedLigandMatrix = voxData(tempMat, transformedElectrons)

        """
            Append all voxelized values to hdf5 file
        """
        molEnergy = mol.data['minimizedAffinity']
        outEnergy = np.asarray(molEnergy, dtype = np.float32)
        d.appendVal(dockedLigandMatrix)  #Appends matrix
        l.appendVal(outEnergy)           #Appends energy
        f.appendVal(np.string_(name))    #Appends file name
    os.chdir(posesPath)


#Uses mpi rank to calculate the max indicies of files to be voxelized
def getTo():
    toNum = ((rank+1)*mult)
    return toNum


#Uses mpi rank to calculate the min indicies of files to be voxelized
def getFrom():
    fromNum = (rank * mult)
    return fromNum


#Run the main fuction
if __name__ == "__main__":
    main()
