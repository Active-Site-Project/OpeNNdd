"""
    Given a directroy of only .hdf5 files, the script externally links all files
    to a newly created master file (user inputs master files name)
    SYSTEM ARGUMENTS
    1) Complete path to DIRECTORY containing all hdf5 files to be externally linked
    2) Desired name for output master hdf5 file
"""
import h5py as h5
import numpy as np
import os
import sys


#Initializes values using system arguments.
pathToHdf5Dir = str(sys.argv[1])  #path to DIRECTORY containing files to be linked
masterName = str(sys.argv[2])     #name for output master file

os.chdir(pathToHdf5Dir)
fileList = []
for f in os.listdir(pathToHdf5Dir):
    if not f.startswith('.'):
        fileList.append(str(f))

#initializes new h5 file
myfile = h5.File(masterName+'.h5','w')
myfile.create_group("ligands")
myfile.create_group("labels")
myfile.create_group("file")

#Links each file to master file
for i in range(len(fileList)):
    myfile['ligands'][str(i)] = h5.ExternalLink(fileList[i], "/ligands")
    myfile['labels'][str(i)] = h5.ExternalLink(fileList[i], "/labels")
    myfile['file'][str(i)] = h5.ExternalLink(fileList[i], "/file")
myfile.close()
