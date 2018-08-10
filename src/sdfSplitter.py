"""
    This program takes a folder of .sdf files that contain many molecules per
    file and splits them such that each molecule is saved into its own .sdf file.
    The program will store the new .sdf files into a directory specified by the
    user.

    SYSTEM ARGUMENTS:
    1) Path to read sdfs containing multiple molecules per sdf
    2) Path to store new sdfs containing only one molecule per sdf
"""

import pybel
import os
import sys


def main():
    #Initializes paths using system arguments
    readPath = str(sys.argv[1])  #the user input the path to read sdf files from
    storePath = str(sys.argv[2]) #the user inputs path to store the poses into


    os.chdir(readPath) #Navigate to the folder containing the sdf files to read
    nameList = []
    #Keep track of all file names inside of the read folder
    for filename in os.listdir(os.getcwd()):
        if not filename.startswith('.'):
            nameList.append(filename)

    i = 0 #Variable used to increment the filenames inside of the output folder

    #Splits the sdf file by each molecule and saves under an incrementing file name
    for j in range(len(nameList)):
        os.chdir(readPath)
        for mol in pybel.readfile('sdf', nameList[j]): #reads each sdf file in readPath
            os.chdir(storePath)
            while os.path.exists('pose%s.sdf' % i): #increments i to find next highest file name
                i += 1
            out = pybel.Outputfile('sdf', 'pose%s.sdf' % i) #creates new .sdf file
            out.write(mol)
            out.close()



#Run the main fuction
if __name__ == "__main__":
    main()
