"""
    Takes in a path to a set of ligands that a network was trained on, and a
    path to a ligand that was not in the training set, then outputs out similar
    the outside ligand is to the training set.
"""


"""imports and constants"""
import numpy as np
import sys
import glob
import math

train_set = str(sys.argv[1]) #path to dir containing all train_set ligand fingerprints
pose = str(sys.argv[2]) #path to the fingerprint of a single pose

def calcSimilarity(set_dir, ligand_path):
    train_files = glob.glob(set_dir + "*.txt") #get all txt files from specified directory
    attributes = (len(open(train_files[0]).readlines()[0].split()) - 1) #get the number of attributes for a given train ligand
    assert (len(open(ligand_path).readlines()[0].split()) - 1), "ligands in train set have more or less fingerprint attributes than the pose to compare"

    #calculate sum of fingerprints of entire training set
    avgs = np.zeros(shape=(len(train_files),attributes), dtype=np.float32) #shape=[#_files, #_attributes_per_ligand]
    for j in range(len(train_files)): #for each training file
        i=0 #number of ligand poses that we have processed
        for line in open(train_files[j]).readlines(): #for each line in the file
            line = line.split()
            for k in range(attributes): #for each attribute per ligand
                avgs[j][k] += float(line[k+1])
            i+=1 #have just processed another pose
        for k in range(attributes): #avg out the fingerprint for this file
            avgs[j][k] /= i

    #calculate the avg train fingerprint and store in the train_fingerprint
    train_fingerprint = np.zeros(shape=attributes, dtype=np.float32) #vector to represent the avg fingerprint in the set
    for avg in avgs:
        train_fingerprint += avg
    for j in range(attributes):
        train_fingerprint[j] /= len(avgs)

    #get the outside ligand fingerprint...pose file should have one single line
    pose_fingerprint = open(ligand_path).readline()
    pose_fingerprint = pose_fingerprint.split()[1:]
    pose_fingerprint = np.asarray(pose_fingerprint, np.float32)

    #compare set fingerprint to the fingerprint of the new pose.. sqrt(sum(x-x^)/num_of_x's)
    sum = 0.0
    for j in range(attributes):
        print(train_fingerprint[j] - pose_fingerprint[j])
        sum += ((train_fingerprint[j] - pose_fingerprint[j]) ** 2)
    error = math.sqrt(sum / attributes)

    print(error)

if __name__ == '__main__':
    calcSimilarity(train_set,pose)
