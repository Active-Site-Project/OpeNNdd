import os
import h5py as h5
import numpy as np
import random
import math
import itertools

def binSearch(arr, target):
    low = 0
    high = len(arr)
    mid = (low + high)//2
    found = False

    if target < arr[0]:
        return 0
                            
    while (not found):
        if (target < arr[mid] and target >=arr[mid-1]):
            found = True
        elif (target >= arr[mid]):
            low = mid+1
            mid = (low+high)//2
        else:
            high = mid-1
            mid = (low+high)//2
    return mid


#class for handling the OpeNNdd dataset.. takes a location to the data and a batch size for initialization
class OpeNNdd_Dataset:
    """data stats"""
    classes = 1 #num of classifications will be one since we want continuous output
    grid_dim = 72
    train_split = .70
    val_split = .10
    test_split = .20

    #instantiate with the hdf5 file and the train_batch_size of your choice
    def __init__(self, hdf5_file, batch_size, channels):
        assert os.path.exists(hdf5_file), 'file does not exist' #make sure the path to the specified file exists
        self.hdf5_file = h5.File(hdf5_file, mode='r') #handle to file
        self.chunk_names = [name for name in self.hdf5_file['labels']]
        self.data_chunks = [len(self.hdf5_file['labels'][partition]) for partition in self.hdf5_file['labels']]
        self.chunk_thresholds = list(itertools.accumulate(self.data_chunks)) 
        self.total_ligands = self.chunk_thresholds[-1]
        self.total_train_ligands = int(round(self.train_split*self.total_ligands))
        self.total_val_ligands = int(round(self.val_split*self.total_ligands))
        self.total_test_ligands = int(round(self.test_split*self.total_ligands))
        self.ligand_indices = list(range(self.total_ligands)) #[0,total_train_ligands)... will be used later to shuffle the data between epochs and when loading initial batch if necessary
        self.train_indices = self.ligand_indices[0:self.total_train_ligands]
        self.val_indices = self.ligand_indices[self.total_train_ligands:self.total_ligands-self.total_test_ligands]
        self.test_indices = self.ligand_indices[self.total_train_ligands+self.total_val_ligands:]
        self.batch_size = batch_size #training batch size for getting next batch in the dataset
        self.total_train_steps = int(math.ceil(self.total_train_ligands /self.batch_size)) #total amount of steps in a single epoch dependent on the batch size
        self.total_val_steps = int(math.ceil(self.total_val_ligands/self.batch_size))
        self.total_test_steps = int(math.ceil(self.total_test_ligands/self.batch_size))
        self.channels = channels

        self.train_ligands_processed = 0
        self.val_ligands_processed = 0
        self.test_ligands_processed = 0

    def shuffle_train_data(self):
        random.shuffle(self.train_indices)
        self.train_ligands_processed = 0

    def shuffle_val_data(self):
        random.shuffle(self.val_indices)
        self.val_ligands_processed = 0

    def shuffle_test_data(self):
        random.shuffle(self.test_indices)
        self.test_ligands_processed = 0

    def next_train_batch(self):
        flag = False
        batch_size = self.batch_size
        #get the next batch
        if (self.total_train_ligands - self.train_ligands_processed) < batch_size:
            flag = True
            batch_size = self.total_train_ligands%batch_size

        batch_ligands = np.zeros([batch_size, self.grid_dim, self.grid_dim, self.grid_dim, self.channels], dtype=np.float32)
        batch_energies = np.zeros([batch_size], dtype=np.float32)
        for i in range(self.train_ligands_processed, self.train_ligands_processed+batch_size):
            file_index = binSearch(self.chunk_thresholds, self.train_indices[i])
            filename = str(self.chunk_names[file_index])
            chunk_index = (self.chunk_thresholds[file_index]-self.chunk_thresholds[file_index-1]-1) if file_index > 0 else self.train_indices[i] 
            batch_ligands[i-self.train_ligands_processed] = self.hdf5_file['ligands'][filename][chunk_index]
            batch_energies[i-self.train_ligands_processed] = self.hdf5_file['labels'][filename][chunk_index]


        if flag:
            self.train_ligands_processed = 0
        else:
            self.train_ligands_processed += batch_size

        #return as np arrays
        return batch_ligands, np.reshape(batch_energies, (batch_size,1))


    def next_val_batch(self):
        flag = False
        batch_size = self.batch_size
        #get the next batch
        if (self.total_val_ligands - self.val_ligands_processed) < batch_size:
            flag = True
            batch_size = self.total_val_ligands%batch_size

        batch_ligands = np.zeros([batch_size, self.grid_dim, self.grid_dim, self.grid_dim, self.channels], dtype=np.float32)
        batch_energies = np.zeros([batch_size], dtype=np.float32)
        for i in range(self.val_ligands_processed, self.val_ligands_processed+batch_size):
            file_index = binSearch(self.chunk_thresholds, self.val_indices[i])
            chunk_index = self.chunk_thresholds[file_index]-self.chunk_thresholds[file_index-1]-1 if file_index > 0 else self.val_indices[i]
            filename = str(self.chunk_names[file_index])
            batch_ligands[i-self.val_ligands_processed] = self.hdf5_file['ligands'][filename][chunk_index]
            batch_energies[i-self.val_ligands_processed] = self.hdf5_file['labels'][filename][chunk_index]


        if flag:
            self.val_ligands_processed = 0
        else:
            self.val_ligands_processed += batch_size

        #return as np arrays
        return batch_ligands, np.reshape(batch_energies, (batch_size,1))

    def next_test_batch(self):
        flag = False
        batch_size = self.batch_size
        #get the next batch
        if (self.total_test_ligands - self.test_ligands_processed) < batch_size:
            flag = True
            batch_size = self.total_test_ligands%batch_size

        batch_ligands = np.zeros([batch_size, self.grid_dim, self.grid_dim, self.grid_dim, self.channels], dtype=np.float32)
        batch_energies = np.zeros([batch_size], dtype=np.float32)
        for i in range(self.test_ligands_processed, self.test_ligands_processed+batch_size):
            file_index = binSearch(self.chunk_thresholds, self.test_indices[i])
            chunk_index = self.chunk_thresholds[file_index]-self.chunk_thresholds[file_index-1]-1 if file_index > 0 else self.test_indices[i]
            filename = str(self.chunk_names[file_index])
            batch_ligands[i-self.test_ligands_processed] = self.hdf5_file['ligands'][filename][chunk_index]
            batch_energies[i-self.test_ligands_processed] = self.hdf5_file['labels'][filename][chunk_index]

        if flag:
            self.test_ligands_processed = 0
        else:
            self.test_ligands_processed += batch_size

        #return as np arrays 
        return batch_ligands, np.reshape(batch_energies, (batch_size,1))
