import os
import tables as tb
import numpy as np
import random
import math

#class for handling the OpeNNDD dataset.. takes a location to the data and a batch size for initialization
class OpeNNDD_Dataset:
    """data stats"""
    classes = 1 #num of classifications will be one since we want continuous output
    grid_dim = 72

    #instantiate with the hdf5 file and the train_batch_size of your choice
    def __init__(self, hdf5_file, batch_size, channels):
        assert os.path.exists(hdf5_file), 'file does not exist' #make sure the path to the specified file exists
        self.hdf5_file = tb.open_file(hdf5_file, mode='r') #handle to file
        self.total_train_ligands = len(self.hdf5_file.root.train_ligands)
        self.total_val_ligands = len(self.hdf5_file.root.val_ligands)
        self.total_test_ligands = len(self.hdf5_file.root.test_ligands)
        self.train_indices = list(range(self.total_train_ligands)) #[0,total_train_ligands)... will be used later to shuffle the data between epochs and when loading initial batch if necessary
        self.val_indices = list(range(self.total_val_ligands)) #[0,total_val_ligands)
        self.test_indices = list(range(self.total_test_ligands)) #[0,total_test_ligands)
        self.batch_size = batch_size #training batch size for getting next batch in the dataset
        self.total_train_steps = int(math.ceil(self.total_train_ligands / batch_size)) #total amount of steps in a single epoch dependent on the batch size
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
            batch_ligands[i-self.train_ligands_processed] = self.hdf5_file.root.train_ligands[self.train_indices[i]]
            batch_energies[i-self.train_ligands_processed] = self.hdf5_file.root.train_labels[self.train_indices[i]]


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
            batch_ligands[i-self.val_ligands_processed] = self.hdf5_file.root.val_ligands[self.val_indices[i]]
            batch_energies[i-self.val_ligands_processed] = self.hdf5_file.root.val_labels[self.val_indices[i]]


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
            batch_ligands[i-self.test_ligands_processed] = self.hdf5_file.root.test_ligands[self.test_indices[i]]
            batch_energies[i-self.test_ligands_processed] = self.hdf5_file.root.test_labels[self.test_indices[i]]


        if flag:
            self.test_ligands_processed = 0
        else:
            self.test_ligands_processed += batch_size

        #return as np arrays
        return batch_ligands, np.reshape(batch_energies, (batch_size,1))
