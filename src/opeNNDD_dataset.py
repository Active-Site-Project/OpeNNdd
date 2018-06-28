import os
import tables as tb
import numpy as np
import random

#class for handling the OpeNNDD dataset.. takes a location to the data and a batch size for initialization
class OpeNNDD_Dataset:
    """data stats"""
    channels = 2 #num of channel for each image
    classes = 1 #num of classifications will be one since we want continuous output
    grid_dim = 72

    #instantiate with the hdf5 file and the train_batch_size of your choice
    def __init__(self, hdf5_file, train_batch_size):
        assert os.path.exists(hdf5_file), 'file does not exist' #make sure the path to the specified file exists
        self.hdf5_file = tb.open_file(hdf5_file, mode='r') #handle to file
        self.total_train_ligands = len(self.hdf5_file.root.train_ligands)
        self.total_val_ligands = len(self.hdf5_file.root.val_ligands)
        self.total_test_ligands = len(self.hdf5_file.root.test_ligands)
        self.train_indices = list(range(self.total_train_ligands)) #[0,total_train_ligands)... will be used later to shuffle the data between epochs and when loading initial batch if necessary
        self.val_indices = list(range(self.total_val_ligands)) #[0,total_val_ligands)
        self.test_indices = list(range(self.total_test_ligands)) #[0,total_test_ligands)
        self.train_batch_size = train_batch_size #training batch size for getting next batch in the dataset
        self.total_train_steps = self.total_train_ligands / train_batch_size #total amount of steps in a single epoch dependent on the batch size
        self.train_ligands_processed = 0

    def shuffle_train_data(self):
        random.shuffle(self.train_indices)

    def shuffle_val_data(self):
        random.shuffle(self.val_indices)

    def shuffle_test_data(self):
        random.shuffle(self.test_indices)


    def next_train_batch(self):
        flag = False
        batch_size = self.train_batch_size
        batch_ligands = np.zeros([batch_size, self.grid_dim, self.grid_dim, self.grid_dim, self.channels])
        batch_energies = np.zeros([batch_size])
        #get the next batch
        if (self.total_train_ligands - self.train_ligands_processed) < self.train_batch_size:
            flag = True
            batch_size = self.total_train_ligands%self.train_batch_size
            print(batch_size)

        for i in range(self.train_ligands_processed, self.train_ligands_processed + batch_size):
            batch_ligands[i] = self.hdf5_file.root.train_ligands[self.train_indices[i]]
            batch_energies[i] = self.hdf5_file.root.train_labels[self.train_indices[i]]


        if flag:
            self.train_ligands_processed = 0
        else:
            self.train_ligands_processed += batch_size

        #return as np arrays
        return np.array(batch_ligands, dtype=np.float32), np.array(batch_energies, dtype=np.float32)


    def val_set(self):
        batch_ligands = np.zeros([self.total_val_ligands, self.grid_dim, self.grid_dim, self.grid_dim, self.channels])
        batch_energies = np.zeros([self.total_val_ligands])
        for i in self.val_indices:
            batch_ligands[i] = self.hdf5_file.root.train_ligands[self.val_indices[i]]
            batch_energies[i] = self.hdf5_file.root.train_labels[self.val_indices[i]]


        #return as np arrays
        return np.array(batch_ligands, dtype=np.float32), np.array(batch_energies, dtype=np.float32)

    def test_set(self):
        batch_ligands = np.zeros([self.total_test_ligands, self.grid_dim, self.grid_dim, self.grid_dim, self.channels])
        batch_energies = np.zeros([self.total_test_ligands])
        for i in self.test_indices:
            batch_ligands[i] = self.hdf5_file.root.train_ligands[self.test_indices[i]]
            batch_energies[i] = self.hdf5_file.root.train_labels[self.test_indices[i]]

        #return as np arrays
        return np.array(batch_ligands, dtype=np.float32), np.array(batch_energies, dtype=np.float32)
