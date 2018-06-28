import os
import random
import tables as tb
import numpy as np
import zipfile

NUM_LIGANDS = 625
NUM_CHANNELS = 3
GRID_DIMENSIONS = 32
HEADER_LENGTH = 6

class opeNN_dd_dataset:
    """ Class constructor that initializes all necessary variables (subject to change) """
    def __init__(self, data_set_dir, num_ligands = NUM_LIGANDS,
                    num_channels = NUM_CHANNELS, grid_dim = GRID_DIMENSIONS):
        self.num_ligands = num_ligands # tracks size of dataset
        self.num_train_ligands = int(0.6*self.num_ligands) # allocates 60% of dataset for training
        self.train_indices = list(range(self.num_train_ligands)) # stores list of indices in training set (makes shuffling data easy)
        self.num_val_ligands = int(0.2*self.num_ligands) # allocates 20% of dataset for validation
        self.val_indices = list(range(self.num_val_ligands)) # stores list of indices in validation set
        self.num_test_ligands = int(0.2*self.num_ligands) # allocates 20% of dataset for testing
        self.test_indices = list(range(self.num_test_ligands)) # stores list of indices in validation set
        self.num_classes = 1 # determines number of output classes in dataset
        self.num_channels = num_channels # determines number of channels in data (currently: 2, nuclei and electron counts)
        self.grid_dim = grid_dim # length of each side of the voxel grid (Length x Height x Width)
        self.hdf5_data_dir = os.path.join(data_set_dir, 'hdf5', 'conformers.hdf5') # directory for HDF5 file where database is stored
        self.hdf5_file = tb.open_file(self.hdf5_data_dir, mode='r') # keep HDF5 file open until training is finished
        self.train_batch_size = 16 # default size of training batch
        self.val_batch_size = 16 # default size of training batch
        self.train_batch_index = 0 # batch index to iterate through training dataset (step = train_batch_size)
        self.val_batch_index = 0 # batch index to iterate through validation dataset (step = val_batch_size)

    """ Functions for shuffling training, validation, and testing data """
    def shuffle_train_data(self):
        random.shuffle(self.train_indices)

    def shuffle_val_data(self):
        random.shuffle(self.val_indices)

    def shuffle_test_data(self):
        random.shuffle(self.test_data)

    """ Functions for fetching next mini-batch in training/validation sets """

    def next_train_batch(self, batch_size = self.batch_size):
        index = self.train_batch_index
        self.train_batch_size = batch_size

        """ Conditional that accounts for case if the total number of training ligands does not divide evenly by minibatch size """
        if (index + batch_size > self.num_train_ligands): # if index + batch_size will
            batch_indices = self.train_indices[index:] # get all indices from the current index to the end of the dataset
            self.shuffle_train_data() # reshuffle data after we reach the end of the dataset
            batch_indices += self.train_indices[:batch_size - (self.num_train_ligands-index)] # append remainder of batch size to batch
            index = batch_size-self.num_train_ligands%batch_size # update local index accordingly
        else:
            batch_indices = self.train_indices[index:index+batch_size] 
            index += batch_size

        self.train_batch_index = index
        batch_ligands = np.zeros([batch_size, self.grid_dim, self.grid_dim, self.grid_dim, self.num_channels])
        batch_energies = np.zeros([batch_size])
        for i in range(batch_size):
            batch_ligands[i] = self.hdf5_file.root.train_ligands[batch_indices[i]]
            batch_energies[i] = self.hdf5_file.root.train_labels[batch_indices[i]]

        batch = [batch_ligands, np.reshape(batch_energies, (batch_size,1))]
        return batch

    def next_val_batch(self, batch_size = self.batch_size):
        index = self.val_batch_index
        self.val_batch_size = batch_size
        if (index + batch_size > self.num_val_ligands):
            batch_indices = self.val_indices[index:]
            self.shuffle_val_data()
            batch_indices += self.val_indices[:batch_size - (self.num_val_ligands-index)]
            index = batch_size-self.num_val_ligands%batch_size
        else:
            batch_indices = self.val_indices[index:index+batch_size]
            index += batch_size

        self.val_batch_index = index
        batch_ligands = np.zeros([batch_size, self.grid_dim, self.grid_dim, self.grid_dim, self.num_channels])
        batch_energies = np.zeros([batch_size])
        for i in range(batch_size):
            batch_ligands[i] = self.hdf5_file.root.val_ligands[batch_indices[i]]
            batch_energies[i] = self.hdf5_file.root.val_labels[batch_indices[i]]

        batch = [batch_ligands, np.reshape(batch_energies, (batch_size,1))]
        return batch
