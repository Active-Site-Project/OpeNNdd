import os
import tables as tb
import numpy as np

#class for handling the OpeNNDD dataset.. takes a location to the data and a batch size for initialization
class OpeNNDD_Dataset:
    """data stats"""
    channels = 3 #num of channel for each image
    classes = 1 #num of classifications will be one since we want continuous output
    total_train_ligands = 0 #placeholders for the real values
    total_val_ligands = 0
    total_test_ligands = 0

    #instantiate with the hdf5 file and the train_batch_size of your choice
    def __init__(self, data_file, train_batch_size):
        assert os.path.exists(data_file), 'file does not exist' #make sure the path to the specified file exists
        self.data_file = tb.open_file(data_file, mode='r') #handle to file
        self.train_indices = list(range(self.total_train_ligands)) #[0,total_train_ligands)... will be used later to shuffle the data between epochs and when loading initial batch if necessary
        self.val_indices = list(range(self.total_val_ligands)) #[0,total_val_ligands)
        self.test_indices = list(range(self.total_test_ligands)) #[0,total_test_ligands)
        self.train_batch_size = train_batch_size #training batch size for getting next batch in the dataset
        self.total_train_steps = self.total_train_ligands / train_batch_size #total amount of steps in a single epoch dependent on the batch size
        assert self.total_train_steps/int(self.total_train_steps) > 0.0, "Did not choose a batch_size that divides evenly into the total training set."
        self.train_ligands_processed = 0

    def shuffle_train_data(self):
        random.shuffle(self.train_indices)

    def shuffle_val_data(self):
        random.shuffle(self.val_indices)

    def shuffle_test_data(self):
        random.shuffle(self.test_indices)


    def next_train_batch(self):
        batch_ligands = []
        batch_energies = []
        #get the next batch
        for i in range(self.train_ligands_processed, self.train_ligands_processed + self.train_batch_size):
            batch_ligands.append(self.data_file.root.train_ligands[self.train_indices[i]])
            batch_energies.append([self.data_file.root.train_labels[self.train_indices[i]]]) #must put brackets to make a list

        #if not train_batch_size ligands left,
        if (self.total_train_ligands - self.train_ligands_processed) < self.train_batch_size:
            self.train_ligands_processed = 0

        self.train_ligands_processed += self.train_batch_size #increment num of ligands we have currently processed

        #return as np arrays
        return np.array(batch_ligands, dtype=np.float32), np.array(batch_energies, dtype=np.float32)


    def val_set(self):
        batch_ligands = []
        batch_energies = []
        for i in self.val_indices:
            batch_ligands.append(self.data_file.root.val_ligands[i])
            batch_energies.append([self.data_file.root.val_labels[i]]) #must put brackets to make a list

        #return as np arrays
        return np.array(batch_ligands, dtype=np.float32), np.array(batch_energies, dtype=np.float32)

    def test_set(self):
        batch_ligands = []
        batch_energies = []
        for i in self.test_indices:
            batch_ligands.append(self.data_file.root.test_ligands[i])
            batch_energies.append([self.data_file.root.test_labels[i]]) #must put brackets to make a list

        #return as np arrays
        return np.array(batch_ligands, dtype=np.float32), np.array(batch_energies, dtype=np.float32)
