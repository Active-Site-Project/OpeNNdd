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
    def __init__(self, data_set_dir, num_ligands = NUM_LIGANDS,
                    num_channels = NUM_CHANNELS, grid_dim = GRID_DIMENSIONS):
        self.num_ligands = num_ligands
        self.num_train_ligands = int(0.6*self.num_ligands)
        self.train_indices = list(range(self.num_train_ligands))
        self.num_val_ligands = int(0.2*self.num_ligands)
        self.val_indices = list(range(self.num_val_ligands))
        self.num_test_ligands = int(0.2*self.num_ligands)
        self.test_indices = list(range(self.num_test_ligands))
        self.num_classes = 1
        self.num_channels = num_channels
        self.grid_dim = grid_dim
        self.json_data_dir = os.path.join(data_set_dir, "json/conformers.zip")
        self.hdf5_data_dir = os.path.join(data_set_dir, "hdf5/conformers.hdf5")
        self.hdf5_file = tb.open_file(self.hdf5_data_dir, mode='r')
        self.train_batch_size = 16
        self.val_batch_size = 16
        self.train_batch_index = 0
        self.val_batch_index = 0

    def load_train_val_test(self):
        self.archive = zipfile.ZipFile(os.path.join(self.json_data_dir), 'r')
        file_names = self.archive.namelist()
        random.shuffle(file_names)
        self.training_data = file_names[:self.num_train_ligands]
        self.validation_data = file_names[self.num_test_ligands:self.num_test_ligands+self.num_val_ligands]
        self.test_data = file_names[self.num_train_ligands+self.num_val_ligands:]

    def shuffle_train_data(self):
        random.shuffle(self.train_indices)

    def shuffle_val_data(self):
        random.shuffle(self.val_indices)

    def shuffle_test_data(self):
        random.shuffle(self.test_data)


    def extract_json_data(self, file):
        try:
            json_file = self.archive.open(file, 'r')
        except AttributeError:
            self.load_train_val_test()
            json_file = self.archive.open(file, 'r')

        json_file.readline()
        grid_dim = int(json_file.readline().split()[2])
        if (self.grid_dim != grid_dim):
            self.grid_dim = grid_dim

        voxel_grid = np.zeros((grid_dim, grid_dim, grid_dim, NUM_CHANNELS))

        for i in range(0, HEADER_LENGTH-4):
            json_file.readline()

        interaction_energy = float(json_file.readline().split()[2])
        b = json_file.readline()
        while(b):
            b = json_file.readline()
            try:
                xpos, ypos, zpos = int(json_file.readline().split()[1][:-1]), int(json_file.readline().split()[1][:-1]), int(json_file.readline().split()[1][:-1])
            except:
                break

            protons, neutrons, electrons = int(json_file.readline().split()[1][:-1]), int(json_file.readline().split()[1][:-1]), int(json_file.readline().split()[1])
            voxel_grid[xpos, ypos, zpos, 0], voxel_grid[xpos, ypos, zpos, 1], voxel_grid[xpos, ypos, zpos, 2] = protons, neutrons, electrons
            b = json_file.readline()

        return voxel_grid, np.asarray([interaction_energy])

    def write_data_to_hdf5(self):
        data_shape = (0, self.grid_dim, self.grid_dim, self.grid_dim, self.num_channels)
        labels_shape = (0,)

        self.lig_dtype = tb.UInt64Atom()
        self.label_dtype = tb.Float32Atom()

        hdf5_file = tb.open_file(self.hdf5_data_dir, mode = 'w')

        self.train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_ligands', self.lig_dtype, shape=data_shape)
        self.val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_ligands', self.lig_dtype, shape=data_shape)
        self.test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_ligands', self.lig_dtype, shape=data_shape)
        self.mean_storage = hdf5_file.create_earray(hdf5_file.root, 'train_mean', self.lig_dtype, shape=data_shape)

        self.train_labels = hdf5_file.create_earray(hdf5_file.root, 'train_labels', self.label_dtype, shape=labels_shape)
        self.val_labels = hdf5_file.create_earray(hdf5_file.root, 'val_labels', self.label_dtype, shape=labels_shape)
        self.test_labels = hdf5_file.create_earray(hdf5_file.root, 'test_labels', self.label_dtype, shape=labels_shape)

        mean = np.zeros(data_shape[1:], np.float32)

        #Write Training Data to HDF5 File
        for i in range(self.num_train_ligands):
            if i %375 == 0 and i > 1:
                print('Train data: %d/%d' % (i, self.num_train_ligands))
            ligand, label = self.extract_json_data(self.training_data[i])
            self.train_storage.append(ligand[None])
            self.train_labels.append(label)

            mean += ligand / float(self.num_train_ligands)

        #Write Validation Data to HDF5 File
        for i in range(self.num_val_ligands):
            if i %125 == 0 and i > 1:
                print('Validation data: %d/%d' % (i, self.num_val_ligands))
            ligand, label = self.extract_json_data(self.validation_data[i])

            self.val_storage.append(ligand[None])
            self.val_labels.append(label)

        #Write Test Data to HDF5 File
        for i in range(self.num_test_ligands):
            if i %125 == 0 and i > 1:
                print('Test data: %d/%d' % (i, self.num_test_ligands))
            ligand, label = self.extract_json_data(self.test_data[i])

            self.test_storage.append(ligand[None])
            self.test_labels.append(label)

        self.mean_storage.append(mean[None])
        hdf5_file.close()

    def next_train_batch(self, batch_size):
        index = self.train_batch_index
        self.train_batch_size = batch_size
        if (index + batch_size > self.num_train_ligands):
            batch_indices = self.train_indices[index:]
            self.shuffle_train_data()
            batch_indices += self.train_indices[:batch_size - (self.num_train_ligands-index)]
            index = batch_size-self.num_train_ligands%batch_size
        else:
            batch_indices = self.train_indices[index:index+batch_size]
            index += batch_size

        self.train_batch_index = index
        batch_ligands = np.zeros([batch_size, self.grid_dim, self.grid_dim, self.grid_dim, self.num_channels])
        batch_energies = np.zeros([batch_size])
        print(len(batch_indices))
        for i in range(batch_size):
            batch_ligands[i] = self.hdf5_file.root.train_ligands[batch_indices[i]]
            batch_energies[i] = self.hdf5_file.root.train_labels[batch_indices[i]]

        batch = [batch_ligands, np.reshape(batch_energies, (batch_size,1))]
        return batch

    def next_val_batch(self, batch_size):
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
