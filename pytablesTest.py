import os
import numpy as np
import tables as tb

""" Wherever is says 'extract_json_data' you would change to 'voxelize' or something like that """
def write_data_to_hdf5(grid_dim=72, num_channels=6):

    test_ligand = np.random.rand(grid_dim,grid_dim,grid_dim,num_channels)
    test_label = [np.random.rand()]
    data_shape = (0, grid_dim, grid_dim, grid_dim, num_channels)
    labels_shape = (0,)
    hdf5_data_dir = os.path.join("D:\\", "dev", "test_db", "test.hdf5")
    num_train_ligands = 70
    num_val_ligands = 10
    num_test_ligands = 20
    FILTERS = tb.Filters(complib='blosc', complevel=5)

    lig_dtype = tb.UInt64Atom()
    label_dtype = tb.Float32Atom()

    hdf5_file = tb.open_file(hdf5_data_dir, mode = 'w')

    train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_ligands', lig_dtype, shape=data_shape, filters=FILTERS, expectedrows=num_train_ligands)
    val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_ligands', lig_dtype, shape=data_shape, filters=FILTERS, expectedrows=num_val_ligands)
    test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_ligands', lig_dtype, shape=data_shape, filters=FILTERS, expectedrows=num_test_ligands)

    train_labels = hdf5_file.create_earray(hdf5_file.root, 'train_labels', label_dtype, shape=labels_shape, filters=FILTERS, expectedrows=num_train_ligands)
    val_labels = hdf5_file.create_earray(hdf5_file.root, 'val_labels', label_dtype, shape=labels_shape, filters=FILTERS, expectedrows=num_val_ligands)
    test_labels = hdf5_file.create_earray(hdf5_file.root, 'test_labels', label_dtype, shape=labels_shape, filters=FILTERS, expectedrows=num_test_ligands)

    #Write Training Data to HDF5 File
    print("Writing Training Ligands to File...")
    for i in range(num_train_ligands):
        ligand, label = [np.full((grid_dim,grid_dim,grid_dim,num_channels), 5)], [np.random.rand()]
        if (i > 0 and i % 1000==0):
            print("Training Ligand #%d" %(i))
        train_storage.append(ligand)
        train_labels.append(label)

    #Write Validation Data to HDF5 File
    print("Writing Validation Ligands to File...")
    for i in range(num_val_ligands):
        ligand, label = [np.full((grid_dim,grid_dim,grid_dim,num_channels), 6)], [np.random.rand()]
        if (i > 0 and i % 1000==0):
            print("Validation Ligand #%d" %(i))
        val_storage.append(ligand)
        val_labels.append(label)

    #Write Test Data to HDF5 File
    print("Writing Testing Ligands to File...")
    for i in range(num_test_ligands):
        ligand, label = [np.full((grid_dim,grid_dim,grid_dim,num_channels), 7)], [np.random.rand()]
        if (i > 0 and i % 1000==0):
            print("Testing Ligand #%d" %(i))
        test_storage.append(ligand)
        test_labels.append(label)

    hdf5_file.close()

write_data_to_hdf5()
hdf5_data_dir = os.path.join("D:\\", "dev", "test_db", "test.hdf5")
hdf5_file = tb.open_file(hdf5_data_dir, mode = 'r')
index = 0

for i in range(len(hdf5_file.root.train_labels)):
    print(hdf5_file.root.train_labels[i])
hdf5_file.close()
