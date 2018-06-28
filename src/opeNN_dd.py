""" Required Imports """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from opeNNDD_dataset import OpeNNDD_Dataset
from tqdm import tqdm

""" Configuration Options for Using GPUs """
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

HOME_DIR = str(Path.home()) # portable function to locate home directory on  a computer
NUM_EPOCHS = 3 # number of passes through data
DATASET_DIR = os.path.join(HOME_DIR, 'dev', 'OpeNN_dd','src', 'data', 'hdf5', 'conformers.hdf5') # directory of the tiny-imagenet-200 database
TRAIN_BATCH_SIZE = 25
VAL_BATCH_SIZE = 125
GRID_DIM = 32

""" Load Database """
opeNN_dd_db = OpeNNDD_Dataset(DATASET_DIR, TRAIN_BATCH_SIZE)

""" Declare Some Constants """
num_train_ligands = opeNN_dd_db.total_train_ligands
num_val_ligands = opeNN_dd_db.total_val_ligands
num_test_ligands = opeNN_dd_db.total_test_ligands
grid_dim = GRID_DIM
num_channels = opeNN_dd_db.channels
num_classes = opeNN_dd_db.classes

""" Declare Model Hyperparameters """
num_pool_layers = 2
num_filters_conv1 = 32
num_filters_conv2 = 64
num_filters_conv3 = 128
num_filters_conv4 = 256
num_nodes_fc1 = 512

filter_size_conv1 = 5
filter_size_conv2 = 5
filter_size_conv3 = 5
filter_size_conv4 = 5

""" Define Model Architecture """

""" Input Layer """
input_layer = tf.placeholder(tf.float32, shape=[None, grid_dim, grid_dim, grid_dim, num_channels])

""" Labels (target values) """
labels = tf.placeholder(tf.float32, shape=[None, 1])

""" Convolutional Layer #1 """
conv1 = tf.layers.conv3d(
    inputs = input_layer,
    filters = num_filters_conv1,
    kernel_size = [filter_size_conv1, filter_size_conv1, filter_size_conv1],
    padding = "same",
    activation = tf.nn.relu
)

""" Convolutional Layer #2 """
conv2 = tf.layers.conv3d(
    inputs = conv1,
    filters = num_filters_conv2,
    kernel_size = [filter_size_conv2, filter_size_conv2, filter_size_conv2],
    padding = "same",
    activation = tf.nn.relu
)

""" Pooling Layer #1 """
pool1 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)

""" Convolutional Layer #3 """
conv3 = tf.layers.conv3d(
    inputs = pool1,
    filters = num_filters_conv3,
    kernel_size = [filter_size_conv3, filter_size_conv3, filter_size_conv3],
    padding = "same",
    activation = tf.nn.relu
)

""" Convolutional Layer #4 """
conv4 = tf.layers.conv3d(
    inputs = conv3,
    filters = num_filters_conv4,
    kernel_size = [filter_size_conv4, filter_size_conv4, filter_size_conv4],
    padding = "same",
    activation = tf.nn.relu
)

""" Pooling Layer #2 """
pool2 = tf.layers.max_pooling3d(inputs=conv4, pool_size = [2,2,2], strides=2)

""" Flatten Final Pooling Layer """
pool2_flat = tf.contrib.layers.flatten(inputs=pool2)

""" Dense Layer #1 """
dense = tf.layers.dense(inputs=pool2_flat, units=num_nodes_fc1, activation=tf.nn.relu)

""" Dropout Regularization for Dense Layer #1 """
dropout = tf.layers.dropout(inputs=dense, rate=0.4)

""" Output Layer """
output_layer = tf.layers.dense(inputs=dropout, units=1, name="output_layer")

""" Loss Calculations """
quadratic_cost = tf.reduce_mean(tf.losses.mean_squared_error(labels = labels, predictions = output_layer), name="quadratic_cost")

""" Loss Minimization Step """
train_step = tf.train.AdamOptimizer(1e-4).minimize(quadratic_cost)

""" Create Object for Saving Model (not fully implemented) """
saver = tf.train.Saver()

""" Initialize TensorFlow Session + Training Loop """
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    opeNN_dd_db.shuffle_train_data()
    for i in range(NUM_EPOCHS):
        for j in tqdm(range(int(num_train_ligands/TRAIN_BATCH_SIZE))):
          train_ligands, train_labels = opeNN_dd_db.next_train_batch()
          train_op, outputs, targets, err = sess.run([train_step, output_layer, labels, quadratic_cost], feed_dict = {input_layer: train_ligands, labels: train_labels})
          print("Target Value: ", targets)
          print("CNN Output: ", outputs)
          print("Quadratic Cost: ", err)
