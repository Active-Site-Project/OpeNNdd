""" Required Imports """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from opeNNDD_dataset import OpeNNDD_Dataset
from tqdm import tqdm
import sys

<<<<<<< HEAD
""" Configuration Options for Using GPUs """
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

HOME_DIR = str(Path.home()) # portable function to locate home directory on  a computer
NUM_EPOCHS = 50 # number of passes through data
DATASET_DIR = os.path.join(HOME_DIR, 'dev', 'OpeNN_dd','src', 'data') # directory of the tiny-imagenet-200 database
TRAIN_BATCH_SIZE = 25
VAL_BATCH_SIZE = 125
=======
"""constants"""
TRAIN_BATCH_SIZE = 5
TRAIN_EPOCHS = 20
>>>>>>> c098cecd4401c3b56485e6edea5d7e0ff167f8bb
GRID_DIM = 32
NUM_POOLS = 2
HDF5_DATA_FILE = str(sys.argv[1])

<<<<<<< HEAD
""" Load Database """
opeNN_dd_db = opeNN_dd_dataset(DATASET_DIR)

""" Declare Some Constants """
num_train_ligands = opeNN_dd_db.num_train_ligands
num_val_ligands = opeNN_dd_db.num_val_ligands
num_test_ligands = opeNN_dd_db.num_test_ligands
grid_dim = GRID_DIM
num_channels = opeNN_dd_db.num_channels
num_classes = opeNN_dd_db.num_classes

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
          train_batch = opeNN_dd_db.next_train_batch()
          train_op, outputs, targets, err = sess.run([train_step, output_layer, labels, quadratic_cost], feed_dict = {input_layer: train_batch[0], labels: train_batch[1]})
          print("Target Value: ", targets)
          print("CNN Output: ", outputs)
          print("Quadratic Cost: ", err)
=======
#-------------------------------------------------------------------------------
"""Helper functions."""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv3D(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding="SAME")

def max_pool3D(x):
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding="SAME")

#-------------------------------------------------------------------------------
"""Network architecture"""

x = tf.placeholder(tf.float32, shape=[None, GRID_DIM, GRID_DIM, GRID_DIM, OpeNNDD_Dataset.channels])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

#layer 1
num_filters_conv1 = 32
filter_size_conv1 = 5
W_conv1 = weight_variable([filter_size_conv1, filter_size_conv1, filter_size_conv1, OpeNNDD_Dataset.channels, num_filters_conv1])
b_conv1 = bias_variable([num_filters_conv1])

h_conv1 = tf.nn.relu(conv3D(x, W_conv1)+b_conv1)
h_pool1 = max_pool3D(h_conv1)

#layer 2
num_filters_conv2 = 64
filter_size_conv2 = 5
W_conv2 = weight_variable([filter_size_conv2, filter_size_conv2, filter_size_conv2, num_filters_conv1, num_filters_conv2])
b_conv2 = bias_variable([num_filters_conv2])
h_conv2 = tf.nn.relu(conv3D(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool3D(h_conv2)

#fc layer 1
pool_reduction = int(GRID_DIM / (2 * pow(2, NUM_POOLS-1))) #l, w of image after n pools
flat_res_2_layer = int(pool_reduction * pool_reduction * pool_reduction * num_filters_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, flat_res_2_layer])
W_fc1 = weight_variable([flat_res_2_layer, 128])
b_fc1 = bias_variable([128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout regularization layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

#logits layer
W_fc2 = weight_variable([128,OpeNNDD_Dataset.classes])
b_fc2 = bias_variable([OpeNNDD_Dataset.classes])
y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

#loss
quadratic_cost = tf.reduce_mean(tf.losses.mean_squared_error(labels = y_, predictions = y_conv))

#train and evaluate model
train_step = tf.train.AdamOptimizer(1e-4).minimize(quadratic_cost)
error_rate = tf.cast(tf.sigmoid(quadratic_cost), tf.float32)

#add ability to save the model
saver = tf.train.Saver()

#-------------------------------------------------------------------------------
"""main code"""
if __name__ == '__main__':
    data_set = OpeNNDD_Dataset(HDF5_DATA_FILE, TRAIN_BATCH_SIZE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAIN_EPOCHS):
            for j in range(int(data_set.total_train_steps)):
                train_ligands, train_labels = data_set.next_train_batch()
                print(train_ligands, train_labels)
                if j % 100:
                    val_ligands, val_labels = data_set.val_set()
                    print(error_rate.eval(feed_dict={x: val_ligands, y_: val_labels}))
                a,b,c = sess.run([train_step, quadratic_cost, error_rate], feed_dict={x: train_ligands, y_: train_labels, keep_prob: 0.5})

        test_ligands, test_labels = data_set.test_set()
        print("Test set error rate:", error_rate.eval(feed_dict={x: test_ligands, y_: test_labels, keep_prob: 0.5}))
>>>>>>> c098cecd4401c3b56485e6edea5d7e0ff167f8bb
