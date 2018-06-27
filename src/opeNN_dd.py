""" Required Imports """
import os
# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from opeNNDD_dataset import OpeNNDD_Dataset
from tqdm import tqdm
import sys

"""constants"""
TRAIN_BATCH_SIZE = 5
TRAIN_EPOCHS = 20
GRID_DIM = 32
num_pools = 2
HDF5_DATA_FILE = str(sys.argv[1])

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
pool_reduction = int(GRID_DIM / (2 * pow(2, num_pools-1))) #l, w of image after n pools
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
