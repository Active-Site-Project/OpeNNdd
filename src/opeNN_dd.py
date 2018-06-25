""" Required Imports """
import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from opeNN_dd_dataset import opeNN_dd_dataset
from tqdm import tqdm

HOME_DIR = str(Path.home()) # portable function to locate home directory on  a computer
NUM_EPOCHS = 3 # number of passes through data
DATASET_DIR = os.path.join(HOME_DIR, 'datasets/active-site/') # directory of the tiny-imagenet-200 database
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 16
GRID_DIM = 32

opeNN_dd_v1 = opeNN_dd_dataset(DATASET_DIR)

opeNN_dd_v1.load_train_val_test()
#opeNN_dd_v1.write_data_to_hdf5()
num_train_ligands = opeNN_dd_v1.num_train_ligands
num_val_ligands = opeNN_dd_v1.num_val_ligands
num_test_ligands = opeNN_dd_v1.num_test_ligands
grid_dim = GRID_DIM
num_channels = opeNN_dd_v1.num_channels
num_classes = opeNN_dd_v1.num_classes
num_layers = 2
num_filters_conv1 = 32
num_filters_conv2 = 64
filter_size_conv1 = 5
filter_size_conv2 = 5

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

x = tf.placeholder(tf.float32, shape=[None, grid_dim, grid_dim, grid_dim, num_channels])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_grid = tf.reshape(x, (-1, grid_dim, grid_dim, grid_dim, num_channels))

W_conv1 = weight_variable([filter_size_conv1, filter_size_conv1, filter_size_conv1, num_channels, num_filters_conv1])
b_conv1 = bias_variable([num_filters_conv1])

h_conv1 = tf.nn.relu(conv3D(x_grid, W_conv1)+b_conv1)
h_pool1 = max_pool3D(h_conv1)

W_conv2 = weight_variable([filter_size_conv2, filter_size_conv2, filter_size_conv2, num_filters_conv1, num_filters_conv2])
b_conv2 = bias_variable([num_filters_conv2])

h_conv2 = tf.nn.relu(conv3D(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool3D(h_conv2)

pool_reduction = int(grid_dim/(2*pow(2, num_layers-1)))
flat_res_2_layer = int(pool_reduction*pool_reduction*pool_reduction*num_filters_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, flat_res_2_layer])

W_fc1 = weight_variable([flat_res_2_layer, 128])
b_fc1 = bias_variable([128])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128,num_classes])
b_fc2 = bias_variable([num_classes])

y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

quadratic_cost = tf.reduce_mean(tf.losses.mean_squared_error(labels = y_, predictions = y_conv))
#quadratic_cost = tf.Print(quadratic_cost_out, [quadratic_cost_out])

train_step = tf.train.AdamOptimizer(1e-4).minimize(quadratic_cost)
error_rate = tf.cast(tf.sigmoid(quadratic_cost), tf.float32)
#accuracy = tf.reduce_mean(tf.metrics.accuracy(labels = y_, predictions = y_conv))


saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    try:
        saver.restore(sess, "./tmp/model.ckpt")
        print("Model restored.")
    except:
        pass

    for i in range(NUM_EPOCHS):
        print("Epoch #%d" % (i))
        opeNN_dd_v1.shuffle_train_data()
        for j in tqdm(range(int(num_train_ligands/TRAIN_BATCH_SIZE))):
            train_batch = opeNN_dd_v1.next_train_batch(TRAIN_BATCH_SIZE)
            print("step: %d" % (j))
            if j % 3 == 0 and j > 0:
                val_batch = opeNN_dd_v1.next_val_batch(VAL_BATCH_SIZE)
                #train_error = accuracy.eval(feed_dict = {x: train_batch[0], y_: train_batch[1], keep_prob: 1.0})
                #val_error = accuracy.eval(feed_dict = {x: val_batch[0], y_: val_batch[1], keep_prob: 1.0})
                train_error = 1-error_rate.eval(feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob: 1.0})
                val_error = 1-error_rate.eval(feed_dict={x: val_batch[0], y_: val_batch[1], keep_prob: 1.0})
                train_loss = quadratic_cost.eval(feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob: 1.0})
                val_loss = quadratic_cost.eval(feed_dict={x: val_batch[0], y_: val_batch[1], keep_prob: 1.0})
                print("step: %d" % (j))
                print("train accuracy: %g" % (train_error))
                print("train loss: %g" % (train_loss))
                print("validation accuracy: %g" % (val_error))
                print("validation loss: %g" % (val_loss))


                save_path = saver.save(sess, "./tmp/model.ckpt")
                print("Model saved in path: %s" % save_path)
            train_step.run(feed_dict = {x: train_batch[0], y_: train_batch[1], keep_prob: 0.5})

    opeNN_dd_v1.hdf5_file.close()
