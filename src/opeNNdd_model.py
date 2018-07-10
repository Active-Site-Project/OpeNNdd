from __future__ import division
from opeNNdd_dataset import OpeNNdd_Dataset as open_data#class for the OpeNNdd dataset
from collections import OrderedDict #dictionary for holding network
import matplotlib.pyplot as plt
import tensorflow as tf #import tensorflow
import numpy as np
import sys #for unit tests
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #disables AVX/FMA warning
from tqdm import tqdm #progress bar
from pathlib import Path #for getting home folder
import random
from datetime import datetime
random.seed(datetime.now())

class OpeNNdd_Model:
    """
        Class to easily declare network models of different architectures and hyperparameters
        for the OpeNNdd_Dataset.
    """

    def __init__(self,
        hdf5_file = None, #complete file path to the hdf5 file where the data is stored
        batch_size = None, #number of images to use for train, val and test batches
        channels = None, #num of channel for each image
        conv_layers = None, #must provide a shape that each dim will specify features per layer.. ex. [32,64,64] -> 3 layers, filters of 32, 64, and 64 features
        conv_kernels = None, #must provide a shape that will specify kernel dim per layer.. ex. [3,5,5] -> 3x3x3 5x5x5 and 5x5x5 filters.. must have same num of dimenions as conv_layers
        pool_layers = None, #must provide a shape that each dim will specify filter size.. ex. [2,2,2] -> 3 pool layers, 2x2x2 filters and stride of 2 is always
        dropout_layers = None, #must be a shape where each dimension is the probability a neuron stays on or gets turned off... must check... ex. [.4,.4,.4] -> 3 layers with keep probability of 0.4
        fc_layers = None, #must provide a shape that each dim will specify units per connected layer.. ex. [1024,256,1] -> 3 layers, 1024 units, 256, units and 1 unit... last fully connected is the logits layer
        loss_function = None, #must be a tensorflow loss function
        optimizer = None, #must be a tensorflow optimizing function with a learning rate already... see unit tests example below
        ordering = None, #must be a string representing ordering of layers by the standard of this class... ex. "cpcpff" -> conv, max_pool, conv1, max_pool, fully connected, fully connected.. and the num of characters must match the sum of all of the dimensions provided in the layers variables
        model_folder = None, #complete path to an existing directory you would like model data stored
        log_folder = None,
        gpu_mode = False #booling for whether or not to enable gpu mode
    ):
        assert (len(conv_layers) + len(pool_layers) + len(dropout_layers) + len(fc_layers) == len(ordering)), "Number of layers does not equal number of entries in the ordering list."
        None if os.path.isdir(model_folder) else os.makedirs(model_folder) #create dir if need be
        self.id = random.randint(100000, 999999)
        self.model_folder = os.path.join(model_folder, str(self.id)) #append / onto model_folder if need be
        os.makedirs(self.model_folder)
        print(self.model_folder)
        self.log_folder = os.path.join(log_folder, str(self.id))
        self.db = open_data(hdf5_file, batch_size, channels) #handle for the OpeNNdd dataset
        self.conv_layers = conv_layers
        self.conv_kernels = conv_kernels
        self.pool_layers = pool_layers
        self.dropout_layers = dropout_layers
        self.fc_layers = fc_layers
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.ordering = ordering.lower() #convert all to lowercase for simplicity
        self.gpu_mode = gpu_mode
        self.flattened = False #flag to know if we have already flattened the data once we come to fully connected layers
        self.network_built = False #flag to see if we have already built the network
        self.epochs = 0 #number of epochs we have currently completed successfully with increasing validation accuracy
        self.stop_threshold = 0
        self.val_mvpe_arr = np.empty([0], dtype=float)
        self.avg_train_mvpe = 0
        self.avg_val_mvpe = 0
        self.avg_test_mvpe = 0

    #3d conv with relu activation
    def conv_3d(self, inputs, filters, kernel_size, name=None):
        out = tf.layers.conv3d(inputs, filters=filters, kernel_size=kernel_size,
                                 padding='same', activation=tf.nn.relu,
                                 name=name)
        tf.summary.histogram('act' + name, out)
        return out

    #max pooling with strides of 2 and same padding
    def max_pool3d(self, inputs, pool_size, name=None):
        out = tf.layers.max_pooling3d(inputs, pool_size=pool_size, strides=(2,2,2),
                                        padding='same', name=name)
        tf.summary.histogram('act' + name, out)
        return out

    #n-dimensions to 1-dimension
    def flatten(self, inputs):
        out = tf.contrib.layers.flatten(inputs)
        return out

    #fully connected layer with relu activation
    def dense_relu(self, inputs, units, name=None):
        out = tf.layers.dense(inputs, units, activation=tf.nn.relu,
                                name=name)

        tf.summary.histogram('act' + name, out)
        return out

    #fully connected no relu, or logits layer
    def dense(self, inputs, units, name=None):
        out = tf.layers.dense(inputs, units,
                                name=name)
        tf.summary.histogram('act' + name, out)
        return out

    #dynamicall build the network
    def build_network(self):
        self.network = OrderedDict({'labels': tf.placeholder(tf.float32, [None, open_data.classes])}) #start a dictionary with first element as placeholder for the labels
        self.network.update({'inputs': tf.placeholder(tf.float32, [None, open_data.grid_dim, open_data.grid_dim, open_data.grid_dim, self.db.channels])}) #append placeholder for the inputs
        c_layer, p_layer, d_layer, f_layer = 0, 0, 0, 0 #counters for which of each type of layer we are on

        #append layers as desired
        for command in self.ordering: #for each layer in network
            if command == 'c': #convolution
                shape = (self.conv_kernels[c_layer], self.conv_kernels[c_layer], self.conv_kernels[c_layer]) #convert dim provided into a tuple
                self.network.update({'conv'+str(c_layer): self.conv_3d(self.network[next(reversed(self.network))], self.conv_layers[c_layer], shape, 'conv'+str(c_layer))}) #append the desired conv layer
                c_layer += 1
            elif command == 'p': #max_pooling
                shape = (self.pool_layers[p_layer], self.pool_layers[p_layer], self.pool_layers[p_layer])
                self.network.update({'pool'+str(p_layer): self.max_pool3d(self.network[next(reversed(self.network))], shape, 'pool'+str(p_layer))})
                p_layer += 1
            elif command == 'd': #dropout
                self.network.update({'dropout'+str(d_layer): tf.nn.dropout(self.network[next(reversed(self.network))], self.dropout_layers[d_layer])})
                d_layer += 1
            elif command == 'f': #fully connected
                if f_layer == self.ordering.count('f') - 1: #we are appending the last fully connected layer.. so use dense no relu
                    if self.flattened:
                        self.network.update({'logits': self.dense(self.network[next(reversed(self.network))], self.fc_layers[f_layer], 'logits')})
                    else:
                        self.network.update({'logits': self.dense(self.flatten(self.network[next(reversed(self.network))]), self.fc_layers[f_layer], 'logits')})
                        self.flattened = True
                else: #dense with relu
                    if self.flattened:
                        self.network.update({'fc'+str(f_layer): self.dense_relu(self.network[next(reversed(self.network))], self.fc_layers[f_layer], 'fc'+str(f_layer))})
                    else:
                        self.network.update({'fc'+str(f_layer): self.dense_relu(self.flatten(self.network[next(reversed(self.network))]), self.fc_layers[f_layer], 'fc'+str(f_layer))})
                        self.flattened = True
                f_layer += 1
        self.network_built = True

        #append loss function and then optimizer
        self.network.update({'loss': tf.reduce_mean(self.loss_function(labels = self.network['labels'], predictions = self.network['logits']), name="quadratic_cost")})
        self.network.update({'optimizer': self.optimizer.minimize(self.network['loss'])})


    def mean_value_percentage_error(self, target, prediction):
        err = 0
        batch_size = target.shape[0]
        for i in range(batch_size):
            err += abs(target[i][0]-prediction[i][0])/abs(target[i][0])
        err *= 100/batch_size
        return err

    #train the model...includes validation
    def train(self):
        None if self.network_built else self.build_network() #Dynamically build the network if need be
        stop_count = 0
        train_iter = 0
        config = tf.ConfigProto()
        if self.gpu_mode == True: # set gpu configurations if specified
            config.gpu_options.allow_growth = True

        saver = tf.train.Saver() #ops to save the model
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer()) #initialize tf variables

            prev_error = float('inf')
            while True: #we are going to fing the number of epochs
                self.db.shuffle_train_data() #shuffle training data between epochs
                for step in tqdm(range(self.db.total_train_steps)):
                    print("Training Model... Step", step, "of", self.db.total_train_steps, "Epoch", self.epochs+1)
                    train_ligands, train_labels = self.db.next_train_batch() #get next training batch
                    train_op = sess.run([self.network['optimizer']], feed_dict={self.network['inputs']: train_ligands, self.network['labels']: train_labels}) #train and return predictions with target values
                    mvpe = self.mean_value_percentage_error(targets, outputs)
                    train_iter+=1

                error = self.validate(sess)
                if prev_error > error: #right now this early stopping only works for errors that will get less, but not accuracies that will become more
                    prev_error = error
                    saver.save(sess, os.path.join(self.model_folder, "model.ckpt"))
                else: #stop training becuase model did not improve with another pass thru the train set, self.epochs is the appropriate num of epochs..might need to change later
                    stop_count+=1
                    if (stop_count > self.stop_threshold):
                        return
                self.epochs += 1

    def validate(self, sess):
        self.db.shuffle_val_data()
        train_writer = tf.summary.FileWriter(os.path.join(self.log_folder, "tensorboard"), sess.graph)
        loss_arr = np.empty([self.db.total_val_steps], dtype=float)
        total_error = 0.0
        total_mvpe = 0
        for step in tqdm(range(self.db.total_val_steps)):
            print("Validating Model... Step", step, "of", self.db.total_val_steps)
            val_ligands, val_labels = self.db.next_val_batch()
            if (step % 10 == 0):
                merge = tf.summary.merge_all()
                summary, err = sess.run([merge, self.network['loss']],  feed_dict={self.network['inputs']: val_ligands, self.network['labels']: val_labels})
                train_writer.add_summary(summary, step)
            else:
                outputs, targets, err = sess.run([self.network['logits'], self.network['labels'], self.network['loss']],  feed_dict={self.network['inputs']: val_ligands, self.network['labels']: val_labels})
            mvpe = self.mean_value_percentage_error(targets, outputs)

            loss_arr[step] = mvpe
            total_error += err

        self.val_mvpe_arr = np.append(self.val_mvpe_arr, loss_arr)
        plt.plot(self.val_mvpe_arr)
        plt.xlabel('Number of Steps')
        plt.ylabel('Mean Absolute Percentage Error (%)')
        plt.show()
        return total_error / self.db.total_val_steps #return the avg error

    #restore the model and test
    def test(self):
        None if self.network_built else self.build_network() #Dynamically build the network if need be

        config = tf.ConfigProto()
        if self.gpu_mode == True: # set gpu configurations if specified
            config.gpu_options.allow_growth = True

        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer()) #initialize tf variables
            saver.restore(sess, self.model_folder)
            self.db.shuffle_test_data() #shuffle training data between epochs
            total_error = 0.0
            total_mvpe = 0
            for step in tqdm(range(self.db.total_test_steps)):
                print("Testing Model... Step", step, "of", self.db.total_test_steps)
                test_ligands, test_labels = self.db.next_test_batch() #get next training batch
                outputs, targets, err = sess.run([self.network['logits'], self.network['labels'], self.network['loss']], feed_dict={self.network['inputs']: test_ligands, self.network['labels']: test_labels}) #train and return predictions with target values
                total_mvpe += self.mean_value_percentage_error(targets, outputs)
                self.avg_test_mvpe = (total_mvpe)/(step+1)
                total_error += err
        return total_error / self.db.total_test_steps
