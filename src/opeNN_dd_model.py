""" Required Imports """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Just disables the warning, doesn't enable AVX/FMA
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from opeNNDD_dataset import OpeNNDD_Dataset
from tqdm import tqdm

class OpeNNdd_Model:
    loss_function = tf.losses.mean_squared_error

    def __init__(
        self,
        db_path = str(sys.argv[1]), # path for database
        tf_config = "gpu", # setting that determines if tensorflow should use GPU accelerated calculations
        input_dim = 72, # dimensions of input (always expecting a cube)
        num_channels = 2, # number of channels in input
        batch_size = 9, # batch_size of model
        num_output_units = 1, # number of outputs that the model should predict
        num_conv_filters = [32,64], # list containing number of convolutional filters for each desired convolutional layer
        uniform_num_seq_conv = None, # number of convolutional layers to do in a row before adding a pooling layer
        num_seq_conv = [1,1],
        uniform_kernel_dim = 5, # setting if user does not want the kernel dimensions to change throughout model.
        kernel_dim = [5,5], # setting to specify custom kernel size for each layer, set uniform_kernel_dim to None and enter a single kernel dimension for each convolutional layer
        uniform_pool_dim = 2, # setting if user does not want the pooling dimensions to change throughout model.
        pool_dim = [2,2], # setting to specify custom pooling dimensions for each convolutional block. Need to set uniform_pool_dim to None
        num_dense_nodes = [128], # setting to specify number of nodes for each dense layer, the length of this list is the number of dense layers
        uniform_dropout_conv = 0.1, # setting to specify uniform dropout across all convolutional layers
        dropout_conv = [0.1, 0.1], # setting to specify custom dropout across different convolutional layers (uniform_dropout_conv must be set to None)
        uniform_dropout_dense = 0.5, # setting to specify uniform dropout across all dense layers
        dropout_dense = [0.5], # setting to specify custom dropout across different dense layers
        optimizer = tf.train.AdamOptimizer, # optimization function for gradient descent
        init_learning_rate = 1e-4, # initial learning rate for optimizer
        id = 0, # unique id for model (used to specify folder for saving the model)
        model_path = "./tmp/best" # general file path to save models
    ):
        if tf_config.lower() == "gpu": # set gpu configurations if specified in model params
            self.config = tf.ConfigProto()
            self.config.gpu_options.allow_growth = True


        """ Initialize some general class variables from function arguments """
        self.db_path = db_path
        self.opeNN_dd_db = OpeNNDD_Dataset(db_path, batch_size) # initialize OpeNNDD_Dataset instantiation with the HDF5 files in db_path and a batch of batch_size
        self.num_output_units = num_output_units
        self.num_conv_filters = num_conv_filters
        self.num_conv_layers = len(num_conv_filters)
        self.num_dense_nodes = num_dense_nodes
        self.num_dense_layers = len(num_dense_nodes)
        self.optimizer = optimizer
        self.init_learning_rate = init_learning_rate
        self.model_path = os.path.join(model_path, "model"+str(id), "model.ckpt") # path to store model parameters

        """ Initialize kernel, pooling, and dropout parameters based on uniform of custom function arguments """
        if (uniform_num_seq_conv):
            self.num_seq_conv = [uniform_num_seq_conv for i in range(int(self.num_conv_layers/uniform_num_seq_conv))]
            print(self.num_seq_conv)
        else:
            self.num_seq_conv = num_seq_conv

        if (uniform_kernel_dim):
            self.kernel_dim = [uniform_kernel_dim for i in range(self.num_conv_layers)]
        else:
            self.kernel_dim = kernel_dim

        if (uniform_pool_dim):
            self.pool_dim = [uniform_pool_dim for i in range(len(num_seq_conv))]
        else:
            self.pool_dim = pool_dim

        if (uniform_dropout_conv):
            self.dropout_conv = [uniform_dropout_conv for i in range(self.num_conv_layers)]
        else:
            self.dropout_conv = manual_dropout_conv

        if (uniform_dropout_dense):
            self.dropout_dense = [uniform_dropout_dense for i in range(self.num_dense_layers)]
        else:
            self.dropout_dense = manual_dropout_dense

        self.layer_list = [tf.placeholder(tf.float32, shape=[None, input_dim, input_dim, input_dim, num_channels])] # instantiate layer list with an input layer
        self.labels = tf.placeholder(tf.float32, shape=[None, num_output_units]) # add a tensor to hold labels for computing loss
        self.layer_count = 0 # layer count will keep track of the index of the most current item in layer list (instantiated at 0 because input layer is at layer_list[0])

        self.add_conv_blocks() # function to add convolutional blocks to model
        self.add_dense_layers() # function to add dense layers to model
        self.optimization_step() # function to add optimizations to model

    def add_conv_blocks(self):
        num_conv = len(self.num_conv_filters) # number of convolutional layers in model
        num_pool = len(self.num_seq_conv)


        # number
        if (num_conv < len(self.num_seq_conv)):
            print("Total number of convolutional layers must be greater than and divisible by number of sequential layers before pooling step. Setting num_seq_conv to 1.")
            self.num_seq_conv = 1

        num_conv_pool_layers = 2*num_conv+num_pool # Need number of convolutional layers + number of convolutional dropout layers + number of pooling layers
        conv_count = 0
        seq_conv_count = 0
        pool_count = 0

        """ Dynamically Build Convolutional and Pooling Layers """
        while self.layer_count < num_conv_pool_layers:
            #Append convolutional layers to list of layers in model
            self.layer_list.append(
                tf.layers.conv3d(
                    inputs = self.layer_list[self.layer_count],
                    filters = self.num_conv_filters[conv_count],
                    kernel_size = [self.kernel_dim[conv_count], self.kernel_dim[conv_count], self.kernel_dim[conv_count]],
                    padding = "same",
                    activation = tf.nn.relu
                ))

            self.layer_count += 1 #increment layer count to stay at the end of the list of layers in the model

            # Append dropout layer to convolutional layer and increment layer list
            self.layer_list.append(tf.layers.dropout(inputs=self.layer_list[self.layer_count], rate=self.dropout_conv[conv_count]))
            self.layer_count += 1

            conv_count += 1 #increment number of convolutional counts to iterate through parameters specific to convolutional + dropout layers in the function params
            seq_conv_count += 1
            if (seq_conv_count == self.num_seq_conv[pool_count]): #if the number of sequential convolutional layers specified divides the number of convolutional layer already inserted, add a pooling layer
                self.layer_list.append(
                    tf.layers.max_pooling3d(
                        inputs=self.layer_list[self.layer_count],
                        pool_size=[self.pool_dim[pool_count], self.pool_dim[pool_count], self.pool_dim[pool_count]],
                        strides=self.pool_dim[pool_count]
                    ))
                seq_conv_count = 0
                pool_count+=1 #increment number of pooling counts to iterate through parameters specific to pooling layers in the function params
                self.layer_count+=1 #increment layer count to stay at the end of the list of layers in the model

        self.layer_list.append(tf.contrib.layers.flatten(inputs=self.layer_list[self.layer_count])) # append layer that flattens pooling layer
        self.layer_count+=1

    def add_dense_layers(self):
        dense_count = 0
        num_dense_layers = len(self.num_dense_nodes)
        while(dense_count < num_dense_layers):
            self.layer_list.append(tf.layers.dense(inputs=self.layer_list[self.layer_count], units=self.num_dense_nodes[dense_count], activation=tf.nn.relu))
            self.layer_count+=1
            self.layer_list.append(tf.layers.dropout(inputs=self.layer_list[self.layer_count], rate=self.dropout_dense[dense_count]))
            self.layer_count+=1
            dense_count+=1


        self.layer_list.append(tf.layers.dense(inputs=self.layer_list[self.layer_count], units=self.num_output_units, name="output_layer"))
        self.layer_count+=1

    def optimization_step(self):
        """ Loss Calculations """
        self.layer_list.append(tf.reduce_mean(tf.losses.mean_squared_error(labels = self.labels, predictions = self.layer_list[self.layer_count]), name="loss"))
        self.layer_count+=1
        """ Loss Minimization Step """
        self.layer_list.append(self.optimizer(self.init_learning_rate).minimize(self.layer_list[self.layer_count]))
        self.layer_count+=1

    def train(self):
        saver = tf.train.Saver()
        best_acc_val = float("inf")
        max_val_steps = 7
        current_val_step = 0
        num_epoch = 1
        #tf.reset_default_graph()
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())

            if os.path.exists(self.model_path):
                saver.restore(sess, self.model_path)
                print("Model restored.")
            else:
                print("No previous model found. New model will be created after first round of validation.")

            print("Starting Training...")
            while current_val_step < max_val_steps:
                print("Train Epoch #%d" % (num_epoch))
                self.opeNN_dd_db.shuffle_train_data()
                self.opeNN_dd_db.shuffle_val_data()

                train_iter = 0
                acc_train = 0

                for j in tqdm(range(int(self.opeNN_dd_db.total_train_steps))):
                    train_ligands, train_labels = self.opeNN_dd_db.next_train_batch()
                    train_op, train_targets, train_outputs, train_err = sess.run([self.layer_list[self.layer_count], self.labels, self.layer_list[self.layer_count-2], self.layer_list[self.layer_count-1]], feed_dict = {self.layer_list[0]: train_ligands, self.labels: train_labels})
                    train_iter+=1
                    acc_train+=train_err
                    avg_train = float(acc_train/train_iter)
                    print("Training - Target Value: \n", train_targets, '\n')
                    print("Training - CNN Output: \n", train_outputs, '\n')
                    print("Training - Quadratic Cost: ", train_err, '\n')
                    print("Training - Average Quadratic Cost: ", avg_train, '\n')

                val_iter = 0
                acc_val = 0
                print("Starting Validation...")
                for j in tqdm(range(opeNN_dd_db.total_val_steps)):
                    val_ligands, val_labels = opeNN_dd_db.next_val_batch()

                    labels = self.labels
                    output_layer = self.layer_list[self.layer_count-2]
                    quadratic_cost = self.layer_list[self.layer_count-1]

                    val_targets, val_outputs, val_err = sess.run([labels, output_layer, quadratic_cost], feed_dict = {input_layer: val_ligands, labels: val_labels})
                    val_iter+=1
                    acc_val+=val_err
                    avg_val = float(acc_val/val_iter)
                    print("Validation - Target Value: ", val_targets, '\n')
                    print("Validation - CNN Output: ", val_outputs, '\n')
                    print("Validation - Quadratic Cost: ", val_err, '\n')
                    print("Validation - Average Quadratic Cost: ", avg_val, '\n')

                if (avg_val < best_acc_val):
                    best_acc_val = avg_val
                    current_val_step = 0
                    save_path = saver.save(sess, self.model_path)
                    print("Best model saved in path: %s" % save_path)
                else:
                    current_val_step+=1

                num_epoch+=1

            print("Model finished training after %d epochs.\n" % (num_epoch))
            print("\nFinal Average Validation Accuracy: %f" % (best_acc_val))

    def test(self):
        tf.reset_default_graph()
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())


            assert saver.restore(sess, self.model_path), "No model exists for testing"

            print("Model restored.")
            test_iter = 0
            acc_test = 0
            print("Starting Testing...")
            for i in tqdm(range(opeNN_dd_db.total_test_steps)):
                test_ligands, test_labels = opeNN_dd_db.next_test_batch()

                labels = self.labels
                output_layer = self.layer_list[self.layer_count-2]
                quadratic_cost = self.layer_list[self.layer_count-1]

                test_targets, test_outputs, test_err = sess.run([labels, output_layer, quadratic_cost], feed_dict = {input_layer: test_ligands, labels: test_labels})
                test_iter+=1
                acc_test+=test_err
                avg_test = float(acc_test/test_iter)
                print("Testing - Target Value: ", test_targets, '\n')
                print("Testing - CNN Output: ", test_outputs, '\n')
                print("Testing - Quadratic Cost: ", test_err, '\n')
                print("Testing - Average Quadratic Cost: ", avg_test, '\n')

            print("Finished Testing. Final Average Testing Accuracy: %f" % (avg_test))

    def print_model_architecture(self):
        for i in range(len(self.layer_list)):
            print(self.layer_list[i])
