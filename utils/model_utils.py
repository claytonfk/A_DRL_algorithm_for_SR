# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 00:24:51 2018

@author: Clayton
"""

import tensorflow as tf

class FCLayer:
    def __init__(self, name, input_size, output_size, seed):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed
        
        with tf.variable_scope(name) as scope:
            # Attempt to initialize weights and biases
            try:
                self.__init_weights_and_biases()
            # If weights and biases already exist, reuse them
            except ValueError:
                scope.reuse_variables()
                self.__init_weights_and_biases()
            
            self.__init_copy_ops()

    def __init_weights_and_biases(self):
        # Defines initializer according to Xavier and Bengio's method
        initializer = tf.contrib.layers.xavier_initializer(seed = self.seed)

        # Initializes weights and biases
        self.weights = tf.get_variable(self.name + "w", [self.input_size, self.output_size], initializer = initializer)
        self.biases  = tf.get_variable(self.name + "b", [self.output_size], initializer = initializer)
        
    def __init_copy_ops(self):
        # Initializes placeholders for copy weight and biases operations
        self.weights_ph = tf.placeholder("float", [self.input_size, self.output_size])
        self.biases_ph  = tf.placeholder("float", [self.output_size])
        # Initializes copy operations
        self.weights_cp = tf.assign(self.weights, self.weights_ph)
        self.biases_cp  = tf.assign(self.biases, self.biases_ph)
        
    def fwd_pass(self, X):
        X = tf.add(tf.matmul(X, self.weights), self.biases)
        return X
    
class LSTMCell:
    def __init__(self, name, time_steps, output_size, seed):
        self.name = name
        self.time_steps = time_steps
        self.output_size = output_size
        self.seed = seed
        
        with tf.variable_scope(name) as scope:
            # Attempt to initialize weights and biases
            try:
                self.__init_weights_and_biases()
            # If weights and biases already exist, reuse them
            except ValueError:
                scope.reuse_variables()
                self.__init_weights_and_biases()

            
    def __init_weights_and_biases(self):
        # Defines initializer according to Xavier and Bengio's method
        initializer = tf.contrib.layers.xavier_initializer(seed = self.seed)


        self.cell = tf.contrib.rnn.LSTMCell(self.output_size, initializer = initializer, forget_bias = 1)
        
        
    def init_copy_ops(self):
        # Initializes placeholders for copy weight and biases operations
        self.kernel_ph = tf.placeholder("float")
        self.biases_ph = tf.placeholder("float")
        # Initializes copy operations
        self.kernel_cp = tf.assign(self.cell.weights[0], self.kernel_ph)
        self.biases_cp = tf.assign(self.cell.weights[1], self.biases_ph)
        
    def fwd_pass(self, X):
        with tf.variable_scope(self.name):
            X, _  = tf.nn.dynamic_rnn(self.cell, X, dtype = tf.float32)
            X = tf.transpose(X, [1, 0, 2])
            # Last output of the LSTM cell
            X = tf.gather(X, int(X.get_shape()[0]) - 1)
        return X