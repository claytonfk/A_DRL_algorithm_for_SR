# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 01:34:52 2018

@author: Clayton
"""

import tensorflow as tf
import utils.model_utils as mu

class Model:
    def __init__(self, name, input_dims, lstm_units, mlp_num_neurons, num_outputs, 
                       time_steps = 1, lr = 0.001, discount_factor = 0.99, seed = 1):
        
        self.name = name                                # Name of the network
        self.input_dims = input_dims                    # Input dimensionality for the model
        self.num_parallel_blocks = len(input_dims)      # Number of parallel blocks
        self.lstm_units = lstm_units                    # Dimensionality of the output of LSTM cell

        self.num_neurons = mlp_num_neurons              # Number of neurons in each FC layer
        self.num_layers = len(mlp_num_neurons)          # Number of FC layers
        self.num_outputs = num_outputs                  # Number of outputs (actions)
        
        self.time_steps = time_steps                    # Number of time steps considered in the LSTM cell
        self.lr = lr                                    # Learning rate for the optimization algorithm
        self.discount_factor = discount_factor          # Discount factor for the expected return
        self.seed = seed                                # Seed for random weight initialization
        
        # Placeholders
        self.X = []
        
        self.X_own = tf.placeholder("float", [None, self.time_steps, self.input_dims[0]])
        self.X_comm = tf.placeholder("float", [None, self.time_steps, sum(input_dims[1:])])
        comm_list = tf.split(self.X_comm, self.num_parallel_blocks - 1, axis = 2)
        
        # Input to the model
        for i in range(0, self.num_parallel_blocks):
            if i == 0:
                self.X.append(self.X_own)            
            else:
                self.X.append(comm_list[i - 1])
            
        self.Y = tf.placeholder("float", [None, 1])                             # Outputs
        self.R = tf.placeholder("float")                                        # Rewards
        self.TO = tf.placeholder("float", [None, self.num_outputs])             # Total output
        self.BS = tf.placeholder(tf.int64)                                      # Batch size
        self.is_training = tf.placeholder(tf.bool)                              # Used for dropout
        self.loss_weights = tf.placeholder("float", [None, 1])                  # Due to prioritized experience replay
        
        # Networks with the same name share the same weights and biases
        with tf.variable_scope(self.name):
            self.__init_model()         # Initilizes model
            self.__init_inference_op()  # Initializes ops for inference
            self.__init_training_op()   # Initializes training operations
            self.__init_RL_op()         # Initializes ops for reinforcement learning 
    
    # Initializes model
    def __init_model(self):
        self.LSTMCells = []
        self.FCBlocks  = []
        for i in range(0, self.num_parallel_blocks):
            self.LSTMCells.append(mu.LSTMCell("LSTM" + str(i), self.time_steps, self.lstm_units, self.seed))
            FCBlock = []
            for j in range(0, self.num_layers):
                if j - 1 >= 0:
                    FCBlock.append(mu.FCLayer("FC" + str(i) + str(j), self.num_neurons[j - 1], self.num_neurons[j], self.seed))
                else:
                    FCBlock.append(mu.FCLayer("FC" + str(i) + str(j), self.lstm_units, self.num_neurons[j], self.seed))
                    
            self.FCBlocks.append(FCBlock)
        
        self.FCConcat = mu.FCLayer("FCConcat", self.num_neurons[-1]*self.num_parallel_blocks, self.num_neurons[-1], 
                                   self.seed)
        
        self.FCAdv = mu.FCLayer("FCAdv", self.num_neurons[-1], self.num_outputs, self.seed)
        #self.FCAdv = mu.FCLayer("FCAdv", self.num_neurons[-1]/2, self.num_outputs, self.seed)
        #self.FCValue = mu.FCLayer("FCValue", self.num_neurons[-1]/2, 1, self.seed)
                    
    # Initializes inference operations
    def __init_inference_op(self):
        # Total output of the network, i.e., action values
        self.total_output = self.forward(self.is_training)
        # For inference, squeeze the total output
        self.squeezed_total_output = tf.squeeze(self.total_output)
        
        # Action_ID with maximum value
        self.max_action_id = tf.argmax(self.total_output, axis = 1)
        self.max_action_id_squeezed = tf.squeeze(self.max_action_id)
        # Value of the action ID with maximum value
        self.max_action_value = tf.reduce_max(self.total_output, axis = 1)
        self.max_action_value_squeezed = tf.squeeze(self.max_action_value)

    # Initializes training operations
    def __init_training_op(self):
        
        # Outputs of the network (each action corresponds to one output)
        self.out = []
        # List of loss functions for each output
        self.loss = []
        # List of optimization operations for each output
        self.optimize = []
            
        for i in range(0, self.num_outputs):
            self.out.append(tf.transpose(tf.gather(tf.transpose(self.total_output), [i])))
            # Loss function for each output
            self.weighted_squared_difference = tf.multiply(self.loss_weights, tf.square(self.Y - self.out[i]))
            self.loss.append(tf.reduce_mean(self.weighted_squared_difference))
            # Optimization operation using Adam to minimize the loss function
            self.optimize.append(tf.train.GradientDescentOptimizer(learning_rate = self.lr).minimize(self.loss[i]))

    
    # Initiliazes operation for DDQN algorithm
    def __init_RL_op(self):
        self.DDQN = self.find_Y_DDQN(self.BS, self.R, self.TO)
    
    # Feedforward network
    def forward(self, is_training):
        parallel_x = []
        
        for i in range(0, self.num_parallel_blocks):
            x = self.X[i]
            x = self.LSTMCells[i].fwd_pass(x)
            for j in range(0, self.num_layers):
                x = self.FCBlocks[i][j].fwd_pass(x)
            
            parallel_x.append(x)
        
        x = tf.concat(parallel_x, axis = 1)
        x = self.FCConcat.fwd_pass(x)
        #stream_adv, stream_value = tf.split(x, 2, axis = 1)
        
        stream_adv   = self.FCAdv.fwd_pass(x)
        #stream_value = self.FCValue.fwd_pass(stream_value)
        
        total_out = stream_adv #stream_value + tf.subtract(stream_adv, tf.expand_dims(tf.reduce_mean(stream_adv, axis = 1), axis = 1))
        
        return total_out
    
    # Operation for DDQN algorithm
    def find_Y_DDQN(self, batch_size, rewards, target_net_total_output):
        # Batch indices
        batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
        # Indices to gather
        gather_idx = tf.concat([batch_idx, tf.expand_dims(self.max_action_id, 1)], axis = 1)
        
        x = tf.gather_nd(target_net_total_output, gather_idx)
        y = tf.add(rewards, tf.multiply(self.discount_factor, x))
        y = tf.expand_dims(y, axis = 1)
        
        return y
    
    # Copy weights between networks
    def copy_weights(self, session, copy_from):
        # Obtain weights from the network to be copied from
        feed_dict = {}
    
        copy_ops = []
        for i in range(0, self.num_parallel_blocks):
            self.LSTMCells[i].init_copy_ops()
            copy_ops += [self.LSTMCells[i].kernel_cp, self.LSTMCells[i].biases_cp]
            kernel, biases = session.run([copy_from.LSTMCells[i].cell.weights[0],
                                          copy_from.LSTMCells[i].cell.weights[1]])
            feed_dict[self.LSTMCells[i].kernel_ph] = kernel
            feed_dict[self.LSTMCells[i].biases_ph] = biases
            for j in range(0, self.num_layers):
                weights, biases = session.run([copy_from.FCBlocks[i][j].weights,
                                               copy_from.FCBlocks[i][j].biases])
                copy_ops += [self.FCBlocks[i][j].weights_cp, self.FCBlocks[i][j].biases_cp]
                feed_dict[self.FCBlocks[i][j].weights_ph] = weights
                feed_dict[self.FCBlocks[i][j].biases_ph]  = biases

        copy_ops += [self.FCConcat.weights_cp, self.FCConcat.biases_cp]
        copy_ops += [self.FCAdv.weights_cp, self.FCAdv.biases_cp]
        #copy_ops += [self.FCValue.weights_cp, self.FCValue.biases_cp]
        
        weights1, biases1, weights2, biases2 = session.run([copy_from.FCConcat.weights,
                                                                                copy_from.FCConcat.biases,
                                                                                copy_from.FCAdv.weights,
                                                                                copy_from.FCAdv.biases])
                                                                                #copy_from.FCValue.weights,
                                                                                #copy_from.FCValue.biases])
        
        feed_dict[self.FCConcat.weights_ph]   = weights1
        feed_dict[self.FCConcat.biases_ph]    = biases1
        feed_dict[self.FCAdv.weights_ph]      = weights2
        feed_dict[self.FCAdv.biases_ph]       = biases2
        #feed_dict[self.FCValue.weights_ph]    = weights3
        #feed_dict[self.FCValue.biases_ph]     = biases3
        
        # Perform the operation of copying weights
        session.run(copy_ops, feed_dict = feed_dict)
    
    # Set learning rate
    def set_learning_rate(self, lr):
        self.lr = lr
        for i in range(0, len(self.optimize)):
            self.optimize[i].learning_rate = lr