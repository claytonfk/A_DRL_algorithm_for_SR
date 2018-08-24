# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 19:18:11 2018

@author: Clayton
"""

import vrep
import numpy as np
import agent as ag
import environment as env
import model as nn
import prioritized_EM as em
import tensorflow as tf
import utils
import time
import random

random.seed(0)

# COMMUNICATION PORT WITH V-REP
comm_port                  = 19999
# RESTORE OPTIONS
restore_model              = 1                                                 # Restore from saved model
restore_em                 = 1                                                 # Restore experience memory from saved file
path_to_model_to_restore   = "./final_models/6_SF_FINAL3.ckpt-80"               # Path to the saved model
path_to_results_to_restore = "./final_results/6_RESULTS_SF_FINAL3.pkl"          # Path to file containing results to restore

# SAVE OPTIONS
save_model_frequency    = 10                                                  # Frequency of saving models (in episodes)
max_to_keep             = 15                                                  # Maximum number of models to keep saved
path_to_model_to_save   = "./final_models/8_SF_FINAL.ckpt"                    # Path to save the model
path_to_results_to_save = "./final_results/8_RESULTS_SF_FINAL.pkl"            # Path to save the results
save_offset             = 0                                                        

# NEURAL NETWORK OPTIONS
training_frequency      = 1                 # Frequency of training (steps)
batch_size              = 2000              # Batch size for training
time_steps              = 4                 # Number of time steps to consider in the recurrent network
num_actions             = 5                 # Number of actions (ALWAYS 5)
num_readings            = 8                 # Number of sensor readings
input_dims              = [13, 10, 10]      # Input dimensionality of the NN
lstm_units              = 50                # Output dimensionality of the LSTM cells
num_nearest_agents_comm = 2                 # Number of neighbors used for communication
num_neurons             = [50, 50]          # Number of neurons in each of the three layers
copy_weights_frequency  = 20                # Frequency for copying weights from policy to target network
discount_factor         = 0.90              # Discount factor
learning_rate           = 0.0008            # Learning rate for training the neural network

# STAGE OPTIONS
num_agents       = 8              # Number of agents in the environment
num_episodes     = 30             # Number of episodes
max_discrepancy  = 200            # Maximum discrepancy per agent
min_discrepancy  = 0.1            # Minimum discrepancy per agent
task             = 1              # 0: dispersion, 1: square formation, 2: aggregation, 3: chain formation
steps_limit      = 600            # Limit of steps per episode
desired_distance = 0.28           # Side of squares in square formation task, 
                                  # or maximum distance to nearest partnet in dispersion task

# REPLAY MEMORY OPTIONS            
em_capacity   = 500000                                   # Capacity of the experience memory
alpha         = 0.9                                      # Exponent alpha
initial_beta  = 0.1                                      # Exponent beta (initially)
final_beta    = 0                                        # Final exponent beta
beta_increase = (final_beta - initial_beta)/num_episodes # Exponent beta increase

# EXPLORATION AND EXPLOITATION OPTIONS
initial_b_temperature = 2    #  Initial boltzmann temperature
final_b_temperature   = 0    #  Final boltzmann temperature
# Boltzmann temperature decay at each epsiode
b_temperature_decay = (initial_b_temperature - final_b_temperature)/num_episodes 

# SEARCH OPTIONS
use_search = 0
search_frequency = 1
num_episodes_search = 20
num_smart_agents = 3

if __name__ == "__main__":
    vrep.simxFinish(-1) # Close all open connections
    client_ID = vrep.simxStart('127.0.0.1', comm_port, True, False, 5000, 5) # Connect to V-REP
    
    assert client_ID != -1, "Could not connect to remote API server."
    print('Connected to remote API server. Agents might move chaotically until the simulation is fully initialized.')
    
    # Start simulation
    vrep.simxStartSimulation(client_ID, vrep.simx_opmode_blocking)
    
    # Initialize all agents
    agent_list = [] # List containing agents
    
    for i in range(0, num_agents):
        if i == 0:
            agent_list.append(ag.Agent(client_ID, i))
        else:
            agent_list.append(ag.Agent(client_ID, i, name = "ePuck#" + str(i - 1)))
                                          
    # Initialize the environment
    print('Creating environment.')
    e = env.Environment(client_ID, agent_list, desired_distance = desired_distance, 
                        success_threshold = min_discrepancy, fail_threshold = max_discrepancy,
                        skip_steps = 1, show_display = 0, steps_limit = steps_limit)
    
    # Initialize replay memory 
    print('Initializing prioritized experience memory.')
    if restore_em:
        _, total_summary_list, p_em = utils.load(path_to_results_to_restore)
    else:
        p_em = em.PrioritizedExperienceMemory(capacity = em_capacity, alpha = alpha, beta = initial_beta)
        total_summary_list     = []  # List containing relevant info about the training

    # Initialize the deep Q-networks
    print('Initializing networks.')
    num_outputs = num_actions
    policy_net = nn.Model('policy', 
                          input_dims = input_dims,
                          lstm_units = lstm_units, 
                          mlp_num_neurons = num_neurons, 
                          num_outputs = num_outputs, 
                          time_steps = time_steps, 
                          lr = learning_rate, 
                          discount_factor = discount_factor)

    target_net = nn.Model('target',
                          input_dims = input_dims,
                          lstm_units = lstm_units, 
                          mlp_num_neurons = num_neurons, 
                          num_outputs = num_outputs, 
                          time_steps = time_steps, 
                          lr = learning_rate, 
                          discount_factor = discount_factor)

    # Initialize session in TensorFlow
    print('Starting session.')
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep = max_to_keep)
    
    # Restore model or not
    if restore_model:
        print('Restoring saved model.')
        saver.restore(session, path_to_model_to_restore)
        # Set learning rate
        policy_net.set_learning_rate(learning_rate)
    else:
        session.run(tf.global_variables_initializer())
        
    total_reward_list      = []  # List containing all rewards of all agents per step
    total_discrepancy_list = []  # List containing discrepancies for each step
    reset_positions        = 1   # Reset positions of the robots
    
    print('Starting simulation.')
    start_time = time.time()
    boltzmann_temperature = initial_b_temperature
    
    for episode in range(0, num_episodes):
        e.start_new_episode(episode, reset_positions)
        
        step                   = 0      # Step in the episode
        end                    = 0      # If true, the episode must end
        total_loss             = 0      # Total loss in the trainings for each episode
        loss_counts            = 0      # Number of times the loss function was minimized
        total_reward_list      = []     # List containing all rewards of all agents per step
        total_discrepancy_list = []     # List containing discrepancies for each step
        
        if episode > 0:
            # Increase exponent beta in replay memory
            p_em.beta += beta_increase
            boltzmann_temperature -= b_temperature_decay
            
        agent_positions =  e.get_agent_positions(step)
        agent_orientations = e.get_agent_orientations(step)
        
        if step % 5 == 0:
            if task == 1:
                best_comb = utils.find_best_square_combination(agent_positions, desired_distance, n_squares = -1)
            else:
                best_comb = -1
        
        if use_search and episode < num_episodes_search:
            _, d, desired_agent_positions = utils.search_all_agent_positions(agent_positions, -1, 
                                                                             best_comb, 0.2, 
                                                                             desired_distance,
                                                                             20, 20, task)
    
        # Start the episode
        while not end:
            # Choose an action for all agents
            own_inputs = []
            comm_inputs = []
            for a in agent_list:
                readings = a.get_sensor_buffer_readings(step, time_steps) # Obtain buffered sensor readings

                actions_buffer = a.get_action_buffer(step, time_steps) # Obtain buffered action IDs
                actions_buffer_encoded = utils.actions_encoded(actions_buffer)
                
                own_inputs.append(utils.concatenate_readings_actions(readings, actions_buffer_encoded)) # Concatenate
                #own_inputs.append(readings)

                # Communication
                comm_inputs.append(e.comm_nearest_agent(a.get_agent_ID(), step, num_nearest_agents_comm, time_steps))
                
            with tf.device('/gpu:0'):
                # Inference dictionary for placeholders
                inference_dict = {target_net.X_own: own_inputs, target_net.X_comm: comm_inputs, target_net.is_training: False}
                # Action values
                q_out = session.run(target_net.squeezed_total_output, feed_dict = inference_dict)
                
            agent_positions =  e.get_agent_positions(step)
            agent_orientations = e.get_agent_orientations(step)
            if task == 1:
                best_comb = utils.find_best_square_combination(agent_positions, 0.28, n_squares = -1)
            
            search_turn = (step % search_frequency == 0) * use_search * (episode < num_episodes_search)
            
            # In case search is being used
            if search_turn:
                smart_agents = []
                for i in range(0, num_smart_agents):
                    done = 0
                    while not done:
                        random_agent = random.randint(0, num_agents)
                        if random_agent not in smart_agents:
                            smart_agents.append(random_agent)
                            done = 1
                smart_actions = utils.search_all_agent_actions(desired_agent_positions, agent_positions, agent_orientations)
                
            for a in agent_list:
                # In case search is being used
                if search_turn:
                    if a.get_agent_ID() in smart_agents:
                        action_ID = smart_actions[a.get_agent_ID()]
                    else:
                        action_ID = utils.boltzmann_exploration(q_out[a.get_agent_ID()], boltzmann_temperature)
                else:
                    action_ID = utils.boltzmann_exploration(q_out[a.get_agent_ID()], boltzmann_temperature)
                
                # Obtain joint speeds according to action_ID
                right_joint_speed, left_joint_speed = utils.action_ID_to_speeds(action_ID) 
                a.actuate(right_joint_speed, left_joint_speed, action_ID) # Actuate agent
                
            # Go to next step and iterate
            step = step + 1
            # Obtain rewards, discrepancy and whether the task has reached its end
            if task == 0:
                rewards, discrepancy, end, reset_positions = e.iterate_training_dispersion(step)
            elif task == 1:
                rewards, discrepancy, end, reset_positions = e.iterate_training_square_formation(step, best_comb)
            elif task == 2:
                rewards, discrepancy, end, reset_positions = e.iterate_training_aggregation(step)
            elif task == 3:
                rewards, discrepancy, end, reset_positions = e.iterate_training_chain_formation(step)
                
            # Stores discrepancy and rewards
            total_discrepancy_list.append(discrepancy)
            total_reward_list.append(sum([rewards[key] for key in rewards]))
                
            # Store sequence in the replay memory
            for a in agent_list:
                new_readings = a.get_sensor_buffer_readings(step, time_steps) # Get newest sensor readings
                new_actions_buffer = a.get_action_buffer(step, time_steps) # Obtain newest buffer of action IDs
                new_actions_buffer_encoded = utils.actions_encoded(new_actions_buffer) # One-hot encoding

                new_own_inputs = utils.concatenate_readings_actions(new_readings, new_actions_buffer_encoded) # Concatenate
                #new_own_inputs = new_readings
                new_comm_inputs = e.comm_nearest_agent(a.get_agent_ID(), step, num_nearest_agents_comm, time_steps) 
                
                new_inputs = np.concatenate([new_own_inputs, new_comm_inputs], axis = 1).tolist()
                
                old_readings = a.get_sensor_buffer_readings(step - 1, time_steps) # Get old sensor readings
                old_actions_buffer = a.get_action_buffer(step - 1, time_steps) # Obtain old buffer of action IDs
                old_actions_buffer_encoded = utils.actions_encoded(old_actions_buffer) # One-hot encoding
                
                old_own_inputs = utils.concatenate_readings_actions(old_readings, old_actions_buffer_encoded) # Concatenate
                #old_own_inputs = old_readings
                old_comm_inputs = e.comm_nearest_agent(a.get_agent_ID(), step - 1, num_nearest_agents_comm, time_steps)
                
                old_inputs = np.concatenate([old_own_inputs, old_comm_inputs], axis = 1).tolist()
                
                agent_ID = a.get_agent_ID() # Get agent ID

                last_action_ID = a.get_last_action_ID() # Obtain last action
                # Store into experience memory (by default with maximum priority)
                p_em.add(old_inputs, a.get_last_action_ID(), rewards[str(agent_ID)], new_inputs, episode)
                
                # Accumulate rewards for the episode
                a.accum_rewards(rewards[str(agent_ID)])
            

            if not step % training_frequency:
                # Sample minibatches from the experience memory
                minibatch, idx, weights = p_em.sample(batch_size)
                # Separate by action_ID
                states, rewards, next_states, idx, weights = utils.organize_minibatch(minibatch, idx, weights, num_actions)
                outputs = [[] for _ in range(0, num_actions)]
                # Train with GPU
                with tf.device('/gpu:0'):
                    for action_ID in range(0, num_actions):
                        if len(states[action_ID]) != 0:
                            states_action_ID = np.array(states[action_ID])
                            next_states_action_ID = np.array(next_states[action_ID])
                            
                            own_states = states_action_ID[:, :, range(0, num_readings + num_actions)]
                            comm_states = states_action_ID[:, :, range(num_readings + num_actions, 
                                                                       num_readings + num_actions + \
                                                                       (num_readings + 2)*num_nearest_agents_comm)]
    
                            next_own_states = next_states_action_ID[:, :, range(0, num_readings + num_actions)]
                            next_comm_states = next_states_action_ID[:, :, range(num_readings + num_actions, 
                                                                                 num_readings + num_actions + \
                                                                                 (num_readings + 2)*num_nearest_agents_comm)]
                            
                            # feed_dict for target network
                            target_dict = {target_net.X_own: next_own_states,
                                           target_net.X_comm: next_comm_states,
                                           target_net.is_training: False}
                            # Obtain total output from target network
                            target_net_outputs = session.run(target_net.total_output, feed_dict = target_dict)

                            # feed_dict for policy network
                            policy_dict = {policy_net.BS: len(own_states),
                                           policy_net.R: rewards[action_ID],
                                           policy_net.TO: target_net_outputs,
                                           policy_net.X_own: next_own_states,
                                           policy_net.X_comm: next_comm_states,
                                           policy_net.is_training: False}
                            # Obtain desired outputs to be trained with
                            outputs[action_ID] = session.run(policy_net.DDQN, feed_dict = policy_dict)

                            # For computing TD error
                            policy_dict = {policy_net.X_own: own_states,
                                           policy_net.X_comm: comm_states,
                                           policy_net.is_training: False}
                            policy_net_outputs = session.run(policy_net.total_output, feed_dict = policy_dict)
                            
                            # Compute TD error
                            policy_net_outputs_action_ID = [q[action_ID] for q in policy_net_outputs]
                            abs_td_error = [abs(int(output) - q) for (output, q) in \
                                            zip(outputs[action_ID], policy_net_outputs_action_ID)]
                            
                            # Update priorities in the experience memory
                            p_em.update_priorities(abs_td_error, idx[action_ID])
                            
                            # feed_dict for training
                            train_dict = {policy_net.X_own: own_states,
                                          policy_net.X_comm: comm_states,
                                          policy_net.Y: outputs[action_ID],
                                          policy_net.is_training: True,
                                          policy_net.loss_weights: np.expand_dims(weights[action_ID], axis = 1)}
                            
                            # Train and obtain the value of the loss
                            _, loss = session.run([policy_net.optimize[action_ID],
                                                   policy_net.loss[action_ID]], feed_dict = train_dict)
                            
                            total_loss = loss + total_loss # Accumulate loss
                            loss_counts = loss_counts + 1  # Count how many losses have been accumulated
                
            # Copy weights from policy network to target network
            if step % copy_weights_frequency == 0 and step != 0:
                with tf.device('/gpu:0'):
                    target_net.copy_weights(session, policy_net)
            
            # In case it is the end of the episode
            if end:
                avg_total_reward = np.mean(total_reward_list)
                avg_total_discrepancy = np.mean(total_discrepancy_list)
                min_total_discrepancy = np.min(total_discrepancy_list)
                max_total_discrepancy = np.max(total_discrepancy_list)
                
                summary_discrepancy_episode = [avg_total_reward, avg_total_discrepancy, min_total_discrepancy, \
                                               max_total_discrepancy]
                summary_discrepancy_episode = [round(content, 3) for content in summary_discrepancy_episode]
                total_summary_list.append(summary_discrepancy_episode)
                
                if loss_counts:
                    avg_loss = total_loss/loss_counts
                else:
                    avg_loss = 0
                
                print("EPISODE %i. Avg D, Min D, Max D: %s. Loss: %f." 
                      % (episode + 1, ', '.join(map(str, summary_discrepancy_episode)), avg_loss))
                
                # Save model
                if (episode + 1) % save_model_frequency == 0:
                    saver.save(session, path_to_model_to_save, global_step = episode + 1 + save_offset)
                    print("Checkpoint saved")

                    info_task_list = [training_frequency, batch_size, time_steps, num_actions, num_neurons, 
                                      lstm_units, copy_weights_frequency, discount_factor, learning_rate, 
                                      num_agents, num_episodes, max_discrepancy, min_discrepancy, task, steps_limit, 
                                      desired_distance, em_capacity, alpha, initial_beta, final_beta, beta_increase, 
                                      initial_b_temperature, final_b_temperature, b_temperature_decay]

                    # Save all information about training
                    utils.save(path_to_results_to_save, [info_task_list, total_summary_list, p_em])
                    print("Training results saved.")
                    
    end_time = time.time()
    elapsed_time = (end_time - start_time)/3600
    print("Elapsed time (hours): %f" % (elapsed_time))
    # Close session in Tensorflow
    session.close()
    # Close all open connections with VREP
    vrep.simxFinish(-1)
    print("Closed connection to VREP")