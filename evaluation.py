# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 19:18:11 2018

@author: Clayton
"""

import vrep
import math
import agent as ag
import environment as env
import model as nn
import tensorflow as tf
import utils

# RESTORE OPTIONS
path_to_model_to_restore   = "./final_models/6_SF_FINAL3.ckpt-80"      # Path to the saved model

# NEURAL NETWORK OPTIONS
time_steps              = 4                 # Number of time steps to consider in the recurrent network
num_actions             = 5                 # Number of actions (ALWAYS 5)
num_readings            = 8                 # Number of sensor readings
input_dims              = [13, 10, 10]      # Input dimensionality of the NN
lstm_units              = 50                # Output dimensionality of the LSTM cells
num_nearest_agents_comm = 2                 # Number of neighbors used for communication
num_neurons             = [50, 50]          # Number of neurons in each of the three layers

# STAGE OPTIONS
num_agents       = 8             # Number of agents in the environment
num_episodes     = 1            # Number of episodes
max_discrepancy  = 200           # Maximum discrepancy per agent
min_discrepancy  = 0             # Minimum discrepancy per agent
task             = 1             # 0: dispersion, 1: square  formation, 2: aggregation, 3: chain formation
steps_limit      = 600           # Limit of steps per episode
desired_distance = 0.28          # Side of squares in square formation task, 
                                 # or maximum distance to nearest partnet in dispersion task

if __name__ == "__main__":
    vrep.simxFinish(-1) # Close all open connections
    client_ID = vrep.simxStart('127.0.0.1', 19999, True, False, 5000, 5) # Connect to V-REP
    
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
                        skip_steps = 1, show_display = 1, steps_limit = steps_limit)

    # Initialize the deep Q-networks
    print('Initializing network.')
    num_outputs = num_actions
                                              
    target_net = nn.Model('target', 
                          input_dims = input_dims,
                          lstm_units = lstm_units, 
                          mlp_num_neurons = num_neurons, 
                          num_outputs = num_outputs, 
                          time_steps = time_steps)

    # Initialize session in TensorFlow
    print('Starting session.')
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    session = tf.Session()
    
    # Restore model or not
    print('Restoring saved model.')
    saver = tf.train.Saver()
    saver.restore(session, path_to_model_to_restore)

    discrepancy_list = []  # List containing discrepancies for each step
    initial_discrepancy_list = [] # List containing initial discrepancies for each episode
    final_discrepancy_list = [] # List containing final discrepancies for each episode
    reset_positions = 1 # Reset positions of the robots
    
    print('Starting simulation.')
    
    final_avg_total_discrepancy = 0
    for episode in range(0, num_episodes):
        e.start_new_episode(episode, reset_positions)
        step = 0    # Step in the episode
        end = 0     # If true, the episode must end
        discrepancy_list = []  # List containing discrepancies for each step
        
        saved_agent_positions = [] # List containing the saved agent positions

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
                
            for a in agent_list: 
                # Action ID according to boltzmann exploration policy
                action_ID = utils.boltzmann_exploration(q_out[a.get_agent_ID()], 0.0)
                
                # Obtain joint speeds according to action_ID
                right_joint_speed, left_joint_speed = utils.action_ID_to_speeds(action_ID) 
                a.actuate(right_joint_speed, left_joint_speed, action_ID) # Actuate agent
            
            if (step % (math.floor(int(steps_limit/5))) == 0 or step == steps_limit) and episode == num_episodes - 1:
                saved_agent_positions.append(e.get_agent_positions(step))
            # Go to next step and iterate
            step = step + 1
            # Obtain rewards, discrepancy and whether the task has reached its end
            if task == 0:
                discrepancy, end, reset_positions = e.iterate_inference_dispersion(step)
            elif task == 1:
                discrepancy, end, reset_positions = e.iterate_inference_square_formation(step)
            elif task == 2:
                discrepancy, end, reset_positions = e.iterate_inference_aggregation(step)
            elif task == 3:
                discrepancy, end, reset_positions = e.iterate_inference_chain_formation(step)

            # Stores discrepancy
            discrepancy_list.append(discrepancy)
            
            # In case it is the end of the episode
            if end:
                initial_discrepancy = discrepancy_list[0]
                initial_discrepancy_list.append(initial_discrepancy)
                final_discrepancy = discrepancy_list[-1]
                final_discrepancy_list.append(final_discrepancy)
                summary_discrepancy_episode = [initial_discrepancy, final_discrepancy]
                summary_discrepancy_episode = [round(content, 3) for content in summary_discrepancy_episode]
                
                print("EPISODE %i. Initial D., Final D.: %s." 
                      % (episode + 1, ', '.join(map(str, summary_discrepancy_episode))))
                
    utils.illustrate_robots2(saved_agent_positions, 0.035, initial_discrepancy_list[-1], 
                             final_discrepancy_list[-1], num_skip_steps = math.floor(steps_limit/5))
    avg_initial_discrepancy = sum(initial_discrepancy_list)/len(initial_discrepancy_list)
    avg_final_discrepancy = sum(final_discrepancy_list)/len(final_discrepancy_list)
    print("Average Initial D., Average Final D.: %f, %f" % (avg_initial_discrepancy, avg_final_discrepancy))                   
    # Close session in Tensorflow
    session.close()
    # Close all open connections with VREP
    vrep.simxFinish(-1)
    print("Closed connection to VREP")