# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 22:46:50 2018

@author: Clayton
"""

import numpy as np
import math

# Given an action ID, returns the joint speeds 
def action_ID_to_speeds(action_ID):
    wheel_diameter = 41
    wheels_distance = 53
    
    translation = 40 # 40 mm/s
    rotation = 1     # 1 rad/s
    
    translation_omega = translation*2/wheel_diameter
    rotation_omega = rotation*wheels_distance/wheel_diameter
    
    if action_ID == 0:      # Still
        right_joint_speed, left_joint_speed = 0, 0
    elif action_ID == 1:    # Fast forward
        right_joint_speed, left_joint_speed = translation_omega, translation_omega
    elif action_ID == 2:    # Fast backward
        right_joint_speed, left_joint_speed = -translation_omega, -translation_omega
    elif action_ID == 3:    # Fast rotate CCW
        right_joint_speed, left_joint_speed = rotation_omega, -rotation_omega
    elif action_ID == 4:    # Fast rotate CW
        right_joint_speed, left_joint_speed = -rotation_omega, rotation_omega

    return right_joint_speed, left_joint_speed

# Apply one hot encoding to a batch of action IDs
def actions_encoded(actions):
    encodings = []
    for action in actions:
        encoding = [0, 0, 0, 0, 0]
        encoding[action] = 1
        encodings.append(encoding)
        
    return encodings

# Apply one hot encoding to a batch of action IDs
def action_ID_buffer_to_speeds(actions):
    speeds = []
    for action in actions:
        speed = action_ID_to_speeds(action)
        speeds.append(speed)
        
    return speeds

# Concatenate sensor readings and last executed actions
def concatenate_readings_actions(readings, actions):
    return np.concatenate((readings, actions), axis = 1).tolist()

# Given an array of action values and a temperature, returns an action ID according
# to Boltzmann exploration policy
def boltzmann_exploration(q_output, temperature):
    # Greedy action
    if temperature <= 0:
        return np.argmax(q_output)
    else:
        # Small (and equivalent) modification to boltzmann function to avoid overflow
        max_q_output = max(q_output)
        q_output  = [q_out - max_q_output for q_out in q_output]
        # Obtains the exponentiation for all action-values in q_output
        boltzmann_exponentials = [math.exp(q_a/temperature) for q_a in q_output]
        sum_boltzmann_exponentials = sum(boltzmann_exponentials)
        # Obtains the probabilities for each action_ID
        boltzmann_probabilities = [boltzmann_exponential/sum_boltzmann_exponentials for \
                                   boltzmann_exponential in boltzmann_exponentials]

        # action_ID chosen 
        action_ID = int(np.random.choice(len(q_output), 1, p = boltzmann_probabilities))
        
        return action_ID
    
# Organize minibatch, indices and weights according to action ID
def organize_minibatch(minibatch, idx, weights, num_actions):
    # For all contents in the minibatch
    states = [[] for _ in range(0, num_actions)]
    rewards  = [[] for _ in range(0, num_actions)]
    next_states = [[] for _ in range(0, num_actions)]
    new_idx = [[] for _ in range(0, num_actions)]
    new_weights = [[] for _ in range(0, num_actions)]

    transition_idx = 0
    # Separating by action ID
    for transition in minibatch:
        state = transition[0]
        action_ID = transition[1]
        reward = transition[2]
        next_state = transition[3]

        states[action_ID].append(state)
        rewards[action_ID].append(reward)
        next_states[action_ID].append(next_state)
        new_idx[action_ID].append(idx[transition_idx])
        new_weights[action_ID].append(weights[transition_idx])
        
        transition_idx += 1
    
    return states, rewards, next_states, new_idx, new_weights
