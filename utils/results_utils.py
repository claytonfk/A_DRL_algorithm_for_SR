# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 23:28:40 2018

@author: Clayton
"""

import pickle
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


# Save variables
def save(filepath, save_variable_list):
    with open(filepath, 'wb') as f:
        pickle.dump(save_variable_list, f)
        
def load(filepath):
    with open(filepath, 'rb') as f:
        info_task_list, total_summary_list, p_em = pickle.load(f)
        
    return info_task_list, total_summary_list, p_em

def load_and_show_results(filepath):
    info_task_list, total_summary_list, _ = load(filepath)
    print_training_info(info_task_list)
    plot_training_results(total_summary_list, 21, 1, stage_division = [150])

# Plot training results
def plot_training_results(total_summary_list, window_size, order, stage_division = -1):
    fig = plt.figure(figsize=(23, 15))
    plt.rcParams.update({'font.size': 17})
    rewards = fig.add_subplot(221)
    avg_d = fig.add_subplot(222)
    min_d = fig.add_subplot(223)
    max_d = fig.add_subplot(224)
    fig.subplots_adjust(wspace = 0.15, hspace = 0.25)
    
    rewards.set_title('Average joint reward per episode')
    rewards.set_xlabel('episode')
    rewards.set_ylabel('reward')

    avg_d.set_title('Average ratio total discrepancy/number of agents per episode')
    avg_d.set_xlabel('episode')
    avg_d.set_ylabel('discrepancy')
    
    min_d.set_title('Minimum ratio total discrepancy/number of agents per episode')
    min_d.set_xlabel('episode')
    min_d.set_ylabel('discrepancy')
    
    max_d.set_title('Maximum ratio total discrepancy/number of agents per episode')
    max_d.set_xlabel('episode')
    max_d.set_ylabel('discrepancy')
    
    rewards.plot(savgol_filter([entry[0] for entry in total_summary_list], window_size, order))
    rewards.plot([entry[0] for entry in total_summary_list], alpha = 0.4)
    avg_d.plot(savgol_filter([entry[1] for entry in total_summary_list], window_size, order))
    avg_d.plot([entry[1] for entry in total_summary_list], alpha = 0.4)
    min_d.plot(savgol_filter([entry[2] for entry in total_summary_list], window_size, order))
    min_d.plot([entry[2] for entry in total_summary_list], alpha = 0.4)
    max_d.plot(savgol_filter([entry[3] for entry in total_summary_list], window_size, order))
    max_d.plot([entry[3] for entry in total_summary_list], alpha = 0.4)

        
    if stage_division != -1:
        for j in stage_division:
            rewards.axvline(x=j, color='black', ls = 'dashed')
            avg_d.axvline(x=j, color='black', ls = 'dashed')
            min_d.axvline(x=j, color='black', ls = 'dashed')
            max_d.axvline(x=j, color='black', ls = 'dashed')
    #rewards.axis([0, 10, 0, 10])

    plt.show()
    
def print_training_info(info_task_list):
    
    training_frequency, batch_size, time_steps, num_actions, num_neurons, num_lstm_units, \
    copy_weights_frequency, discount_factor, learning_rate, num_agents, num_episodes, \
    max_discrepancy, min_discrepancy, task, steps_limit, desired_distance, em_capacity, \
    alpha, initial_beta, final_beta, beta_increase, initial_b_temperature, final_b_temperature, \
    b_temperature_decay = info_task_list

    if task == 0:
        print('Dispersion task')
        print('---------------')
    elif task == 1:
        print('Square formation task')
        print('---------------------')
    
    print('Number of episodes:     %d' % num_episodes)
    print('Time steps:             %d' % time_steps)
    print('Number of agents:       %d' % num_agents)
    print('Batch size:             %d' % batch_size)
    print('Initial Boltzmann temp: %d' % initial_b_temperature)
    print('Final Boltzmann temp:   %d' % final_b_temperature)
    print('Steps limit:            %d' % steps_limit)
    print('Discount factor:        %.2f' % discount_factor)
    
def illustrate_robots(agent_positions, time_step = -1, enclosing_circle = -1):
    circles = []
    
    # Find limits for the axis
    offset = 3
    min_x = min([pos[0] for pos in agent_positions]) - offset
    min_y = min([pos[1] for pos in agent_positions]) - offset
    max_x = max([pos[0] for pos in agent_positions]) + offset
    max_y = max([pos[1] for pos in agent_positions]) + offset
    
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    
    max_delta = max(delta_x, delta_y)
    max_x = min_x + max_delta
    max_y = min_y + max_delta
    
    # Create circles for all robots in agent_positions
    for pos in agent_positions:
        circles.append(plt.Circle((pos[0], pos[1]), 0.5, color='black', alpha = 0.8, clip_on = False))
        if enclosing_circle != -1:
            circles.append(plt.Circle((pos[0], pos[1]), 0.5 + enclosing_circle, color='black', alpha = 0.2, clip_on = False))
    
    plt.axis('equal') # Set equal length of axis
    ax = plt.gca() # Get axis
    ax.cla() # Clear axis
    
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False) 
    
    ax.set_xlim((min_x, max_x)) # Set x limits
    ax.set_ylim((min_y, max_y)) # Set y limits
    
    # Draw circles
    for cir in circles:
        ax.add_artist(cir)
    
    if time_step != -1:
        plt.text(0.5*(min_x + max_x) - max_delta*0.1, min_y + max_delta*0.1, r'Time step $t$ = ' + str(time_step))
        
def illustrate_robots2(multiple_agent_positions, enclosing_circle = -1, initial_dis = -1, final_dis = -1, num_skip_steps = -1):
    fig, flattened_axes = plt.subplots(1, 5)
    plt.rcParams.update({'font.size': 17})
    #flattened_axes = [ax for sublist in axes for ax in sublist]
    fig.set_size_inches(18, 3.2)
    
    # Find limits for the axis
    offset = 0.3
    min_x = []
    max_x = []
    min_y = []
    max_y = []
    
    for agent_positions in multiple_agent_positions:
        min_x.append(min([pos[0] for pos in agent_positions]) - offset)
        min_y.append(min([pos[1] for pos in agent_positions]) - offset)
        max_x.append(max([pos[0] for pos in agent_positions]) + offset)
        max_y.append(max([pos[1] for pos in agent_positions]) + offset)
        
    for i in range(0, len(min_x)):     
        delta_x = max_x[i] - min_x[i]
        delta_y = max_y[i] - min_y[i]
            
        max_delta = max(delta_x, delta_y)
        max_x[i] = min_x[i] + max_delta
        max_y[i] = min_y[i] + max_delta

    i = 0
    for axis in flattened_axes:
        axis.cla() 
        axis.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
        axis.tick_params(axis='y', which='both', left=False, right=False, labelleft=False) 
        axis.set_xlim((min(min_x), max(max_x))) # Set x limits
        axis.set_ylim((min(min_y), max(max_y))) # Set y limits
        axis.set_title('(%d)' % ((i+1)))
        i += 1
    
    i = 0
    for agent_positions in multiple_agent_positions:
        circles = []
        # Create circles for all robots in agent_positions
        for pos in agent_positions:
            circles.append(plt.Circle((pos[0], pos[1]), 0.035, color='black', alpha = 0.8, clip_on = False))
            if enclosing_circle != -1:
                circles.append(plt.Circle((pos[0], pos[1]), enclosing_circle, color='black', alpha = 0.2, clip_on = False))
        
        for cir in circles:
            flattened_axes[i].add_artist(cir)
        i += 1
    
    if initial_dis != -1 and final_dis != -1:
        plt.suptitle(r'Initial ' + r'$H_T/n$' + ' (1): %.2f. Final ' % initial_dis + r'$H_T/n$' + ' (5): %.2f. No. of steps from plot to plot: %d.' % (final_dis, num_skip_steps), x = 0.5, y = 1.15)
        
    
    
    