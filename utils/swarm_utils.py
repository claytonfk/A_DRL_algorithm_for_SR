# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 22:51:49 2018

@author: Clayton
"""

import utils.geom_utils as gu
import numpy as np
import random
import math
import copy
from itertools import combinations

random.seed(0)

# Find Euclidean distance between an agent and other agents
def find_distance_to_other_robots(agent_positions, main_agent_position):
    distances_to_other_robots = []
    # Find the distance
    for agent_position in agent_positions:
        distances_to_other_robots.append(gu.find_distance_between_points(agent_position, main_agent_position))
    
    return distances_to_other_robots

# Find most distant robots and the distance between them 
def find_most_distant_robots(agent_positions):
    largest_distance = 0
    robot1 = -1
    robot2 = -1
    for i in range(0,len(agent_positions)):
        for j in range(0,len(agent_positions)):
            if j > i:
                distance = gu.find_distance_between_points(agent_positions[i], agent_positions[j])
                
                if largest_distance < distance:
                    largest_distance = distance
                    robot1 = i
                    robot2 = j
    
    return robot1, robot2, largest_distance      
                

# Find the position, distance, IDs and angles of the nearest robots of a specific robot
def find_n_nearest_robots(agent_positions, agent_ID, num_nearest_robots):
    num_agents = len(agent_positions)
    
    assert num_agents > num_nearest_robots
    
    # Find distances to all other robots
    distances_to_other_robots = find_distance_to_other_robots(agent_positions, agent_positions[agent_ID])
    distances_to_other_robots_cp = distances_to_other_robots.copy()
    # Sort in ascending order from left to right
    distances_to_other_robots.sort()
    
    # Find num_nearest_robots IDs
    nearest_robots_IDs = []
    
    for x in distances_to_other_robots[1:num_nearest_robots + 1]:
        if distances_to_other_robots_cp.index(x) not in nearest_robots_IDs:
            nearest_robots_IDs.append(distances_to_other_robots_cp.index(x))
        else:
            distances_to_other_robots_cp_cp = copy.deepcopy(distances_to_other_robots_cp)
            distances_to_other_robots_cp_cp[distances_to_other_robots_cp_cp.index(x)] = -1
            nearest_robots_IDs.append(distances_to_other_robots_cp_cp.index(x))

    # Find their positions
    nearest_robots_positions = [agent_positions[x] for x in nearest_robots_IDs]

    # Find  the angles between the vectors formed starting from the agent with ID agent_ID to its
    # num_nearest_robots IDs
    vectors = [np.subtract(x, agent_positions[agent_ID]).tolist() for x in nearest_robots_positions]
    angles = [gu.find_vectors_angle(vectors[i], vectors[j]) for (i, j) in combinations(range(0, num_nearest_robots), 2)]

    return distances_to_other_robots[1:num_nearest_robots + 1], nearest_robots_positions, nearest_robots_IDs, angles

# Detect possible collision between agents and return their IDs
def detect_possible_collision(agent_positions, tolerance):
    num_positions = len(agent_positions)
    
    for i in range(0, num_positions):
        for j in range(0, num_positions):
            if j > i:
                distance = gu.find_distance_between_points(agent_positions[i], agent_positions[j])
                if distance < tolerance:
                    return True, i, j
    
    return False, -1, -1

# Find discrepancy in square formation - when number of robots is four
def find_discrepancy_square_formation_four(agent_ID, agent_positions, desired_distance):
    R = 1000 # Upscale factor
    num_agents = len(agent_positions)
    distances_to_nearest_robots, nearest_robots_positions, nearest_robots_IDs, _ = find_n_nearest_robots(agent_positions, 
                                                                                                         agent_ID, 
                                                                                                         num_agents - 1)
    # Error in distance to nearest robots
    distance_error_nearest = [math.pow(x - y, 2) for (x, y) in \
                               zip(distances_to_nearest_robots[:2], [desired_distance]*2)]
    
    # Error in distance to diagonal robot
    distance_error_diagonal = [math.pow(x - y, 2) for (x, y) in \
                               zip(distances_to_nearest_robots[2:], [desired_distance*1.41]*(num_agents - 3))]

    distance_error_diagonal = min(distance_error_diagonal)

    # Sum of all errors in distance to nearest robots
    distance_error_nearest  = sum(distance_error_nearest)

    discrepancy = (2*distance_error_nearest + distance_error_diagonal)*R
    
    return discrepancy

 # Find total discrepancy in square formation - when number of robots is four
def find_total_discrepancy_square_formation_four(agent_positions, square_IDs, desired_distance):
    new_agent_positions = [agent_positions[ID] for ID in square_IDs]
    
    total_discrepancy = 0
    for i in range(0, 4):
        total_discrepancy += find_discrepancy_square_formation_four(i, new_agent_positions, desired_distance)
    
    return total_discrepancy

# Find best square combination in the square formation task
def find_best_square_combination(agent_positions, desired_distance, n_squares = -1):
    # Number of squares
    if len(agent_positions) == 4:
        n_squares = 1
    elif len(agent_positions) == 6:
        n_squares = 2
    elif len(agent_positions) == 8:
        n_squares = 3
    elif len(agent_positions) == 9:
        n_squares = 4
    elif len(agent_positions) == 11:
        n_squares = 5
    elif len(agent_positions) == 12:
        n_squares = 6
    else:
        raise NotImplementedError
    
    # Possible squares
    squares = list(combinations(range(0, len(agent_positions)), 4))
    # Number of possible squares
    n_possible_squares = len(squares)
    # All position square combinations
    possible_comb_squares = [list(entry) for entry in list(combinations(range(0, n_possible_squares), n_squares))]
    
    # Find all plausible square combinations
    good_comb_squares = []
    for possible_comb in possible_comb_squares:
        flattened_IDs = []
        for comb_ID in possible_comb:
            flattened_IDs.append(squares[comb_ID])
        flattened_IDs = np.array(flattened_IDs).flatten()
        unique_flattened_IDs = np.unique(flattened_IDs).tolist()
        n_shared_IDs = len(flattened_IDs) - len(unique_flattened_IDs)
        
        if n_squares*4 - n_shared_IDs == len(agent_positions):
            good_comb_squares.append(possible_comb)

    # Find best square combinations
    best_dis = math.inf
    best_comb = -1
    for good_comb in good_comb_squares:
        tot_dis = 0
        for comb_ID in good_comb:
            tot_dis += find_total_discrepancy_square_formation_four(agent_positions, squares[comb_ID], desired_distance)
        
        if tot_dis < best_dis:
            best_dis = tot_dis
            best_comb = good_comb
    
    # Best square combination
    best_comb_squares = [squares[comb_ID] for comb_ID in best_comb]

    return best_comb_squares

# A square discrepancy function - for any number of robots
def find_discrepancy_square_formation_any(agent_positions, desired_distance, best_comb_squares, agent_ID):
    square_partners = [best_comb_square for best_comb_square in best_comb_squares if agent_ID in best_comb_square]
    
    discrepan = []
    for s in square_partners:
        alt_agent_positions = [agent_positions[ID] for ID in s]
        alt_agent_ID = alt_agent_positions.index(agent_positions[agent_ID])
        discrepan.append(find_discrepancy_square_formation_four(alt_agent_ID, alt_agent_positions, desired_distance))
    
    agent_dis = sum(discrepan)/len(discrepan)
    return agent_dis

# Find total discrepancy in square formation - for any number of robots
def find_total_discrepancy_square_formation_any(agent_positions, best_comb_squares, desired_distance):
    total_discrepancy = 0
    for i in range(0, len(agent_positions)):
        total_discrepancy += find_discrepancy_square_formation_any(agent_positions, desired_distance, best_comb_squares, i)
    
    return total_discrepancy
  
## Find discrepancy in dispersion
def find_discrepancy_dispersion(agent_ID, agent_positions, desired_distance):
    R = 3000 # Upscale factor

    distance_nearest_robot, _, _, _  = find_n_nearest_robots(agent_positions, agent_ID, 1)
    error_distance = math.pow(distance_nearest_robot[0] - desired_distance, 2)

    return error_distance*R

### Find discrepancy in aggregation
def find_discrepancy_aggregation(agent_ID, agent_positions, desired_distance):     
    R = 250 # Upscale factor
    distance_nearest_robot, _, _, _  = find_n_nearest_robots(agent_positions, agent_ID, len(agent_positions) - 1)
    distance_nearest_robot_range = [d for d in distance_nearest_robot if d < 0.5]
    num_agents_range = len(distance_nearest_robot_range) + 1
    
    if len(distance_nearest_robot_range) != 0:
        largest_diam = max(distance_nearest_robot_range)
    else:
        largest_diam = 0.5
        
    largest_diam += desired_distance
    largest_radius = largest_diam/2
    smallest_radius = (desired_distance)/2
    density = num_agents_range*(smallest_radius**2)/(largest_radius**2)
    
    error_density = math.pow(density - 0.6, 2)

    return R*error_density

## Find total discrepancy in aggregation
def find_total_discrepancy_aggregation(agent_positions, desired_distance):
    total_discrepancy = 0
    for i in range(0, len(agent_positions)):
        total_discrepancy += find_discrepancy_aggregation(i, agent_positions, desired_distance)
        
    return total_discrepancy

## Find total discrepancy in dispersion
def find_total_discrepancy_dispersion(agent_positions, desired_distance):
    total_discrepancy = 0
    for i in range(0, len(agent_positions)):
        total_discrepancy += find_discrepancy_dispersion(i, agent_positions, desired_distance)
        
    return total_discrepancy

# Find discrepancy in chain formation
def find_discrepancy_chain_formation(agent_ID, agent_positions, desired_distance):
    R1 = 10 # Upscale factor
    R2 = 1000
    distance_nearest_robot, nrp, _, ang = find_n_nearest_robots(agent_positions, agent_ID, 2)
    should_be_inside = 0
    
    partners_distance = gu.find_distance_between_points(nrp[0], nrp[1])
    if distance_nearest_robot[0] < partners_distance and distance_nearest_robot[1] < partners_distance:
        should_be_inside = 1
    
    if should_be_inside:
        error = (ang[0] - math.pi)**2
    else:
        error = (ang[0])**2

    error_distance = math.pow(distance_nearest_robot[0] - desired_distance, 2)

    return R1*error + R2*error_distance, False

# Find total discrepancy in chain formation
def find_total_discrepancy_chain_formation(agent_positions, desired_distance):
    total_discrepancy = 0
    for agent_ID in range(0, len(agent_positions)):
        total_discrepancy += find_discrepancy_chain_formation(agent_ID, agent_positions, desired_distance)[0]
        
    return total_discrepancy

def create_random_positions(num_random_positions):
    ul = 0.05*num_random_positions
    ll = -0.05*num_random_positions
    
    final_agent_positions = []
    agent_positions = []
    for i in range(0, num_random_positions):
        done = 0
        while not done:
            x_coord, y_coord = round(random.uniform(ll, ul), 3), round(random.uniform(ll, ul), 3)
            agent_positions.append([x_coord, y_coord, 0.0191])
            collision, _, _ = detect_possible_collision(agent_positions, 0.15)
            
            if not collision:
                done = 1
                final_agent_positions.append([x_coord, y_coord, 0.0191])
            else:
                del agent_positions[-1]
   
    return final_agent_positions