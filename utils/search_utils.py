# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 03:58:49 2018

@author: Clayton
"""


import utils.action_utils as au
import utils.geom_utils as gu
import math
import utils.swarm_utils as su
from copy import deepcopy

#import action_utils as au
#import geom_utils as gu
#import math
#import swarm_utils as su
#from copy import deepcopy

class Path:
    # Initializes path
    def __init__(self, agent_positions, agent_orientations, agent_ID, best_comb, time_step, desired_distance,
                 steps = [], cost = 0, available_actions = [0, 5, 6, 7, 8], last_action = -1, current_discrepancy = math.inf,
                 task = 1):
        self.__steps = steps                                        # Actions taken
        self.__cost  = cost                                         # Cost of the path
        self.__agent_positions = deepcopy(agent_positions)          # Agent positions
        self.__agent_orientations = deepcopy(agent_orientations)    # Agent orientations
        self.agent_ID = agent_ID                                    # Agent ID
        self.best_comb = best_comb                                  # Best agent square combination
        self.time_step = time_step                                  # Time step
        self.desired_distance = desired_distance                    # Desired distance
        self.available_actions = available_actions                  # Available actions
        self.last_action = last_action                              # Last taken action
        self.current_discrepancy = current_discrepancy              # Discrepancy
        self.task = task                                            # Task (1 for square formation, 0 for dispersion, 2 for chain formation)
    
    # Take action - irreversible
    def iterate_irreversible(self, action_ID):
        # Last discrepancy
        if self.task == 1:
            last_discrepancy = su.find_discrepancy_square_formation_any(self.__agent_positions, 
                                                                        self.desired_distance, 
                                                                        self.best_comb, 
                                                                        self.agent_ID)    
        elif self.task == 0:
            last_discrepancy = su.find_discrepancy_dispersion(self.agent_ID, 
                                                              self.__agent_positions, 
                                                              self.desired_distance)
        elif self.task == 2:
            last_discrepancy = su.find_discrepancy_aggregation(self.agent_ID, 
                                                              self.__agent_positions, 
                                                              self.desired_distance)
        elif self.task == 3:
            last_discrepancy = su.find_discrepancy_chain_formation(self.agent_ID, 
                                                                   self.__agent_positions, 
                                                                   self.desired_distance)

        # Iterate model
        self.__agent_positions = iterate_model_transition_single2(self.__agent_positions, 
                                                                 self.agent_ID, 
                                                                 action_ID)
        
        # Current discrepancy
        if self.task == 1:
            self.current_discrepancy = su.find_discrepancy_square_formation_any(self.__agent_positions, 
                                                                                self.desired_distance, 
                                                                                self.best_comb, 
                                                                                self.agent_ID)
        elif self.task == 0:
            self.current_discrepancy = su.find_discrepancy_dispersion(self.agent_ID, 
                                                                      self.__agent_positions, 
                                                                      self.desired_distance)
        elif self.task == 2:
            self.current_discrepancy = su.find_discrepancy_aggregation(self.agent_ID, 
                                                                      self.__agent_positions, 
                                                                      self.desired_distance)
        elif self.task == 3:
            self.current_discrepancy = su.find_discrepancy_chain_formation(self.agent_ID, 
                                                                           self.__agent_positions, 
                                                                           self.desired_distance)
            
        # Costs
        if (last_discrepancy - self.current_discrepancy) < 0:
            non_reward_action = 16
        elif (last_discrepancy - self.current_discrepancy) == 0:
            non_reward_action = 2
        elif (last_discrepancy - self.current_discrepancy) > 0:
            non_reward_action = 1
        
        if action_ID == 0:
            non_reward_action = 2
            
        # Update costs, actions taken and last action taken
        self.__cost = self.__cost  + non_reward_action 
        self.__steps.append(action_ID)
        self.last_action = action_ID
    
    # Take action - reversible
    def _iterate_reversible(self, action_ID):
        if action_ID == 0:
            return 2
        
        # Last discrepancy
        if self.task == 1:
            last_discrepancy = su.find_discrepancy_square_formation_any(self.__agent_positions, 
                                                                        self.desired_distance, 
                                                                        self.best_comb, 
                                                                        self.agent_ID)
        elif self.task == 0:
            last_discrepancy = su.find_discrepancy_dispersion(self.agent_ID, 
                                                              self.__agent_positions, 
                                                              self.desired_distance)
        elif self.task == 2:
            last_discrepancy = su.find_discrepancy_aggregation(self.agent_ID, 
                                                              self.__agent_positions, 
                                                              self.desired_distance)
        elif self.task == 3:
            last_discrepancy = su.find_discrepancy_chain_formation(self.agent_ID, 
                                                                   self.__agent_positions, 
                                                                   self.desired_distance)


        
        # Iterate model
        temp_agent_positions = iterate_model_transition_single2(self.__agent_positions,
                                                               self.agent_ID, 
                                                               action_ID)
            
        # Current discrepancy
        if self.task == 1:
            current_discrepancy = su.find_discrepancy_square_formation_any(temp_agent_positions, 
                                                                           self.desired_distance, 
                                                                           self.best_comb, 
                                                                           self.agent_ID)
        elif self.task == 0:
            current_discrepancy = su.find_discrepancy_dispersion(self.agent_ID, 
                                                                 temp_agent_positions, 
                                                                 self.desired_distance)
        elif self.task == 2:
            current_discrepancy = su.find_discrepancy_aggregation(self.agent_ID, 
                                                                  temp_agent_positions, 
                                                                  self.desired_distance)
        elif self.task == 3:
            current_discrepancy = su.find_discrepancy_chain_formation(self.agent_ID, 
                                                                      temp_agent_positions, 
                                                                      self.desired_distance)
            
        #  Costs
        if (last_discrepancy - current_discrepancy) < 0:
            non_reward_action = 16
        elif (last_discrepancy - current_discrepancy) == 0:
            non_reward_action = 2
        elif (last_discrepancy - current_discrepancy) > 0:
            non_reward_action = 1
        
        # Return cost
        cost = self.__cost + non_reward_action
         
        return cost  
    
    def iterate_reversible_(self):
        # Minimum possible path cost in the next action 
        # and action for minimum possible cost
        min_path_cost = math.inf
        min_path_cost_action_ID = math.inf
        
        # Do not consider opposite actions
        if self.last_action == 5:
            opposite_action_ID = 6
        elif self.last_action == 6:
            opposite_action_ID = 5
        elif self.last_action == 7:
            opposite_action_ID = 8
        elif self.last_action == 8:
            opposite_action_ID = 7
        else:
            opposite_action_ID = -1
        
        # Perform - reversibly - all available actions
        for action_ID in self.available_actions:
            if action_ID != opposite_action_ID:
                cost = self._iterate_reversible(action_ID)
                
                # Keep in memory best cost and its respective action
                if cost < min_path_cost:
                    min_path_cost = cost
                    min_path_cost_action_ID = action_ID
                
        return min_path_cost, min_path_cost_action_ID
    
    # Get information from path
    def get_cost(self):
        return self.__cost
        
    def get_path(self):
        return self.__steps
    
    def get_agent_positions(self):
        return self.__agent_positions
    
    def get_current_discrepancy(self):
        return self.current_discrepancy
    
    def __len__(self):
        return len(self.__steps)
    
    # Copy path
    def copy(self, remove_action):
        available_actions = self.available_actions.copy()
        available_actions.remove(remove_action)
        return Path(self.__agent_positions, self.__agent_orientations, self.agent_ID, self.best_comb, 
                    self.time_step, self.desired_distance, deepcopy(self.__steps), self.__cost,
                    available_actions, self.last_action, self.current_discrepancy, self.task)
    
# Model with rotation and orientations
def iterate_model_transition_single(agent_positions, agent_orientations, agent_ID, action_ID, time_step):
    # Deep copy positions and orientations
    agent_positions_cp = deepcopy(agent_positions)
    agent_orientations_cp = deepcopy(agent_orientations)
    # Robot parameters
    wheels_distance = 53/1000
    wheel_diameter = 41/1000
    
    # If agent has rotated clockwise (action_ID 4) or counterclockwise (action_ID 3)
    if action_ID == 4 or action_ID == 3:
        r, _ = au.action_ID_to_speeds(action_ID)
        agent_orientations_cp[agent_ID] += round(r*wheel_diameter*time_step/(wheels_distance*3.14159), 4)
        if agent_orientations_cp[agent_ID] > 1:
            agent_orientations_cp[agent_ID] = -1 + (agent_orientations_cp[agent_ID] - 1)
        elif agent_orientations_cp[agent_ID] < -1:
            agent_orientations_cp[agent_ID] = 1 + (agent_orientations_cp[agent_ID] + 1)
    # If agent has moved forward (action_ID 1) or backward (action_ID 2)
    elif action_ID == 1 or action_ID == 2:
        r, _ = au.action_ID_to_speeds(action_ID)
        agent_positions_cp[agent_ID][0] -= round(0.5*r*wheel_diameter*time_step*math.sin(agent_orientations_cp[agent_ID]*3.14159), 4)
        agent_positions_cp[agent_ID][1] += round(0.5*r*wheel_diameter*time_step*math.cos(agent_orientations_cp[agent_ID]*3.14159), 4)
    
    return agent_positions_cp, agent_orientations_cp

# Model without rotation and orientation
def iterate_model_transition_single2(agent_positions, agent_ID, action_ID):
    # Deep copy positions
    agent_positions_cp = deepcopy(agent_positions)
    
    # Go up
    if action_ID == 5:
        vert = 1
        hor = 0
    # Go down
    elif action_ID == 6:
        vert = -1
        hor = 0
    # Go left
    elif action_ID == 7:
        vert = 0
        hor = -1
    # Go right
    elif action_ID == 8:
        vert = 0
        hor = 1
    else:
        vert, hor = 0, 0
    
    # Change positions
    agent_positions_cp[agent_ID][0] += hor*0.01
    agent_positions_cp[agent_ID][1] += vert*0.01
    # Round positions
    agent_positions_cp[agent_ID][0] = round(agent_positions_cp[agent_ID][0], 3)
    agent_positions_cp[agent_ID][1] = round(agent_positions_cp[agent_ID][1], 3)
    
    return agent_positions_cp


# Uniform-cost search for a single agent
def search_single(agent_positions, agent_orientations, agent_ID, best_comb, time_step, desired_distance, iterations, task):
    # List of paths
    path_list = []
    available_actions = [0, 5, 6, 7, 8]
    
    # Initial path
    initial_path = Path(agent_positions, agent_orientations, agent_ID, 
                        best_comb, time_step, desired_distance, 
                        steps = [],  cost = 0, available_actions = available_actions,
                        task = task)
    
    path_list.append(initial_path)

    # Perform iterations
    for j in range(0, iterations):
        # Info about path to expand
        expand_path_index = 0
        expand_path_cost  = math.inf
        expand_path_action_ID = 0
        # Find path to expand
        for i in range(0, len(path_list)):
            cost, action_ID = path_list[i].iterate_reversible_()
            if cost < expand_path_cost:
                expand_path_cost = cost
                expand_path_index = i
                expand_path_action_ID = action_ID
        
        # Make a copy of the path
        path_list.append(path_list[expand_path_index].copy(expand_path_action_ID))
        # Expand the path
        path_list[expand_path_index].iterate_irreversible(expand_path_action_ID)
    
    # Best path (path that has lowest discrepancy)
    best_path_index = 0
    best_discrepancy = math.inf
    
    # Find best path
    for i in range(0, len(path_list)):
        discrepancy = path_list[i].get_current_discrepancy()
        
        if discrepancy < best_discrepancy:
            best_discrepancy = discrepancy
            best_path_index = i
    
    # Get actions from the best path
    path = path_list[best_path_index].get_path()
    
    return path, path_list[best_path_index].get_current_discrepancy(), path_list[best_path_index].get_agent_positions()

# Uniform-cost search for a single agent
def search_all_agent_positions(agent_positions, agent_orientations, best_comb, time_step, desired_distance, passes, iterations, task):
    # For all agents
    for j in range(0, passes):
        total_dis = 0
        for ID in range(0, len(agent_positions)):
            path, dis, agent_positions = search_single(agent_positions, agent_orientations, ID, best_comb, 
                                                       time_step, desired_distance, iterations, task)
            total_dis += dis

    return path, total_dis, agent_positions

# Find best agent actions
def search_all_agent_actions(desired_agent_positions, agent_positions, agent_orientations):
    action_list = []
    
    for i in range(0, len(agent_positions)):
        # Distance between desired agent position and given agent position
        distance = gu.find_distance_between_points(desired_agent_positions[i], agent_positions[i])
        # Vector from give agent position to desired agent position
        vector = [desired_agent_positions[i][0] - agent_positions[i][0], desired_agent_positions[i][1] - agent_positions[i][1]]
        # Angle between vector and unit vector [0, 1]
        angle = gu.find_vectors_angle([0, 1], vector)


        if desired_agent_positions[i][0] - agent_positions[i][0] > 0:
            angle = -angle

        # Difference between desired orientation and given orientation
        angle_diff =  agent_orientations[i] - (angle/math.pi)

        if abs(angle_diff) > 0.05 and abs(angle_diff) < 0.95:
            if angle_diff > 0 and angle_diff < 0.5:
                action_list.append(4)
            elif angle_diff >= 0.5 and angle_diff < 0.95:
                action_list.append(3)
            elif angle_diff > -0.5 and angle_diff < 0:
                action_list.append(3)
            elif angle_diff > -0.95 and angle_diff <= -0.5:
                action_list.append(4)
        elif abs(angle_diff) <= 0.05 and distance > 0.04:
            action_list.append(1)
        elif abs(angle_diff) >= 0.95 and distance > 0.04:
            action_list.append(2)
        else:
            action_list.append(0)
        
    return action_list