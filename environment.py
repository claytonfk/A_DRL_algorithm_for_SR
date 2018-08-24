# -*- coding: utf-8 -*-
"""
Created on Fri May 11 00:05:06 2018

@author: Clayton
"""

import vrep
import time
import numpy as np
import utils

class Environment:
    def __init__(self, client_ID, agents, desired_distance, success_threshold = 5, fail_threshold = 500,
                 skip_steps = 1, show_display = 1, steps_limit = 300, steps_before_evaluation = 1):
        
        self.__client_ID = client_ID # VREP Client ID
        self.__agents = agents  # List containing all agents
        self.__num_agents = len(agents) # Number of agents
        self.__desired_distance = desired_distance # Desired distance, either for dispersion or pattern formation
        self.__success_threshold = success_threshold # Discrepancy below which the episode succesfully finishes
        self.__fail_threshold = fail_threshold # Discrepancy above which the episode fails
        self.__skip_steps = skip_steps # Number of steps to skip during iteration in VREP
        self.__show_display = show_display # Show the display in VREP. Set to false to increase performance
        self.__steps_limit = steps_limit # Maximum number of steps, above which the task reaches its end
        self.__steps_before_evaluation = steps_before_evaluation # How often (in steps) rewards are given

        # Start in synchronous mode
        error = vrep.simxSynchronous(client_ID, 1)
        assert error == 0, "Could not start simulation in synchronous mode."
        
        self.__step = 0
        
    ## Getters and setters
    def get_client_ID(self):
        return self.__client_ID
    
    def get_num_agents(self):
        return self.__num_agents
    
    def get_agents(self):
        return self.__agents
    
    def get_success_threshold(self):
        return self.__success_threshold
    
    def set_success_threshold(self, success_threshold):
        self.__success_threshold = success_threshold
        
    def get_fail_threshold(self):
        return self.__fail_threshold
    
    def set_fail_threshold(self, fail_threshold):
        self.__fail_threshold = fail_threshold

    def get_skip_steps(self):
        return self.__skip_steps
    
    def set_skip_steps(self, skip_steps):
         self.__skip_steps = skip_steps
         
    def get_show_display(self):
        return self.__show_display
    
    def set_show_display(self, show_display):
        self.__show_display = show_display
        
    def get_steps_limit(self):
        return self.__steps_limit
    
    def set_steps_limit(self, steps_limit):
        self.__steps_limit = steps_limit
        
    def get_steps_before_evaluation(self):
        return self.__steps_before_evaluation
    
    def set_steps_before_evaluation(self, steps_before_evaluation):
        self.__steps_before_evaluation = steps_before_evaluation
        
        
    def randomize_positions(self):
        # Set random positions to all agents
        random_pos = utils.create_random_positions(self.__num_agents)
        for i in range(0, self.__num_agents):
            self.__agents[i].set_position(random_pos[i])
    
    def start_new_episode(self, episode, reset_positions):
        if reset_positions:
            # Stop the simulation if it is running
            vrep.simxStopSimulation(self.__client_ID, vrep.simx_opmode_blocking)
            time.sleep(2) # One second dela
            self.randomize_positions()
            
            time.sleep(1) # One second delay
            # Start simulation
            vrep.simxStartSimulation(self.__client_ID, vrep.simx_opmode_blocking)
            # Set display to enabled or not
            vrep.simxSetBooleanParameter(self.__client_ID, vrep.sim_boolparam_display_enabled, self.__show_display, vrep.simx_opmode_oneshot)
            # Enable threaded rendering
            vrep.simxSetBooleanParameter(self.__client_ID, vrep.sim_boolparam_threaded_rendering_enabled, 1, vrep.simx_opmode_oneshot)
            
            # Catch error
            error = vrep.simxSynchronous(self.__client_ID, 1)
            assert error == 0, "Could not start simulation in synchronous mode."

        # Reset all agents 
        for agent in self.__agents:
            agent.restart()
            
     
    # Get the agent positions for a specific step
    def get_agent_positions(self, step):
        agent_positions = []
        
        # Get and store the agent positions
        for agent in self.__agents:
            agent_positions.append(agent.get_position(step))
            
        return agent_positions
    
    # Get the agent orientations for a specific step
    def get_agent_orientations(self, step):
        agent_orientations = []
        
        # Get and store the agent positions
        for agent in self.__agents:
            agent_orientations.append(agent.get_orientation(step))
            
        return agent_orientations
    
    
    def __iterate(self, step):
        # Continue the simulation for the number of steps defined in skip_steps
        for i in range(0, self.__skip_steps):
            vrep.simxSynchronousTrigger(self.__client_ID)
        
        # Current agents' positions
        agent_positions = self.get_agent_positions(step)
        self.get_agent_orientations(step)
        
        # Last agents' positions
        if step - self.__steps_before_evaluation >= 0:
            last_agent_positions = self.get_agent_positions(step - self.__steps_before_evaluation)
        else:
            last_agent_positions = self.get_agent_positions(0)
            
        return agent_positions, last_agent_positions
    
    
    def iterate_training_square_formation(self, step, best_comb_squares):
        # Iterate, get current and last agent positions
        agent_positions, last_agent_positions = self.__iterate(step)
        
        total_discrepancy = 0
        rewards = {}
        
        # For all agents
        for agent_ID in range(0, self.__num_agents):
            current_agent_positions_frozen = last_agent_positions.copy()
            current_agent_positions_frozen[agent_ID] = agent_positions[agent_ID]
            last_agent_positions_frozen = last_agent_positions.copy()
            last_best_dis = utils.find_discrepancy_square_formation_any(last_agent_positions_frozen, self.__desired_distance, best_comb_squares, agent_ID)
            current_best_dis = utils.find_discrepancy_square_formation_any(current_agent_positions_frozen, self.__desired_distance, best_comb_squares, agent_ID)
            diff_best_dis = last_best_dis - current_best_dis
            total_discrepancy += utils.find_discrepancy_square_formation_any(agent_positions, self.__desired_distance, best_comb_squares, agent_ID)
            
            multiplier = max(0.5, -0.03*current_best_dis + 5)
            
            # Reward the agent according to change in discrepancies
            if step % self.__steps_before_evaluation == 0:
                action_ID = self.__agents[agent_ID].get_last_action_ID()
                
                if action_ID == 1 or action_ID == 2:
                    rewards[str(agent_ID)] = (np.sign(diff_best_dis) - 0.025)*multiplier
                elif action_ID == 3 or action_ID == 4:
                    rewards[str(agent_ID)] = -0.025*multiplier
                else:
                    rewards[str(agent_ID)] = 0
                
            elif step % self.__steps_before_evaluation != 0:
                rewards[str(agent_ID)] = 0
            

        collision, ID1, ID2 = utils.detect_possible_collision(agent_positions, 0.08)
        
        discrepancy = total_discrepancy/self.__num_agents
        # Possible collision situation
        if collision:
            end = 1
            reset = 1
            rewards[str(ID1)] = -5
            rewards[str(ID2)] = -5
        # Decide whether or not the training should end at the next step
        elif step >= self.__steps_limit:
            end = 1
            reset = 1
        elif ((discrepancy > self.__fail_threshold) or (discrepancy < self.__success_threshold)) and discrepancy > 0:
            end = 1
            reset = 1
        else:
            end = 0
            reset = 0

        return rewards, discrepancy, end, reset
        
    def iterate_training_dispersion(self, step):
        # Iterate, get current and last agent positions
        agent_positions, last_agent_positions = self.__iterate(step)
        
        total_discrepancy = 0
        rewards = {}
        
        # For all agents
        for agent_ID in range(0, self.__num_agents):
            current_agent_positions_frozen = last_agent_positions.copy()
            current_agent_positions_frozen[agent_ID] = agent_positions[agent_ID]
            last_agent_positions_frozen = last_agent_positions.copy()
            
            last_best_dis = utils.find_discrepancy_dispersion(agent_ID, last_agent_positions_frozen, self.__desired_distance)
            current_best_dis = utils.find_discrepancy_dispersion(agent_ID, current_agent_positions_frozen, self.__desired_distance)
            diff_best_dis = last_best_dis - current_best_dis
            total_discrepancy += utils.find_discrepancy_dispersion(agent_ID, agent_positions, self.__desired_distance)

            multiplier = max(0.5, -0.03*current_best_dis + 5)
            
            # Reward the agent according to change in discrepancies
            if step % self.__steps_before_evaluation == 0:
                action_ID = self.__agents[agent_ID].get_last_action_ID()
                
                if action_ID == 1 or action_ID == 2:
                    rewards[str(agent_ID)] = (np.sign(diff_best_dis) - 0.10)*multiplier
                elif action_ID == 3 or action_ID == 4:
                    rewards[str(agent_ID)] = -0.10*multiplier
                else:
                    rewards[str(agent_ID)] = 0
                
            elif step % self.__steps_before_evaluation != 0:
                rewards[str(agent_ID)] = 0
            
        # Detect possible collision
        collision, ID1, ID2 = utils.detect_possible_collision(agent_positions, 0.08)
        
        discrepancy = total_discrepancy/self.__num_agents
        
        # In case a possible collision will happen
        if collision:
            end = 1
            reset = 1
            rewards[str(ID1)] = -5
            rewards[str(ID2)] = -5 
        # Decide whether or not the training should end at the next step
        elif step >= self.__steps_limit:
            end = 1
            reset = 1
        elif ((discrepancy > self.__fail_threshold) or (discrepancy < self.__success_threshold)) and discrepancy > 0:
            end = 1
            reset = 1
        else:
            end = 0
            reset = 0
            
        return rewards, discrepancy, end, reset
    
    def iterate_training_chain_formation(self, step):
        # Iterate, get current and last agent positions
        agent_positions, last_agent_positions = self.__iterate(step)
        
        total_discrepancy = 0
        rewards = {}
        
        # For all agents
        for agent_ID in range(0, self.__num_agents):
            current_agent_positions_frozen = last_agent_positions.copy()
            current_agent_positions_frozen[agent_ID] = agent_positions[agent_ID]
            last_agent_positions_frozen = last_agent_positions.copy()

            last_best_dis, too_far = utils.find_discrepancy_chain_formation(agent_ID, last_agent_positions_frozen, self.__desired_distance)
            current_best_dis, too_far = utils.find_discrepancy_chain_formation(agent_ID, current_agent_positions_frozen, self.__desired_distance)
            diff_best_dis = last_best_dis - current_best_dis
            total_discrepancy += utils.find_discrepancy_chain_formation(agent_ID, agent_positions, self.__desired_distance)[0]

            multiplier = max(0.5, -0.03*current_best_dis + 5)
            
            # Reward the agent according to change in discrepancies
            if step % self.__steps_before_evaluation == 0:
                action_ID = self.__agents[agent_ID].get_last_action_ID()
                
                if action_ID == 1 or action_ID == 2:
                    rewards[str(agent_ID)] = (np.sign(diff_best_dis) - 0.1)*multiplier
                elif action_ID == 3 or action_ID == 4:
                    rewards[str(agent_ID)] = -0.1*multiplier
                else:
                    rewards[str(agent_ID)] = 0
                
                if too_far:
                    rewards[str(agent_ID)] = -2
            elif step % self.__steps_before_evaluation != 0:
                rewards[str(agent_ID)] = 0
            
        # Detect possible collision
        collision, ID1, ID2 = utils.detect_possible_collision(agent_positions, 0.08)
        
        discrepancy = total_discrepancy/self.__num_agents
        
        # In case a possible collision will happen
        if collision:
            end = 0
            reset = 0
            rewards[str(ID1)] = -5
            rewards[str(ID2)] = -5    
        # Decide whether or not the training should end at the next step
        elif step >= self.__steps_limit:
            end = 1
            reset = 1
        elif ((discrepancy > self.__fail_threshold) or (discrepancy < self.__success_threshold)) and discrepancy > 0:
            end = 1
            reset = 1
        else:
            end = 0
            reset = 0
            
        return rewards, discrepancy, end, reset
    
     
    def iterate_training_aggregation(self, step):
        # Iterate, get current and last agent positions
        agent_positions, last_agent_positions = self.__iterate(step)
        
        total_discrepancy = 0
        rewards = {}
        
        # For all agents
        for agent_ID in range(0, self.__num_agents):
            current_agent_positions_frozen = last_agent_positions.copy()
            current_agent_positions_frozen[agent_ID] = agent_positions[agent_ID]
            last_agent_positions_frozen = last_agent_positions.copy()
            
            last_best_dis = utils.find_discrepancy_aggregation(agent_ID, last_agent_positions_frozen, self.__desired_distance)
            current_best_dis = utils.find_discrepancy_aggregation(agent_ID, current_agent_positions_frozen, self.__desired_distance)
            diff_best_dis = last_best_dis - current_best_dis
            total_discrepancy += utils.find_discrepancy_aggregation(agent_ID, agent_positions, self.__desired_distance)

            multiplier = max(0.5, -0.03*current_best_dis + 5)
            
            # Reward the agent according to change in discrepancies
            if step % self.__steps_before_evaluation == 0:
                
                action_ID = self.__agents[agent_ID].get_last_action_ID()
                if action_ID == 1 or action_ID == 2:
                    rewards[str(agent_ID)] = (np.sign(diff_best_dis) - 0.1)*multiplier
                elif action_ID == 3 or action_ID == 4:
                    rewards[str(agent_ID)] = -0.1*multiplier
                else:
                    rewards[str(agent_ID)] = 0

            elif step % self.__steps_before_evaluation != 0:
                rewards[str(agent_ID)] = 0
            
            
        # Detect possible collision
        collision, ID1, ID2 = utils.detect_possible_collision(agent_positions, 0.08)
        
        discrepancy = total_discrepancy/self.__num_agents
        
        # In case a possible collision will happen
        if collision:
            end = 1
            reset = 1
            rewards[str(ID1)] = -30
            rewards[str(ID2)] = -30  
        # Decide whether or not the training should end at the next step
        elif step >= self.__steps_limit:
            end = 1
            reset = 1
        elif (discrepancy > self.__fail_threshold) and discrepancy > 0:
            end = 1
            reset = 1
        else:
            end = 0
            reset = 0
            
        return rewards, discrepancy, end, reset
    
    def iterate_inference_square_formation(self, step):
        end = 0
        # Current agents' positions
        agent_positions = self.get_agent_positions(step)
#        if step == 1 or step == 800:
#            best_comb_squares = utils.find_best_square_combination(agent_positions, self.__desired_distance)
#        else:
#            best_comb_squares = -1
        total_discrepancy = 0
        # Calculate the discrepancy of every agent and sum them all
        for agent_ID in range(0, self.__num_agents):
            if step == 1 or step == 600:
                total_discrepancy += utils.find_discrepancy_square_formation_four(agent_ID, agent_positions, self.__desired_distance) 
                #utils.find_discrepancy_square_formation_any(agent_positions, self.__desired_distance, best_comb_squares, agent_ID)
            else:
                total_discrepancy += 0
           
        
        discrepancy = total_discrepancy/self.__num_agents    

        # Continue the simulation for the number of steps defined in skip_steps
        for i in range(0, self.__skip_steps):
            vrep.simxSynchronousTrigger(self.__client_ID)
            
        # Decide whether or not the training should end at the next step
        if step >= self.__steps_limit:
            end = 1
            reset = 1
        elif ((discrepancy > self.__fail_threshold) or (discrepancy < self.__success_threshold)) and discrepancy > 0:
            end = 1
            reset = 1
        else:
            end = 0
            reset = 0
    
        return discrepancy, end, reset

    def iterate_inference_dispersion(self, step):
        end = 0
        # Current agents' positions
        agent_positions = self.get_agent_positions(step)
        total_discrepancy = 0
        # Calculate the discrepancy of every agent and sum them all
        for agent_ID in range(0, self.__num_agents):
            total_discrepancy += utils.find_discrepancy_dispersion(agent_ID, agent_positions, 
                                                                   self.__desired_distance)
        
        
        discrepancy = total_discrepancy/self.__num_agents    
        # Continue the simulation for the number of steps defined in skip_steps
        for i in range(0, self.__skip_steps):
            vrep.simxSynchronousTrigger(self.__client_ID)
            
        # Decide whether or not the training should end at the next step
        if step >= self.__steps_limit:
            end = 1
            reset = 1
        elif ((discrepancy > self.__fail_threshold) or (discrepancy < self.__success_threshold)) and discrepancy > 0:
            end = 1
            reset = 1
        else:
            end = 0
            reset = 0
            
        return discrepancy, end, reset
    
    def iterate_inference_chain_formation(self, step):
        end = 0
        # Current agents' positions
        agent_positions = self.get_agent_positions(step)
        total_discrepancy = 0
        # Calculate the discrepancy of every agent and sum them all
        for agent_ID in range(0, self.__num_agents):
            total_discrepancy += utils.find_discrepancy_chain_formation(agent_ID, agent_positions, 
                                                                        self.__desired_distance)[0]
        
        
        discrepancy = total_discrepancy/self.__num_agents    
        # Continue the simulation for the number of steps defined in skip_steps
        for i in range(0, self.__skip_steps):
            vrep.simxSynchronousTrigger(self.__client_ID)
            
        # Decide whether or not the training should end at the next step
        if step >= self.__steps_limit:
            end = 1
            reset = 1
        elif ((discrepancy > self.__fail_threshold) or (discrepancy < self.__success_threshold)) and discrepancy > 0:
            end = 1
            reset = 1
        else:
            end = 0
            reset = 0
            
        return discrepancy, end, reset
    
    def iterate_inference_aggregation(self, step):
        end = 0
        # Current agents' positions
        agent_positions = self.get_agent_positions(step)
        total_discrepancy = 0
        # Calculate the discrepancy of every agent and sum them all
        for agent_ID in range(0, self.__num_agents):
            total_discrepancy += utils.find_discrepancy_aggregation(agent_ID, agent_positions, 
                                                                        self.__desired_distance)
        
        
        discrepancy = total_discrepancy/self.__num_agents    
        # Continue the simulation for the number of steps defined in skip_steps
        for i in range(0, self.__skip_steps):
            vrep.simxSynchronousTrigger(self.__client_ID)
        
        # Decide whether or not the training should end at the next step
        if step >= self.__steps_limit:
            end = 1
            reset = 1
        elif ((discrepancy > self.__fail_threshold) or (discrepancy < self.__success_threshold)) and discrepancy > 0:
            end = 1
            reset = 1
        else:
            end = 0
            reset = 0
            
        return discrepancy, end, reset
    
    def comm_nearest_agent(self, agent_ID, step, num_agents, time_steps):
        communication = []
        for j in range(0, time_steps):
            step = step - j
            if step < 0:
                step = 0
            agent_positions = self.get_agent_positions(step)
            agent_orientations = self.get_agent_orientations(step)
            distances, _, IDs, _ = utils.find_n_nearest_robots(agent_positions, agent_ID, num_agents)
            
            distances = [distance*2 for distance in distances]

            readings = [self.__agents[ID].get_sensor_readings(step) for ID in IDs]
            orientations = [(agent_orientations[agent_ID] - agent_orientations[ID] + 2)/4 for ID in IDs]

            communication_step = []
            for i in range(0, num_agents):
                communication_step = communication_step + readings[i] + [distances[i], orientations[i]]
            
            communication.append(communication_step)
        
        return communication