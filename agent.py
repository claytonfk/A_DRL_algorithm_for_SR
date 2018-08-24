# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 18:52:31 2018

@author: Clayton
"""

import vrep
import numpy as np

class Agent:
    # Initializes agent
    def __init__(self, client_ID, agent_ID, name = "ePuck", min_detection_dist = 0.01, max_detection_dist = 0.5, enable_actuation = 1):
        
        self.__client_ID = client_ID # Client ID of VREP 
        self.__agent_ID = agent_ID
        # Obtains the name of the agent
        self.__name = name
        if name != "ePuck" and len(name) >= 7:
            self.__suffix = name[-(len(name) - 5):]
        else:
            self.__suffix = ""
            
        self.__min_detection_dist = min_detection_dist # Minimum detection distance for the sensors
        self.__max_detection_dist = max_detection_dist # Maximum detection distance for the sensors
        self.__enable_actuation   = enable_actuation   # If set to true, the agent moves, otherwise it doesn't
        
        # For processing sensor measurements
        self.__m = 1/(max_detection_dist - min_detection_dist)
        self.__n = -self.__m*min_detection_dist
        
        # Initialize base, joints and sensors
        self.init_base()
        self.init_joints()
        self.init_sensors()
        
        # Initialize sensor readings buffer, positions buffer, accumulated rewards variable 
        # and last action variable
        self.restart()
        
    ## Getters and setters
    def get_client_ID(self):
        return self.__client_ID
        
    def get_agent_ID(self):
        return self.__agent_ID
    
    def get_base_ID(self):
        return self.__base
        
    def get_min_detection_dist(self):
        return self.__min_detection_dist
        
    def get_max_detection_dist(self):
        return self.__max_detection_dist
    
    def is_actuation_enabled(self):
        return self.__enable_actuation
   
    def get_step(self):
        return self.__step
    
    def set_actuation(self, enable_actuation):
        self.__enable_actuation = enable_actuation
    
    ## Initialize components 
    
    # Initialize base (used for obtaining the position of the robot with respect to the global frame)
    def init_base(self):
        error_code, self.__body = vrep.simxGetObjectHandle(self.__client_ID, "ePuck" + self.__suffix, vrep.simx_opmode_blocking)
        assert error_code == 0, "Robot handle could not be obtained."
        
        error_code, self.__base = vrep.simxGetObjectHandle(self.__client_ID, "ePuck_base" + self.__suffix, vrep.simx_opmode_blocking)
        assert error_code == 0, "Base handle could not be obtained."
        
        vrep.simxGetObjectPosition(self.__client_ID, self.__base, -1, vrep.simx_opmode_streaming)
        vrep.simxGetObjectOrientation(self.__client_ID, self.__base, -1, vrep.simx_opmode_streaming)
    
    # Initialize the left and right joints
    def init_joints(self):
        error_code, self.__right_joint = vrep.simxGetObjectHandle(self.__client_ID, "ePuck_rightJoint" + self.__suffix, vrep.simx_opmode_blocking)    
        assert error_code == 0, "Joint handle could not be obtained."
        
        error_code, self.__left_joint  = vrep.simxGetObjectHandle(self.__client_ID, "ePuck_leftJoint" + self.__suffix, vrep.simx_opmode_blocking)
        assert error_code == 0, "Joint handle could not be obtained."

    # Initialize the sensors
    def init_sensors(self):
        self.__sensors = []
        
        for i in range(1, 9):
            error_code, sensor = vrep.simxGetObjectHandle(self.__client_ID, "ePuck_proxSensor" + str(i) + self.__suffix, vrep.simx_opmode_blocking)    
            assert error_code == 0, "Sensor handle could not be obtained."
            self.__sensors.append(sensor)
            vrep.simxReadProximitySensor(self.__client_ID, self.__sensors[i - 1], vrep.simx_opmode_streaming)     
    

    # Actuate
    def actuate(self, right_joint_speed, left_joint_speed, action_ID):
        # If the actuation is not enabled, then the agent will not move
        if not self.__enable_actuation:
            right_joint_speed = 0
            left_joint_speed = 0
            action_ID = 0
        
        # Save action ID
        self.__last_action_ID = action_ID
        self.__actions_buffer.append(action_ID)
        # Set speeds to the joints
        vrep.simxSetJointTargetVelocity(self.__client_ID, self.__right_joint, right_joint_speed, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(self.__client_ID, self.__left_joint,  left_joint_speed, vrep.simx_opmode_streaming)
        

    # Retrieve position of the robot
    def __get_position(self):
        error_code, position = vrep.simxGetObjectPosition(self.__client_ID, self.__base, -1, vrep.simx_opmode_blocking)
        assert error_code == 0, "Position of base could not be obtained."
        position.pop(2)  # Do not consider z axis
        position = [round(z, 3) for z in position] # Round to 3 digits
        
        return position
    
    # Get position of the robot with respect to the step
    def get_position(self, step):
        if len(self.__positions_buffer) == 0 or step > self.__position_step:
            self.__positions_buffer.append(self.__get_position())
            if step > self.__position_step:
                self.__position_step = self.__position_step + 1
 
        return self.__positions_buffer[step]
    
    # Set position of the robot 
    def set_position(self, position):
        error_code = vrep.simxSetObjectPosition(self.__client_ID, self.__body, -1, position, vrep.simx_opmode_blocking)
        assert error_code == 0, "Position of robot could not be set."
    
    # Retrieve orientation of the robot
    def __get_orientation(self):
        error_code, orientation = vrep.simxGetObjectOrientation(self.__client_ID, self.__base, -1, vrep.simx_opmode_blocking)
        assert error_code == 0, "Orientation of base could not be obtained."
        orientation = orientation[2]/3.141596  # Do not consider x and y axis and normalize
        orientation = round(orientation, 3) # Round to 3 digits
        
        return orientation
    
    # Get orientation of the robot with respect to the step
    def get_orientation(self, step):
        if len(self.__orientations_buffer) == 0 or step > self.__orientation_step:
            self.__orientations_buffer.append(self.__get_orientation())
            if step > self.__orientation_step:
                self.__orientation_step = self.__orientation_step + 1
 
        return self.__orientations_buffer[step]
    
    # Retrieve sensor readings of the robot
    def __get_sensor_readings(self):
        readings = []
        
        for i in range(0, 8):
            error_code, detection_state, detected_point, _, _ = vrep.simxReadProximitySensor(self.__client_ID, self.__sensors[i], vrep.simx_opmode_buffer)
            assert error_code == 0, "Could not obtain readings." 
           
            if detection_state == 0:
                distance = 0
            else:                
                distance = np.linalg.norm(detected_point)
                # Preprocesses the readings
                if distance < self.__min_detection_dist:
                    distance = 0
                else:
                    distance = self.__m*distance + self.__n
            
            readings.append(distance)
            
        readings = [round(z, 3) for z in readings] # Rounds to 3 digits
        return readings
    
    # Get sensor readings of the robot with respect to the step
    def get_sensor_readings(self, step):
        if len(self.__readings_buffer) == 0 or step > self.__sensor_readings_step:
            self.__readings_buffer.append(self.__get_sensor_readings())
            if step > self.__sensor_readings_step:
                self.__sensor_readings_step = self.__sensor_readings_step + 1
            
        return self.__readings_buffer[step]
    
    # Get a batch of sensor readings. Last num_readings readings
    def get_sensor_buffer_readings(self, step, num_readings):
        self.get_sensor_readings(step)
        if step + 1 - num_readings >= 0:
            buffered_readings = self.__readings_buffer[step + 1 - num_readings: step + 1]
        else:
            buffered_readings = self.__readings_buffer[0: step + 1]
            buffered_readings  = buffered_readings + [buffered_readings[-1]]*(num_readings - len(buffered_readings))

        return buffered_readings
    
    def get_action_buffer(self, step, num_actions):
        buffered_actions = self.__actions_buffer[step - num_actions + 1: step + 1]
        
        if len(buffered_actions) < num_actions:
            buffered_actions = [0]*(num_actions - len(buffered_actions)) + buffered_actions
            
        return buffered_actions

    # Returns last executed action
    def get_last_action_ID(self):
        return self.__last_action_ID
    
    # Returns rewards accumulated in the episode
    def get_accum_rewards(self):
        return self.__reward
    
    # Reset all the buffers and variables (called for every episode)
    def restart(self):
        self.__readings_buffer = []
        self.__positions_buffer = []
        self.__orientations_buffer = []
        self.__actions_buffer = []
        self.__reward = 0
        self.__last_action_ID = 0
        self.__position_step = 0
        self.__orientation_step = 0
        self.__sensor_readings_step = 0
        
        # Reset joint speeds
        self.actuate(0, 0, 0)        
        # Obtain current position and sensor readings
        self.get_position(0)
        self.get_orientation(0)
        self.get_sensor_readings(0)
    
    # Accumulate rewards
    def accum_rewards(self, reward):
        self.__reward = self.__reward + reward