# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 22:37:41 2018

@author: Clayton
"""

import numpy as np
import math

# Find center of mass of certain agents
def find_com(agent_positions):
    transposed_positions = np.transpose(agent_positions)
    num_agents = len(agent_positions)
    # Center of mass coordinates
    x_com = round(sum(transposed_positions[0])/num_agents, 3)
    y_com = round(sum(transposed_positions[1])/num_agents, 3)
    
    return x_com, y_com

# Find Euclidean distance between two points
def find_distance_between_points(point1, point2):
    distance = math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))
    return distance

# Find cosine of the angle between two vectors
def find_vectors_cosine(v0, v1):
    return (np.dot(v0, v1))/(np.linalg.norm(v0)*np.linalg.norm(v1))

# Find sine of the angle between two vectors
def find_vectors_sine(v0, v1):
    return (np.linalg.norm(np.cross(v0, v1)))/(np.linalg.norm(v0)*np.linalg.norm(v1))

# Find angle between two vectors
def find_vectors_angle(v0, v1):
    return math.atan2(find_vectors_sine(v0, v1), find_vectors_cosine(v0, v1))

# Find angle between two vectors - between 0 and 2pi
def find_vectors_angle2(v0, v1):
    angle = math.atan2(find_vectors_sine(v0, v1), find_vectors_cosine(v0, v1))
    
    if v1[1] < 0:
        angle = 2*math.pi - angle
    
    return angle

# Calculate area within polygon and its perimeter
def polygon_area_and_perimeter(vertices):
    sequential_vertices = []
    sequential_vertices.append(vertices[0])

    min_angle_index = -1
    min_angle = 2*math.pi
    index_now = 0
    indices = [_ for _ in range(1, len(vertices))]
    perimeter = 0
    
    while len(indices) != 0:
        for i in indices:
            angle = find_vectors_angle2([1, 0], [vertices[i][0] - vertices[index_now][0], vertices[i][1] - vertices[index_now][1]])
            if angle < min_angle:
                min_angle = angle
                min_angle_index = i
        
        sequential_vertices.append(vertices[min_angle_index])
        perimeter += find_distance_between_points(vertices[index_now], vertices[min_angle_index])
        index_now = min_angle_index
        indices.remove(index_now)
        min_angle_index = -1
        min_angle = 2*math.pi
        
    perimeter += find_distance_between_points(sequential_vertices[0], sequential_vertices[-1])

    x = [_[0] for _ in sequential_vertices]
    y = [_[1] for _ in sequential_vertices]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))), perimeter