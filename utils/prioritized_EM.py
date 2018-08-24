# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:15:08 2018
@author: Clayton
"""

import math
import numpy as np

class PrioritizedExperienceMemory():
    # Initialize prioritized experience memory
    def __init__(self, capacity = 10000, alpha = 0.9, beta = 0.1):
        self.capacity = capacity
        self.contents = [] # Content of the memory
        self.priorities = [] # Priorities
        # Parameters for the experience memory
        self.alpha = alpha
        self.beta = beta
        self.head = 0

    # Add experience to the memory, with defined priority
    def add(self, state, action, reward, next_state, episode):
        # Obtain maximal priority
        if len(self.priorities) != 0:
            priority = max(self.priorities)
        else:
            priority = 1
        # If the memory is not full, add a new experience
        if len(self.contents) < self.capacity:
            self.contents.append((state, action, reward, next_state))
            self.priorities.append(math.pow(priority, self.alpha))
        # If the memory is full, replace the least prioritized experience with the newest experience
        else:
            # Find index of the least prioritized experience
            idx = np.argmin(self.priorities)
            self.contents[idx]  = (state, action, reward, next_state)
            self.priorities[idx] = math.pow(priority, self.alpha)
            
        self.head = self.head + 1

    # Obtain a sample of experiences from the memory. Each experience has its own probability
    # of being included in the sample
    def sample(self, batch_size):
        # If the batch size is greater than the length of the memory, set the batch size to 
        # be equal to the length of the memory
        if batch_size > len(self.contents):
            batch_size = len(self.contents)
        
        # Find probability distribution
        priorities_np = np.array(self.priorities) # conversion to numpy (for performance)
        sum_of_priorities = np.sum(priorities_np)
        p = np.round(priorities_np/sum_of_priorities, 3) + 0.001
        sum_p = np.sum(p)

        # Renormalize
        p = p/sum_p

        # Indexes sampled from memory
        idxs = np.random.choice(len(self.contents), batch_size, p = p, replace = False)
        # Obtain sample
        sample = [self.contents[idx] for idx in idxs]
        
        # Before computing importance-sampling weights
        p_beta = np.power(p, self.beta)
        min_p_beta = np.min(p_beta)
        # Importance-sampling weights for sampled experiences
        weights_sample = [min_p_beta/p_beta[idx] for idx in idxs]

        return sample, list(idxs), weights_sample
    
    # Update priorities
    def update_priorities(self, priorities, idxs):
        idx_priority = 0
        for idx in idxs:
            priority = priorities[idx_priority] 
            self.priorities[idx] = priority if not math.isnan(priority) and not math.isinf(priority) \
                                   else max(self.priorities)
            idx_priority += 1
    
    # Empty the memory
    def empty(self):
        self.contents = []
        self.priorities = []
    
    # Obtain the length of the memory
    def __len__(self):
        return len(self.contents)