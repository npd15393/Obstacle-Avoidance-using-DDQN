# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:44:45 2017

@author: Nishant
"""
from AirSimClient import *

from argparse import ArgumentParser

import numpy as np
from cntk.core import Value
from cntk.initializer import he_uniform
from cntk.layers import Sequential, Convolution2D, Dense, default_options
from cntk.layers.typing import Signature, Tensor
from cntk.learners import adam, learning_rate_schedule, momentum_schedule, UnitType
from cntk.logging import TensorBoardProgressWriter
from cntk.ops import abs, argmax, element_select, less, relu, reduce_max, reduce_sum, square
from cntk.ops.functions import CloneMethod, Function
from cntk.train import Trainer

class DroneEnv:
    def __init__(self,startpt,targetpt,client):
        self.startpt=startpt
        self.targetpt=targetpt
        self.nA=8
        self.client=client
        
        
    def step(self,action):
        """ This allows the agent to select the next action to perform in regard of the current state of the environment.
        It follows the terminology used in the Nature paper.

        Attributes:
            state (Tensor[input_shape]): The current environment state

        Returns: Int >= 0 : Next action to do
        """
        # Append the state to the short term memory (ie. History)

        # Keep track of interval action counter
        self._num_actions_taken += 1
        
        responses = self.client.simGetImages([ImageRequest(3, AirSimImageType.DepthPerspective, True, False)])
        next_state = self.transform_input(responses)
        
        collision_info = self.client.getCollisionInfo()
        quad_state = self.client.getPosition()
        quad_vel = self.client.getVelocity()
        
        reward = self.compute_reward(quad_state, quad_vel, collision_info)
        done = self.isDone(reward)
        
        return next_state,reward,done
        
    def reset(self):
        self.client.reset()
        time.sleep(0.5)
    
    def interpret_action(action):
        scaling_factor = 0.25
        if action == 0:
            quad_offset = (0, 0, 0)
        elif action == 1:
            quad_offset = (scaling_factor, 0, 0)
        elif action == 2:
            quad_offset = (0, scaling_factor, 0)
        elif action == 3:
            quad_offset = (0, 0, scaling_factor)
        elif action == 4:
            quad_offset = (-scaling_factor, 0, 0)    
        elif action == 5:
            quad_offset = (0, -scaling_factor, 0)
        elif action == 6:
            quad_offset = (0, 0, -scaling_factor)
        
        return quad_offset
    
    
    def compute_reward(self,quad_state, quad_vel,targetpt, collision_info):
#        thresh_dist = 7
#        beta = 1
#    
#        #z = -10
#        #pts = [np.array([-.55265, -31.9786, -19.0225]), np.array([48.59735, -63.3286, -60.07256]), np.array([193.5974, -55.0786, -46.32256]), np.array([369.2474, 35.32137, -62.5725]), np.array([541.3474, 143.6714, -32.07256])]
#        
#        
#        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
#    
#        if collision_info.has_collided:
#            reward = -100
#        elif quad_pt==np.array(self.targetpt):
#            reward=+100
#        else:    
#            dist = 10000000
#            for i in range(0, len(pts)-1):
#                dist = min(dist, np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))
#    
#            #print(dist)
#            if dist > thresh_dist:
#                reward = -10
#            else:
#                reward_dist = (math.exp(-beta*dist) - 0.5) 
#                reward_speed = (np.linalg.norm([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val]) - 0.5)
#                reward = reward_dist + reward_speed
    
        #        thresh_dist = 7
        beta = 1
    
        #z = -10
        #pts = [np.array([-.55265, -31.9786, -19.0225]), np.array([48.59735, -63.3286, -60.07256]), np.array([193.5974, -55.0786, -46.32256]), np.array([369.2474, 35.32137, -62.5725]), np.array([541.3474, 143.6714, -32.07256])]
        
        
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))
    
        if collision_info.has_collided:
            reward = -100
        elif quad_pt[0]==np.array(targetpt)[0] and quad_pt[2]==np.array(targetpt)[2] and quad_pt[1]==np.array(targetpt)[1]:
            reward=+100
        else:    
#            dist = 10000000

            dist = np.linalg.norm((quad_pt - targetpt))
    
            #print(dist)
#            if dist > thresh_dist:
#                reward = -10
#            else:
            reward_dist = (math.exp(-beta*dist) - 0.5) 
            reward_speed = (np.linalg.norm([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val]) - 0.5)
            reward = reward_dist + reward_speed

        return reward
    
    def isDone(self):
        done=self.client.getCollisionInfo().has_collided
        return done

    def transform_input(responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
    
        from PIL import Image
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert('L')) 
    
        return im_final