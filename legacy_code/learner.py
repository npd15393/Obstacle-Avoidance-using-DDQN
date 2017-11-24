# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 19:22:41 2017

@author: Nishant
"""
import numpy as np
import random as rand
import itertools
from collections import defaultdict
from lib import plotting

class Learner():
    def __init__(self,alpha=0.1,epsi=0.2,discount=1):
        self.alpha=alpha
        self.epsi=epsi
        self.discount=discount
        
    def running_policy(self,Q,pos):
        A = np.ones(8, dtype=float) * self.epsi / 8
        best_action = np.argmax(Q[pos])
        A[best_action] += (1.0 - self.epsi)
        return A
    
    def greedy_policy(self,Q,pos):
        return np.argmax(Q[pos])
    
    def SARSA(self,envr,nE):
        Q= defaultdict(lambda: np.zeros(envr.action_space.n))
        
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(nE),
            episode_rewards=np.zeros(nE))
        
        for i in range(nE):  
            current_pos=envr.reset()
            act_prob=self.running_policy(Q,current_pos)
            act = np.random.choice(np.arange(len(act_prob)), p=act_prob)
    
            for t in itertools.count():        
                next_state, reward, done, _= envr.step(act)
    
                stats.episode_rewards[i] += reward
                stats.episode_lengths[i] = t
    
                next_act_prob=self.running_policy(Q, current_pos)
                next_act = np.random.choice(np.arange(len(next_act_prob)), p=next_act_prob)
                
                TD_err=reward+self.discount*Q[next_state][next_act]-Q[current_pos][act]
                Q[current_pos][act]+=self.alpha*(TD_err)
                
                if done:
                    break
    
                current_pos=next_state
                act=next_act
        return Q,stats
    
    def Q_learn(self,envr,nE,Q=[]):
        if Q==[]:
            Q= defaultdict(lambda: np.zeros(envr.action_space.n))
        
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(nE),
            episode_rewards=np.zeros(nE))
        
        for i in range(nE):  
            
            current_pos=envr.reset()
    
            act_prob=self.running_policy(Q,current_pos)
            act = np.random.choice(np.arange(len(act_prob)), p=act_prob)
    
            for t in itertools.count():        
                next_state, reward, done, _= envr.step(act)
    
                stats.episode_rewards[i] += reward
                stats.episode_lengths[i] = t
    
                next_act_prob=self.running_policy(Q, current_pos)
                next_act = np.random.choice(np.arange(len(next_act_prob)), p=next_act_prob)
                up_act = self.greedy_policy(Q,current_pos)
                
                TD_err=reward+self.discount*Q[next_state][up_act]-Q[current_pos][act]
                Q[current_pos][act]+=self.alpha*(TD_err)
    
                if done:
                    break
                    

                current_pos=next_state
                act=next_act
        return Q,stats