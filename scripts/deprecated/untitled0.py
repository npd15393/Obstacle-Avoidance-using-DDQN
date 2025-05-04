# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:05:53 2017

@author: Nishant
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 21:43:01 2017

@author: Nishant
"""

import tensorflow as tf
from keras import backend as K
import numpy as np
import itertools
import matplotlib.pyplot as plt
import random as rand

num_cores = 4
GPU=True
CPU=False

if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)


from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier

class DDPG():
    
    rewards=[]
    temp_buffer=[]
    def running_policy(self,epsi,q,pos):
        A = np.ones(4, dtype=float) * epsi /4
        best_action = np.argmax(q)
        A[best_action] += (1.0 - epsi)
        return A
    
    def greedy_policy(self,Q):
        return np.argmax(Q)
    
    def create_model(self,env):
        #Create a nn of fc and dropout layers
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim=1, kernel_initializer="normal"))
        #model.add(Dropout(0.1))
        model.add(Dense(32, activation="relu", kernel_initializer="normal"))
        #model.add(Dropout(0.1))
        model.add(Dense(env.nA, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print("NN Architecture created")
        return model

    
#    def QLearn(self,envr,model,nE,pR=False):
#        #Q= defaultdict(lambda: np.zeros(envr.action_space.n))
#    
#        assert not (model == None or envr == None)
#        
#        rb=self.new_qlearn_epoch(envr,model,nE,True)
#        model=self.new_qlearn_epoch(envr,model,nE,False,rb)
#        
#        return model
    
    def trainPG(envr):
        
        model=create_model(envr)
        for i in range(MAX_EPISODES):
            states,rewards,acts=rollout(envr,model)
            delta=policy_grad(states,rewards,acts)
            
            new_acts=
    
    def rollout(self,envr,model):
        e_rewards=[]
        e_states=[]
        e_acts=[]
        c_st=envr.reset()
        e_states.append(c_st)
        while n<self.MAX_TRIALS:
            
            act=self.get_action()
            e_acts.append(act)
            ns,re,done,_=envr.step(act)
 
            e_states.append(ns)
            e_rewards.append(re)
            
            if done:
                return e_states,e_rewards,e_acts
            
            c_st=ns
            
        return e_states,e_rewards,e_acts
    
    def policy_grad(states,rewards,acts):
        
    def new_qlearn_epoch(self,envr,model,nE,refill=False,rb=[],buffersize=128):
        
        epsilon=1
        GAMMA=0.99
        temp_buffer=[]

        for i in range(nE):  
            
            current_pos=envr.reset()
            done=False
            treward=0
            
            while not done:        
                
                #envr._render()
                q=model.predict(np.reshape(current_pos,(1,4)))[0]
                
                epsilon*=0.99
                if not refill:
                    act=self.running_policy(epsilon,q)
                else:
                    act=rand.choice([0,1])
                
                next_state, reward, done, _= envr.step(act)
                reward if not done else -100
                
                treward+=reward                
                
                q1=model.predict(np.reshape(next_state,(1,4)))[0]
                target= model.predict(np.reshape(current_pos,(1,4)))[0]
                    
                if done:
                                target[act]=reward
                                self.rewards.append(treward)
                            
                else:
                                target[act]=reward+GAMMA*np.amax(q1)
                            
                #model.fit(np.reshape(current_pos,(1,4)),np.reshape(target,(1,2)),epochs=1,verbose=0)
                temp_buffer.append([current_pos,act,next_state,target])
                
                if done:
                    print('Episode:{0}/{2} - Total reward={1}'.format(i+1,treward,nE))
                                 
                current_pos=next_state
                
            if len(temp_buffer)>=buffersize:
                    if not refill:
                        #rbs=rand.sample(range(len(rb)), 32 )
                        
                        X=[]
                        targets=[]
                        for u in rb:
                            X.append(u[0])
                            targets.append(u[3])
                            
                            model.fit(np.reshape(u[0],(1,4)),np.reshape(u[3],(1,2)),epochs=1,verbose=0)
                        rb=self.copy_buffer(temp_buffer)
                        temp_buffer=[]
                        
                    else:
                        return temp_buffer
                        print('Initial trial done')

    
    def copy_buffer(self,rb):
            nb=[]
            for i in rb:
                nb.append(i)
            return nb
        
    def run_policy(self,env,model):
            current_pos=env.reset()
            q=model.predict(current_pos)
            
            act=np.argmax(q)
            for t in itertools.count():        
                next_state, reward, done, _= env.step(act)
                if done:
                    break
                else:
                    current_pos=next_state
            