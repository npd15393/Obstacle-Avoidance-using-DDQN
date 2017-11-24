# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 20:34:16 2017

@author: Nishant
"""
import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Lambda,merge,Input
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import random as rand
from collections import deque

class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


class DQN:
    # Methods
    # Init: Inits network; *requires gym based env as arg 
    # train:Train network
    # plot_behavior: Plots total episodic rewards per episode
    def __init__(self,envr,buffersize=1000,batch_size=64,target_update_threshold=40):
        self.rewards=[]
        self.buffersize=buffersize
        self.batch_size=batch_size
        self.target_update_threshold=target_update_threshold
        model=self.create_model(envr)
        self.targetmodel=keras.models.clone_model(model)
        self.temp_buffer=Memory(buffersize)
        
    def create_model(nA,sD,hiddenLayers=[32,32],learning_rate=0.0001):
        #Create network with below params
        model = Sequential()
        for i in range(len(hiddenLayers)):
            if i==0:
                model.add(Dense(i, activation="relu", input_dim=sD))
            elif hiddenLayers[i]<0:
                model.add(Dropout(-hiddenLayers[i]))
            else:
                model.add(Dense(hiddenLayers[i], activation="relu"))

        model.add(Dense(nA, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
        print(model.summary())
        return model

    def running_policy(epsi,q):
        if np.random.rand()<=max([epsi,0.01]):
            return rand.choice([0,1])
        return np.argmax(q)
    
    def initial_run(self,envr):
        epsi=1
        current_pos=envr.reset()
        for i in range(1000):
            epsi*=0.99
            act=rand.choice([0,1])
            next_state, reward, done, _= envr.step(act)
            
            self.temp_buffer.add([current_pos,act,next_state,reward,done])
            if done:
                current_pos=envr.reset()
            else:
                current_pos=next_state

    def train(self,nE,simu=False):
        
        self.initial_run(self.envr)
        epsilon=1
        GAMMA=0.99
        

        for i in range(nE):  
            
            current_pos=self.envr.reset()
            done=False
            treward=0
            
            while not done:        
                
                if simu:
                    self.envr._render()
                
                q=self.model.predict(np.reshape(current_pos,(1,4)))[0]               
 
                epsilon*=0.99

                act=self.running_policy(epsilon,q)
                    
                next_state, reward, done, _= self.envr.step(act)

                self.temp_buffer.add([current_pos,act,next_state,reward,done])
                treward+=reward                
                

                X=[]
                targets=[]

                us=self.temp_buffer.sample(self.batch_size)
                for u in us:
                                X.append(u[0])
                                
                                q1=self.model.predict(np.reshape(u[2],(1,4)))[0]
                                target= self.model.predict(np.reshape(u[0],(1,4)))[0]
                        
                                if u[4]:
                                    target[u[1]]=u[3]
                                else:
                                    target[u[1]]=u[3]+GAMMA*np.amax(q1)
                                    
                                targets.append(target)

                self.model.fit(np.reshape(X,(self.batch_size,4)),np.reshape(targets,(self.batch_size,2)),epochs=1,verbose=0)


                if done :#and (i+1)%10==0:
                    print('Episode:{0}/{2} - Total reward={1}'.format(i+1,treward,nE))
                    self.rewards.append(treward)

                current_pos=next_state
         
        return self.model
    
    def plot_behavior(self):
        plt.plot(range(len(self.rewards)),self.rewards)