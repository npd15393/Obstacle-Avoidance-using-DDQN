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


class DDQN:
    # Methods
    # Init: Inits duel networks; *requires gym based env as arg 
    # train:Train networks
    # plot_behavior: Plots total episodic rewards per episode
    def __init__(self,envr,buffersize=1000,batch_size=64,target_update_threshold=40):
        self.rewards=[]
        self.buffersize=buffersize
        self.batch_size=batch_size
        self.target_update_threshold=target_update_threshold
        model=self.create_model(envr)
        self.targetmodel=keras.models.clone_model(model)
        self.temp_buffer=Memory(buffersize)
        
    def create_model(nA,sD,learning_rate=0.0001):
        #Create network with below params
        h1size=64
        h2size=64
        svfh1=64
        avfh1=64
        
        
        input = Input(shape=(sD,))
        x=Dense(h1size, activation="relu")(input)
        x=Dense(h2size, activation="relu")(x)
        #x=Flatten()(x)
        state_value = Dense(svfh1, activation='relu', init='uniform')(x)
        state_value = Dense(1, init='uniform')(state_value)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0]), output_shape=(nA,))(state_value)
        # action advantage tower - A
        action_advantage = Dense(avfh1, activation='relu', init='uniform')(x)
        action_advantage = Dense(nA, init='uniform')(action_advantage)
        action_advantage = Lambda(lambda a: a[:, :] - K.max(a[:, :], keepdims=True), output_shape=(nA,))(action_advantage)
        # merge to state-action value function Q
        state_action_value = merge([state_value, action_advantage], mode='sum')
        model = Model(input=input, output=state_action_value)
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
                                
                                q1=self.targetmodel.predict(np.reshape(u[2],(1,4)))[0]
                                target= self.model.predict(np.reshape(u[0],(1,4)))[0]
                        
                                if u[4]:
                                    target[u[1]]=u[3]
                                else:
                                    target[u[1]]=u[3]+GAMMA*np.amax(q1)
                                    
                                targets.append(target)

                self.model.fit(np.reshape(X,(self.batch_size,4)),np.reshape(targets,(self.batch_size,2)),epochs=1,verbose=0)

                            #temp_buffer=[]
                if i%self.target_update_threshold==0:
                                for j in range(len(self.model.layers)):
                                    self.targetmodel.layers[j].set_weights(self.model.layers[j].get_weights())

                if done :#and (i+1)%10==0:
                    print('Episode:{0}/{2} - Total reward={1}'.format(i+1,treward,nE))
                    self.rewards.append(treward)
                    
#                if len(temp_buffer)%target_update_threshold==0:
#                    for j in range(len(model.layers)):
#                        targetmodel.layers[j].set_weights(model.layers[j].get_weights())
                        
                current_pos=next_state
         
        return self.model
    
    def plot_behavior(self):
        plt.plot(range(len(self.rewards)),self.rewards)