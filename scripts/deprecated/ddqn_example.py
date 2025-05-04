# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:43:48 2017

@author: Nishant
"""

import gym
import itertools
import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Lambda,merge,Input,Flatten
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import random as rand
from collections import deque

rewards=[]
buffersize=10000
batch_size=64
target_update_threshold=40


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

temp_buffer=Memory()
def create_model(env):
        learning_rate=0.001
        #Create a nn of 4 fc and dropout layers
        input = Input(shape=(4,))
        x=Dense(64, activation="relu")(input)
        x=Dense(64, activation="relu")(x)
        #x=Flatten()(x)
        state_value = Dense(64, activation='relu', init='uniform')(x)
        state_value = Dense(1, init='uniform')(state_value)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0]), output_shape=(2,))(state_value)
        # action advantage tower - A
        action_advantage = Dense(64, activation='relu', init='uniform')(x)
        action_advantage = Dense(2, init='uniform')(action_advantage)
        action_advantage = Lambda(lambda a: a[:, :] - K.max(a[:, :], keepdims=True), output_shape=(2,))(action_advantage)
        # merge to state-action value function Q
        state_action_value = merge([state_value, action_advantage], mode='sum')
        model = Model(input=input, output=state_action_value)
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
        print("NN Architecture created")
        return model


def running_policy(epsi,q):
        if np.random.rand()<=max([epsi,0.01]):
            return rand.choice([0,1])
        return np.argmax(q)

def QLearn(envr,nE,pR=False):
        assert not ( envr == None)   
        model=train(envr,nE)
       
        return model

def initial_run(envr):
    epsi=1
    current_pos=envr.reset()
    for i in range(1000):
        epsi*=0.99
        act=rand.choice([0,1])
        next_state, reward, done, _= envr.step(act)
        
        temp_buffer.add([current_pos,act,next_state,reward,done])
        if done:
            current_pos=envr.reset()
        else:
            current_pos=next_state
        

def train(envr,nE):
        epsilon=1
        GAMMA=0.99
        

        for i in range(nE):  
            
            current_pos=envr.reset()
            done=False
            treward=0
            
            while not done:        
                
                #envr._render()
                q=model.predict(np.reshape(current_pos,(1,4)))[0]               
 
                epsilon*=0.99

                act=running_policy(epsilon,q)
                    
                next_state, reward, done, _= envr.step(act)
                #reward=reward if not done else -100
                
                temp_buffer.add([current_pos,act,next_state,reward,done])
                treward+=reward                
                
                #if len(temp_buffer)==buffersize:
                X=[]
                targets=[]
                        
                #while len(temp_buffer)>buffersize-batch_size:
                            #u=fetch_random_value(temp_buffer)
                us=temp_buffer.sample(batch_size)
                for u in us:
                                X.append(u[0])
                                
                                q1=targetmodel.predict(np.reshape(u[2],(1,4)))[0]
                                target= model.predict(np.reshape(u[0],(1,4)))[0]
                        
                                if u[4]:
                                    target[u[1]]=u[3]
                                else:
                                    target[u[1]]=u[3]+GAMMA*np.amax(q1)
                                    
                                targets.append(target)

                model.fit(np.reshape(X,(batch_size,4)),np.reshape(targets,(batch_size,2)),epochs=1,verbose=0)

                            #temp_buffer=[]
                if i%target_update_threshold==0:
                                for j in range(len(model.layers)):
                                    targetmodel.layers[j].set_weights(model.layers[j].get_weights())
                
                
                #model.fit(np.reshape(current_pos,(1,4)),np.reshape(target,(1,2)),epochs=1,verbose=0)
                
                if done :#and (i+1)%10==0:
                    print('Episode:{0}/{2} - Total reward={1}'.format(i+1,treward,nE))
                    rewards.append(treward)
                    
#                if len(temp_buffer)%target_update_threshold==0:
#                    for j in range(len(model.layers)):
#                        targetmodel.layers[j].set_weights(model.layers[j].get_weights())
                        
                current_pos=next_state
         
        return model
    
def fetch_random_value(rb):
    u=rand.choice(range(len(rb)))
    t=rb[u]
    rb.pop(u)
    return t

def copy_buffer(rb):
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
                                
env=gym.make('CartPole-v1')
model=create_model(env)
initial_run(env)
targetmodel=keras.models.clone_model(model)
model=QLearn(env,1000)
plt.plot(range(len(rewards)),rewards)