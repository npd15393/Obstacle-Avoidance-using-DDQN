# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 19:03:59 2017

@author: Nishant
"""
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Lambda,merge,Input,MaxPooling2D,Conv2D
from keras import backend as K
from keras.optimizers import Adam
import keras

class model_creator:
    
    def __init__(self):
        self.ConvLayerSizes=[[16,8,4],[32,4,2],[32,3,1]]

    def createDDQN(self,input_shape,nA):
        learning_rate=0.001
        #Create a nn of 4 fc and dropout layers
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4),
                 activation='relu',
                 input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2),
                 activation='relu',
                 input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(nA, activation='softmax'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
        print("NN Architecture created")      
        target=keras.models.clone_model(model, model.get_weights())      
        return model,target

        
    def createDuelDDQN(self,input_shape,nA):
        learning_rate=0.001
        input = Input(shape=(input_shape,))
        x=Conv2D(16, kernel_size=(8, 8), strides=(4, 4),
                 activation='relu',
                 input_shape=input_shape)(input)
        x=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x=Conv2D(32, kernel_size=(4, 4), strides=(2, 2),
                 activation='relu',
                 input_shape=input_shape)(x)
        x=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x=Conv2D(32, kernel_size=(4, 4), strides=(2, 2),
                 activation='relu',
                 input_shape=input_shape)(x)
        x=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        state_value = Dense(256, activation='relu', init='uniform')(x)
        state_value = Dense(1, init='uniform')(state_value)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0]), output_shape=(nA,))(state_value)
        # action advantage tower - A
        action_advantage = Dense(128, activation='relu', init='uniform')(x)
        action_advantage = Dense(nA, init='uniform')(action_advantage)
        action_advantage = Lambda(lambda a: a[:, :] - K.max(a[:, :], keepdims=True), output_shape=(nA,))(action_advantage)
        # merge to state-action value function Q
        state_action_value = merge([state_value, action_advantage], mode='sum')
        model = Model(input=input, output=state_action_value)
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
        print("NN Architecture created")
        target=keras.models.clone_model(model, model.get_weights())
        return model,target