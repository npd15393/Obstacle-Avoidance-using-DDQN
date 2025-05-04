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
from cntk.ops.functions import CloneMethod, Function, load_model
from cntk.train import Trainer
import csv
from PIL import Image
import pprint
import matplotlib.pyplot as plt


class DeepQAgent(object):
    def __init__(self):
        self._load_models()
        self._history = History((4,84,84))
        self._num_actions_taken = 0

    def _load_models(self):
        self._target_net = load_model('trainedModels/Target_net')
        print ('**Target model loaded**')

    def act(self, state):
        self._history.append(state)
        env_with_history = self._history.value
        q_values = self._target_net.eval(env_with_history.reshape((1,) + env_with_history.shape))
        action = q_values.argmax()
        self._num_actions_taken += 1
        return action


class History(object):
    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        return self._buffer

    def append(self, state):
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = state

    def reset(self):
        self._buffer.fill(0)
 
        
def transform_input(responses):
    margin = 10
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    img1d = 255/np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((84, 84)).convert('L'))  
    #cropped_image = im_final[42-margin: 42+margin , :]
    return im_final

def interpret_action(action):
    scaling_factor = 0.75
    if action == 0:
        quad_offset = (0, 0, 0)
    elif action == 1:
        quad_offset = (scaling_factor, 0, 0)
    elif action == 2:
        quad_offset = (-scaling_factor, 0, 0)
    elif action == 3:
        quad_offset = (0, scaling_factor, 0)
    elif action == 4:
        quad_offset = (0, -scaling_factor, 0)    
    # elif action == 5:
    #     quad_offset = (0, -scaling_factor, 0)
    # elif action == 6:
    #     quad_offset = (0, 0, -scaling_factor)
    
    return quad_offset

def compute_reward(quad_state, quad_vel, collision_info):
    thresh_dist = 500  #thershold distance for reward function
    beta = 0.05
    z = -10
    pts = [np.array([-50, 77.5, -5])]
    quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))

    # if collision_info.has_collided:
    if (collision_info.position[b'x_val'] != 0) or (collision_info.position[b'y_val'] != 0) or (collision_info.position[b'z_val'] != 0):
        reward = -200  #reward for collision
        print ('The drone has collided')    
    else:    
        dist = 10000000
        for i in range(0, len(pts)):
            distance_actual = np.linalg.norm(quad_pt - pts[i])
            dist = min(dist, distance_actual )
            # print ('Distance : {} '.format(distance_actual))
        if dist > thresh_dist:
            reward = -160
            # print ('Distance : {} , reward =  : {} '.format(dist,reward))
        elif (dist < 10 ):
            reward = 200 
            print ('The goal is reached ')    
        else:
            # loss_dist = (math.exp(beta*dist) - 0.5) 
            reward_speed = (np.linalg.norm([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val]) - 0.5)
            # reward =  reward_speed - loss_dist
            reward = -sum(np.absolute(pts[0] - quad_pt)) 
            # print ('Distance : {} , reward_dist : {} , reward_speed : {}  , reward =  : {} '.format(dist,loss_dist,reward_speed,reward))
        if (quad_state.z_val < -100):
            reward -= 100
            print ('The pilot is going above z range')
    return reward




def isDone(reward):
    done = 0
    if  reward <= -150: #collide
        done = 1
        print ('The drone failed to reach the goal')
    if (reward  > -5):
        done = 1
        print ('The drone reached nice.')     
    return done



# The Execution of the program starts from here

# Connect to the AirSim simulator 
client = MultirotorClient()
client.confirmConnection()
print ('Connection with AirSim established')

quad_pose = client.getPosition();
print('QUAD_POS X: {0} Y: {1} Z: {2}'.format(quad_pose.x_val, quad_pose.y_val, quad_pose.z_val))


n_episodes = 5
episode_length = []
avg_reward = []

# Make a RL agent
agent = DeepQAgent()

# Training Loop
for episode in range(n_episodes):
    print ('Starting Episode :', episode)
    client.reset()  # Reset the environment
    client.enableApiControl(True)   # Request API control to AirSim 

    # Takeoff
    client.takeoff()
    print ('Quad takeoff')
    
    client.moveToPosition(0, 0, -5, 5) # Move 5 meter above home
    # Reset loop variables

    r = []

    # Get the initial image from the camera
    responses = client.simGetImages([ImageRequest(1, AirSimImageType.DepthPerspective, True, False)])
    current_state = transform_input(responses)
    
    for steps in range(1000):
        
        if steps == (999):
            print('Max Steps reached')
        
        
        action = agent.act(current_state)
        quad_offset = interpret_action(action)
        quad_vel = client.getVelocity()
    
        client.moveByVelocityZ(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], -5, 5)
    
        # Get the state of the agent
        quad_pose = client.getPosition()
        quad_vel = client.getVelocity()
        collision_info = client.getCollisionInfo()
        
        # Compute rewawd
        reward = compute_reward(quad_pose, quad_vel, collision_info)

        # Check if the episode is done
        done = isDone(reward)
        print('Episode {} Action {}  Reward {}  Done: {} '.format(episode, action, reward, done))

        r.append(reward)     

        # If episode is done then break
        if done:
            print ('The episode {0} is done.'.format(episode))
            break

        responses = client.simGetImages([ImageRequest(1, AirSimImageType.DepthPerspective, True, False)])
        current_state = transform_input(responses)

    
    # Log episode length and average reward for every episode
    episode_length.append(len(r))    
    avg_reward.append(sum(r)/len(r))

    plt.figure()
    plt.plot(r)
    plt.ylabel('Reward')
    plt.xlabel('Steps')
    plt.title('Reward vs Episode Steps')
    plt.savefig('savedRewards/EP' + str(episode))
    plt.close()

    # Save avg_reward vs episodes
    plt.figure()
    plt.plot(avg_reward)
    plt.ylabel('Average Reward')
    plt.xlabel('Episodes')
    plt.title('Average Reward vs Episodes')
    plt.savefig('savedRewards/AvgRewardVsEp')
    plt.close()

print('******************************END***********************************')
plt.subplot(1,2,1)
plt.plot(episode_length)
plt.subplot(1,2,2)
plt.plot(avg_reward)
plt.show()
