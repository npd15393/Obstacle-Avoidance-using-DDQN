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
    if action == 0:
        quad_yaw_offset = -10   # Yaw left 10 degrees
    elif action == 1:
        quad_yaw_offset = -5    # Yaw left 5 degrees
    elif action == 2:
        quad_yaw_offset = 0     # Do not change yaw
    elif action == 3:
        quad_yaw_offset = 5     # Yaw right 5 degrees
    elif action == 4:
        quad_yaw_offset = 10    # Yaw right 10 degrees
    
    return quad_yaw_offset

def compute_reward(quad_state, quad_vel, collision_info):
    thresh_dist = 500  #thershold distance for reward function
    beta = 0.05
    z = -10
    p2 = np.array(list((-50, 77.5))) #goal
    p1 = np.array(list((0,0))) #starting pt
    p3 = np.array(list((quad_state.x_val, quad_state.y_val))) #current_point  
    d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1) #perpendicular distance
    
    # if collision_info.has_collided:
    if (collision_info.position[b'x_val'] != 0) or (collision_info.position[b'y_val'] != 0) or (collision_info.position[b'z_val'] != 0):
        reward = -200  #reward for collision
        print ('The drone has collided')    

    else:    
        #get the perpendicular distance from the line
        perpendicular_reward = d
        # print ('The perpendicular distance obtained is {} and the reward is {}'.format(d,perpendicular_reward))

        #get the goal distance:
        goal_dist = norm(p3-p2)
        goal_reward = goal_dist*1.0 
        # print ('The goal distance obtained is {} and the reward is {}'.format(goal_dist,goal_reward))
        if goal_dist < 10 :
            reward  = 1000    
        else:
            reward = 100 - perpendicular_reward*1.0 - goal_dist
            # print ('reward : {} '.format(reward))
    return reward


def isDone(reward):
    done = 0
    if  reward <=  -10 : #collide
        done = 1
        print ('The drone failed to reach the goal')
    if (reward  > 90):
        done = 1
        print ('The drone reached nice.')     
    return done


def moveInHeading(client, velocity = 1, heading = 0, altitude = -5, duration = 1):
    # Move the Quad in 'heading' direction pointing forwards with velocity
    Vx = math.cos(math.radians(heading)) * velocity
    Vy = math.sin(math.radians(heading)) * velocity
    (pitch, roll, yaw) = client.getPitchRollYaw()
    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)
    yaw_err = heading - yaw
    # print('Roll: {0} Pitch: {1} Yaw: {2}'.format(roll, pitch, yaw))
    client.moveByVelocityZ(Vx, Vy, altitude, duration, DrivetrainType.MaxDegreeOfFreedom, YawMode(False, yaw_err/pi)) # Move 5,5,5 meter in heading mode
    (pitch, roll, yaw) = client.getPitchRollYaw()
    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)
    yaw_err = heading - yaw
    # print('Roll: {0} Pitch: {1} Yaw: {2}'.format(roll, pitch, yaw))

# The Execution of the program starts from here

# Connect to the AirSim simulator 
client = MultirotorClient()
client.confirmConnection()
print ('Connection with AirSim established')

quad_pose = client.getPosition();
print('QUAD_POS X: {0} Y: {1} Z: {2}'.format(quad_pose.x_val, quad_pose.y_val, quad_pose.z_val))

# Initialize API control
#client.enableApiControl(True)

# Arm the quadcopter
#client.armDisarm(True)

# Takeoff
#client.takeoff()
#print ('Quad takeoff')
#time.sleep(0.5) # sleep till the quad completes takeoff
#client.hover()
#print ('Quad hover')

# Training variables
n_episodes = 1000
eps_save_model = 1

# RL Agent Settings
NumBufferFrames = 4
SizeRows = 84
SizeCols = 84
NumActions = 5

plt_save_after = 100
episode_length = []
avg_reward = []
# Make a RL agent
agent = DeepQAgent()

V = 5   # Velocity of the Quad

pi = 3.14

quad_yaw_offset = 0

#Training Loop
for episode in range(n_episodes):
    print ('Starting Episode :', episode)
    client.reset()  # Reset the environment
    client.enableApiControl(True)   # Request API control to AirSim 

    # Takeoff
    client.takeoff()
    print ('Quad takeoff')
    
    client.moveToPosition(0, 0, -5, 5) # Move 5 meter above home
    client.rotateToYaw(90)
    
    
    # Reset loop variables
    i = 1
    r = []
    # Reset yaw angle
    yaw = 90;
    quad_yaw_offset = 0;

    # Get the initial image from the camera
    responses = client.simGetImages([ImageRequest(1, AirSimImageType.DepthPerspective, True, False)])
    current_state = transform_input(responses)
    
    for steps in range(1000):
        
        if steps == (999):
            print('Max Steps reached')
        
        
        action = agent.act(current_state)

        # Get yaw offset based upon current image
        quad_yaw_offset = interpret_action(action)

        # Constrain yaw between +- 180 degrees
        yaw = yaw + quad_yaw_offset
        if (yaw > 180):
            yaw = yaw - 360

        if (yaw < -180):
            yaw = yaw + 360

        moveInHeading(client, V, yaw, -5, 5)


        # Get the state of the agent
        quad_pose = client.getPosition()
        quad_vel = client.getVelocity()
        collision_info = client.getCollisionInfo()
        
        # Compute rewawd
        reward  = compute_reward(quad_pose, quad_vel, collision_info, quad_yaw_offset)

        # Check if the episode is done
        done = isDone(reward)
        print('Episode {} Action {}  Reward {}  Done: {} '.format(episode, action, reward, done))

        r.append(reward)

        # If episode is done then break
        if done:
            print ('The episode {0} is done.'.format(episode))
            break

        # Get the image and gaet the next state
        responses = client.simGetImages([ImageRequest(1, AirSimImageType.DepthPerspective, True, False)])
        current_state = transform_input(responses)


    # Log episode length and average reward for every episode
    episode_length.append(len(r))    
    avg_reward.append(sum(r)/len(r))

    # Save model after every eps_save_model episodes
    if (episode % eps_save_model) == 0:
        agent._save_models()

        # Save reward plot for the episode
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

        # Save Episode Length vs episodes
        plt.figure()
        plt.plot(episode_length)
        plt.ylabel('Episode Length')
        plt.xlabel('Episodes')
        plt.title('Episode Length vs Episodes')
        plt.savefig('savedRewards/EpiLengthVsEp')
        plt.close()

print('******************************END***********************************')
print ('The Training has been completed for {} episodes with model saved '.format(n_episodes))
print ('Stay Calm and Test the model')

plt.subplot(1,2,1)
plt.plot(episode_length)
plt.subplot(1,2,2)
plt.plot(avg_reward)
plt.show()