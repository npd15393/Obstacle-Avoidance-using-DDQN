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
from cntk.cntk_py import combine
import csv
from PIL import Image
import pprint
import matplotlib.pyplot as plt
from numpy.linalg import norm
import CreateModel

class ReplayMemory(object):
    """
    ReplayMemory keeps track of the environment dynamic.
    We store all the transitions (s(t), action, s(t+1), reward, done).
    The replay memory allows us to efficiently sample minibatches from it, and generate the correct state representation
    (w.r.t the number of previous frames needed).
    """
    def __init__(self, size, sample_shape, history_length=4):
        self._pos = 0
        self._count = 0
        self._max_size = size
        self._history_length = max(1, history_length)
        self._state_shape = sample_shape
        self._states = np.zeros((size,) + sample_shape, dtype=np.float32)
        self._actions = np.zeros(size, dtype=np.uint8)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._terminals = np.zeros(size, dtype=np.float32)

    def __len__(self):
        """ Returns the number of items currently present in the memory
        Returns: Int >= 0
        """
        return self._count

    def append(self, state, action, reward, done):
        """ Appends the specified transition to the memory.

        Attributes:
            state (Tensor[sample_shape]): The state to append
            action (int): An integer representing the action done
            reward (float): An integer representing the reward received for doing this action
            done (bool): A boolean specifying if this state is a terminal (episode has finished)
        """
        assert state.shape == self._state_shape, \
            'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = done

        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def sample(self, size):
        """ Generate size random integers mapping indices in the memory.
            The returned indices can be retrieved using #get_state().
            See the method #minibatch() if you want to retrieve samples directly.

        Attributes:
            size (int): The minibatch size

        Returns:
             Indexes of the sampled states ([int])
        """

        # Local variable access is faster in loops
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []

        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            if index not in indexes:

                # if not wrapping over current pointer,
                # then check if there is terminal state wrapped inside
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        return indexes

    def minibatch(self, size):
        """ Generate a minibatch with the number of samples specified by the size parameter.

        Attributes:
            size (int): Minibatch size

        Returns:
            tuple: Tensor[minibatch_size, input_shape...], [int], [float], [bool]
        """
        indexes = self.sample(size)

        pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        dones = self._terminals[indexes]

        return pre_states, actions, post_states, rewards, dones

    def get_state(self, index):
        """
        Return the specified state with the replay memory. A state consists of
        the last `history_length` perceptions.

        Attributes:
            index (int): State's index

        Returns:
            State at specified index (Tensor[history_length, input_shape...])
        """
        if self._count == 0:
            raise IndexError('Empty Memory')

        index %= self._count
        history_length = self._history_length

        # If index > history_length, take from a slice
        if index >= history_length:
            return self._states[(index - (history_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)

class History(object):
    """
    Accumulator keeping track of the N previous frames to be used by the agent
    for evaluation
    """

    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        """ Underlying buffer with N previous states stacked along first axis

        Returns:
            Tensor[shape]
        """
        return self._buffer

    def append(self, state):
        """ Append state to the history

        Attributes:
            state (Tensor) : The state to append to the memory
        """
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = state

    def reset(self):
        """ Reset the memory. Underlying buffer set all indexes to 0

        """
        self._buffer.fill(0)

class LinearEpsilonAnnealingExplorer(object):
    """
    Exploration policy using Linear Epsilon Greedy

    Attributes:
        start (float): start value
        end (float): end value
        steps (int): number of steps between start and end
    """

    def __init__(self, start, end, steps):
        self._start = start
        self._stop = end
        self._steps = steps

        self._step_size = (end - start) / steps

    def __call__(self, num_actions):
        """
        Select a random action out of `num_actions` possibilities.

        Attributes:
            num_actions (int): Number of actions available
        """
        return np.random.choice(num_actions)

    def _epsilon(self, step):
        """ Compute the epsilon parameter according to the specified step

        Attributes:
            step (int)
        """
        if step < 0:
            return self._start
        elif step > self._steps:
            return self._stop
        else:
            return self._step_size * step + self._start

    def is_exploring(self, step):
        """ Commodity method indicating if the agent should explore

        Attributes:
            step (int) : Current step

        Returns:
             bool : True if exploring, False otherwise
        """
        e=self._epsilon(step)
#        print('Epsilon = '+str(e))
        return np.random.rand() < e

def huber_loss(y, y_hat, delta):
    """ Compute the Huber Loss as part of the model graph

    Huber Loss is more robust to outliers. It is defined as:
     if |y - y_hat| < delta :
        0.5 * (y - y_hat)**2
    else :
        delta * |y - y_hat| - 0.5 * delta**2

    Attributes:
        y (Tensor[-1, 1]): Target value
        y_hat(Tensor[-1, 1]): Estimated value
        delta (float): Outliers threshold

    Returns:
        CNTK Graph Node
    """
    half_delta_squared = 0.5 * delta * delta
    error = y - y_hat
    abs_error = abs(error)

    less_than = 0.5 * square(error)
    more_than = (delta * abs_error) - half_delta_squared
    loss_per_sample = element_select(less(abs_error, delta), less_than, more_than)

    return reduce_sum(loss_per_sample, name='loss')

class DeepQAgent(object):
    """
    Implementation of Deep Q Neural Network agent like in:
        Nature 518. "Human-level control through deep reinforcement learning" (Mnih & al. 2015)
    """
    def __init__(self, input_shape, nb_actions,
                 gamma=0.99, explorer=LinearEpsilonAnnealingExplorer(0.5, 0.1, 400000),
                 learning_rate=0.001, momentum=0.95, minibatch_size=32,
                 memory_size=50000, train_after= 500, train_interval=500, target_update_interval=1000,
                 monitor=True): 
        self.input_shape = input_shape
        self.nb_actions = nb_actions
        self.gamma = gamma

        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_interval = target_update_interval

        self._explorer = explorer
        self._minibatch_size = minibatch_size
        self._history = History(input_shape)
        self._memory = ReplayMemory(memory_size, input_shape[1:], 4)
        self._num_actions_taken = 0

        # Metrics accumulator
        self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

        # Action Value model (used by agent to interact with the environment)
        modelFactory=CreateModel()
        self._action_value_net, self._target_net =  modelFactory.createDDQN(input_shape,nb_actions)
        #load_model("trainedModels/Action_vale_net")

        # Target model used to compute the target Q-values in training, updated
        # less frequently for increased stability.
        #self._target_net = load_model("trainedModels/Target_net")

        # Function computing Q-values targets as part of the computation graph
#        @Function
#        @Signature(post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
#        def compute_q_targets(post_states, rewards, terminals):
#            return element_select(
#                terminals,
#                rewards,
#                gamma * reduce_max(self._target_net(post_states), axis=0) + rewards,
#            )

    def target_qvals(self,us,model,targetmodel):
            X=[]
            targets=[]
            GAMMA=0.995
            for u in range(len(us[0])):
                                X.append(us[0][u])
                                
                                q1=targetmodel.predict(np.reshape(us[2][u],(1,self.nb_actions)))[0]
                                target= model.predict(np.reshape(us[0][u],(1,self.nb_actions)))[0]
                        
                                if us[4][u]:
                                    target[us[1][u]]=us[3][u]
                                else:
                                    target[us[1][u]]=us[3][u]+GAMMA*np.amax(q1)
                                    
                                targets.append(target)
            return X,targets

    def act(self, state):
        """ This allows the agent to select the next action to perform in regard of the current state of the environment.
        It follows the terminology used in the Nature paper.

        Attributes:
            state (Tensor[input_shape]): The current environment state

        Returns: Int >= 0 : Next action to do
        """
        # Append the state to the short term memory (ie. History)
        self._history.append(state)

        # If policy requires agent to explore, sample random action
        if self._explorer.is_exploring(self._num_actions_taken):
            action = self._explorer(self.nb_actions)
        else:
            # Use the network to output the best action
            env_with_history = self._history.value
            q_values = self._action_value_net.predict(
                # Append batch axis with only one sample to evaluate
                env_with_history.reshape((1,) + env_with_history.shape)
            )[0]

            self._episode_q_means.append(np.mean(q_values))
            self._episode_q_stddev.append(np.std(q_values))

            # Return the value maximizing the expected reward
            action = q_values.argmax()

        # Keep track of interval action counter
        self._num_actions_taken += 1
        return action

    def observe(self, old_state, action, reward, done):
        """ This allows the agent to observe the output of doing the action it selected through act() on the old_state

        Attributes:
            old_state (Tensor[input_shape]): Previous environment state
            action (int): Action done by the agent
            reward (float): Reward for doing this action in the old_state environment
            done (bool): Indicate if the action has terminated the environment
        """
        self._episode_rewards.append(reward)

        # If done, reset short term memory (ie. History)
        if done:
            # Plot the metrics through Tensorboard and reset buffers
            if self._metrics_writer is not None:
                self._plot_metrics()
            self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

            # Reset the short term memory
            self._history.reset()

        # Append to long term memory
        self._memory.append(old_state, action, reward, done)

    def train(self):
        """ This allows the agent to train itself to better understand the environment dynamics.
        The agent will compute the expected reward for the state(t+1)
        and update the expected reward at step t according to this.

        The target expectation is computed through the Target Network, which is a more stable version
        of the Action Value Network for increasing training stability.

        The Target Network is a frozen copy of the Action Value Network updated as regular intervals.
        """
        agent_step = self._num_actions_taken

        if agent_step >= self._train_after:
            if (agent_step % self._train_interval) == 0:
                pre_states, actions, post_states, rewards, terminals = self._memory.minibatch(self._minibatch_size)
                
                print ('Training the agent')
                x,t=self.target_qvals([pre_states,
                        Value.one_hot(actions.reshape(-1, 1).tolist(), self.nb_actions),
                        post_states,
                        rewards,
                        terminals],self._action_value_net,self._target_net)
                self._action_value_net.fit(np.reshape(x,(len(pre_states),self.input_shape)),np.reshape(t,(len(pre_states),self.nb_actions)),epochs=1,verbose=0)

                # Update the Target Network if needed
                if (agent_step % self._target_update_interval) == 0:
                    print ('Updating the target Network')
                    self._target_net = self._action_value_net.clone(CloneMethod.freeze)
                    filename = "models\model%d" % agent_step
                    self._trainer.save_checkpoint(filename)

    def _plot_metrics(self):
        """Plot current buffers accumulated values to visualize agent learning
        """
        if len(self._episode_q_means) > 0:
            mean_q = np.asscalar(np.mean(self._episode_q_means))
            self._metrics_writer.write_value('Mean Q per ep.', mean_q, self._num_actions_taken)

        if len(self._episode_q_stddev) > 0:
            std_q = np.asscalar(np.mean(self._episode_q_stddev))
            self._metrics_writer.write_value('Mean Std Q per ep.', std_q, self._num_actions_taken)

        self._metrics_writer.write_value('Sum rewards per ep.', sum(self._episode_rewards), self._num_actions_taken)

    def _save_models(self):
        self._target_net.save('trainedModels/Target_net')
        self._action_value_net.save('trainedModels/Action_vale_net')
        print ('Action value and Target model saved')
        
    def _load_models(self):
        self._target_net = load_model('trainedModels/Target_net')
        self._action_value_net = load_model('trainedModels/Action_vale_net')
        print ('Action value and Target model loaded')
        
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
        quad_yaw_offset = - 2 # Yaw left 10 degrees
    elif action == 1:
        quad_yaw_offset = 0    # Yaw left 5 degrees
    elif action == 2:
        quad_yaw_offset = 2     # Do not change yaw
#    elif action == 3:
#        quad_yaw_offset = 2.5     # Yaw right 5 degrees
#    elif action == 4:
#        quad_yaw_offset = 5   # Yaw right 10 degrees
    
    return quad_yaw_offset

def compute_reward(quad_state, quad_vel, collision_info,collisions=0):
    thresh_dist = 500  #thershold distance for reward function
    beta = 0.05
    z = -10
    p2 = np.array(list((150, 0))) #goal
    p1 = np.array(list((0,0))) #starting pt
    ys=[30,60,90,120]
    checks=[1,10,50,100]
    p3 = np.array(list((quad_state.x_val, quad_state.y_val))) #current_point  
    d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1) #perpendicular distance
    
    # if collision_info.has_collided:
    if ((collision_info.position[b'x_val'] != 0) or (collision_info.position[b'y_val'] != 0) or (collision_info.position[b'z_val'] != 0)) and collisions==0:
        reward = -200  #reward for collision
        print ('The drone has collided')    

    else:    
        #get the perpendicular distance from the line
        perpendicular_reward = d
        # print ('The perpendicular distance obtained is {} and the reward is {}'.format(d,perpendicular_reward))
        ch=0
        for i in range(len(ys)):    
            if quad_state.x_val<ys[i]+1 and quad_state.x_val>ys[i]-1:
                ch=checks[i]
                
        #get the goal distance:
        goal_dist = norm(p3-p2)
        goal_reward = goal_dist*1.0 
        # print ('The goal distance obtained is {} and the reward is {}'.format(goal_dist,goal_reward))
        if goal_dist < 10 :
            reward  = 500  
        else:
            reward = ch#- perpendicular_reward*1.0 - goal_dist
            # print ('reward : {} '.format(reward))
    return reward



def isDone(reward,collisions):
    done = 0
    if  reward<=-200 : #collide
        done = 1
#        print ('The drone failed to reach the goal')
    if (reward  > 490):
        done = 1
        print ('The drone reached nice.')     
    return done




def get_directional_velocity(Vel = 1 , theta = 0):
    #return the directional velocity 
    return (Vel*cos(theta),Vel*sin(theta))
    # quad_pose = client.getPosition();
    # quad_position = np.array(quad_pose.x_val, quad_pose.y_val )


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
n_episodes = 5000
eps_save_model = 5

# RL Agent Settings
NumBufferFrames = 4
SizeRows = 84
SizeCols = 84
NumActions = 3
collisions=0
plt_save_after = 100
episode_length = []
avg_reward = []
# Make a RL agent
agent = DeepQAgent((NumBufferFrames, SizeRows, SizeCols), NumActions, monitor=True)

V = 3   # Velocity o fhte Quad
#client.enableApiControl(True)

log = open("savedLogs/Quad_log.txt", "w")
log.write("Episode, Average_Reward, Episode_Length,\n")

pose_log = open("savedLogs/Quad_pose_log.txt", "w")
pose_log.write("Episode, Roll, Pitch, Yaw,\n")

#Training Loop
for episode in range(n_episodes):
    print ('Starting Episode :', episode)

    max_steps = int(200*episode) + 10000   
    client.reset()  # Reset the environment
    client.enableApiControl(True)   # Request API control to AirSim 
    #print('reset')

    # Takeoff
    client.takeoff()
    print ('Quad takeoff')
    
    client.moveToPosition(client.getPosition().x_val, client.getPosition().y_val, -5, 5) # Move 5 meter above home
    # client.rotateToYaw(90)    
    # client.moveByVelocityZ(5, 5, -5, 60, DrivetrainType.ForwardOnly, YawMode(False, 0)) # Move 5,5,5 meter in heading mode
    
    # Reset loop variables
    i = 1
    r = []
    # Reset yaw angle
    yaw = 90;
    done=0
    # Get the initial image from the camera
    responses = client.simGetImages([ImageRequest(1, AirSimImageType.DepthPerspective, True, False)])
    current_state = transform_input(responses)
    
#    for steps in range(max_steps):
    while done==0:   
#        if steps == (max_steps -1):
#            print('Max Steps reached')
#        
        # Testing space
        # client.moveToPosition(0, 0, -5, 5) # Move 5 meter above home
        # quad_pose = client.getPosition();
        # print('QUAD_POS X: {0} Y: {1} Z: {2}'.format(quad_pose.x_val, quad_pose.y_val, quad_pose.z_val))
        
        action = agent.act(current_state)

        # Get yaw offset based upon current image
        quad_yaw_offset = interpret_action(action)

        # Constrain yaw between +- 180 degrees
        yaw = yaw + quad_yaw_offset
        if (yaw > 180):
            yaw = yaw - 360

        if (yaw < -180):
            yaw = yaw + 360

        # Calculate Vx and Vy components of velocity V
        Vx = V#*(math.cos(math.pi/8*quad_yaw_offset))#math.radians(yaw)))
        Vy = V*(math.sin(math.pi/4*quad_yaw_offset))#math.radians(yaw)))

#        print('Yaw: {0} Vx: {1} Vy: {2}'.format(yaw, Vx, Vy))

        client.moveByVelocityZ(Vx, Vy,-5, 1 )#, DrivetrainType.MaxDegreeOfFreedom, YawMode(False, 0))
#        if action==1:
#            client.rotateToYaw(0)
#            client.
        # Testing move Commands
        # client.moveByVelocity(0.0, +5.0, -2, 20)
        # client.moveByVelocityZ(-1, 5, -5, 1)
        # client.moveByVelocity(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], 1)
        # client.moveByVelocityZ(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], -5, 5)
        # client.moveToPosition(quad_pose.x_val+quad_offset[0], quad_pose.y_val+quad_offset[1], -5, 5, 1, DrivetrainType.ForwardOnly, YawMode(False, 0)) # Move in heading mode
        # client.moveToPosition(quad_pose.x_val+quad_offset[0], quad_pose.y_val+quad_offset[1], -5, 5) # Move in heading mode
        # time.sleep(0.1) 
        # quad_pose = client.getPosition();
        # print('QUAD_POS X: {0} Y: {1} Z: {2}'.format(quad_pose.x_val, quad_pose.y_val, quad_pose.z_val))        

        # Get the state of the agent
        quad_pose = client.getPosition()
        quad_vel = client.getVelocity()
        collision_info = client.getCollisionInfo()
        
        (pitchd, rolld, yawd) = client.getPitchRollYaw()
        rolld = math.degrees(rolld)
        pitchd = math.degrees(pitchd)
        yawd = math.degrees(yawd)    
        # print('Roll: {0} Pitch: {1} Yaw: {2}'.format(roll, pitch, yaw))
        pose_log.write("{0}, {1}, {2}, {3},\n".format(episode, rolld, pitchd, yawd))
        
        
        # Compute rewawd
        reward  = compute_reward(quad_pose, quad_vel, collision_info,collisions)
#        if collision_info.has_collided:
#            collisions+=1
#            client.moveByVelocityZ(-Vx, -Vy,-5, 1)
#            time.sleep(0.5)

            
        pitch,roll,yaw=client.getPitchRollYaw()
        client.rotateToYaw(-yaw*180/math.pi)
            
            
        
        # Check if the episode is done
        done = isDone(reward,collisions)
        #print('Episode {} Action {}  Reward {}  Done: {} '.format(episode, action, reward, done))

        if done==1:
            collisions=0
            
        r.append(reward)
        agent.observe(current_state, action, reward, done)
        agent.train()

        # If episode is done then break
        if done:
            print ('The episode {0} is done.'.format(episode))
            break

        # Get the image and gaet the next state
        responses = client.simGetImages([ImageRequest(1, AirSimImageType.DepthPerspective, True, False)])
        current_state = transform_input(responses)
        # plt.imshow(cropped_state,cmap = 'gray')

        # plt.figure()
        # plt.plot(r)
        # plt.ylabel('Reward')
        # plt.xlabel('Episode Length')
        # plt.show()
        # plt.close()

    print ('Will train after {}'.format(( 500 - agent._num_actions_taken ) % 500))
    
    # plt.plot(r)
    # plt.show()

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
        
        
        # Save avg_reward vs episodes
        plt.figure()
        plt.plot(episode_length)
        plt.ylabel('Episode Length')
        plt.xlabel('Episodes')
        plt.title('Episode Length vs Episodes')
        plt.savefig('savedRewards/EpLengthVsEp')
        plt.close()

    log.write("{0}, {1}, {2},\n".format(episode, sum(r)/len(r), len(r)))

print('******************************END***********************************')
print ('The Training has been completed for {} episodes with model saved '.format(n_episodes))
print ('Stay Calm and Test the model')

plt.subplot(1,2,1)
plt.plot(episode_length)
plt.subplot(1,2,2)
plt.plot(avg_reward)
plt.show()
