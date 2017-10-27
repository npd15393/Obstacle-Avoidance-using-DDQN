import gym
import numpy as np
import sys
import random as rand
import matplotlib
import itertools

if "../" not in sys.path:
  sys.path.append("../") 

from lib.envs.DroneWorldEnv import DroneGridworldEnvmt
#from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting
from collections import defaultdict

matplotlib.style.use('ggplot')

env = DroneGridworldEnvmt((20,20),(16,12),(1,16),np.zeros([20,20]))
#env=WindyGridworldEnv()

#env._inject([(5,5),(8,7),(16,13),(2,16)])

alpha=0.1
epsi=0.1
discount=1

def running_policy(Q,pos):
    A = np.ones(8, dtype=float) * epsi / 8
    best_action = np.argmax(Q[pos])
    A[best_action] += (1.0 - epsi)
    return A

def greedy_policy(Q,pos):
    return np.argmax(Q[pos])

def SARSA(envr,nE):
    Q= defaultdict(lambda: np.zeros(env.action_space.n))
    
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(nE),
        episode_rewards=np.zeros(nE))
    
    for i in range(nE):  
        current_pos=env.reset()
        act_prob=running_policy(Q,current_pos)
        act = np.random.choice(np.arange(len(act_prob)), p=act_prob)

        for t in itertools.count():        
            next_state, reward, done, _= envr.step(act)

            stats.episode_rewards[i] += reward
            stats.episode_lengths[i] = t

            next_act_prob=running_policy(Q, current_pos)
            next_act = np.random.choice(np.arange(len(next_act_prob)), p=next_act_prob)
            
            TD_err=reward+discount*Q[next_state][next_act]-Q[current_pos][act]
            Q[current_pos][act]+=alpha*(TD_err)
            
            if done:
                break
                env._render()

            current_pos=next_state
            act=next_act
    return Q,stats

def Q_learn(envr,nE):
    Q= defaultdict(lambda: np.zeros(env.action_space.n))
    
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(nE),
        episode_rewards=np.zeros(nE))
    
    for i in range(nE):  
        
        current_pos=env.reset()

        act_prob=running_policy(Q,current_pos)
        act = np.random.choice(np.arange(len(act_prob)), p=act_prob)

        for t in itertools.count():        
            next_state, reward, done, _= envr.step(act)

            stats.episode_rewards[i] += reward
            stats.episode_lengths[i] = t

            next_act_prob=running_policy(Q, current_pos)
            next_act = np.random.choice(np.arange(len(next_act_prob)), p=next_act_prob)
            up_act = greedy_policy(Q,current_pos)
            
            TD_err=reward+discount*Q[next_state][up_act]-Q[current_pos][act]
            Q[current_pos][act]+=alpha*(TD_err)

            if done:
                break
                env._render()

            current_pos=next_state
            act=next_act
    return Q,stats

q,stats=SARSA(env,200)
plotting.plot_episode_stats(stats)
env._renderLegacy()
#env._renderpolicy(q,False,'human')
trace=[]
cs=env._reset()
#done=False
#while not done:
#    act=np.argmax(q[cs])
#    cs, _, done, _=env.step(act)
#    trace.append(cs)
#env._render(trace)
#q,stats=Q_learn(env,200)
#plotting.plot_episode_stats(stats)