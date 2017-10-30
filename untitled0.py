import gym
import numpy as np
import sys
import random as rand
import matplotlib

if "../" not in sys.path:
  sys.path.append("../") 

from learner import Learner
from lib.envs.DroneWorldEnv import DroneGridworldEnvmt
#from lib.envs.windy_gridworld import WindyGridworldEnv
from lib import plotting
matplotlib.style.use('ggplot')

env = DroneGridworldEnvmt((10,10),(6,9),(1,1),np.zeros([10,10]))
#env=WindyGridworldEnv()

env._inject([(5,5),(8,7),(16,13),(2,16)])

alpha=0.1
epsi=0.2
discount=0.9

ai=Learner(alpha,epsi,discount)
q,stats=ai.Q_learn(env,5000)
plotting.plot_episode_stats(stats)
#env._renderLegacy()
env._renderpolicy(q,False,'human')
q,stats=ai.SARSA(env,5000)
plotting.plot_episode_stats(stats)
#trace=[]
#cs=env._reset()
#done=False
#while not done:
#   act=np.argmax(q[cs])
#   cs, _, done, _=env.step(act)
#   print(done)
#   trace.append(cs)
#env._render(trace)
#q,stats=Q_learn(env,200)
#plotting.plot_episode_stats(stats)