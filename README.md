# Welcome to Obstacle Avoidance using DDQN

## Background

One of the challenges faced by Autonomous Aerial Vehicles is reliable navigation through urban environments. Factors like reduction in precision of Global Positioning System (GPS), narrow spaces and dynamically moving obstacles make the path planning of an aerial robot a complicated task. One of the skills required for the agent to effectively navigate through such an environment is to develop an ability to avoid collisions using information from onboard depth sensors. 
However, there lies a perceivable gap for robust and collision free autonomous quadcopter navigation in urban environments. One of the key limiting factors is the imprecise performance GPS localization systems when surrounded by high rise structures such as tall buildings. In such a hostile environment, obstacle avoidance becomes a necessary skill for such an agent.

## How did we do it
We trained a Quadcopter Agent using Airsim ina custom Unity based training arena to avoid obstacles such as tall buildings, civilian structures like traffic poles, posts while navigating from a start position to the goal position using solely a monocular camera. 

## How to set things up
This code is an add on to the Airsim source code.
* Clone <a href="https://github.com/Microsoft/AirSim">Airsim</a> repo
* Build from source by following the instructions in 'How to Get It' section.
* Now clone(or download and copy) this repo in the (your cloned path)/PythonClient/
* Copy merge <a href="https://www.dropbox.com/s/modl4yevcjcnzzf/Unreal.rar?dl=0">this arena</a> to (your cloned path)/Unreal

## How to train
* Start Blocks as indicated <a href="https://github.com/Microsoft/AirSim/blob/master/docs/unreal_blocks.md">here</a> repo
* Run better_drl_dqn3.py

[[Demo Video]](https://www.youtube.com/watch?v=WZe6jF1GAxk)
