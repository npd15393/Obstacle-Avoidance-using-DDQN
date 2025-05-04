# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:50:41 2017

@author: Nishant
"""
class QLearner:
    
    def __init__(self,RewardMap,stepsize=0.1,GAMMA=0.9):
        self.rewards=RewardMap
        self.stepsize=stepsize
        self.GAMMA=GAMMA
        
    def chooseAction(state,Qvals):
        
    def qLearn(env,stateActionValues, stepSize=ALPHA,startState, goalState):
        stateActionValues=np.zeros([env.nS,env.nA])
        #state in this case is {heading angle,velocity}
        for i in range(100):
            currentState = startState
            #every action is {roll,pitch,throttle,yaw,time}
            currentAction = chooseAction(currentState, stateActionValues)
            rewards = 0.0
            while currentState != goalState:
                #env can be uncertain, turning policy is learnt based on env physics.
                newState,reward,done = env.step(currentAction)
                rewards +=reward 
                #newState = [currentState[0]][currentState[1]][currentAction]
                acts=[(key,val) for key,val in stateActionValues.items() if key[0]=newState]
                act_max=max(acts,key=lambda item:item[1])
                # Q-Learning update
                stateActionValues[(currentState,currentAction)] += self.stepSize * (reward + self.GAMMA * act_max[1] - stateActionValues[(currentState, currentAction)])
                if done:
                    break
                currentState = newState
                currentAction=act_max[0]
        return stateActionValues,rewards
