import gym
import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
NW=7
NE=1
SW=5
SE=3
RIGHT = 2
DOWN =4 
LEFT = 6
action_reps={0:' ^  ',1:' NE ',2:' >  ',3:' SE ',4:' v  ',5:' SW ',6:' <  ',7:' NW '}
class DroneGridworldEnvmt(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}
    
    def check_danger(self,coord):
        t=((coord[0]<0) or (coord[0]>=self.shape[0]) or (coord[1]<0) or (coord[1]>=self.shape[1]))
        return (t)
    
    def check_closecall(self,coord):
        t=((coord[0]==0) or (coord[0]==self.shape[0]-1) or (coord[1]==0) or (coord[1]==self.shape[1]-1))
        return (t)
        
    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = (np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]).astype(int)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == self.goalpt
        
        #Replace with RewardMap
        if new_state in self.sinks:
            return [(1.0, current, -100.0, True)]
        else:
            return [(1.0, new_state, -1.0, is_done)]
        
        
#        oc=self.check_danger(new_position)
#        if not oc:
#            new_state = np.ravel_multi_index(tuple(new_position), self.shape)
#            if (new_state in self.sinks) :
#                oc=True
#               
#        if oc:
#            is_done=True
#        else:
#            is_done = (tuple(new_position) == self.goalpt)
#        
#        #Replace with RewardMap
#        cc=self.check_closecall(new_position)
#        
#        if oc:
#            return [(1.0, current, -100.0,is_done)]     
#        elif cc:
#            return [(1.0, new_state, -2.0,is_done )]
#        else:
#            return [(1.0, new_state, -1.0, is_done)]


    def __init__(self,shape,startpt,goalpt,winds):
        
        assert ((startpt[0]<shape[0] and startpt[1]<shape[1]) and (goalpt[0]<shape[0] and goalpt[1]<shape[1]))

        self.sinks=[]
        self.shape = shape
        self.startpt=startpt
        self.goalpt=goalpt

        nS = np.prod(self.shape)
        nA = 8
        
        self.winds = winds

        #winds[:,[3,4,5,8]] = 1
        #winds[:,[6,7]] = 2

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][NE] = self._calculate_transition_prob(position, [-1, 1], winds)
            P[s][SE] = self._calculate_transition_prob(position, [1, 1], winds)
            P[s][SW] = self._calculate_transition_prob(position, [1, -1], winds)
            P[s][NW] = self._calculate_transition_prob(position, [-1, -1], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index(startpt, self.shape)] = 1.0
        super(DroneGridworldEnvmt, self).__init__(nS, nA, P, isd)


    def _inject(self,obstacle):
        for i in obstacle:
            self.sinks.append(i)
    
    def _renderLegacy(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " ' "
            elif position in self.sinks:
                output = " X "
            elif position == self.goalpt:
                output = " G "
            else:
                output = " 0 "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")
        
    def _render(self,colorTrace=[], mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if position == self.startpt:
                output = " O "
            elif position in self.sinks:
                output = " X "
            elif position in colorTrace:
                output=" T "
            elif position == self.goalpt:
                output = " G "
            else:
                output=" _ "
            

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")
        
    def _renderpolicy(self,Q, close=False, mode='human'):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if position == self.startpt:
                output = " O  "
            elif position in self.sinks:
                output = " X  "
            elif position == self.goalpt:
                output = " G  "
            else:
                output = action_reps[np.argmax(Q[s])]

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")