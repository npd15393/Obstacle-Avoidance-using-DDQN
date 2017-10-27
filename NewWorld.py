# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:55:20 2017

@author: Nishant
"""

class NewWorld:
    RewardMap={}
    def __init__(self,Map,greyScaleThreshold=0.5):
        RewardMap=np.zeros([len(Map),len(Map[0])])
        for i in range(len(Map)):
            for j in range(len(Map[0]))
                if Map[i][j]<greyScaleThreshold      
                    RewardMap[(i,j)]=-1
                else:
                    RewardMap[(i,j)]=-200
    