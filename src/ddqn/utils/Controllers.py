# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 21:14:01 2017

@author: Nishant
"""

#Controllers
class controllers:
    def __init__(self,client):
        self.Go2Goal=Go2Goal(client)
        self.turn_right=tright(client)
        self.turn_left=tleft(client)
        self.thr_right=trightthrottle(client)
        self.thr_left=tleftthrottle(client)
        self.stop=stop(client)
        
#class Controller(object):
#    def __init__(self,client):
#        self.client=client
#        
#    def act():
#        return
#    
class Go2Goal:#(Controller):
    def __init__(self,client):
        self.client=client
    
    def act(self,targetpt,vel=5):
        self.client.moveToPosition(targetpt[0], targetpt[1], targetpt[2], vel)

class tright:#(Controller):
    def __init__(self,client):
        self.client=client
        
    def act(self):  
        return
    
class tleft:#(Controller):
    def __init__(self,client):
        self.client=client
        
    def act(self): 
        return
        
class trightthrottle:#(Controller):
    def __init__(self,client):
       self.client=client
        
    def act(self):
        return
        
class tleftthrottle:#(Controller):
    def __init__(self,client):
       self.client=client
        
    def act(self):  
        return
        
class stop:#(Controller):
    def __init__(self,client):
       self.client=client
        
    def act(self):
        self.client.hover()
        
