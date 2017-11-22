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
        self.stop=stop(client)
        
class Controller:
    def __init__(self,client):
        self.client=client
        
    def act():
        return
    
class Go2Goal(Controller):
    def __init__(self,client):
        Controller.__init__(client)
    
    def act(targetpt,vel=5):
        client.moveToPosition(targetpt[0], targetpt[1], targetpt[2], vel)

class tright(Controller):
     def __init__(self,client):
        Controller.__init__(client)
        
    def act():  
        
class tleft(Controller):
     def __init__(self,client):
        Controller.__init__(client)
        
    def act():  
        
class stop(Controller):
     def __init__(self,client):
        Controller.__init__(client)
        
    def act():
        client.hover()
        
