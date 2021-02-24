#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:22:33 2021

@author: workstation
"""

# a simple NN built with pytorch that has one 128 neurons hidden layers 

import torch.nn as nn #import module toruch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T


class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        
        #run superclass tf.Module constructor
        super(LinearDeepQNetwork, self).__init__() 
        #Class Atributes
        self.fc1 =nn.Linear(*input_dims, 128) # layer 1 input 
        self.fc2 = nn.Linear(128, n_actions) # layer 2 output the action values
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss() #minimum square error loss function
        
        #select device  if have gpu
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        #send the network to the device
        self.to(self.device)
        
        
    # our feedforward function take input as current state of env 
    def forward(self, state):
        #pass the state to the first layer with relu activation function
        #pass that quantity fo 2nd fully connected layer without activate it
        
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)
        
        return actions
    
