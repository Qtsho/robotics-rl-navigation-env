#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:25:45 2021

@author: workstation
"""
import naive_dqn as ndqn
import numpy as np
import torch as T
import gym
import utils



#Agent_naiev class not inlcude the Q network (OOP common sense)
class Agent_naive():
    def __init__ (self, input_dims, n_actions, lr, gamma = 0.99,
                  epsilon= 1.0, eps_de=1e-5, eps_min= 0.01):
        
        #save at member variables for Agent class
        self.lr= lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_de = eps_de
        self.eps_min = eps_min
        
        #action space (only number for easy): a list
        self.action_space = [i for i in range(self.n_actions)]
        
              
        # Q value for agent (will be the deepQnetwork)    
        self.Q = ndqn.LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
        
         
            
        
    #using epsilon greedy to choice action for exploration purpose    
    def choose_action (self, observation):
    
        #take greedy action if (>epsilon) else take random action
        if np.random.random() > self.epsilon:
            #make sure as observation is pytorch tensor, Q.device variable 
            #store in self.Q object
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
            
        return action
    
    #decrement epsilon overtime.
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_de \
                        if self.epsilon > self.eps_min else self.eps_min
                            

    def learn (self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        #converge nm array to pytorch cuda tensor
        states = T.tensor(state, dtype= T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype= T.float).to(self.Q.device)
        
        
        # the feedfoward to calcutlate the update equation for our Q estimate
        q_prediction = self.Q.forward(states)[actions] #we want the delta between
        
        #similar to Q-update from Q-learning but use lose in NN 
        q_next = self.Q.forward(states_).max() # prediction
        
        q_target = rewards + self.gamma*q_next
        loss = self.Q.loss(q_target, q_prediction).to(self.Q.device) # is the TD error
        
        #backward propergation to update the layers' weights
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()
        
 
#main function of the programe.
if __name__ == '__main__':
    '''make the CarPole env
    actions: 
    
    '''
    env = gym.make ('CartPole-v1')
    n_games = 10000 #10k game
    scores = []
    eps_history = []
    
    agent = Agent_naive(lr= 0.0001,input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n)
    
    #running for 10000 games
    for i in range (n_games):
        '''reset'''
        score = 0 
        done = False
        obs = env.reset()
        #if not done, in one game (hear is falling cartpole)
        while not done:
            env.render()
            action = agent.choose_action(obs) #choose action base on current obs(state)
            obs_, reward, done, info = env.step(action) #get nextstate,reward, doneflag, debug info from gym
            score += reward #add cumulative reward
            agent.learn(obs, action, reward, obs_) #learn from sars' tuple
            obs = obs_ # set curent state as next state
        
        scores.append(score) #append for plotting
        eps_history.append(agent.epsilon) # append for plotting
        
        if i% 100 == 0: #print avg score in 100 games on terminal
            avg_score = np.mean(scores[-100:])
            print ('episode', i, 'score %.1f avg score %.1f epsilon %.2f' % 
                   (score, avg_score,agent.epsilon))
            
    #save result under name        
    filename = 'cartpole_naive_dqn.png'
    x = [i+1 for i in range (n_games)] 
    utils.plot_learning_curve (x, scores, eps_history, filename)         
    