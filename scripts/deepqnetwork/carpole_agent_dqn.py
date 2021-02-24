#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:48:56 2021

@author: workstation
"""

import numpy as np
import torch as T
import gym
import utils
import dql

class Agent():
    
    def __init__ (self, input_dims, n_actions, lr, batch_size, gamma = 0.99,
                  epsilon= 1.0, eps_de=1e-5, eps_min= 0.01,
                  mem_size = 1000000):
        #action space (only number for easy): a list
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        
        
        self.lr= lr
        self.input_dims = input_dims
    
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_de = eps_de
        self.eps_min = eps_min
        
        self.batch_size = batch_size
        
        ' A memory for dqn of the Agent: Agent memmory.'
        self.mem_size = mem_size
        self.memory = dql.ReplayBuffer(mem_size, input_dims, self.batch_size) # creat a instance of class replaybuffer
        
        'Q eval network: is an object DQN'
        self.q_eval = dql.LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
   
    
    def choose_action (self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.forward(state)
            action = np.argmax(actions)
        return action
         
    
    #interface function (oop) for dqn.py        
    def store_trasition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    
    def learn (self):
        'start learning only if filling the batch_size'
        if self.memory.mem_cntr < self.memory.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()
        
        states, actions, rewards, states_ ,terminals, batch_index= self.memory.sample_buffer(self.memory.batch_size)
        
        #converge np array to pytorch cuda tensor and send to device
        states = T.tensor(states, dtype= T.float).to(self.q_eval.device)
        #actions = T.tensor(actions).to(self.q_eval.device) # dont need to be a tensor
        rewards = T.tensor(rewards).to(self.q_eval.device)
        states_ = T.tensor(states_, dtype= T.float).to(self.q_eval.device)
        terminals = T.tensor(terminals).to(self.q_eval.device)
          
        # the feedfoward to calcutlate the update equation for our Q estimate
        q_prediction = self.q_eval.forward(states)[batch_index, actions] #we want the delta between action the agent actually took and max action
        #similar to Q-update from Q-learning but use lose in NN 
        q_next = self.q_eval.forward(states_).max()
        q_next [terminals] = 0
        
        # clone network
        q_target = np.copy(q_prediction)
        
        q_target = rewards + self.gamma*q_next
        
        'loss funtion' 
        loss = self.q_eval.loss(q_target, q_prediction).to(self.q_eval.device) # is the TD error
        #backward propergation to update the layers' weights -> learn
        loss.backward()
        self.Q.optimizer.step()
        
        self.epsilon = self.epsilon = self.epsilon - self.eps_de \
                        if self.epsilon > self.eps_min else self.eps_min
        
if __name__ == '__main__':
    'make trainnign env'
    
    env = gym.make('CartPole-v1')
    num_games = 10000 
    rewards = []
    episode_hist = []
    
    agent = Agent (lr = 0.003, input_dims = env.observation_space.shape,
                  n_actions = env.action_space.n, batch_size = 64,gamma=0.99)
    
    for i in range(num_games):
        game_score= 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_,reward,done,info = env.step (action)
            game_score += reward
            agent.store_trasition(observation, action, reward, observation_, done)
            agent.learn()
            observation_= observation
            
        rewards.append(game_score)
        episode_hist.append(agent.epsilon)
        
        if i% 100 == 0: #print avg score in 100 games on terminal
           avg_score = np.mean(rewards[-100:])
           print ('episode', i, 'score %.1f avg score %.1f epsilon %.2f' % 
                  (game_score, avg_score,agent.epsilon))
            
   #save result under name        
    filename = 'cartpole_dqn.png'
    x = [i+1 for i in range (num_games)] 
    utils.plot_learning_curve (x, rewards, episode_hist, filename)  
    
    