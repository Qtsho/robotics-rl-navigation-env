import numpy as np
import torch as T
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
class ReplayBuffer():
    def __init__(self, max_size, input_dims, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0 #keep track position of position of first unsave mem
        self.batch_size = batch_size
        #mem for states unpack input dims as a list *
        self.state_memory  = np.zeros((self.mem_size, *input_dims),
                                      dtype = np.float32)
        #mem for state trasition
        self.new_state_memory = np.zeros ((self.mem_size, *input_dims),
                                          dtype =np.float32)
        #mem for action   
        self.action_memory = np.zeros (self.mem_size, dtype =np.int32)
        #reward memory
        
        self.reward_memory = np.zeros (self.mem_size, dtype =np.float32)
        #mem for terminal
        self.terminal_memory = np.zeros (self.mem_size, dtype =np.bool)
    
    #add the trasition to the memory buffer
    def store_transition (self, state, action, reward, new_state , done):
        
        #check the unocupied mem position
        index = self.mem_cntr % self.mem_size
        
        #update the replay buffer
        self.state_memory[index]= state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int (done) #want to multiply the reward by 0 (can also do it in learn function)
        self.mem_cntr+=1 # increase the memory
        
    #function to sample from the memmory from the batch size    
    def sample_buffer (self, batch_size):
        #have we fill up the mem or not? 
        max_mem = min (self.mem_cntr, self.mem_size)
        
        #select randomly from batch
        batch = np.random.choice(max_mem, batch_size, replace = False)# wont select again with replay False
        # book keeping for proper batch slicing
        batch_index = np.arange (self.batch_size, dtype=np.int32)
        
        
        #return the data in batch
        states = self.state_memory[batch]
        states_=  self.new_state_memory[batch]
        rewards =  self.reward_memory[batch]
        actions =  self.action_memory[batch] 
        terminal = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, terminal, batch_index

class LinearDeepQNetwork(nn.Module):
    
    #constructor 
    def __init__(self, lr, n_actions, input_dims):
        # run the constructor of the parent class.
        super(LinearDeepQNetwork,self).__init__()
        
        # 3 steps:
        # 1: define the layers
        # 2: define the optimizer, and lose funtion: torch.optim, torch.nn
        # 3: define the training devices and send it to the device.: torch.device
        
        self.fc1 = nn.Linear(*input_dims, 256)
        self.fc2 = nn.Linear (256, n_actions)
        #choose gradient decent method for backward propergation
        self.optimizer = optim.Adam(self.parameters(),lr = lr)
        self.loss = nn.MSELoss()
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #send the network to the device
        self.to(self.device)
        
        
    #forward propergation: activation function
    def forward(self, state):
        
        layer1 = F.relu (self.fc1(state))
        out_actions = self.fc2(layer1)
        
        return out_actions
