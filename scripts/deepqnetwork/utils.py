#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:59:40 2021

@author: workstation
"""

import matplotlib.pyplot as plt
import numpy as np
#to plot the learning curve of the RL agent.
def plot_learning_curve (x, scores, epsilons, filename):
    ''' PARAMS
    x: game/episode index
    scores: the reward of that episode
    epsilon: epsilon greedy control
    filename: which file will export the plot
    '''
    
    #create a figure
    fig= plt.figure()
    #create subplot
    ax = fig.add_subplot (111, label ="1")
    ax2 = fig.add_subplot (111, label = "2", frame_on = False)
    
    
    #Plot in first figure: epsilon - game
    ax.plot (x, epsilons, color = "C0")
    
    ax.set_xlabel ("Training Steps", color = "C0")
    ax.set_ylabel ("Epsilon", color = "C0")
    
    ax.tick_params(axis='x', color = "C0")
    ax.tick_params(axis='y', color = "C0")
    
    
    N = len (scores)
    running_avg = np.empty(N)
    for t in range (N):
        running_avg[t] = np.mean(scores [max(0, t-100): (t+1)])
    
    # plot in second figure: avg100reward - game
    ax2.scatter(x, running_avg, color = "C1")
    ax2.axes.get_xaxis().set_visible(False)
    
    
    ax2.yaxis.tick_right()
    ax2.set_ylabel ("Score", color = "C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', color = "C1")
    plt.show()
    plt.savefig(filename)
    
    