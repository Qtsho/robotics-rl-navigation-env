# robotics-rl-navigation-env

A model based RL approach for robot navigation.
## Table of contents
* [General Info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General Info


Navigation RL environment for turtlebot3 and innok heros skidsteer robot. This code is adapted from Open AI Gym and ROBOTIS.

For testing this env, I used DQN [] for robot mapless navigation. The implementation is in Pytorch with Replay Buffer and Target network.


## Technologies

Dependencies: 
- ROS noetic

- Gazebo >9.x

- Turtlebot description

- innoks heros description


## Setup



Tutorial:
- run gazebo simulation env: turtlebot/innok
- run agent node which use dqn. Can resume from trained episode
