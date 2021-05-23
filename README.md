# robotics-rl-navigation-env

Navigation RL environment for turtlebot3 and innok heros skidsteer robot. This code is adapted from Open AI Gym and ROBOTIS.

For testing this env, I used DQN [] for robot mapless navigation. The implementation is in Pytorch with Replay Buffer and Target network.





Dependencies: 
- ROS noetic

- Gazebo >9.x

- Turtlebot description

- innoks heros description

Tutorial:
- run gazebo simulation env: turtlebot/innok
- run agent node which use dqn. Can resume from trained episode
