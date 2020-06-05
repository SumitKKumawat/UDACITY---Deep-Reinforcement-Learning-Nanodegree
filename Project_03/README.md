## Project : Collaboration and Competition
### Description

For this project, we train a pair of agents to play tennis.

![Image](https://github.com/SumitKKumawat/Images/blob/master/tennis.gif)

### Problem Statement
A reward of +0.1 is provided for each step that one of the two agent hits the ball over the net. A reward of -0.01 is provided an agent lets a nall hit the ground or hits the ball out of bounds. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 24 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic. In order to solve the environment, one of the agent must get an average score of +0.5 over 100 consecutive episodes.

### Files
* Tennis.ipynb: Notebook used to control and train the agent
* maddpg_agents.py: Create an MADDPG_Agents class that interacts with and learns from the environment
* model.py: Actor and Critic classes
* report.pdf: Technical report

### Dependencies
To be able to run this code, you will need an environment with Python 3 and the dependencies are listed as per below.

* [numpy:](https://scipy.org/install.html)
* [matplotlib:](https://matplotlib.org/users/installing.html)
* [PyTorch:](https://pytorch.org/get-started/locally/)
* [Unity ML:](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)

Furthermore, you need to download the environment from one of the links below. You need only to select the environment that matches your operating system:

Linux : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
MAC OSX : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
Windows : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

### Running
Run the cells in the notebook `Tennis.ipynb` to train an agent that solves our required task of moving the double-jointed arm.