## Project 1: Navigation

#### Introduction
For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

* Random agent
![Image](https://github.com/SumitKKumawat/Images/blob/master/UDACITY%20random_agent.gif)


* Trained agent
![Image](https://github.com/SumitKKumawat/Images/blob/master/UDACITY%20trained_agent.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

#### Getting Started

1. Download the environment from one of the links below. You need only select the environment that matches your operating system:

   * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
   * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
   * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
   * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the environment.


2. Place the file in this folder, unzip (or decompress) the file and then write the correct path in the argument for creating the environment under the notebook `Navigation.ipynb`:

```python
env = env = UnityEnvironment(file_name="VisualBanana.app")
```

#### Description
* dqn_agent.py: code for the agent used in the environment
* model.py: code containing the Q-Network used as the function approximator by the agent
* checkpoint.pth: saved model weights for the original DQN model
* Navigation.ipynb: notebook containing the solution

#### Dependencies
* Numpy
* Matplotylib
* PyTorch
* UnityAgents

#### Running the code
After all packages have been installed in the environment you should open Jupyter Notebook using Anaconda find and open `Navigation.iypnb` archive. To run the cells you can simply click on the first one and press Shift + Enter. This can be made through the whole Notebook.
