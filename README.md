# Deep Reinforcement Learning Project: Navigation

![banana](images/banana-intro.gif)

## Project details

Project created as 1st project on Udacity Deep Reinforcement Learning nanodegree. The goal of the agent is to gather `yellow` bananas while avoiding the `blue` ones. Here are Unity details of the environment:

```
Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , , 
```

That means we work with state vector containing 37 continous values and 4 discrete actions representing moves (forward, backward, turn left, turn right). The environment is considered solved when agents reaches average score of 13.0 on 100 consecutive episodes.

## Getting started

Make sure you have `python 3.7` installed and virtual environment of your choosing activated.
You have to install `jupyter notebook` and view `navigation.ipynb`.
Unity has to be installed on your system.

File `model.py` contains neural network class used as a Q function in which we define the forward and file `agent.py` contains agent code in which it contains different actions of the agent along with the creation of buffer.

## Instructions

To start you need to run `navigation.ipynb`
