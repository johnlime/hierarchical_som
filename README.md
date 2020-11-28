
# Hierarchical Self Organizing Map Reinforcement Learning

Attempts of modeling functionalities of supplementary motor cortex, premotor cortex, and primary motor cortex using Kohonen's self organizing maps

## Background

Understanding the functionalities of primary motor cortex and premotor cortex, mainly supplementary motor cortex or supplementary motor area (SMA), of the human brain has been a major focus of research in the field of neuroscience, which has provided data regarding synaptic activation regions corresponding to sensory inputs and motor outputs, stemming numerous hypotheses. We focus on the hypothesis that the SMA manages the sequence in which the motor neurons within the primary motor cortex are activated in order to execute diverse tasks.

Recent preliminary works mainly focus on modeling using gradient-based deep neural networks. While these methods succeed in accurately mimicking and predicting the motor output given a synaptic input, they do not represent the self organization and clustering of data known to exist in both of the cortices. We attempt to utilize Kohonen's self organizing maps and have agents learn tasks using hierarchical reinforcement learning in a similar fashion as such seen in motor cortices.

## Approach

- Model the cluster mapping and overlapping of motor units into a form of motor neurons in primary motor, premotor, and supplementary motor cortex

- Cluster map model using hierarchically structured Kohonen's self organizing maps

- Model must be able to perform well in given tasks and environments

  - Should start by making the agent learn to perform specific tasks explicitly defined by reward function

  - How well does it capture the dynamics of the system at hand?

- Unsupervised learning and sequential execution of the tasks

## Implemented Prototype Models

### Pose-Somatotopic Model

- Hierarchical self organizing map, where the high-level manager layer chooses an action given the activated node on the low-level worker layer, which maps the action space.

  - Worker is trained using randomized vectors extracted from the action space.

  - Manger is trained by clustering the state space by running the trained worker.

  - The state space in the manager is concatenated with one-hot vectors or positions representing the activated nodes of a separate layer for mapping the state space

### SMC-Premotor-PID Model

Hierarchical self organizing map with a similar structure as the aforementioned model, where the worker layer maps the appropriate reference signal for the PID controller instead of the action space.

## Current Poor Results

https://github.com/johnlime/hierarchical_som/blob/master/demo/Visualization.ipynb

## Analysis of the Current Poor Results

1. Using an approximation of state and action spaces may not be appropriate when the task is to arrive to an unstable fixed point.

2. Since the goal of the algorithm is to mimic the cerebral cortex, research results in that field should be taken into consideration for better understanding and engineering.

3. Hierarchical structure could be used for more complex sequential tasks

## Open Problem

- PID control is used for lower-level (premotor and primary motor cortex) control in order for less complication of the problem

- Currently trying the problem on pole-balancing. But is conceptualizing and controlling the cart good enough for this task? More "biological" tasks such as locomotion and inverse kinematics should be taken into consideration

- Can SOM be adapted to deal with unstable dynamical systems?

- May be a good idea to attempt implementation of FALCON for reference
