
# Hierarchical Self Organizing Map Reinforcement Learning

Attempts of modeling functionalities of supplementary motor cortex, premotor cortex, and primary motor cortex using Kohonen's self organizing maps

**Poor Implementation. NOT for application. Merely a presentation of prototype/concept**

## Background

Understanding the functionalities of primary motor cortex and premotor cortex, mainly supplementary motor cortex or supplementary motor area (SMA), of the human brain has been a major focus of research in the field of neuroscience, which has provided data regarding synaptic activation regions corresponding to sensory inputs and motor outputs, stemming numerous hypotheses [0, 1]. We focus on the hypothesis that the SMA manages the sequence in which the motor neurons within the primary motor cortex are activated in order to execute diverse tasks.

Preliminary works mainly focus on modeling using gradient-based deep neural networks [2]. While these methods succeed in accurately mimicking and predicting the motor output given a synaptic input, they do not represent the self organization and clustering of data known to exist in both of the cortices [0]. We attempt to utilize Kohonen's self organizing maps and have agents learn tasks using hierarchical reinforcement learning in a similar fashion as such seen in motor cortices.

## Approach

- Model the cluster mapping and overlapping of motor units into a form of motor neurons in primary motor, premotor, and supplementary motor cortex

- Cluster map model using hierarchically structured Kohonen's self organizing maps

- Model must be able to perform well in given tasks and environments

  - Have the agent learn to perform specific tasks explicitly defined by reward function

  - Examine how well does it capture the dynamics of the system at hand

- Unsupervised learning and sequential execution of the tasks

## Implemented Prototype Models

### Pose-Somatotopic Model

- Hierarchical self organizing map, where the high-level manager layer chooses a node in the low-level worker layer, which maps the action space, given the activated node on a separate layer which maps the state space

  - Worker is trained using randomized vectors extracted from the action space, representing somatotopic mapping of the primary motor cortex.

  - A separate layer for mapping the state space is trained by running a random policy
  
  - Manger's synaptic weights are composed of the q-values of the worker's nodes concatenated with a one hot vector representing the activated node of the separate state layer.
  
  - Q-values representing the worker's nodes are trained using Q-learning.

### SMC-Premotor-PID Model

- Hierarchical self organizing map with a similar structure as the Pose-Somatotopic model

- Worker layer maps the appropriate reference signal for the PID controller instead of the action space.

- PID control is used for lower-level (premotor and primary motor cortex) control in order for less complication of the problem

## Current Poor Results

https://github.com/johnlime/hierarchical_som/blob/master/demo/Visualization.ipynb

## Analysis of the Current Poor Results

- Using an approximation of state and action spaces may not be appropriate when the task is to arrive to an unstable fixed point.

- Can SOM be adapted to deal with unstable dynamical systems?

## Open Problem

- Currently trying the problem on pole-balancing. But is conceptualizing and controlling the cart good enough for this task? More "biological" tasks such as locomotion and inverse kinematics should be taken into consideration

- Since the goal of the algorithm is to mimic the cerebral cortex, research results in that field should be taken into consideration for better understanding and engineering.

## References

[0] Michael S.A. Graziano, Tyson N. Aflalo,
Mapping Behavioral Repertoire onto the Cortex,
Neuron,
Volume 56, Issue 2,
2007,
Pages 239-251,
ISSN 0896-6273,
https://doi.org/10.1016/j.neuron.2007.09.013.

[1] Nachev, P., Kennard, C. & Husain, M. Functional role of the supplementary and pre-supplementary motor areas. Nat Rev Neurosci 9, 856–869 (2008). https://doi.org/10.1038/nrn2478

[2] Akbar, M. N., Yarossi, M., Martinez-Gost, M., Sommer, M. A., Dannhauer, M., Rampersad, S., ... & Erdoğmuş, D. (2020). Mapping Motor Cortex Stimulation to Muscle Responses: A Deep Neural Network Modeling Approach. arXiv preprint arXiv:2002.06250.
https://doi.org/10.1145/3389189.3389203
