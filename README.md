
# Hierarchical Self Organizing Map Reinforcement Learning

Attempts of modeling functionalities of supplementary motor cortex, premotor cortex, and primary motor cortex using Kohonen's self organizing maps

**NOT for application. Merely a presentation of prototype/concept**

## Background

Understanding the functionalities of primary motor cortex and premotor cortex, mainly supplementary motor cortex or supplementary motor area (SMA), of the human brain has been a major focus of research in the field of neuroscience, which has provided data regarding synaptic activation regions corresponding to sensory inputs and motor outputs, stemming numerous hypotheses [0, 1]. We focus on modeling the hypothesis that the SMA manages the sequence in which the motor neurons within the primary motor cortex are activated in order to execute diverse tasks.

Preliminary works mainly focus on modeling using gradient-based deep neural networks [2]. While these methods succeed in accurately mimicking and predicting the motor output given a synaptic input, they do not represent the topologically preserved hierarchical self organization and clustering of input data known to exist in both of the cortices [0]. Alongside this, works of self organizing reinforcement learning methods, which include usage of adaptive resonance control (ART) [3, 4], should be mentioned since their architecture strongly resemble the ones presented here; however, such lack the ability to depict preservation of topology of the input data. We attempt to utilize Kohonen's self organizing maps (SOM) and have agents learn tasks using hierarchical reinforcement learning in a similar fashion as such seen in motor cortices.

SOM-based algorithms additionally has a better visualization scheme than the feed-forward deep neural network counterparts.

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

  - Manger's synaptic weights are composed of the q-values of the worker's nodes concatenated with a one hot vector or position representing the activated node of the separate state layer.

    - Note that the position-based strategy takes inspiration from the hypothesis that the motor cortices encode input signals as cortical directions [5]

  - Q-values representing the worker's nodes are trained using Q-learning.

  - Since we found that updating only the weights representing the state space performs significantly better than both state space and q-values, weight vector segments containing only the state space are updated

### SMC-Premotor-PID Model

- Hierarchical self organizing map with a similar structure as the Pose-Somatotopic model

- Worker layer maps the appropriate reference signal for the PID controller instead of the action space.

- PID control is used for lower-level (premotor and primary motor cortex) control in order for less complication of the problem

## Evaluation
We compare the performance of all of the aforementioned models using the two tasks below.

- NavigationTask

  - An environment where an agent controlled by such has to navigate itself from one point to another.

  - Rewards are generated every timestep as a negative value of the distance between the goal and the agent's positions

- NavigationTaskV2

  - Same as NavigationTask, but the reward are calculated using the absolute value of the radian difference between the vector pointing from the current position to the target position, and the vector of the action taken in the current step.

  - Rewards' range is standardized to have the range: `[-math.pi, math.pi]`

- Cartpole-v1 task in OpenAI Gym

In addition to this, we observe the effect of training both sensory cognition and motor generation modules concurrently as opposed to pre-training the sensory cognition module and training the motor generation module afterwords. This is in consideration of the concept of affordance [6], which hypothesizes that both modules are structurally and functionally intertwined with each other and are involved in the decision making process of the agent in question.

## Dependencies

- PyTorch
- Gym
- Numpy
- Pickle
- Random

## Running demo
All of the demonstrations are for the cartpole-v1 task in OpenAI Gym

1. Clone repository

2. Set current directory to the project path
```
cd hierarchical_som
```

3. Set `PYTHONPATH` to current directory
```
export PYTHONPATH=$PWD
```

4. Train worker layer and state map
```
python3 demo/<MODEL_NAME>/cartpole/cartpole.py
```

5. Train manager layer using one-hot vectors for worker node representation
```
python3 demo/<MODEL_NAME>/cartpole/manager_cartpole.py
```

Alternatively, it is possible to use position as worker node representation
```
python3 demo/<MODEL_NAME>/cartpole/manager_cartpole_position.py
```

## Current (Poor) Results

(A) https://github.com/johnlime/hierarchical_som/blob/master/demo/Visualization.ipynb

- Visualization of both models (including both one-hot vector based and position based state space representations) tested on Cartpole-v1

(B) https://github.com/johnlime/hierarchical_som/blob/master/demo/NavigationTask%20w%20ModifiedManagerSOM.ipynb

- Visualization of both one-hot vector based and position based state space representation of the SMC-Premotor-PID Model tested on NavigationTask.

- Attempts to replicate (C) by modifying the bellman function for Q-learning in ManagerSOM.

- Performance of one-hot-vector-based selection and position-based selection are compared

(B-2) https://github.com/johnlime/hierarchical_som/blob/master/demo/NavigationTaskV2%20w%20ManagerSOM.ipynb

- Visualization of the position-based state space representation of the SMC-Premotor-PID Model using NavigationTaskV2

(C) https://github.com/johnlime/hierarchical_som/blob/master/demo/Hierarchical%20Self%20Organizing%20Map.ipynb

- Initial testing of the SMC-Premotor-PID model, where PID is replaced with linear locomotion on NavigationTask

(D-1) https://github.com/johnlime/hierarchical_som/blob/master/data/smc_premotor_pid/cartpole/cartpole_affordance_returns.png

- Training returns of implementation of affordance using SMC-Premotor-PID model on Cartpole-v1

(D-2) https://github.com/johnlime/hierarchical_som/blob/master/data/smc_premotor_pid/cartpole/cartpole_affordance_all_neighbor_returns.png

- Same as (D-1), but weights representing both state SOM positions and q-values are updated

## Analysis of the Current Poor Results

- Better performance when the Bellman equation is modified to utilize max_a Q(s, a) instead of max_a Q(s', a); (A)

  - It should be noted that the similar results could be seen with NavigationTask

  - The original version of the model, which had no changes to the Bellman equation, led to the model's inability to learn the solution of the task.

- **The problem regarding the Bellman equation seen in (A) can be resolved with a better representation of the reward function (B-2)**

  - SMC-Premotor-PID model was successfully able to produce good results for navigation an agent from one point to another using NavigationTaskV2

- Clustering one hot vectors of state layer leads to significant decrease in performance compared to clustering positions of each nodes; (B-1; B-2)

- Implementations with affordance taken into consideration (A; D-1; D-2)

  - More stable convergence of return to a specific value during training than the alternatives

  - Higher return during said convergence

- **Using an approximation of state and action spaces may not be appropriate when the task is to arrive to an unstable fixed point.** (A)

  - Can SOM be adapted to deal with unstable dynamical systems?

## Open Problem

- Struggling to solve the pole-balancing problem. But is conceptualizing and controlling the cart good enough for this task? More "biological" tasks such as locomotion and inverse kinematics should be taken into consideration

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

[3] Ah-Hwee Tan, "FALCON: a fusion architecture for learning, cognition, and navigation," 2004 IEEE International Joint Conference on Neural Networks (IEEE Cat. No.04CH37541), Budapest, 2004, pp. 3297-3302 vol.4, https://doi.org/0.1109/IJCNN.2004.1381208.

[4] Ah-Hwee Tan. 2006. Self-organizing neural architecture for reinforcement learning. In Proceedings of the Third international conference on Advances in Neural Networks - Volume Part I (ISNN'06). Springer-Verlag, Berlin, Heidelberg, 470–475. https://doi.org/10.1007/11759966_70

[5] Teka WW, Hamade KC, Barnett WH, Kim T, Markin SN, Rybak IA, et al. (2017) From the motor cortex to the movement and back again. PLoS ONE 12(6): e0179288. https://doi.org/10.1371/journal.pone.0179288

[6] Cisek P, Kalaska JF. Neural mechanisms for interacting with a world full of action choices. Annu Rev Neurosci. 2010;33:269-98. https://doi.org/10.1146/annurev.neuro.051508.135409. PMID: 20345247.
