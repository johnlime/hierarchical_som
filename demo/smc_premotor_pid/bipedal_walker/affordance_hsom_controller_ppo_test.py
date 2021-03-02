import sys
sys.path.append("libraries/RlkitExtension")

import gym
import torch
import numpy as np
import pickle
import random

import matplotlib.pyplot as plt

"""
Hyperparameters
"""
maxtime = 10 ** 3

"""
Models
"""
filehandler = open("data/smc_premotor_pid/bipedal_walker/affordance_wsm_all_neighbors_ppo.obj", 'rb')
worker_som, state_som, manager_som = pickle.load(filehandler)
# trained on 447 epochs
filehandler = open("libraries/RlkitExtension/data/PPOBipedalWalkerV2Poses/PPOBipedalWalkerV2Poses_2021_03_01_12_33_33_0000--s-0/params.pkl", 'rb')
policy = torch.load(filehandler)["evaluation/policy"]

"""
Motor field online training return recording
"""
cumulative_return = []
tmp_cum_return = 0
tmp_epoch_count = 0

"""
Testing
"""
env = gym.make("BipedalWalker-v3")
obs = env.reset()

total_return = 0
action = torch.empty(4)

for t in range(0, maxtime):
    env.render()
    state_index = state_som.select_winner(obs)
    current_state_location = state_som.location[state_index]
    action_index = manager_som.get_action(current_state_location) # deterministic
    policy_input = np.append(worker_som.w[action_index].detach(), obs)
    action = policy.get_action(policy_input)[0]

    next_obs, reward, done, _ = env.step(action)

    next_state_location = state_som.location[state_som.select_winner(next_obs)]

    total_return += reward
    obs = next_obs

    # if done:
    #     print("Episode finished after {} timesteps".format(t+1))
    #     break

env.close()
