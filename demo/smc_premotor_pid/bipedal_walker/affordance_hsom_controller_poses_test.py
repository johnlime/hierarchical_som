from custom_env.bipedal_walker_poses import BipedalWalkerPoses
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
filehandler = open("data/smc_premotor_pid/bipedal_walker/affordance_wsm.obj", 'rb')
worker_som, state_som, manager_som = pickle.load(filehandler)

"""
Motor field online training return recording
"""
cumulative_return = []
tmp_cum_return = 0
tmp_epoch_count = 0

"""
Testing
"""
env = BipedalWalkerPoses()
obs = env.reset()

total_return = 0
action = torch.empty(4)

for t in range(0, maxtime):
    env.render()
    current_state_location = state_som.location[state_som.select_winner(obs)]
    action_index = manager_som.get_action(current_state_location) # deterministic

    # PD control
    # Gains estimated via CMA-ES
    k_p = 2.175604023818439
    k_d = 1.2390217586889263
    action[0] = k_p * (worker_som.w[action_index][0] - obs[4]) + k_d * obs[5]
    action[1] = k_p * (worker_som.w[action_index][1] - obs[6]) + k_d * obs[7]
    action[2] = k_p * (worker_som.w[action_index][2] - obs[9]) + k_d * obs[10]
    action[3] = k_p * (worker_som.w[action_index][3] - obs[11]) + k_d * obs[12]

    for i in range(4):
        if action[i] > 1:
            action[i] = 1
        elif action[i] < -1:
            action[i] = -1
        else:
            pass

    next_obs, reward, done, _ = env.step(action)

    next_state_location = state_som.location[state_som.select_winner(next_obs)]

    total_return += reward
    obs = next_obs

    # if done:
    #     print("Episode finished after {} timesteps".format(t+1))
    #     break

env.close()
