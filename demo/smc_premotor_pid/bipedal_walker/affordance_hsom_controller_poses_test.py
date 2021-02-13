from custom_env.bipedal_walker_poses import BipedalWalkerPoses
import torch
import numpy as np
import pickle
import random

import gym

"""
Hyperparameters
"""
maxtime = 10 ** 2

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
env = BipedalWalkerPoses() #gym.make("BipedalWalker-v2")
obs = env.reset()

total_return = 0
action = torch.empty(4)

for action_index in range(state_som.total_nodes):
    print(action_index)
    for t in range(0, maxtime):
        env.render()
        # current_state_location = state_som.location[state_som.select_winner(obs)]
        # action_index = manager_som.get_action(current_state_location) # deterministic

        # PD control
        # Gains estimated via CMA-ES
        # k_p = 2.175604023818439
        # k_d = 1.2390217586889263
        # action[0] = k_p * (worker_som.w[action_index][0] - obs[4]) + k_d * obs[5]
        # action[1] = k_p * (worker_som.w[action_index][1] - obs[6]) + k_d * obs[7]
        # action[2] = k_p * (worker_som.w[action_index][2] - obs[9]) + k_d * obs[10]
        # action[3] = k_p * (worker_som.w[action_index][3] - obs[11]) + k_d * obs[12]

        x = [1.1856849882386575, 1.5903066105130703, 1.1236952114939938, 3.0433557288087694, -0.3027671732001401, 0.39535286688944876, -0.08675309470444716, 0.4490997629403532]
        k_p = []
        k_d = []
        for i in range(4):
            k_p.append(x[i])
            k_d.append(x[2 * i])

        action[0] = k_p[0] * (worker_som.w[action_index][0] - obs[4]) + k_d[0] * obs[5]
        action[1] = k_p[1] * (worker_som.w[action_index][1] - obs[6]) + k_d[1] * obs[7]
        action[2] = k_p[2] * (worker_som.w[action_index][2] - obs[9]) + k_d[2] * obs[10]
        action[3] = k_p[3] * (worker_som.w[action_index][3] - obs[11]) + k_d[3] * obs[12]

        for i in range(4):
            if action[i] > 1:
                action[i] = 1
            elif action[i] < -1:
                action[i] = -1
            else:
                pass

        next_obs, _, _, _ = env.step(action)
        obs = next_obs

        # if done:
        #     print("Episode finished after {} timesteps".format(t+1))
        #     break
    env.reset()

env.close()
