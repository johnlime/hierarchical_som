import sys
sys.path.append("libraries/RlkitExtension")

import gym
import torch
import numpy as np
import pickle
import random
from model.kohonen_som import KohonenSOM
from model.manager_som_position import ManagerSOMPositionAllNeighbor

import matplotlib.pyplot as plt

"""
Hyperparameters
"""
maxitr = 10 ** 3
maxtime = 10 ** 3
gamma = 0.99
epsilon = 0.3

"""
Models
"""
worker_som = KohonenSOM(total_nodes=100, node_size=4, update_iterations=maxitr)
state_som = KohonenSOM(total_nodes=100, node_size=24, update_iterations=maxitr)
manager_som = ManagerSOMPositionAllNeighbor(total_nodes = 100,
                        state_som = state_som,
                        worker_som = worker_som,
                        update_iterations=maxitr)

filehandler = open("libraries/RlkitExtension/data/PPOBipedalWalkerV2Poses/PPOBipedalWalkerV2Poses_2021_03_01_12_33_33_0000--s-0/params.pkl", 'rb')
policy = torch.load(filehandler)["evaluation/policy"]

"""
Sensor field online training pool
"""
sampled_length = 0
sample_iter = maxtime * maxitr
worker_pool = torch.empty(sample_iter * 200, 4)
state_pool = torch.empty(sample_iter * 200, 24)
action = torch.empty(4)

"""
Motor field online training return recording
"""
cumulative_return = []
tmp_cum_return = 0
tmp_epoch_count = 0

"""
Training
"""

env = gym.make("BipedalWalker-v3")
obs = env.reset()
action_index = 0

for epoch in range(maxitr):
    total_return = 0
    print(epoch, " / ", maxitr)

    for t in range(0, maxtime):
        # env.render()
        if t % 5 == 0:
            # sample observations from environment
            worker_pool[sampled_length][0] = torch.tensor(obs[4])
            worker_pool[sampled_length][1] = torch.tensor(obs[6])
            worker_pool[sampled_length][2] = torch.tensor(obs[9])
            worker_pool[sampled_length][3] = torch.tensor(obs[11])
            state_pool[sampled_length] = torch.tensor(obs)
            sampled_length += 1

            current_state_location = state_som.location[state_som.select_winner(obs)]

            # epsilon greedy
            if random.random() > epsilon:
                action_index = manager_som.get_action(current_state_location) # deterministic

            else:
                action_index = random.randrange(worker_som.total_nodes)

        policy_input = np.append(worker_som.w[action_index].detach(), obs)
        action = policy.get_action(policy_input)[0]

        for i in range(4):
            if action[i] > 1:
                action[i] = 1
            elif action[i] < -1:
                action[i] = -1
            else:
                pass

        next_obs, reward, done, _ = env.step(action)

        next_state_location = state_som.location[state_som.select_winner(next_obs)]

        # online training
        manager_som.action_q_learning(
            current_state_position = current_state_location,
            action_index = action_index,
            reward = reward,
            next_state_position = next_state_location,
            t = epoch,
            gamma = gamma)

        worker_som.update(worker_pool[:sampled_length], epoch)
        state_som.update(state_pool[:sampled_length], epoch)

        total_return += (gamma ** t) * reward
        obs = next_obs

        if done:
            # print("Episode finished after {} timesteps".format(t+1))
            # print(epoch, total_return)

            # tmp_cum_return += total_return
            # tmp_epoch_count += 1
            #
            # cumulative_return.append(tmp_cum_return / tmp_epoch_count)
            # tmp_cum_return = 0
            # tmp_epoch_count = 0
            break

    cumulative_return.append(total_return)
    obs = env.reset()

plt.plot(np.linspace(0, len(cumulative_return), num = len(cumulative_return)), np.array(cumulative_return), marker='.', linestyle='-', color='blue')
plt.savefig("data/smc_premotor_pid/bipedal_walker/affordance_controller_all_neighbors_ppo.png")

filehandler = open("data/smc_premotor_pid/bipedal_walker/affordance_wsm_all_neighbors_ppo.obj", 'wb')
pickle.dump([worker_som, state_som, manager_som], filehandler)
