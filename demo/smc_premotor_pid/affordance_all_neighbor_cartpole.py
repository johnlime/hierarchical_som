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
maxitr = 5 * 10 ** 3
maxtime = 10 ** 2
gamma = 0.99
epsilon = 0.3

"""
Models
"""
worker_som = KohonenSOM(total_nodes=100, node_size=2, update_iterations=maxitr)
state_som = KohonenSOM(total_nodes=100, node_size=4, update_iterations=maxitr)
manager_som = ManagerSOMPositionAllNeighbor(total_nodes = 100,
                        state_som = state_som,
                        worker_som = worker_som,
                        update_iterations=maxitr)

"""
Sensor field online training pool
"""
sampled_length = 0
sample_iter = maxtime * maxitr
worker_pool = torch.empty(sample_iter * 200, 2)
state_pool = torch.empty(sample_iter * 200, 4)

"""
Motor field online training return recording
"""
cumulative_return = []
tmp_cum_return = 0
tmp_epoch_count = 0

"""
Training
"""

env = gym.make("CartPole-v1")
obs = env.reset()

for epoch in range(maxitr):
    total_return = 0

    for t in range(0, maxtime):
        # env.render()
        # sample observations from environment
        worker_pool[sampled_length] = torch.tensor(obs[:2])
        state_pool[sampled_length] = torch.tensor(obs)
        sampled_length += 1

        current_state_location = state_som.location[state_som.select_winner(obs)]

        # epsilon greedy
        if random.random() > epsilon:
            action_index = manager_som.get_action(current_state_location) # deterministic

        else:
            action_index = random.randrange(worker_som.total_nodes)

        if worker_som.w[action_index][0] >= 0.5:
            action = 1
        else:
            action = 0

        # Pseudo-PD control
        k_p = 1.0
        k_d = 0.05
        action = k_p * (worker_som.w[action_index][0] - obs[0]) + k_d * (worker_som.w[action_index][1] - obs[1])
        if (action > 0): action = 1
        else: action = 0

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

            tmp_cum_return += total_return
            tmp_epoch_count += 1

            if epoch % 99 == 0:
                cumulative_return.append(tmp_cum_return / tmp_epoch_count)
                tmp_cum_return = 0
                tmp_epoch_count = 0
            break

    obs = env.reset()


plt.plot(np.linspace(0, len(cumulative_return), num = len(cumulative_return)), np.array(cumulative_return), marker='.', linestyle='-', color='blue')
plt.savefig("data/smc_premotor_pid/cartpole_affordance_all_neighbor_returns.png")

filehandler = open("data/smc_premotor_pid/affordance_all_neighbor_wsm.obj", 'wb')
pickle.dump([worker_som, state_som, manager_som], filehandler)
