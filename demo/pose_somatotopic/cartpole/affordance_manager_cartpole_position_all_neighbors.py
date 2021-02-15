import gym
import torch
import numpy as np
import pickle
import random
from model.kohonen_som import KohonenSOM
from model.manager_som_position import ManagerSOMPositionAllNeighbor

import matplotlib.pyplot as plt

maxitr = 5 * 10 ** 3
maxtime = 10 ** 2
sampled_length = 0
sample_iter = maxtime * maxitr
gamma = 0.99
epsilon = 0.3
cumulative_return = []

state_som = KohonenSOM(total_nodes=100, node_size=3, update_iterations=maxitr) # angle and angular velocity only
worker_som = KohonenSOM(total_nodes=2, node_size=1, update_iterations=maxitr)

manager_som = ManagerSOMPositionAllNeighbor(total_nodes = 100,
                        state_som = state_som,
                        worker_som = worker_som,
                        update_iterations=maxitr)

env = gym.make("CartPole-v1")
obs = env.reset()

worker_pool = torch.empty(sample_iter * 200, 1)
state_pool = torch.empty(sample_iter * 200, 3)

tmp_cum_return = 0
tmp_epoch_count = 0

for epoch in range(maxitr):
    total_return = 0

    for t in range(0, maxtime):
#         env.render()
        current_state_location = state_som.location[state_som.select_winner(obs[1:])]

        # epsilon greedy
        if random.random() > epsilon:
            action_index = manager_som.get_action(current_state_location) # deterministic

        else:
            action_index = random.randrange(worker_som.total_nodes)

        if worker_som.w[action_index][0] >= 0.5:
            action = 1
        else:
            action = 0

        next_obs, reward, done, _ = env.step(action)
        next_state_location = state_som.location[state_som.select_winner(next_obs[1:])]

        # sample observations from environment
        worker_pool[sampled_length][0] = torch.tensor(action)
        state_pool[sampled_length][0] = torch.tensor(obs[1])
        state_pool[sampled_length][1] = torch.tensor(obs[2])
        state_pool[sampled_length][2] = torch.tensor(obs[3])
        sampled_length += 1

        worker_som.update(worker_pool[:sampled_length], epoch)
        state_som.update(state_pool[:sampled_length], epoch)

        # online training
        manager_som.action_q_learning(
            current_state_position = current_state_location,
            action_index = action_index,
            reward = reward,
            next_state_position = current_state_location,
            t = epoch,
            gamma = gamma)

        total_return += (gamma ** t) * reward
        obs = next_obs

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print(epoch, total_return)

            tmp_cum_return += total_return
            tmp_epoch_count += 1

            if epoch % 99 == 0:
                cumulative_return.append(tmp_cum_return / tmp_epoch_count)
                tmp_cum_return = 0
                tmp_epoch_count = 0
            break

    obs = env.reset()

plt.plot(np.linspace(0, len(cumulative_return), num = len(cumulative_return)), np.array(cumulative_return), marker='.', linestyle='-', color='blue')
plt.savefig("data/pose_somatotopic/cartpole/affordance_cartpole_positions_all_neighbors_returns.png")

filehandler = open("data/pose_somatotopic/cartpole/affordance_wsm_position_all_neighbors.obj", 'wb')
pickle.dump([state_som, worker_som, manager_som], filehandler)
