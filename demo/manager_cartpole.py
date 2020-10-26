import gym
import torch
import numpy as np
import pickle
import random
from model.manager_som import ManagerSOM

import matplotlib.pyplot as plt

manager_maxitr = 10 ** 3
maxtime = 10 ** 2
gamma = 0.99
epsilon = 0.3
cumulative_return = []

pos_filehandler = open("data/som_pos.obj", 'rb')
som_pos = pickle.load(pos_filehandler)
angle_filehandler = open("data/som_ang.obj", 'rb')
som_ang = pickle.load(angle_filehandler)

manager_som = ManagerSOM(total_nodes=100, worker_som=[som_pos, som_ang], update_iterations=manager_maxitr)
env = gym.make("CartPole-v1")
obs = env.reset()

for epoch in range(manager_maxitr):
    total_return = 0

    for t in range(0, maxtime):
#         env.render()
        if t % 5 == 0:
            current_winner_indices = []
            current_winner_indices.append(som_pos.select_winner(obs[:2]))
            current_winner_indices.append(som_ang.select_winner(obs[2:]))

            # create one-hot vector for winner worker som
            current_position = torch.zeros(manager_som.state_indices[0] + manager_som.state_indices[1])
            current_position[current_winner_indices[0]] = 1
            current_position[manager_som.state_indices[0]:][current_winner_indices[1]] = 1

            # epsilon greedy
            if random.random() > epsilon:
                action = manager_som.get_action(current_position) # deterministic

            else:
                action = random.randrange(manager_som.action_indices[0])

        next_obs, reward_value, done, _ = env.step(action)

        # online training
        manager_som.action_q_learning(
            current_winner_indices = current_winner_indices,
            next_winner_index = action,
            reward = reward_value,
            t = epoch,
            gamma = gamma)

        total_return += (gamma ** t) * reward_value
        obs = next_obs

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print(epoch, total_return)
            if epoch % 99 == 0:
                cumulative_return.append(total_return)
            break

    obs = env.reset()

plt.plot(np.linspace(0, len(cumulative_return), num = len(cumulative_return)), np.array(cumulative_return), marker='.', linestyle='-', color='blue')
# plt.savefig()

# filehandler = open(, 'wb')
pickle.dump(manager_som, filehandler)
