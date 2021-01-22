import gym
import torch
import numpy as np
import pickle
import random
from model.manager_som import ManagerSOM

import matplotlib.pyplot as plt

manager_maxitr = 5 * 10 ** 3
maxtime = 10 ** 2
gamma = 0.99
epsilon = 0.3
cumulative_return = []

state_filehandler = open("data/pose_somatotopic/cartpole/state_som.obj", 'rb')
state_som = pickle.load(state_filehandler)
worker_filehandler = open("data/pose_somatotopic/cartpole/worker_som.obj", 'rb')
worker_som = pickle.load(worker_filehandler)

manager_som = ManagerSOM(total_nodes = 100,
                        state_som = state_som,
                        worker_som = worker_som,
                        update_iterations=manager_maxitr)
env = gym.make("CartPole-v1")
obs = env.reset()

tmp_cum_return = 0
tmp_epoch_count = 0
for epoch in range(manager_maxitr):
    total_return = 0

    for t in range(0, maxtime):
#         env.render()
        state_vector = torch.zeros(state_som.total_nodes)
        state_vector[state_som.select_winner(obs)] = 1

        # epsilon greedy
        if random.random() > epsilon:
            action_index = manager_som.get_action(state_vector) # deterministic

        else:
            action_index = random.randrange(worker_som.total_nodes)

        if worker_som.w[action_index][0] >= 0.5:
            action = 1
        else:
            action = 0

        next_obs, reward, done, _ = env.step(action)

        # online training
        manager_som.action_q_learning(
            current_state_index = state_som.select_winner(obs),
            action_index = action_index,
            reward = reward,
            next_state_index = state_som.select_winner(next_obs),
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
plt.savefig("data/pose_somatotopic/cartpole/cartpole_returns.png")

filehandler = open("data/pose_somatotopic/cartpole/manager_som.obj", 'wb')
pickle.dump(manager_som, filehandler)
