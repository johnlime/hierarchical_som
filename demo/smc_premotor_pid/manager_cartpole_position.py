import gym
import torch
import numpy as np
import pickle
import random
from model.manager_som_position import ManagerSOMPosition

import matplotlib.pyplot as plt

manager_maxitr = 5 * 10 ** 3
maxtime = 10 ** 2
gamma = 0.99
epsilon = 0.3
cumulative_return = []

state_filehandler = open("data/smc_premotor_pid/state_som.obj", 'rb')
state_som = pickle.load(state_filehandler)
worker_filehandler = open("data/smc_premotor_pid/worker_som.obj", 'rb')
worker_som = pickle.load(worker_filehandler)

manager_som = ManagerSOMPosition(total_nodes = 100,
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
plt.savefig("data/smc_premotor_pid/cartpole_positions_returns.png")

filehandler = open("data/smc_premotor_pid/manager_position_som.obj", 'wb')
pickle.dump(manager_som, filehandler)
