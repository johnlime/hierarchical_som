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

filehandler = open("data/som.obj", 'rb')
som = pickle.load(filehandler)

manager_som = ManagerSOM(total_nodes=49, worker_som=som, additional_state_space = 2, update_iterations=manager_maxitr)
env = gym.make("CartPole-v1")
obs = env.reset()

for epoch in range(manager_maxitr):
    total_return = 0

    for t in range(0, maxtime):
#         env.render()
        if t % 20 == 0:
            # create one-hot vector for winner worker som
            current_position = torch.zeros(manager_som.state_indices + 2)
            current_position[som.select_winner(obs[:1])] = 1

            # additional state
            current_position[-2:] = torch.tensor(obs[2:])
            
            # epsilon greedy
            if random.random() > epsilon:
                action_index = manager_som.get_action(current_position) # deterministic
                
            else:
                action_index = random.randrange(10)
                

        # Pseudo-PD control
        k_p = 1.0
        k_d = 1.0
        action = k_p * (som.w[action_index][0] - obs[0]) + k_d * (som.w[action_index][1] - obs[1])
        if (action > 0): action = 1
        else: action = 0

        next_obs, reward_value, done, _ = env.step(action)

        # online training
        manager_som.action_q_learning(
            current_winner_index = som.select_winner(current_position[-2:].float()),
            additional_states = obs[2:],
            next_winner_index = action_index,
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
plt.show()

filehandler = open("data/manager_som.obj", 'wb')
pickle.dump(manager_som, filehandler)
