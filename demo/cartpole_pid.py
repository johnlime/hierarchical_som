import gym
import numpy as np
import random

import matplotlib.pyplot as plt

manager_maxitr = 1
maxtime = 10 ** 3
gamma = 0.99
cumulative_return = []

env = gym.make("CartPole-v1")

for epoch in range(manager_maxitr):
    total_return = 0
    obs_trajectory = []
    obs = env.reset()

    # coefficients for PD control
    k_p = 0.1 # Does not matter in this case due to the fact that the action is binary
    k_d = 0.0

    for t in range(0, maxtime):
        env.render()
        obs_trajectory.append(obs[0])
        # PD control
        action = k_p * (0 - obs[0]) + k_d * (0 - obs[1])
        if (action > 0): action = 1
        else: action = 0

        next_obs, reward_value, done, _ = env.step(action)

        total_return += (gamma ** t) * reward_value
        obs = next_obs

    plt.plot(np.linspace(0, len(obs_trajectory), num = len(obs_trajectory)), np.array(obs_trajectory), marker='.', linestyle='-', color='blue')
    plt.title('k_p = ' + str(k_p))
    # plt.savefig()
    plt.clf()
