import gym
import torch
import numpy as np
import pickle
from model.manager_som import ManagerSOM

manager_maxitr = 10**3
maxtime = 10 ** 2

current_state = []
current_action = []
reward = []
next_state = []

filehandler = open("data/som.obj", 'rb')
som = pickle.load(filehandler)

manager_som = ManagerSOM(total_nodes=100, worker_som=som, additional_state_space = 2, update_iterations=manager_maxitr)
env = gym.make("CartPole-v1")
obs = env.reset()

for epoch in range(manager_maxitr):
    # for visualization
    total_return = 0

    for t in range(maxtime):
        # create one-hot vector for winner worker som
        current_position = torch.zeros(manager_som.state_indices + 2)
        current_position[som.select_winner(obs[:1])] = 1
        # additional state
        current_position[-2:] = torch.tensor(obs[2:])
        action_index = manager_som.get_action(current_position) # deterministic

        # Pseudo-PD control
        k_p = 0.5
        k_d = 1.2
        action = k_p * som.w[action_index][0] + k_d * som.w[action_index][1]
        if (action > 0): action = 1
        else: action = 0

        next_obs, reward_value, done, _ = env.step(action)

        # trajectory sampling
        current_state.append(np.array(current_position))
        current_action.append(np.array(action_index))
        reward.append(np.array(reward_value))
        next_state.append(np.array(next_obs))

        # online training
        manager_som.action_q_learning(
            current_winner_index = som.select_winner(current_position[-2:].float()),
            additional_states = obs[2:],
            next_winner_index = action_index,
            reward = reward_value,
            t = t)

        total_return += (0.9 ** t) * reward_value
        obs = next_obs

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

    # plt.plot(np.array(next_state)[:, 0], np.array(next_state)[:, 1], marker='.', linestyle='-', color='blue')
    # plt.plot(0.0, 0.0, marker='v', linestyle='None', color='orange')
    # plt.show()
    # print(epoch, total_return)

    obs = env.reset()
    current_state = []
    current_action = []
    reward = []
    next_state = []

filehandler = open("data/manager_som.obj", 'wb')
pickle.dump(manager_som, filehandler)
