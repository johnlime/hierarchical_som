import gym
import torch
from model.manager_som import ManagerSOM

manager_maxitr = 50 # 10**3
maxtime = 100

current_state = []
current_action = []
reward = []
next_state = []

manager_som = ManagerSOM(total_nodes=100, worker_som=som, update_iterations=manager_maxitr)
env = gym.make("CartPole-v1")
obs = env.reset()

#############################

for epoch in range(manager_maxitr):
    # for visualization
    total_return = 0

    for t in range(maxtime):
        current_position = task.state()
        winner_one_hot = torch.zeros(manager_som.state_indices)
        winner_one_hot[som.select_winner(current_position)] = 1
        action_index = manager_som.get_action(winner_one_hot) # deterministic

        # step forward
        reward_value, next_position = task.step(som.w[action_index])
        reward_vector = torch.zeros(100)
        reward_vector[action_index] = reward_value

        # trajectory sampling
        current_state.append(np.array(current_position))
        current_action.append(np.array(action_index))
        reward.append(np.array(reward_value))
        next_state.append(np.array(next_position))

        # online training
        manager_som.action_q_learning(
            current_winner_index = som.select_winner(current_position.float()),
            next_winner_index = action_index,
            reward = reward_value,
            t = t)

        total_return += (0.9 ** t) * reward_value

    plt.plot(np.array(next_state)[:, 0], np.array(next_state)[:, 1], marker='.', linestyle='-', color='blue')
    plt.plot(0.0, 0.0, marker='v', linestyle='None', color='orange')
    plt.plot(np.array(task.goal)[0], np.array(task.goal)[1], marker='v', linestyle='None', color='red')
    plt.show()
    print(epoch, total_return)

    task.reset()
    current_state = []
    current_action = []
    reward = []
    next_state = []

import pickle
filehandler = open("data/som.obj", 'wb')
pickle.dump(som, filehandler)
