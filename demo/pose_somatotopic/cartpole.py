import gym
import torch
from model.kohonen_som import KohonenSOM

env = gym.make("CartPole-v1")
observation = env.reset()
maxitr = 10 ** 3
state_som = KohonenSOM(total_nodes=100, node_size=4, update_iterations=maxitr)
worker_som = KohonenSOM(total_nodes=2, node_size=1, update_iterations=maxitr)

sampled_length = 0
sample_iter = 100
pose_pool = torch.empty(sample_iter * 200, 4)
action_pool = torch.empty(sample_iter * 200, 1)

for __ in range(sample_iter):
    for _ in range(200):
        action = env.action_space.sample() # random actions for exploration
        action_pool[sampled_length][0] = action
        obs, reward, done, info = env.step(action)
        pose_pool[sampled_length] = torch.tensor(obs)
        sampled_length += 1

        if done:
            # sampled_length is also saved in addition
            obs = env.reset()
            break
env.close()

for iteration in range(maxitr):
    state_som.update(pose_pool[:sampled_length], iteration)
    worker_som.update(action_pool[:sampled_length], iteration)

import pickle
state_filehandler = open("data/pose_somatotopic/state_som.obj", 'wb')
pickle.dump(state_som, state_filehandler)
worker_filehandler = open("data/pose_somatotopic/worker_som.obj", 'wb')
pickle.dump(worker_som, worker_filehandler)
