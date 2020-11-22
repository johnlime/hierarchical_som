import gym
import torch
from model.kohonen_som import KohonenSOM

env = gym.make("CartPole-v1")
observation = env.reset()
maxitr = 10 ** 3
worker_som = KohonenSOM(total_nodes=100, node_size=2, update_iterations=maxitr)
state_som = KohonenSOM(total_nodes=100, node_size=4, update_iterations=maxitr)

sampled_length = 0
sample_iter = 100
worker_pool = torch.empty(sample_iter * 200, 2)
state_pool = torch.empty(sample_iter * 200, 4)

for __ in range(sample_iter):
    for _ in range(200):
        action = env.action_space.sample() # random actions for exploration
        obs, reward, done, info = env.step(action)
        worker_pool[sampled_length] = torch.tensor(obs[:2])
        state_pool[sampled_length] = torch.tensor(obs)
        sampled_length += 1

        if done:
            # sampled_length is also saved in addition
            obs = env.reset()
            break
env.close()

for iteration in range(maxitr):
    worker_som.update(worker_pool[:sampled_length], iteration)
    state_som.update(state_pool[:sampled_length], iteration)

import pickle
worker_filehandler = open("data/smc_premotor_pid/worker_som.obj", 'wb')
pickle.dump(worker_som, worker_filehandler)
state_filehandler = open("data/smc_premotor_pid/state_som.obj", 'wb')
pickle.dump(state_som, state_filehandler)
