import gym
import torch
from model.kohonen_som import KohonenSOM

env = gym.make("CartPole-v1")
observation = env.reset()
maxitr = 10 ** 3
som = KohonenSOM(total_nodes=10, node_size=2, update_iterations=maxitr)

sampled_length = 0
sample_iter = 100
pose_pool = torch.empty(sample_iter * 200, 2)

for __ in range(sample_iter):
    for _ in range(200):
        action = env.action_space.sample() # random actions for exploration
        observation, reward, done, info = env.step(action)
        pose_pool[sampled_length][0] = observation[0]
        pose_pool[sampled_length][1] = observation[1]
        sampled_length += 1

        if done:
            # sampled_length is also saved in addition
            observation = env.reset()
            break
env.close()

for iteration in range(maxitr):
    som.update(pose_pool[:sampled_length], iteration)

import pickle
filehandler = open("data/som.obj", 'wb')
pickle.dump(som, filehandler)
