import gym
import torch
from model.kohonen_som import KohonenSOM

env = gym.make("CartPole-v1")
observation = env.reset()
maxitr = 10 ** 3
som_pos = KohonenSOM(total_nodes=10, node_size=2, update_iterations=maxitr)
som_ang = KohonenSOM(total_nodes=10, node_size=2, update_iterations=maxitr)

sampled_length = 0
sample_iter = 100
pose_pool = torch.empty(sample_iter * 200, 2)
angle_pool = torch.empty(sample_iter * 200, 2)

for __ in range(sample_iter):
    for _ in range(200):
        action = env.action_space.sample() # random actions for exploration
        obs, reward, done, info = env.step(action)
        pose_pool[sampled_length][0] = obs[0]
        pose_pool[sampled_length][1] = obs[1]
        angle_pool[sampled_length][0] = obs[2]
        angle_pool[sampled_length][1] = obs[3]
        sampled_length += 1

        if done:
            # sampled_length is also saved in addition
            obs = env.reset()
            break
env.close()

for iteration in range(maxitr):
    som_pos.update(pose_pool[:sampled_length], iteration)
    som_ang.update(angle_pool[:sampled_length], iteration)

import pickle
pos_filehandler = open("data/som_pos.obj", 'wb')
pickle.dump(som_pos, pos_filehandler)
ang_filehandler = open("data/som_ang.obj", 'wb')
pickle.dump(som_ang, ang_filehandler)
