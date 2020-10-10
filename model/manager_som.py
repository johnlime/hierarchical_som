import torch
import torch.nn.functional as f
import math
import random
import kohonen_som

class ManagerSOM (KohonenSOM):
    
    """
    Input state vector:
    {current_state_indices, current_states, reward}

    Weight vector:
    {current_state_indices, current_states, return_estimates_for_next_state_indices}
    """
    
    def __init__(self, total_nodes = None, worker_som = None, additional_state_space = 0, update_iterations = 100):
        self.worker_som = worker_som
        self.state_indices = worker_som.total_nodes
        self.add_state = additional_state_space
        
        # self.w consists of a worker_som-sized one hot encoder and a worker_som-sized categorical distribution
        super().__init__(total_nodes = total_nodes, node_size = self.state_indices * 2 + self.add_state)   
        
    def select_winner(self, x):
        # x must consist of a one-hot vector for current state indices and additional states
        return torch.argmin(torch.norm(torch.sqrt((x - self.w[:, :self.state_indices + self.add_state]) ** 2), p=1, dim=1), dim=0)
    
    def get_action(self, x):
        return torch.argmax(self.w[self.select_winner(x)][self.state_indices + self.add_state:], dim=0)
    
    def get_softmax(self, x):
        return torch.max(f.softmax(self.w[
            self.select_winner(x)
            ][self.state_indices + self.add_state:]))[0]
        
    def action_q_learning(
            self, 
            current_winner_index = None, 
            additional_states = None,
            next_winner_index = None, # selected action
            reward = 0,
            t = None, 
            htype = 0,
            lr = 0.9,
            gamma = 0.9):
        current_winner = torch.zeros(self.worker_som.total_nodes)
        current_winner[current_winner_index] = 1
        
        current_state_space = None
        
        if additional_states != None:
            current_state_space = torch.cat((current_winner, additional_states))
            
        else:
            current_state_space = current_winner
            
        # update q-value using new reward and largest est. prob of action
        self.w[self.select_winner(current_state_space)][self.state_indices + self.add_state + next_winner_index] += lr * (
            reward 
            + gamma * self.get_softmax(current_state_space) 
            - self.w[self.select_winner(current_state_space)][self.state_indices + self.add_state + next_winner_index]
            )