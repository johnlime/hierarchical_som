import torch
import torch.nn.functional as f
import math
import random
try:
    import kohonen_som
except:
    print("path exception")
finally:
    from model.kohonen_som import KohonenSOM

class ManagerSOM (KohonenSOM):

    """
    Input state vector:
    {current_state_indices, reward}

    Weight vector:
    {current_state_indices, return_estimates_per_action}
    """

    def __init__(self, total_nodes = None, state_som = None, worker_som = None, update_iterations = 100):
        self.state_som = state_som
        self.worker_som = worker_som

        node_size = self.state_som.total_nodes + self.worker_som.total_nodes

        # self.w consists of a state_som-sized one hot encoder and a worker_som-sized categorical distribution
        super().__init__(total_nodes = total_nodes, node_size = node_size)

    def select_winner(self, x):
        # x must consist of a one-hot vector for current state indices
        return torch.argmin(torch.norm(torch.sqrt((x - self.w[:,:self.state_som.total_nodes]) ** 2), p=1, dim=1), dim=0)

    def get_action(self, x):
        return torch.argmax(self.w[self.select_winner(x)][-self.worker_som.total_nodes:], dim=0)

    def get_value(self, x):
        return torch.max(self.w[self.select_winner(x)][-self.worker_som.total_nodes:])[0]
    
    def get_softmax(self, x):
        return f.softmax(self.w[self.select_winner(x)][-self.worker_som.total_nodes:])[torch.argmax(x)]

    def action_q_learning(self,
                        current_state_index = None,
                        action_index = None,
                        reward = 0,
                        next_state_index = None,
                        t = None,
                        htype = 0,
                        lr = 0.9,
                        gamma = 0.9):
        current_state_space = torch.zeros(self.state_som.total_nodes)
        current_state_space[current_state_index] = 1

        next_state_space = torch.zeros(self.state_som.total_nodes)
        next_state_space[next_state_index] = 1

        winner_c = self.select_winner(current_state_space)

        # update q-value using new reward and largest est. prob of action
        self.w[winner_c][self.state_som.total_nodes + action_index] += lr * (
            reward
            + gamma * self.get_value(next_state_space)
            - self.w[winner_c][self.state_som.total_nodes + action_index]
            )

        # update weights by neighboring the state spaces
        if htype==0:
            self.w[:, :self.state_som.total_nodes] += self.h0(winner_c, t) * (
                current_state_space
                - self.w[:, :self.state_som.total_nodes]
                )

        elif htype==1:
            self.w[:, :self.state_som.total_nodes] += self.h1(winner_c, t) * (
                current_state_space
                - self.w[:, :self.state_som.total_nodes]
                )

    def update(self, x=None, t=None, htype=0):
        raise NameError('Use action_q_learning() instead of update()')
