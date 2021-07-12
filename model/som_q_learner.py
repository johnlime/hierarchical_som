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

class SOMQLearner (KohonenSOM):

    """
    Input state vector:
    {current_state, reward}

    Weight vector:
    {current_state, return_estimates_per_action}
    """

    def __init__(self, total_nodes = None, state_dim = None, action_som = None, update_iterations = 100):
        self.state_dim = state_dim
        self.action_som = action_som

        node_size = self.state_dim + self.action_som.total_nodes

        # self.w consists of a state_som-sized one hot encoder and a worker_som-sized categorical distribution
        super().__init__(total_nodes = total_nodes, node_size = node_size)

    def select_winner(self, x):
        x = torch.tensor(x)
        return torch.argmin(torch.norm(torch.sqrt((x - self.w[:, :self.state_dim])**2), p=1, dim=1), dim=0)

    def get_action(self, x):
        return torch.argmax(self.w[self.select_winner(x)][-self.action_som.total_nodes:], dim=0)

    def get_value(self, x):
        return torch.max(self.w[self.select_winner(x)][-self.action_som.total_nodes:])

    def get_softmax(self, x):
        return torch.max(f.softmax(self.w[
            self.select_winner(x)
            ][-self.action_som.total_nodes:]))[0]

    def action_q_learning(self,
                        current_state = None,
                        action_index = None,
                        reward = 0,
                        next_state = None,
                        t = None,
                        htype = 0,
                        lr = 0.9,
                        gamma = 0.9):
        winner_c = self.select_winner(current_state)

        # update q-value using new reward and largest est. prob of action
        self.w[winner_c][self.state_dim + action_index] += lr * (
            reward
            + gamma * self.get_value(next_state)
            - self.w[winner_c][self.state_dim + action_index]
            )

        # update weights by neighboring the state spaces
        if htype==0:
            self.w[:, :self.state_dim] += self.h0(winner_c, t) * (
                current_state - self.w[:, :self.state_dim]
                )

        elif htype==1:
            self.w[:, :self.state_dim] += self.h1(winner_c, t) * (
                current_state - self.w[:, :self.state_dim]
                )

    def update(self, x=None, t=None, htype=0):
        raise NameError('Use action_q_learning() instead of update()')



class SOMQLearnerAllNeighbor (SOMQLearner):
    def action_q_learning(self,
                        current_state = None,
                        action_index = None,
                        reward = 0,
                        next_state = None,
                        t = None,
                        htype = 0,
                        lr = 0.9,
                        gamma = 0.9):
        winner_c = self.select_winner(current_state)

        # update q-value using new reward and largest est. prob of action
        self.w[winner_c][self.state_dim + action_index] += lr * (
            reward
            + gamma * self.get_value(next_state)
            - self.w[winner_c][self.state_dim + action_index]
            )

        # update weights by neighboring both the state space and the rewards
        target_weights = torch.empty(self.state_dim + self.action_som.total_nodes)
        target_weights[:self.state_dim] = current_state
        target_weights[self.state_dim:] = self.w[winner_c][self.state_dim:]
        if htype==0:
            self.w += self.h0(winner_c, t) * (
                target_weights
                - self.w
                )

        elif htype==1:
            self.w += self.h1(winner_c, t) * (
                target_weights
                - self.w
                )
