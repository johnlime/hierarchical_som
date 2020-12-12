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

class ManagerSOMPosition (KohonenSOM):

    """
    Input state vector:
    {current_state_indices, reward}

    Weight vector:
    {current_state_indices, return_estimates_per_action}
    """

    def __init__(self, total_nodes = None, state_som = None, worker_som = None, update_iterations = 100):
        self.state_som = state_som
        self.worker_som = worker_som

        node_size = 2 + self.worker_som.total_nodes # self.state_som.total_nodes + self.worker_som.total_nodes

        # self.w consists of a state_som-sized one hot encoder and a worker_som-sized categorical distribution
        super().__init__(total_nodes = total_nodes, node_size = node_size)

    def select_winner(self, x):
        # x must consist of the POSITION for current state indices
        return torch.argmin(torch.norm(torch.sqrt((x - self.w[:,:2]) ** 2), p=1, dim=1), dim=0)

    def get_action(self, x):
        return torch.argmax(self.w[self.select_winner(x)][-self.worker_som.total_nodes:], dim=0)

    def get_value(self, x):
        return torch.max(self.w[self.select_winner(x)][-self.worker_som.total_nodes:])[0]

    def get_softmax(self, x):
        return torch.max(f.softmax(self.w[
            self.select_winner(x)
            ][-self.worker_som.total_nodes:]))[0]

    def action_q_learning(self,
                        current_state_position = None,
                        action_index = None,
                        reward = 0,
                        next_state_position = None,
                        t = None,
                        htype = 0,
                        lr = 0.9,
                        gamma = 0.9):
        winner_c = self.select_winner(current_state_position)

        # update q-value using new reward and largest est. prob of action
        self.w[winner_c][2 + action_index] += lr * (
            reward
            + gamma * self.get_value(next_state_position)
            - self.w[winner_c][2 + action_index]
            )

        # update weights by neighboring the state spaces
        if htype==0:
            self.w[:, :2] += self.h0(winner_c, t) * (
                current_state_position
                - self.w[:, :2]
                )

        elif htype==1:
            self.w[:, :2] += self.h1(winner_c, t) * (
                current_state_position
                - self.w[:, :2]
                )

    def update(self, x=None, t=None, htype=0):
        raise NameError('Use action_q_learning() instead of update()')



class ManagerSOMPositionAllNeighbor (ManagerSOMPosition):
    def action_q_learning(self,
                        current_state_position = None,
                        action_index = None,
                        reward = 0,
                        next_state_position = None,
                        t = None,
                        htype = 0,
                        lr = 0.9,
                        gamma = 0.9):
        winner_c = self.select_winner(current_state_position)

        # update q-value using new reward and largest est. prob of action
        self.w[winner_c][2 + action_index] += lr * (
            reward
            + gamma * self.get_value(next_state_position)
            - self.w[winner_c][2 + action_index]
            )

        # update weights by neighboring both the state space and the rewards
        target_weights = torch.empty(2 + self.worker_som.total_nodes)
        target_weights[:2] = current_state_position
        target_weights[2:] = self.w[winner_c][2:]
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
