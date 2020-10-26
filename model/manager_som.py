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
    {current_state_indices, additional_state_indices, additional_states reward}

    Weight vector:
    {current_state_indices, additional_state_indices, additional_states, return_estimates_per_action}
    """

    def __init__(self, total_nodes = None, worker_som = None, additional_state_space = 0, update_iterations = 100):
        # array of worker_som references and their total nodes
        # first index should be the PID action space
        self.worker_som = worker_som
        self.state_indices = []
        self.add_state = additional_state_space

        node_size = self.worker_som[0].total_nodes + self.add_state # action and additional states
        # calculate state space size
        for i in range(len(self.worker_som)):
            self.state_indices.append(self.worker_som[i].total_nodes)
            node_size += self.state_indices[i]

        # self.w consists of a worker_som-sized one hot encoder and a worker_som-sized categorical distribution
        super().__init__(total_nodes = total_nodes, node_size = node_size)

    def select_winner(self, x):
        # x must consist of a one-hot vector for current state indices, additional SOM indices and additional states
        return torch.argmin(torch.norm(torch.sqrt((x - self.w[:,: - self.state_indices[0]]) ** 2), p=1, dim=1), dim=0)

    def get_action(self, x):
        return torch.argmax(self.w[self.select_winner(x)][- self.state_indices[0]:], dim=0)

    def get_softmax(self, x):
        return torch.max(f.softmax(self.w[
            self.select_winner(x)
            ][- self.state_indices[0]:]))[0]

    def action_q_learning(
            self,
            current_winner_indices = None, # should be an array input
            additional_states = None,
            next_winner_index = None, # selected action
            reward = 0,
            t = None,
            htype = 0,
            lr = 0.9,
            gamma = 0.9):
        current_winner = torch.zeros(self.node_size - self.state_indices[0])
        prev_index = 0
        for i in range(len(current_winner_indices)):
            current_winner[prev_index: prev_index + self.state_indices[i]][current_winner_indices[i]] = 1
            prev_index += self.state_indices[i]

        current_state_space = None

        if self.add_state != 0:
            current_state_space = torch.cat((current_winner, torch.tensor(additional_states).float()))

        else:
            current_state_space = current_winner

        winner_c = self.select_winner(current_state_space)

        # update q-value using new reward and largest est. prob of action
        self.w[winner_c][- self.state_indices[0] + next_winner_index] += lr * (
            reward
            + gamma * self.get_softmax(current_state_space)
            - self.w[winner_c][- self.state_indices[0] + next_winner_index]
            )

        part_x = torch.cat((current_state_space, self.w[winner_c][- self.state_indices[0] :]))

        if htype==0:
            self.w[:, : -self.state_indices[0]] += self.h0(winner_c, t) * (
                current_state_space
                - self.w[:, : -self.state_indices[0]]
                )

            # # reward neighboring
            # self.w[:, next_winner_index - self.state_indices[0]] += torch.transpose(self.h0(winner_c, t), 0, 1)[0] * (
            #     self.w[winner_c][next_winner_index - self.state_indices[0]]
            #     - self.w[:, next_winner_index - self.state_indices[0]]
            #     )

        elif htype==1:
            self.w[:, : -self.state_indices[0]] += self.h1(winner_c, t) * (
                current_state_space
                - self.w[:, : -self.state_indices[0]]
                )

            # # reward neighboring
            # self.w[:, next_winner_index - self.state_indices[0]] += torch.transpose(self.h1(winner_c, t), 0, 1)[0] * (
            #     self.w[winner_c][next_winner_index - self.state_indices[0]]
            #     - self.w[:, next_winner_index - self.state_indices[0]]
            #     )

    def update(self, x=None, t=None, htype=0):
        raise NameError('Use action_q_learning() instead of update()')
        return None, None
