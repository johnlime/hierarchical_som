import torch
import math
import random

class KohonenSOM:
    def __init__(self, total_nodes=None, node_size=2, update_iterations=100):
        self.total_nodes = total_nodes
        self.node_size = node_size
        self.tau = update_iterations
        dim = int(math.sqrt(self.total_nodes))
        self.w = torch.randn(self.total_nodes, self.node_size)
        self.location = torch.empty(self.total_nodes, 2)
        for i in range(dim):
            for j in range(dim):
                self.location[i*dim+j][0] = i
                self.location[i*dim+j][1] = j
        self.w /= torch.norm(self.w)

    def select_winner(self, x):
        x = torch.tensor(x)
        return torch.argmin(torch.norm(torch.sqrt((x - self.w)**2), p=1, dim=1), dim=0)

    def select_winner_by_location(self, x):
        x = torch.tensor(x)
        return torch.argmin(torch.norm(torch.sqrt((x - self.location[:,:2]) ** 2), p=1, dim=1), dim=0)

    def sigma(self, t):
        s0 = 0.9
        return s0 * math.exp(- t / self.tau)

    def h0(self, c, t): # neighborhood function
        # As t gets larger, less neurons are selected
        rangen = int((self.total_nodes / 2 - (self.total_nodes / 2 - 1) * t / self.tau)) # range of neighboring nodes
        chosen = torch.zeros(self.total_nodes)

        # Sort neurons by closest in weight distances to input
        chosen_indices = torch.sort(torch.norm((self.location[c] - self.location), p=2, dim=1), descending=False)[1][:rangen]
        for i in chosen_indices:
            chosen[i] = 1

        # vector of which competitive neurons are updated
        return 1 / math.sqrt(t + 1) * chosen.reshape(self.total_nodes, 1)

    def h1(self, c, t): # deciding neighbors by distance
        return 1 / math.sqrt(t + 1) * torch.exp(-1 / 2 * torch.norm((self.location[c] - self.location), p=2, dim=1).reshape(self.total_nodes, 1) / self.sigma(t) ** 2)

    def update(self, x=None, t=None, htype=0):
        x = x.float() # Resizing input to a compatible dimension
        index = random.randint(0, x.size()[0] - 1) # Choose one input index
        part_x = x[index] # Extract input vector
        c = self.select_winner(part_x) # winner index

        if htype==0:
            self.w += self.h0(c, t) * (part_x - self.w)
        elif htype==1:
            self.w += self.h1(c, t) * (part_x - self.w)

        return index, c
