import torch

class NavigationTask():
    def __init__(self):
        self.current_position = torch.zeros(2)
        self.speed = 0.01
        self.goal = torch.tensor([0.5, 0.5]).float()
        
    def reset(self):
        self.current_position[0] = 0
        self.current_position[1] = 0
        
    def step(self, target):
        self.current_position += self.speed * ((target - self.current_position) / torch.sqrt(torch.sum((target - self.current_position) ** 2)))
        reward = -torch.sum((self.goal - self.current_position) ** 2)
        return reward, self.current_position
    
    def state(self):
        return self.current_position