import torch
import math

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

class NavigationTaskV2():
    def __init__(self):
        self.current_position = torch.zeros(2)
        self.speed = 0.01
        self.goal = torch.tensor([0.5, 0.5]).float()

    def reset(self):
        self.current_position[0] = 0
        self.current_position[1] = 0

    def step(self, target):
        step_vector = (target - self.current_position) / torch.sqrt(torch.sum((target - self.current_position) ** 2))
        optimal_vector = (self.goal - self.current_position) / torch.sqrt(torch.sum((self.goal - self.current_position) ** 2))

        step_radian = math.asin(step_vector[1]) # positive or negative radian
        if math.acos(step_vector[0]) < 0:
            step_radian = step_radian / abs(step_radian) * math.pi - step_radian
            
        optimal_radian = math.asin(optimal_vector[1])
        if math.acos(optimal_vector[0]) < 0:
            optimal_radian = optimal_radian / abs(optimal_radian) * math.pi - optimal_radian

        reward = abs(step_radian - optimal_radian)
        if reward > math.pi:
            reward = 2 * math.pi - reward
        reward = math.pi / 4 - reward
        
        self.current_position += self.speed * step_vector

        return reward, self.current_position

    def state(self):
        return self.current_position
