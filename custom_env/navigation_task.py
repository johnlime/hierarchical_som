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

    
class NavigationTaskDirectional(NavigationTask):
    def step(self, step_radian):
        optimal_vector = (self.goal - self.current_position) / torch.sqrt(torch.sum((self.goal - self.current_position) ** 2))

        if step_radian >= 2.0 * math.pi or step_radian < 0:
            raise Exception("step_radian should be 0 <= x < 2pi")

        optimal_radian = math.asin(optimal_vector[1])
        if math.acos(optimal_vector[0]) < 0:
            optimal_radian = optimal_radian / abs(optimal_radian) * math.pi - optimal_radian

        reward = abs(step_radian - optimal_radian)
        if reward > math.pi:
            reward = 2 * math.pi - reward
        reward = math.pi / 4 - reward

        self.current_position += self.speed * step_vector

        return reward, self.current_position
    

class NavigationTaskMultiTarget(NavigationTask):
    def __init__(self):
        super().__init__()
        self.all_goals = torch.tensor([[0.5, 0.5], [0.4, 0.2]]).float()
        self.current_goal_index = 0
        self.goal = self.all_goals[self.current_goal_index]
        
    def step(target):
        reward, return_position = super().step(target)
        
        if torch.sqrt(torch.sum((self.goal - self.current_position) ** 2)) < 0.05:
            self.current_goal_index += 1
            if self.current_goal_index < self.all_goals.shape[0]:
                self.goal = self.all_goals[self.current_goal_index]
            else:
                pass
        
    