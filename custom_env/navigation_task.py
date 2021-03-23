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

        # Angle between two vectors in 2D space can be determined via dot product and the magnitude of the two vectors
        # Magnitude of the two vectors are already normalized
        dot_product = torch.dot(step_vector, optimal_vector)
        if (dot_product > 1): 
            dot_product = 1
        elif (dot_product < -1):
            dot_product = -1
        dif_radian = math.acos(dot_product)

        reward = math.pi - dif_radian

        self.current_position += self.speed * step_vector

        return reward, self.current_position

    def state(self):
        return self.current_position

    
class NavigationTaskDirectional(NavigationTask):
    def step(self, step_radian):
        optimal_vector = (self.goal - self.current_position) / torch.sqrt(torch.sum((self.goal - self.current_position) ** 2))

        if step_radian >= 2.0 * math.pi or step_radian < 0:
            raise Exception("step_radian should be 0 <= x < 2pi")

        optimal_radian = math.acos(optimal_vector[1])
        if math.asin(optimal_vector[0]) < 0:
            optimal_radian = 2 * math.pi - optimal_radian

        reward = abs(step_radian - optimal_radian)
        if reward > math.pi:
            reward = 2 * math.pi - reward

        self.current_position += self.speed * step_vector

        return reward, self.current_position
    

class NavigationTaskMultiTarget(NavigationTask):
    def __init__(self):
        super().__init__()
        self.all_goals = torch.tensor([[0.4, 0.2], [0.5, 0.5]]).float()
        self.goal_completed = [False, False]
        self.current_goal_index = 0
        self.goal = self.all_goals[self.current_goal_index]
        
    def reset(self):
        super().reset()
        self.goal_completed = [False, False]
        self.current_goal_index = 0
        self.goal = self.all_goals[self.current_goal_index]
        
    def step(self, target):
        reward, return_position = super().step(target)
        
        if torch.sqrt(torch.sum((self.goal - self.current_position) ** 2)) < 0.1:
            self.goal_completed[self.current_goal_index] = True
            self.current_goal_index += 1
            if self.current_goal_index < self.all_goals.shape[0]:
                self.goal = self.all_goals[self.current_goal_index]
            else:
                pass
        
        return reward, return_position
    