from custom_env.cartpole_clone import CartPoleEnv

class CartPoleEnvReal (CartPoleEnv):
    def step(self, action):
        if action >= 0.5:
            action = 1
        else:
            action = 0
        return super().step(action)
