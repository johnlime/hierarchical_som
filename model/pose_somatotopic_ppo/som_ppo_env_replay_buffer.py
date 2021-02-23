from rlkit.torch.ppo.ppo_env_replay_buffer import PPOEnvReplayBuffer
import numpy as np

class SOMPPOEnvReplayBuffer(PPOEnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None,
    ):
        super().__init__(
                max_replay_buffer_size,
                env,
                env_info_sizes=None,
        )
        observation_dim = 2
        self._observation_dim = observation_dim
        # self._action_dim = action_dim
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        # self._actions = np.zeros((max_replay_buffer_size, action_dim))
