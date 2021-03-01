from rlkit.torch.ppo.ppo_env_replay_buffer import PPOEnvReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np

class PosePPOEnvReplayBuffer(PPOEnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            pose_dim,
            env_info_sizes=None,
    ):
        super().__init__(
                max_replay_buffer_size,
                env,
                env_info_sizes=None,
        )
        self._observation_dim += pose_dim
        self._observations = np.zeros((max_replay_buffer_size, self._observation_dim))
        self._next_obs = np.zeros((max_replay_buffer_size, self._observation_dim))
