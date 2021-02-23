from rlkit.torch.ppo.ppo_env_replay_buffer import PPOEnvReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np

class SOMPPOEnvReplayBuffer(PPOEnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            action_dim,
            som_action_dim=None,
            env_info_sizes=None,
    ):
        super().__init__(
                max_replay_buffer_size,
                env,
                env_info_sizes=None,
        )
        observation_dim = 2
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))

        self._som_observations = np.zeros((max_replay_buffer_size, get_dim(self.env.observation_space)))
        if som_action_dim == None:
            self._som_actions = np.zeros((max_replay_buffer_size, get_dim(self.env.action_space)))
        else:
            self._som_actions = np.zeros((max_replay_buffer_size, som_action_dim))

    def add_path(self, path):
        for i, (
                obs,
                action,
                reward,
                next_obs,
                som_obs,
                som_action,
                terminal,
                agent_info,
                env_info,
                advantage,
                returns,
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["som_observations"],
            path["som_actions"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
            path["advantages"],
            path["returns"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                som_observation=som_obs,
                som_action=som_action,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
                advantage=advantage,
                returns=returns
            )
        self.terminate_episode()

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, agent_info, advantage, returns,
                   som_observation, som_action, **kwargs):

        self._som_observations[self._top] = som_observation
        self._som_actions[self._top] = som_action

        return super().add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            agent_info=agent_info,
            advantage=advantage,
            returns=returns,
            **kwargs
        )

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            log_prob=self._log_prob[indices],
            advantage=self._advantage[indices],
            returns=self._return[indices],
            som_observations=self._som_observations[indices],
            som_actions=self._som_actions[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def end_epoch(self, epoch):
        self._som_observations.fill(0)
        self._som_actions.fill(0)
        super().end_epoch(epoch)
