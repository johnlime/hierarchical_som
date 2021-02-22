from libraries.RlkitExtension.rlkit.torch.ppo.ppo_path_collector import PPOMdpPathCollector
from model.pose_somatotopic_ppo.rollout import som_rollout

class SOMPPOMdpPathCollector (PPOMdpPathCollector):
    def __init__(
            self,
            env,
            state_som,
            worker_som,
            policy,

            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,

            calculate_advantages = False,
            vf = None,
            discount=0.99,
            gae_lambda=0.95
    ):
        self._state_som = state_som
        self._worker_som = worker_som

        super().__init__(
            env,
            policy,

            max_num_epoch_paths_saved=max_num_epoch_paths_saved,
            render=render,
            render_kwargs=render_kwargs,

            calculate_advantages = calculate_advantages,
            vf = vf,
            discount=discount,
            gae_lambda=gae_lambda
        )

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = som_rollout(
                self._env,
                self._state_som,
                self._worker_som,
                self._policy,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])

            # calculate advantages and add column to path
            path = self.add_advantages(path, path_len, self.calculate_advantages)

            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths
