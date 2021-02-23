from libraries.RlkitExtension.rlkit.torch.ppo.discrete_ppo import DiscretePPOTrainer
import torch.optim as optim

class SOMPPOTrainer(DiscretePPOTrainer):
    def __init__(
            self,
            env,
            state_som,
            worker_som,
            policy,
            vf,

            epsilon=0.05,
            reward_scale=1.0,

            lr=1e-3,
            optimizer_class=optim.Adam,

            plotter=None,
            render_eval_paths=False,
    ):
        self.state_som = state_som
        self.worker_som = worker_som
        self.epoch = 0

        super().__init__(
                env,
                policy,
                vf,

                epsilon=0.05,
                reward_scale=1.0,

                lr=1e-3,
                optimizer_class=optim.Adam,

                plotter=None,
                render_eval_paths=False
        )

    def train_from_torch(self, batch):
        """
        (next_)observations: 2-dimensional position representation of state_som
        actions: One-hot vector representation of worker_som
        advantage: Advantage for each obs-action pairs

        som_obervations: Original Gym observation
        som_actions: Original Gym actions
        """
        som_obs = batch["som_observations"]
        som_actions = batch["som_actions"]

        self.state_som.update(som_obs, self.epoch)
        self.worker_som.update(som_actions, self.epoch)
        self.epoch += 1

        super().train_from_torch(batch)
