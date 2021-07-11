from libraries.RlkitExtension.rlkit.torch.ppo.discrete_ppo import DiscretePPOTrainer
import torch.optim as optim

class SOMPPOTrainer(DiscretePPOTrainer):
    def __init__(
            self,
            env,
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
        actions: One-hot vector representation of worker_som
        advantage: Advantage for each obs-action pairs

        obervations: Original Gym observation
        som_actions: Action SOM Output (Original Gym actions)
        """
        som_actions = batch["som_actions"]

        self.worker_som.update(som_actions, self.epoch)
        self.epoch += 1

        super().train_from_torch(batch)
