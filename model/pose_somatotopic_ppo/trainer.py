from libraries.RlkitExtension.rlkit.torch.ppo.ppo import PPOTrainer

class SOMPPOTrainer(PPOTrainer):
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
        

        # obs = batch['observations']
        # old_log_pi = batch['log_prob']
        # advantage = batch['advantage']
        # returns = batch['returns']
        # actions = batch['actions']
        #
        # self._n_train_steps_total += 1

        super().train_from_torch(self, batch)
