import libraries.RlkitExtension.rlkit.torch.pytorch_util as ptu
from libraries.RlkitExtension.rlkit.torch.ppo.ppo_env_replay_buffer import PPOEnvReplayBuffer
from libraries.RlkitExtension.rlkit.envs.wrappers import NormalizedBoxEnv
from libraries.RlkitExtension.rlkit.launchers.launcher_util import setup_logger
from model.pose_somatotopic_ppo.path_collector import SOMPPOMdpPathCollector
from libraries.RlkitExtension.rlkit.torch.ppo.policies import DiscretePolicy, TanhGaussianPolicy, MakeDeterministic
from model.pose_somatotopic_ppo.trainer import SOMPPOTrainer
from libraries.RlkitExtension.rlkit.torch.networks import FlattenMlp
from libraries.RlkitExtension.rlkit.torch.ppo.ppo_torch_batch_rl_algorithm import PPOTorchBatchRLAlgorithm
from model.kohonen_som import KohonenSOM

import torch
import pickle

def experiment(variant):
    torch.autograd.set_detect_anomaly(True)
    expl_env = NormalizedBoxEnv(gym.make("Cartpole-v1"))
    eval_env = NormalizedBoxEnv(gym.make("Cartpole-v1"))
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    som_max_update_iterations = variant['algorithm_kwargs']['num_iter']
                            * variant['algorithm_kwargs']['num_eval_steps_per_epoch']
                            * variant['algorithm_kwargs']['um_trains_per_train_loop']

    M = variant['layer_size']
    vf = FlattenMlp(
        input_size=2,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = DiscretePolicy(
        obs_dim=2,
        action_dim=100,
        hidden_sizes=[M, M],
    )
    state_som = KohonenSOM(total_nodes=100, node_size=obs_dim, update_iterations=)
    worker_som = KohonenSOM(total_nodes=2, node_size=action_dim, update_iterations=)
    eval_policy = MakeDeterministic(policy)
    eval_step_collector = SOMPPOMdpPathCollector(
        eval_env,
        state_som,
        worker_som,
        eval_policy,
        calculate_advantages=False
    )
    expl_step_collector = SOMPPOMdpPathCollector(
        expl_env,
        state_som,
        worker_som,
        policy,
        calculate_advantages=True,
        vf=vf,
        gae_lambda=0.97,
        discount=0.995,
    )
    replay_buffer = PPOEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SOMPPOTrainer(
        env=eval_env,
        state_som=state_som,
        worker_som=worker_som,
        policy=policy,
        vf=vf,
        **variant['trainer_kwargs']
    )
    algorithm = PPOTorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_step_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    # noinspection PyTypeChecker
    T = 2048
    max_ep_len = 1000
    epochs = 10
    minibatch_size = 64

    variant = dict(
        algorithm="PPO",
        version="normal",
        layer_size=64,
        replay_buffer_size=T,
        algorithm_kwargs=dict(
            num_iter=int(1e6 // T),
            num_eval_steps_per_epoch=max_ep_len,
            num_trains_per_train_loop=T // minibatch_size * epochs,
            num_expl_steps_per_train_loop=T,
            min_num_steps_before_training=0,
            max_path_length=max_ep_len,
            minibatch_size=minibatch_size,
        ),
        trainer_kwargs=dict(
            epsilon=0.2,
            reward_scale=1.0,
            lr=3e-4,
        ),
    )
    setup_logger('PPOCartpole-v1', variant=variant)
    #ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)

    state_filehandler = open("data/pose_somatotopic/cartpole/ppo/state_som.obj", 'wb')
    pickle.dump(state_som, state_filehandler)
    worker_filehandler = open("data/pose_somatotopic/cartpole/ppo/worker_som.obj", 'wb')
    pickle.dump(worker_som, worker_filehandler)