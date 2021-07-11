import numpy as np

def som_rollout(
        env,
        worker_som,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    if render_kwargs is None:
        render_kwargs = {}

    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    agent_infos = []
    env_infos = []

    som_actions = []

    o = env.reset()
    agent.reset()
    next_som_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        a = np.argmax(a)
        som_a = worker_som.w[a]
        next_som_o, r, d, env_info = env.step(som_a)

        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)

        som_actions.append(som_a)

        path_length += 1

        if max_path_length == np.inf and d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,

        som_actions=som_actions,
    )
