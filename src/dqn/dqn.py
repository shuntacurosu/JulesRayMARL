from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


def get_idqn_config(env: ParallelPettingZooEnv, framework: str = "torch") -> DQNConfig:
    """
    IDQN (Independent DQN) の設定を生成します。
    各エージェントが独立したポリシー（モデル）を持ちます。
    """
    policies = {
        agent_id: PolicySpec(
            observation_space=env.observation_space[agent_id],
            action_space=env.action_space[agent_id],
        )
        for agent_id in env.get_agent_ids()
    }

    policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id

    config = (
        DQNConfig()
        .environment(env=env)
        .framework(framework)
        .env_runners(num_env_runners=0, batch_mode="complete_episodes")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size=256,
            hiddens=[64, 64],
            dueling=False,
            double_q=False,
            replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 50000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
            },
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )
    return config


def get_shared_dqn_config(
    env: ParallelPettingZooEnv, framework: str = "torch"
) -> DQNConfig:
    """
    Shared DQN の設定を生成します。
    すべてのエージェントが単一の共有ポリシー（モデル）を使用します。
    """
    agent_id = env.get_agent_ids().pop()
    shared_policy = PolicySpec(
        observation_space=env.observation_space[agent_id],
        action_space=env.action_space[agent_id],
    )

    policy_mapping_fn = lambda agent_id, *args, **kwargs: "shared_policy"

    config = (
        DQNConfig()
        .environment(env=env)
        .framework(framework)
        .env_runners(num_env_runners=0, batch_mode="complete_episodes")
        .multi_agent(
            policies={"shared_policy": shared_policy},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size=256,
            hiddens=[64, 64],
            dueling=False,
            double_q=False,
            replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 50000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
            },
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )
    return config
