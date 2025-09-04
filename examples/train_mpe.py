import mlflow
import ray
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.mpe import simple_spread_v3

from src.ma_poca.algorithm import MAPOCAConfig

def create_env(config):
    """Wrapper function to create the PettingZoo environment."""
    # Using the AEC API (env) is more compatible with the RLlib wrapper
    env = simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=25)
    return env

def main():
    # Register the custom environment creator function.
    tune.register_env("mpe_simple_spread", lambda config: PettingZooEnv(create_env(config)))

    # Initialize Ray.
    ray.init(num_cpus=4, include_dashboard=False)

    # Get the environment to extract action and observation spaces.
    temp_env = PettingZooEnv(create_env({}))
    # For a shared policy, we use the individual agent's space.
    obs_space = temp_env.observation_space['agent_0']
    act_space = temp_env.action_space['agent_0']
    agent_ids = temp_env.possible_agents

    # Configure the MA-POCA algorithm.
    config = (
        MAPOCAConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .environment(env="mpe_simple_spread")
        .framework("torch")
        .env_runners(num_env_runners=0)
        .multi_agent(
            policies={
                "shared_policy": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=(lambda agent_id, episode, **kwargs: "shared_policy"),
        )
        .training(
            gamma=0.99,
            lr=1e-5,
            lambda_=0.9,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
            clip_param=0.2,
            train_batch_size=4096,
        )
        .resources(num_gpus=0)
        .debugging(log_level="INFO")
    )

    # In this version of RLlib, some PPO-specific parameters must be set as direct
    # attributes on the config object, not inside the .training() method.
    config.sgd_minibatch_size = 128
    config.num_sgd_iter = 10
    config.normalize_advantages = True
    config.grad_clip = 40.0

    # Update the custom model config
    config.model["custom_model_config"]["max_agents"] = len(agent_ids)

    # Set up MLflow tracking.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ma-poca-mpe-experiment")

    # Build the algorithm.
    algo = config.build()

    # Training loop.
    print("Starting training...")
    with mlflow.start_run(run_name="ma_poca_run") as parent_run:
        for i in range(10):
            result = algo.train()
            print(f"Iteration: {i+1}, Episode Reward Mean: {result.get('episode_reward_mean', float('nan'))}")

            # Log metrics to MLflow in a nested run for clarity
            with mlflow.start_run(run_name=f"iter_{i+1}", nested=True):
                metrics_to_log = {}
                metrics_to_log["episode_reward_mean"] = result.get("episode_reward_mean")

                learner_stats = result.get("info", {}).get("learner", {}).get("shared_policy", {}).get("learner_stats", {})
                metrics_to_log["policy_loss"] = learner_stats.get("total_loss")
                metrics_to_log["vf_loss"] = learner_stats.get("vf_loss")
                metrics_to_log["entropy"] = learner_stats.get("entropy")

                # Filter out None values before logging
                filtered_metrics = {k: v for k, v in metrics_to_log.items() if v is not None}
                mlflow.log_metrics(filtered_metrics, step=result["training_iteration"])

    print("Training finished.")

    # Clean up.
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()
