import os

import mlflow
import ray
import torch
from pettingzoo.mpe import simple_spread_v3
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.schedulers import ASHAScheduler

from ma_poca.algorithm import MAPOCAConfig


def create_env(config):
    """PettingZoo環境を作成するためのラッパー関数。"""
    # AEC API (env) を使用すると、RLlibラッパーとの互換性が高まります
    env = simple_spread_v3.env(
        N=config.get("N", 3),
        local_ratio=config.get("local_ratio", 0.5),
        max_cycles=config.get("max_cycles", 25),
    )
    return env


def train_ma_poca(config, checkpoint_dir=None):
    """ハイパーパラメーターチューニング用のMA-POCA学習関数。"""
    # カスタム環境作成関数を登録します。
    tune.register_env(
        "mpe_simple_spread", lambda env_config: PettingZooEnv(create_env(env_config))
    )

    # 行動空間と観測空間を抽出するために環境を取得します。
    temp_env = PettingZooEnv(create_env(config))
    temp_env.reset()
    # 共有ポリシーの場合、個々のエージェントの空間を使用します。
    agent_ids = list(temp_env.get_agent_ids())
    agent_id = agent_ids[0]
    # 各エージェントの観測空間と行動空間を使用するように修正
    obs_space = temp_env.observation_space[agent_id]
    act_space = temp_env.action_space[agent_id]

    # MA-POCAアルゴリズムを設定します。
    ma_poca_config = (
        MAPOCAConfig()
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .environment(env="mpe_simple_spread", env_config=config)
        .framework("torch")
        .env_runners(num_env_runners=0)
        .multi_agent(
            policies={
                "shared_policy": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=(lambda agent_id, episode, **kwargs: "shared_policy"),
        )
    )

    # 学習パラメーターを設定します
    ma_poca_config.training(
        gamma=config.get("gamma", 0.99),
        lr=config.get("lr", 1e-4),
        grad_clip=config.get("grad_clip", 40.0),
        train_batch_size=config.get("train_batch_size", 4096),
    )

    # PPO固有のパラメーターを直接属性として設定します
    ma_poca_config.lambda_ = config.get("lambda_", 0.9)
    ma_poca_config.clip_param = config.get("clip_param", 0.2)
    ma_poca_config.vf_loss_coeff = config.get("vf_loss_coeff", 0.5)
    ma_poca_config.entropy_coeff = config.get("entropy_coeff", 0.01)
    ma_poca_config.sgd_minibatch_size = config.get("sgd_minibatch_size", 128)
    ma_poca_config.num_sgd_iter = config.get("num_sgd_iter", 10)
    ma_poca_config.normalize_advantages = config.get("normalize_advantages", True)

    # カスタムモデル設定を更新します
    if "custom_model_config" not in ma_poca_config.model:
        ma_poca_config.model["custom_model_config"] = {}
    ma_poca_config.model["custom_model_config"]["max_agents"] = len(agent_ids)

    # GPUが利用可能な場合はGPUを使用する設定に変更
    ma_poca_config.resources(num_gpus=1 if torch.cuda.is_available() else 0)
    ma_poca_config.debugging(log_level="INFO")

    # アルゴリズムをビルドします。
    algo = ma_poca_config.build()

    # 学習ループ。
    # 試行回数を減らして早期に終了するように修正
    for _i in range(config.get("training_iterations", 2)):  # 2回に変更
        result = algo.train()
        # 結果からエピソード報酬平均を抽出します
        episode_reward_mean = result.get("env_runners", {}).get("episode_reward_mean")
        if episode_reward_mean is None:
            # ラーナー統計から取得を試みます
            episode_reward_mean = (
                result.get("info", {})
                .get("learner", {})
                .get("shared_policy", {})
                .get("learner_stats", {})
                .get("episode_reward_mean")
            )

        # それでもNoneの場合は、一貫した出力のためにnanを設定します
        if episode_reward_mean is None:
            episode_reward_mean = float("nan")

        # Tuneにメトリクスを報告します
        # episode_reward_meanをmetricsディクショナリに含めるように修正
        tune.report(metrics={"episode_reward_mean": episode_reward_mean})

    # クリーンアップ。
    algo.stop()


def main():
    # Rayを初期化します。
    ray.init(num_cpus=4, include_dashboard=False)

    # MLflowトラッキングを設定します。
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ma-poca-hpo-experiment")

    # ハイパーパラメーターの探索空間を定義します
    config = {
        # 環境パラメーター
        "N": tune.choice([3]),  # 3に固定
        "local_ratio": tune.uniform(0.5, 0.5),  # 0.5に固定
        "max_cycles": tune.choice([10]),  # 10に変更
        # 学習パラメーター
        "gamma": tune.uniform(0.99, 0.99),  # 0.99に固定
        "lr": tune.loguniform(1e-4, 1e-4),  # 1e-4に固定
        "lambda_": tune.uniform(0.9, 0.9),  # 0.9に固定
        "vf_loss_coeff": tune.uniform(0.5, 0.5),  # 0.5に固定
        "entropy_coeff": tune.loguniform(0.01, 0.01),  # 0.01に固定
        "clip_param": tune.uniform(0.2, 0.2),  # 0.2に固定
        "train_batch_size": tune.choice([2048]),  # 2048に変更
        # PPO固有のパラメーター
        "sgd_minibatch_size": tune.choice([64]),  # 64に変更
        "num_sgd_iter": tune.choice([5]),  # 5に変更
        "grad_clip": tune.uniform(40.0, 40.0),  # 40.0に固定
        # その他のパラメーター
        "training_iterations": 2,  # 2に変更
    }

    # スケジューラーを定義します（metricとmodeはTunerで指定するため削除）
    scheduler = ASHAScheduler(
        max_t=2,  # 2に変更
        grace_period=1,
        reduction_factor=2,
    )

    # 絶対パスを取得します
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "..", "results")

    # ハイパーパラメーターチューニングを実行します
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_ma_poca),
            resources={"cpu": 1, "gpu": 0 if not torch.cuda.is_available() else 1},
        ),
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=scheduler,
            num_samples=2,  # 2に変更
        ),
        param_space=config,
        run_config=tune.RunConfig(
            name="ma_poca_hpo",
            storage_path=results_dir,  # 絶対パスを使用
            stop={"training_iteration": 2},  # 2に変更
        ),
    )

    results = tuner.fit()

    # 最良の結果を出力します
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"最良の試行設定: {best_result.config}")
    metrics = best_result.metrics or {}
    print(
        "最良の試行最終エピソード報酬平均: {}".format(
            metrics.get("episode_reward_mean", "N/A")
        )
    )

    # Rayをシャットダウンします
    ray.shutdown()


if __name__ == "__main__":
    main()
