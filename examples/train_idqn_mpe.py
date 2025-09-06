import ray
from ray import tune

from dqn.dqn import get_idqn_config
from dqn.utils import create_env


def main():
    """IDQNの学習を実行するメイン関数。"""
    # MPE環境を指定
    # 利用可能な環境: "simple_v3", "simple_spread_v3"
    env_name = "simple_spread_v3"

    # RLlibのAlgo/Tune APIは環境名を直接受け取ることを想定しているため、
    # env_creatorを登録する
    tune.register_env(env_name, lambda config: create_env(env_name, **config))

    # IDQN用の設定を取得
    # 環境インスタンスを渡して、ポリシー定義に必要な情報を取得させる
    config = get_idqn_config(create_env(env_name))

    # 環境名をコンフィグに設定
    config.environment(env=env_name)

    # 結果の出力設定
    # ログは ~/ray_results/idqn_simple_spread_v3 に保存される
    # TensorBoardで確認する場合: tensorboard --logdir ~/ray_results/idqn_simple_spread_v3
    stop_conditions = {
        "training_iteration": 100,  # 100回イテレーションを実行
        "timesteps_total": 200000,  # 合計200,000タイムステップ
    }
    results = tune.run(
        "DQN",
        name="idqn_simple_spread_v3",
        stop=stop_conditions,
        config=config.to_dict(),
        verbose=1,  # 0: silent, 1: results, 2: detailed results
        checkpoint_freq=10,
        checkpoint_at_end=True,
    )

    print("Training finished.")
    print("You can view results with TensorBoard by running:")
    print(f"tensorboard --logdir {results.get_best_logdir('episode_reward_mean', 'max')}")


if __name__ == "__main__":
    # Rayを初期化
    # local_mode=Trueにすると、デバッグが容易になります
    ray.init(local_mode=False)
    main()
    ray.shutdown()
