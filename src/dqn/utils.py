import numpy as np
from mpe2 import simple_v3, simple_spread_v3
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from supersuit import dtype_v0

SUPPORTED_MPE_ENVS = {
    "simple_v3": simple_v3,
    "simple_spread_v3": simple_spread_v3,
}


def create_env(env_name: str, **kwargs):
    """
    指定された名前のMPE ParallelEnv環境を生成し、データ型を修正し、
    RLlib用のラッパーを適用します。

    Args:
        env_name (str): simple_v3 や simple_spread_v3 などの環境名。
        **kwargs: 環境の初期化に渡す追加の引数。

    Returns:
        ParallelPettingZooEnv: RLlibで扱えるようにラップされた環境インスタンス。
    """
    if env_name not in SUPPORTED_MPE_ENVS:
        raise ValueError(
            f"Unsupported MPE environment: {env_name}. "
            f"Supported environments are: {list(SUPPORTED_MPE_ENVS.keys())}"
        )

    # parallel_env() を使って環境を生成
    env_module = SUPPORTED_MPE_ENVS[env_name]
    env = env_module.parallel_env(**kwargs)

    # supersuitラッパーで観測のデータ型をfloat32に変換
    env = dtype_v0(env, dtype=np.float32)

    # RLlibのParallelPettingZooEnvラッパーを適用
    rllib_env = ParallelPettingZooEnv(env)
    return rllib_env
