# APIリファレンス

<cite>
**このドキュメントで参照されているファイル**   
- [algorithm.py](../../src/ma_poca/algorithm.py)
- [policy.py](../../src/ma_poca/policy.py)
- [train_mpe.py](../../examples/train_mpe.py)
- [hpo_mpe.py](../../examples/hpo_mpe.py)
- [README.md](../../README.md)
</cite>

## 目次
1. [MAPOCAConfigクラス](#mapocaconfigクラス)
2. [MAPOCAPolicyクラス](#mapocapolicyクラス)
3. [MAPOCATorchModelクラス](#mapocatorchmodelクラス)
4. [SelfAttentionクラス](#selfattentionクラス)
5. [ma_poca_postprocessing関数](#ma_poca_postprocessing関数)

## MAPOCAConfigクラス

MAPOCAConfigクラスは、MA-POCA（Multi-Agent Posthumous Credit Assignment）アルゴリズムの構成設定を定義するためのクラスです。このクラスはRLlibのPPOConfigクラスを拡張しており、MA-POCAアルゴリズムに特化したカスタムモデルとポリシー設定を提供します。

### 構成オプション

MAPOCAConfigクラスは、以下の構成オプションを提供します。

```python
class MAPOCAConfig(PPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or MAPOCA)

        self.framework("torch")
        self.model = {
            "custom_model": MAPOCATorchModel,
            "custom_model_config": {}, # Let user specify max_agents
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "max_seq_len": 20,
        }

```

#### フレームワーク設定

- **目的**: 使用する深層学習フレームワークを指定します。
- **デフォルト値**: `"torch"`
- **有効な範囲**: `"torch"`（PyTorch）
- **使用例**: 
  ```python
  config.framework("torch")

 ```

#### モデル設定

- **目的**: ニューラルネットワークモデルのアーキテクチャとパラメータを定義します。
- **デフォルト値**: 
  ```python
  {
      "custom_model": MAPOCATorchModel,
      "custom_model_config": {},
      "fcnet_hiddens": [256, 256],
      "fcnet_activation": "relu",
      "max_seq_len": 20,
  }
  ```
- **詳細**:
  - `custom_model`: MA-POCAアルゴリズムで使用するカスタムモデルクラス（MAPOCATorchModel）を指定します。
  - `custom_model_config`: カスタムモデルに渡される追加設定を含む辞書です。特に`max_agents`パラメータをここで指定します。
  - `fcnet_hiddens`: 全結合層の隠れ層のサイズを指定します。デフォルトでは256ユニットの隠れ層が2つあります。
  - `fcnet_activation`: 活性化関数を指定します。デフォルトはReLUです。
  - `max_seq_len`: 最大シーケンス長を指定します。デフォルトは20です。
#### 使用例
```python
from ma_poca.algorithm import MAPOCAConfig

config = MAPOCAConfig()
config.environment(env="mpe_simple_spread")
config.training(gamma=0.99, lr=1e-5)
config.model["custom_model_config"]["max_agents"] = 3

```

#### トレーニング結果の取得方法

- **目的**: MA-POCAアルゴリズムのトレーニング結果からepisode reward meanを正しく取得する方法を示します。
- **詳細**: 
  - トレーニング結果からepisode reward meanを取得する際は、`result["env_runners"]["episode_reward_mean"]`を使用します。
  - 過去のバージョンでは`result["episode_reward_mean"]`を使用していましたが、RLlibのバージョンアップに伴い、キーが変更されました。
- **使用例**:
  ```python
  result = algo.train()
  episode_reward_mean = result.get("env_runners", {}).get("episode_reward_mean")
  print(f"Episode Reward Mean: {episode_reward_mean}")
  ```

## MAPOCAPolicyクラス

MAPOCAPolicyクラスは、MA-POCAアルゴリズムのポリシー実装を提供するクラスです。このクラスはRLlibのPPOTorchPolicyクラスを継承しており、MA-POCAアルゴリズムに特化したポリシー設定を提供します。

### クラス定義

```python
class MAPOCAPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        from ray.rllib.algorithms.ppo.ppo_torch_policy import ValueNetworkMixin
        ValueNetworkMixin.__init__(self, config)

    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        batch_with_critic_obs = ma_poca_postprocessing(self, sample_batch, other_agent_batches, episode)
        return super().postprocess_trajectory(batch_with_critic_obs, other_agent_batches, episode)

    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        """Adds the critic observations to the action output for debugging."""
        # Return an empty dict to avoid issues with the action output
        return {}
```

### メソッド

#### `__init__(self, observation_space, action_space, config)`

- **目的**: MAPOCAPolicyクラスの初期化を行います。
- **引数**:
  - `observation_space`: 観測空間
  - `action_space`: 行動空間
  - `config`: 設定
- **詳細**: 
  - 親クラスの初期化を行った後、ValueNetworkMixinの初期化を行います。

#### `postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None)`

- **目的**: トレーニングデータの後処理を行います。
- **引数**:
  - `sample_batch`: サンプルバッチ
  - `other_agent_batches`: 他のエージェントのバッチ（オプション）
  - `episode`: エピソード（オプション）
- **詳細**: 
  - `ma_poca_postprocessing`関数を使用して、トレーニングデータの後処理を行います。

#### `extra_action_out(self, input_dict, state_batches, model, action_dist)`

- **目的**: アクション出力に追加情報を加えます。
- **引数**:
  - `input_dict`: 入力辞書
  - `state_batches`: 状態バッチ
  - `model`: モデル
  - `action_dist`: アクション分布
- **詳細**: 
  - デバッグ目的で、クリティック観測をアクション出力に追加します。

## MAPOCATorchModelクラス

MAPOCATorchModelクラスは、MA-POCAアルゴリズムのモデル実装を提供するクラスです。このクラスはRLlibのTorchModelV2クラスを継承しており、MA-POCAアルゴリズムに特化したモデル設定を提供します。

### クラス定義

```python
class MAPOCATorchModel(TorchModelV2, nn.Module):
    """MA-POCA model with a self-attention based centralized critic."""
    def __init__(
        self,
        obs_space: Box,
        action_space: Box,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        obs_dim = obs_space.shape[0]

        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.action_branch = nn.Linear(256, num_outputs)

        self.attention_layer = SelfAttention(input_dim=obs_dim, d_model=128)

        self.critic_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self._value_out = None
```

### メソッド

#### `__init__(self, obs_space, action_space, num_outputs, model_config, name)`

- **目的**: MAPOCATorchModelクラスの初期化を行います。
- **引数**:
  - `obs_space`: 観測空間
  - `action_space`: 行動空間
  - `num_outputs`: 出力数
  - `model_config`: モデル設定
  - `name`: モデル名
- **詳細**: 
  - 親クラスの初期化を行った後、アクターネットワーク、アテンションレイヤー、クリティックネットワークを初期化します。

#### `forward(self, input_dict, state, seq_lens)`

- **目的**: 順伝播計算を行います。
- **引数**:
  - `input_dict`: 入力辞書
  - `state`: 状態
  - `seq_lens`: シーケンス長
- **戻り値**: 
  - `action_logits`: アクションロジット
  - `state`: 状態
- **詳細**: 
  - アクターネットワークを使用してアクションロジットを計算します。
  - クリティック観測が存在する場合は、アテンションレイヤーとクリティックネットワークを使用して価値関数を計算します。

#### `value_function(self)`

- **目的**: 価値関数を返します。
- **戻り値**: 
  - `_value_out`: 価値関数
- **詳細**: 
  - 事前に計算された価値関数を返します。

## SelfAttentionクラス

SelfAttentionクラスは、自己注意機構を実装するクラスです。

### クラス定義

```python
class SelfAttention(nn.Module):
    """A simple self-attention module."""
    def __init__(self, input_dim, d_model=128, n_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads

        self.query = nn.Linear(input_dim, d_model)
        self.key = nn.Linear(input_dim, d_model)
        self.value = nn.Linear(input_dim, d_model)

        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
```

### メソッド

#### `__init__(self, input_dim, d_model=128, n_heads=4)`

- **目的**: SelfAttentionクラスの初期化を行います。
- **引数**:
  - `input_dim`: 入力次元
  - `d_model`: モデル次元（デフォルト: 128）
  - `n_heads`: ヘッド数（デフォルト: 4）
- **詳細**: 
  - クエリ、キー、バリューの線形層とマルチヘッドアテンションを初期化します。

#### `forward(self, x, mask=None)`

- **目的**: 順伝播計算を行います。
- **引数**:
  - `x`: 入力
  - `mask`: マスク（オプション）
- **戻り値**: 
  - `attn_output`: アテンション出力
- **詳細**: 
  - クエリ、キー、バリューを計算し、マルチヘッドアテンションを適用します。

## ma_poca_postprocessing関数

ma_poca_postprocessing関数は、MA-POCAアルゴリズムのトレーニングデータの後処理を行う関数です。

### 関数定義

```python
def ma_poca_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    # Initialize vf_preds if not present
    if SampleBatch.VF_PREDS not in sample_batch:
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)
    
    if not policy.loss_initialized():
        obs_dim = policy.observation_space.shape[0]
        max_agents = policy.config["model"]["custom_model_config"]["max_agents"]
        sample_batch["critic_obs"] = torch.zeros(len(sample_batch), max_agents, obs_dim).float().to(policy.device)
        sample_batch["critic_obs_mask"] = torch.ones(len(sample_batch), max_agents).bool().to(policy.device)
        sample_batch["critic_obs_mask"][:, 0] = False
        return sample_batch

    max_agents = policy.config["model"]["custom_model_config"]["max_agents"]
    obs_dim = policy.observation_space.shape[0]

    if other_agent_batches is None:
        obs_by_timestep = defaultdict(list)
        for i in range(len(sample_batch)):
            eps_id = sample_batch[SampleBatch.EPS_ID][i]
            t = sample_batch[SampleBatch.T][i]
            obs_by_timestep[(eps_id, t)].append((sample_batch[SampleBatch.OBS][i], i))

        critic_obs_list, critic_mask_list = [], []
        for i in range(len(sample_batch)):
            eps_id = sample_batch[SampleBatch.EPS_ID][i]
            t = sample_batch[SampleBatch.T][i]
            my_obs = sample_batch[SampleBatch.OBS][i]

            all_obs_with_indices = obs_by_timestep.get((eps_id, t), [])

            padded_obs = np.zeros((max_agents, obs_dim), dtype=np.float32)
            mask = np.ones(max_agents, dtype=bool)

            padded_obs[0] = my_obs
            other_obs = [obs for obs, index in all_obs_with_indices if index != i]
            num_others = len(other_obs)

            if num_others > 0:
                num_to_fill = min(num_others, max_agents - 1)
                padded_obs[1:1+num_to_fill] = np.array(other_obs[:num_to_fill])

            num_filled = 1 + min(num_others, max_agents - 1)
            mask[:num_filled] = False

            critic_obs_list.append(padded_obs)
            critic_mask_list.append(mask)

        sample_batch["critic_obs"] = torch.from_numpy(np.array(critic_obs_list)).float().to(policy.device)
        sample_batch["critic_obs_mask"] = torch.from_numpy(np.array(critic_mask_list)).bool().to(policy.device)
    else:
        raise NotImplementedError(
            "MA-POCA multi-worker postprocessing is not implemented in this example."
        )

    return sample_batch
```

### 関数の詳細

- **目的**: MA-POCAアルゴリズムのトレーニングデータの後処理を行います。
- **引数**:
  - `policy`: ポリシー
  - `sample_batch`: サンプルバッチ
  - `other_agent_batches`: 他のエージェントのバッチ（オプション）
  - `episode`: エピソード（オプション）
- **戻り値**: 
  - `sample_batch`: 後処理されたサンプルバッチ
- **詳細**: 
  - クリティック観測とマスクを計算し、サンプルバッチに追加します。