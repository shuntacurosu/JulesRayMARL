from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Dict
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor
from ray.rllib.utils.typing import List, ModelConfigDict, TensorType


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

    def forward(self, x, mask=None):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_output, _ = self.mha(q, k, v, key_padding_mask=mask)
        return attn_output


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
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 1)
        )

        self._value_out = None

    def forward(self, input_dict: Dict, state: List[TensorType], seq_lens: TensorType):
        obs_tensor = flatten_inputs_to_1d_tensor(
            input_dict[SampleBatch.OBS], self.obs_space
        )
        actor_features = self.actor_net(obs_tensor)
        action_logits = self.action_branch(actor_features)

        if "critic_obs" in input_dict:
            critic_obs = input_dict["critic_obs"]
            critic_mask = input_dict["critic_obs_mask"]

            attention_out = self.attention_layer(critic_obs, mask=critic_mask)
            my_attention_out = attention_out[:, 0, :]
            self._value_out = self.critic_net(my_attention_out).squeeze(-1)
        else:
            batch_size = obs_tensor.shape[0]
            device = obs_tensor.device
            self._value_out = torch.zeros(batch_size, device=device)

        return action_logits, state

    def value_function(self) -> TensorType:
        assert self._value_out is not None, "must call forward() first"
        return self._value_out


def ma_poca_postprocessing(
    policy, sample_batch, other_agent_batches=None, episode=None
):
    # Initialize vf_preds if not present
    if SampleBatch.VF_PREDS not in sample_batch:
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32
        )

    if not policy.loss_initialized():
        obs_dim = policy.observation_space.shape[0]
        max_agents = policy.config["model"]["custom_model_config"]["max_agents"]
        sample_batch["critic_obs"] = (
            torch.zeros(len(sample_batch), max_agents, obs_dim)
            .float()
            .to(policy.device)
        )
        sample_batch["critic_obs_mask"] = (
            torch.ones(len(sample_batch), max_agents).bool().to(policy.device)
        )
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
                padded_obs[1 : 1 + num_to_fill] = np.array(other_obs[:num_to_fill])

            num_filled = 1 + min(num_others, max_agents - 1)
            mask[:num_filled] = False

            critic_obs_list.append(padded_obs)
            critic_mask_list.append(mask)

        sample_batch["critic_obs"] = (
            torch.from_numpy(np.array(critic_obs_list)).float().to(policy.device)
        )
        sample_batch["critic_obs_mask"] = (
            torch.from_numpy(np.array(critic_mask_list)).bool().to(policy.device)
        )
    else:
        raise NotImplementedError(
            "MA-POCA multi-worker postprocessing is not implemented in this example."
        )

    return sample_batch


class MAPOCAPolicy(PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        from ray.rllib.algorithms.ppo.ppo_torch_policy import ValueNetworkMixin

        ValueNetworkMixin.__init__(self, config)

    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        batch_with_critic_obs = ma_poca_postprocessing(
            self, sample_batch, other_agent_batches, episode
        )
        return super().postprocess_trajectory(
            batch_with_critic_obs, other_agent_batches, episode
        )
