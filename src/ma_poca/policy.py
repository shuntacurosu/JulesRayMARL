import torch
import torch.nn as nn
import numpy as np
import ray
from gymnasium.spaces import Box, Dict
from collections import defaultdict

from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.utils.typing import TensorType, List, ModelConfigDict

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

        actor_input_dim = 18
        critic_attention_input_dim = 18

        self.actor_net = nn.Sequential(
            nn.Linear(actor_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.action_branch = nn.Linear(256, num_outputs)

        self.attention_layer = SelfAttention(input_dim=critic_attention_input_dim, d_model=128)

        self.critic_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self._value_out = None

    def forward(self, input_dict: Dict, state: List[TensorType], seq_lens: TensorType):
        obs_tensor = flatten_inputs_to_1d_tensor(input_dict[SampleBatch.OBS], self.obs_space)
        local_obs_tensor = obs_tensor[:, :18]
        actor_features = self.actor_net(local_obs_tensor)
        action_logits = self.action_branch(actor_features)

        if "critic_obs" in input_dict:
            critic_obs = input_dict["critic_obs"]
            critic_mask = input_dict["critic_obs_mask"]
        else:
            batch_size = obs_tensor.shape[0]
            max_agents = self.model_config["custom_model_config"]["max_agents"]
            obs_dim = 18
            device = obs_tensor.device
            critic_obs = torch.zeros(batch_size, max_agents, obs_dim, device=device)
            critic_mask = torch.ones(batch_size, max_agents, dtype=torch.bool, device=device)
            critic_mask[:, 0] = False

        attention_out = self.attention_layer(critic_obs, mask=critic_mask)
        my_attention_out = attention_out[:, 0, :]

        self._value_out = self.critic_net(my_attention_out).squeeze(-1)

        return action_logits, state

    def value_function(self) -> TensorType:
        assert self._value_out is not None, "must call forward() first"
        return self._value_out

def ma_poca_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    if not policy.loss_initialized():
        obs_dim = 18
        max_agents = policy.config["model"]["custom_model_config"]["max_agents"]
        sample_batch["critic_obs"] = torch.zeros(len(sample_batch), max_agents, obs_dim).float().to(policy.device)
        sample_batch["critic_obs_mask"] = torch.ones(len(sample_batch), max_agents).bool().to(policy.device)
        sample_batch["critic_obs_mask"][:, 0] = False # Avoid NaNs
        return sample_batch

    max_agents = policy.config["model"]["custom_model_config"]["max_agents"]
    obs_dim = policy.observation_space.shape[0]

    # In single worker mode, all data is in `sample_batch`.
    # We need to group it by timestep to build the centralized observation.
    if other_agent_batches is None:
        obs_by_timestep = defaultdict(list)
        # Use a unique identifier for each sample to handle multiple episodes in one batch
        sample_batch["unique_id"] = np.arange(len(sample_batch))

        for i in range(len(sample_batch)):
            eps_id = sample_batch[SampleBatch.EPS_ID][i]
            t = sample_batch[SampleBatch.T][i]
            obs_by_timestep[(eps_id, t)].append(sample_batch[SampleBatch.OBS][i])

        critic_obs_list, critic_mask_list = [], []
        for i in range(len(sample_batch)):
            eps_id = sample_batch[SampleBatch.EPS_ID][i]
            t = sample_batch[SampleBatch.T][i]
            my_obs = sample_batch[SampleBatch.OBS][i]

            # Get all observations at this specific timestep
            all_obs_at_ts = obs_by_timestep.get((eps_id, t), [])

            padded_obs = np.zeros((max_agents, obs_dim), dtype=np.float32)
            mask = np.ones(max_agents, dtype=bool)

            # Place current agent's observation at the first position
            padded_obs[0] = my_obs

            # Fill the rest with other agents' observations
            other_obs = [obs for obs in all_obs_at_ts if not np.array_equal(obs, my_obs)]
            num_others = len(other_obs)
            if num_others > 0:
                padded_obs[1:1+num_others] = np.array(other_obs)

            num_filled = 1 + num_others
            mask[:num_filled] = False

            critic_obs_list.append(padded_obs)
            critic_mask_list.append(mask)

        sample_batch["critic_obs"] = torch.from_numpy(np.array(critic_obs_list)).float().to(policy.device)
        sample_batch["critic_obs_mask"] = torch.from_numpy(np.array(critic_mask_list)).bool().to(policy.device)
    else:
        # Multi-worker logic would be needed here. For this example, we assume single worker.
        pass

    return sample_batch

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
