from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.policy.policy import Policy
from typing import Type

from src.ma_poca.policy import MAPOCAPolicy, MAPOCATorchModel

class MAPOCAConfig(PPOConfig):
    """
    Configuration class for the MAPOCA algorithm.
    Extends PPOConfig to set custom model and policy settings.
    """
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


class MAPOCA(PPO):
    @classmethod
    def get_default_config(cls) -> "MAPOCAConfig":
        return MAPOCAConfig()

    @classmethod
    def get_default_policy_class(cls, config: "PPOConfig") -> Type[Policy]:
        return MAPOCAPolicy
