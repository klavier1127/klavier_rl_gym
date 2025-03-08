import torch
from typing import Tuple, Union
from abc import ABC, abstractmethod

# minimal interface of the environment
class VecEnv(ABC):
    num_envs: int
    num_obs: int
    num_obs_history: int
    num_privileged_obs: int
    num_env_obs: int
    num_rma_obs: int
    num_actions: int
    max_episode_length: int
    privileged_obs_buf: torch.Tensor
    obs_buf: torch.Tensor
    obs_noise_buf: torch.Tensor
    obs_history_buf: torch.Tensor
    rma_obs_buf: torch.Tensor
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    episode_length_buf: torch.Tensor # current episode duration
    extras: dict
    device: torch.device
    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[
        torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        pass

    @abstractmethod
    def reset(self, env_ids: Union[list, torch.Tensor]):
        pass

    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_obs_noise(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_obs_history(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        pass

    @abstractmethod
    def get_rma_observations(self) -> Union[torch.Tensor, None]:
        pass

    @abstractmethod
    def get_amp_observations(self) -> Union[torch.Tensor, None]:
        pass

    @abstractmethod
    def get_env_observations(self) -> Union[torch.Tensor, None]:
        pass