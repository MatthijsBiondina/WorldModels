import numpy as np
from abc import ABC, abstractmethod


class Environment(ABC):
    def __init__(self, save_loc: str):
        self.save_loc = save_loc

    @abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def render(self) -> np.ndarray:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def sample_random_action(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def obs_size(self) -> int:
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        pass
