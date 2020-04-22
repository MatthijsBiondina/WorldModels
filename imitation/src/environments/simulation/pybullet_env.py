import gym
import pybulletgym
import numpy as np
from src.environments.general.environment_template import Environment
from src.utils import config as cfg

_ = pybulletgym
PREP_VECTORS = {'InvertedPendulumSwingupPyBulletEnv-v0': np.array([1, 0.2, 1, 1, 0.067], dtype=np.float16)}


def preprocess_observation(obs):
    """
    :param obs: unprocessed observation
    :return: normalized observation
    """
    return np.clip(obs * PREP_VECTORS[cfg.env_name], -1., 1.)


class SimEnv(Environment):

    def __init__(self, save_loc: str):
        super().__init__(save_loc)
        self.env = gym.make(cfg.env_name)
        self.t = 0
        self.actions = [np.zeros(self.action_size)] * cfg.latency

    def reset(self):
        """
        Reset environment

        :return: observation at t=0
        """
        self.t = 0
        self.actions = [np.zeros(self.action_size)] * cfg.latency
        return preprocess_observation(self.env.reset())

    def step(self, action: np.ndarray):
        """
        Perform action and observe next state. Action is repeated 'action_repeat' times.

        :param action: the action to take
        :return: next observation, reward, terminal state
        """
        obs, done = None, None
        reward = 0
        self.actions.append(action)
        for k in range(cfg.action_repeat):
            obs, reward_k, done, _ = self.env.step(self.actions[0])
            reward += reward_k
            done = done or self.t == cfg.max_episode_length
            if done:
                break
        self.actions.pop(0)
        return preprocess_observation(obs), reward, done

    def render(self) -> np.ndarray:
        """
        Renders the environment to RGB array

        :return: frame capture of environment
        """
        return self.env.render(mode='rgb_array')

    def close(self):
        """
        Cleanup

        :return: n/a
        """
        self.env.close()

    def sample_random_action(self) -> np.ndarray:
        """
        Sample an action randomly from a uniform distribution over all valid actions

        :return: random action
        """
        return self.env.action_space.sample()

    @property
    def obs_size(self) -> int:
        """
        GETTER METHOD

        :return: size of observations in this environment
        """
        return self.env.observation_space.shape[0]

    @property
    def action_size(self):
        """
        GETTER METHOD

        :return: size of actions in this environment
        """
        return self.env.action_space.shape[0]
