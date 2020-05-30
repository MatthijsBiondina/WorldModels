from typing import Optional

import src.utils.config as cfg
from src.environments.general.environment_template import Environment
from src.environments.realworld.pendulum_env import Pendulum
from src.environments.simulation.pybullet_env import SimEnv


def new_environment(save_loc: Optional[str] = None) -> Environment:
    if cfg.simulation:
        return SimEnv(save_loc)
    else:
        return Pendulum(save_loc)