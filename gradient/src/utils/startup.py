import logging

import json
import os
import random
import shutil
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

import src.utils.config as cfg
from src.data.memory import ExperienceReplay
from src.environments.init_environment import new_environment
from src.models.planner.cem_planner import Planner
from src.models.policy.policy_model import PolicyModel
from src.models.worldmodel.world_model import WorldModel
from src.training.trainer import Trainer
from src.utils.tools import pyout


def init_printoptions():
    torch.set_printoptions(precision=3, threshold=25, edgeitems=5, linewidth=60)
    np.set_printoptions(precision=3, threshold=25, edgeitems=5, linewidth=60, sign=' ')


def init_seeds():
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)


def init_logging(path=None):
    ii = 0
    try:
        while ii in [int(x.split('__')[1]) for x in os.listdir('res/results')]:
            ii += 1
    except FileNotFoundError:
        pass
    save_loc = os.path.join('res/results', '__'.join((cfg.id, str(ii).zfill(2),
                                                      datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))))
    if os.path.exists(save_loc):
        shutil.rmtree(save_loc)
    os.makedirs(save_loc)
    with open(os.path.join(save_loc, 'hyperparameters.txt'), 'w+') as f:
        f.write(cfg.hyperparameters())
    if path is None:
        metrics = {'steps': [], 'episodes': [], 'rewards': [], 't_scores': [], 't_quartz': [],
                   'o_loss': [], 'r_loss': [], 'kl_loss': [], 'p_loss': []}
    else:
        with open(path, 'r') as f:
            metrics = json.load(f)
    logger = logging.getLogger('planet')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(save_loc, 'planet.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:\n%(message)s'))
    logger.addHandler(fh)
    return save_loc, metrics


def init_environment(save_loc):
    return new_environment(save_loc)


def init_memory(save_loc, mem_load=None):
    return ExperienceReplay(save_loc, mem_load)


def init_models(env):
    worldmodel = WorldModel(env.obs_size, env.action_size)
    planner = Planner(worldmodel, env.action_size)
    policy = nn.DataParallel(PolicyModel(env.action_size).cuda(), device_ids=list(range(torch.cuda.device_count())))
    return worldmodel, planner, policy


def init_trainer(env, worldmodel, planner, policy):
    trainer = Trainer(env, worldmodel, planner, policy)
    return trainer


def init_seed_episodes(D: ExperienceReplay, trainer: Trainer, metrics: dict):
    epoch = 0
    while epoch < cfg.seed_episodes or D.len < cfg.batch_size:
        trainer.collect_interval(metrics, D, epoch)
        epoch += 1
    for key in metrics:
        while len(metrics[key]) < len(metrics['episodes']):
            metrics[key].append(None)

    # metrics['o_loss'].append(None)
    # metrics['r_loss'].append(None)
    # metrics['kl_loss'].append(None)
    # metrics['p_loss'].append(None)
    # metrics['t_scores'].append(None)
    # epoch += 1
    return epoch


def init_hyperparameters():
    global_prior = Normal(torch.zeros(cfg.batch_size, 1, cfg.state_size).cuda(),
                          torch.ones(cfg.batch_size, 1, cfg.state_size).cuda())
    free_nats = torch.full((1,), cfg.free_nats).cuda()
    return global_prior, free_nats
