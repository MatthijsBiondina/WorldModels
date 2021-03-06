import os

import torch

from src.utils.plotting import save_metrics
from src.utils.startup import init_printoptions, init_seeds, init_logging, init_environment, init_memory, init_models, \
    init_trainer, init_seed_episodes, init_hyperparameters
import src.utils.config as cfg

# {k:tuple(f.size()) for k,f in sorted(locals().items()) if type(f).__name__ == 'Tensor'}
### dev

fresh = (cfg.models == '')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
###

init_printoptions()
init_seeds()
save_loc, metrics = init_logging(None if fresh else os.path.join(cfg.models, 'metrics.json'))
env = init_environment(save_loc)
D = init_memory(save_loc, None if fresh else os.path.join(cfg.models, 'memory'))
worldmodel, planner, policy = init_models(env)
trainer = init_trainer(env, worldmodel, planner, policy)
if fresh:
    epoch = init_seed_episodes(D, trainer, metrics)
else:
    epoch = cfg.seed_episodes
global_prior, free_nats = init_hyperparameters()

while epoch < cfg.episodes:
    trainer.train_interval(metrics, D, epoch, global_prior, free_nats)
    if epoch % 25 == 0:
        trainer.test_interval(metrics, save_loc, epoch)
    else:
        metrics['t_scores'].append(None)
        metrics['t_quartz'].append(None)
    trainer.collect_interval(metrics, D, epoch, save_loc)
    save_metrics(metrics, save_loc)
    worldmodel.save_state_dicts(save_loc)
    policy.module.save_state_dicts(save_loc)
    epoch += 1

# D = init_memory()
#
# for ii in range(10):
#     D.push(torch.tensor([ii, ii+0.1, ii+0.2, ii+0.3]), torch.tensor([ii]), ii, ii == 5)
#
#
# O, A, R, M = D.sample()

print("bye " * 2)
