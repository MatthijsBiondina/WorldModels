import traceback
import numpy as np

action_noise = 0.5
action_repeat = 4
action_delay = 0.02
activation_function = 'relu'
adam_epsilon = 1e-4
batch_size = 5
belief_size = 200
bit_depth = 5
candidates = 1000
checkpoint_experience = False
checkpoint_interval = 1
chunk_size = 50
collect_interval = 100
disable_cuda = False
embedding_size = 100
env_name = 'InvertedPendulumSwingupPyBulletEnv-v0'
episodes = 1000
experience_replay = ''
experience_size = 1000
free_nats = 2
global_kl_beta = 0.1
grad_clip_norm = 1.
hidden_size = 200
id = 'default'
learning_rate = 5e-4
learning_rate_schedule = 0.1
max_episode_length = 1000
min_std_dev = 1e-1
models = ''
optimisation_iters = 10
overshooting_distance = 24
overshooting_kl_beta = 1.
overshooting_reward_scale = 0
planning_horizon = 12
render = False
seed = 1
seed_episodes = 5
simulation = True
state_size = 30
top_candidates = 100




def hyperparameters():
    mname = traceback.extract_stack(None, 1)[0].name
    keys = [key for key in globals().keys() if '__' not in key and key is not mname and key is not 'traceback']
    return '\n'.join([f'{key} := {eval(key)}' for key in keys])
