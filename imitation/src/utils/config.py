import random
import traceback
batch_mult = 1

linear_policy_update = 0.1
action_noise = 0.1
action_noise_schedule = 0.01
action_repeat = 2
action_delay = 0.02
activation_function = 'relu'
adam_epsilon = 1e-4
batch_size = 50 * batch_mult # <-----------------------------------------------
minib_size = batch_size
belief_size = 200
bit_depth = 5
candidates = 1000
checkpoint_experience = False
checkpoint_interval = 1
chunk_size = 50
collect_interval_worm = 100 // batch_mult # <----------------------------------------
collect_interval_plcy = 50
disable_cuda = False
embedding_size = 100
env_name = 'InvertedPendulumSwingupPyBulletEnv-v0'
episodes = 1000
experience_replay = ''
experience_size = 1000000
free_nats = 1.
global_kl_beta = 0.0
grad_clip_norm = 1.
hidden_size = 200
id = 'policy_aggregation'
latency=0
learning_rate = 5e-4  / batch_mult # <------------------------------------------
learning_rate_schedule = 0.
max_episode_length = 1000
min_std_dev = 1e-1
models = ''
optimisation_iters = 10  # <- debug speedup
overshooting_distance = 24
overshooting_kl_beta = 0.5
overshooting_reward_scale = 0
planning_horizon = 12
render = False
seed = random.randint(0,1000000)
seed_episodes = 5  # <- debug speedup
simulation = True
state_size = 30
top_candidates = 100


def hyperparameters():
    mname = traceback.extract_stack(None, 1)[0].name
    keys = [key for key in globals().keys() if '__' not in key and key is not mname and
            key is not 'traceback' and key is not 'np']
    return '\n'.join([f'{key} := {eval(key)}' for key in keys])
