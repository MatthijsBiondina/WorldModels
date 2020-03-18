import random
import torch
from src.environments.init_environment import new_environemnt
from src.data.memory import ExperienceReplay

env = new_environemnt()
memory = ExperienceReplay()

for k in range(0, 100):
    a = torch.tensor([k], dtype=torch.float16)
    o, r = torch.tensor([k,k+0.1], dtype=torch.float16), k
    done = random.random() > 0.95
    memory.push(o, a, r, done)

batch = memory.sample()

print("bye " * 2)
