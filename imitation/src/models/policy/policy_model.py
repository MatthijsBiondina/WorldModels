import torch
from torch import nn
import src.utils.config as cfg
from src.utils.tools import init_weights


class PolicyModel(nn.Module):
    def __init__(self, action_size):
        super(PolicyModel, self).__init__()
        self.func = getattr(nn.functional, cfg.activation_function)
        self.line_1 = nn.Linear(cfg.belief_size + cfg.state_size, cfg.hidden_size)
        self.line_2 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.line_3 = nn.Linear(cfg.hidden_size, action_size)
        self.apply(init_weights)

    def forward(self, b, s):
        h = self.func(self.line_1(torch.cat((b, s), dim=1)))
        h = self.func(self.line_2(h))
        return self.line_3(h)
