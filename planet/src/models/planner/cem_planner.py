import torch
import torch.nn as nn
from torch import Tensor
import src.utils.config as cfg
from src.models.worldmodel.world_model import WorldModel


class Planner(nn.Module):
    def __init__(self, world_model: WorldModel, action_size):
        super(Planner, self).__init__()
        self.worldmodel = world_model
        self.action_size = action_size

    def forward(self, b: Tensor, s: Tensor):
        """
        Laten planning with CEM

        :param b: deterministic state [batch x belief_size]
        :param s: stochastic state [batch x state_size]
        :return:
        """
        minibs = b.size(0)
        rollup = cfg.candidates * torch.arange(0, minibs, dtype=torch.int64).unsqueeze(dim=1).cuda()

        b = b.unsqueeze(dim=1).expand(minibs, cfg.candidates, cfg.belief_size).reshape(-1, cfg.belief_size)
        s = s.unsqueeze(dim=1).expand(minibs, cfg.candidates, cfg.state_size).reshape(-1, cfg.state_size)

        # Initialize factorized belief over action sequences: q(a_t:t+H) ~ N(0,I)
        MU_a = torch.zeros(minibs, 1, cfg.planning_horizon, self.action_size)
        STD_a = torch.ones(minibs, 1, cfg.planning_horizon, self.action_size)

        for _ in range(cfg.optimisation_iters):
            # Evaluate J action sequences from the current belief
            # A = torch.clamp(MU_a + STD_a * torch.randn(minibs, cfg.candidates, cfg.planning_horizon, self.action_size),
            #                 -1., 1.).view(minibs * cfg.candidates, cfg.planning_horizon, self.action_size)

            A = (MU_a + STD_a * torch.randn(minibs, cfg.candidates, cfg.planning_horizon, self.action_size))
            A = A.view(minibs * cfg.candidates, cfg.planning_horizon, self.action_size)

            # Sample next state predictions
            B, S, _, _ = self.worldmodel.t_model(s, torch.clamp(A, -1., 1.), b)

            # Calculate expected returns
            R: Tensor = self.worldmodel.r_model(B.view(-1, cfg.belief_size), S.view(-1, cfg.state_size)
                                                ).view(-1, cfg.planning_horizon).sum(dim=1)

            # Re-fit belief to K best action sequences
            _, topk = R.view(minibs, cfg.candidates).topk(cfg.top_candidates, dim=1, largest=True, sorted=False)

            # Fix for unrolled indices
            A_top = A[(topk + rollup).view(-1)].view(minibs, cfg.top_candidates, cfg.planning_horizon, self.action_size)
            MU_a = A_top.mean(dim=1, keepdim=True)
            STD_a = A_top.std(dim=1, unbiased=False, keepdim=True)

        return torch.clamp(MU_a[:, :, 0].squeeze(dim=1), -1., 1.)
