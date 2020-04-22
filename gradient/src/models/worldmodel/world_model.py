import os
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

import src.utils.config as cfg
from src.utils.tools import pyout


def init_weights(m):
    try:
        nn.init.xavier_normal_(m.weight)
        nn.init.normal_(m.bias, 0., 0.1)
    except AttributeError:
        pass


def bottle(function: nn.Module, inputs: Tuple[Tensor, ...]) -> Tensor:
    """
    unroll inputs: [time x batch x *input_features] -> [(time*batch) x *input_features]
    apply function: F([(time*batch) x *input_features]) = [(time*batch) x *output_features]
    restore dims: [(time*batch) x *output_features] -> [time x batch x *output_features]

    :param function:
    :param inputs:
    :return:
    """
    in_shapes = tuple(map(lambda x: x.size(), inputs))
    # args = tuple(map(lambda args: args[0].reshape(args[1][0] * args[1][1], *args[1][2:]), zip(inputs, in_shapes)))
    y = function(*map(lambda args: args[0].reshape(args[1][0] * args[1][1], *args[1][2:]), zip(inputs, in_shapes)))
    ou_shape = y.size()
    return y.view(in_shapes[0][0], in_shapes[0][1], *ou_shape[1:])


# b_t = f(b_t-1, s_t-1, a_t-1)
# s_t ~ p(s_t | b_t)
class TransitionModel(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(self, action_size: int):
        super(TransitionModel, self).__init__()
        self.func = getattr(F, cfg.activation_function)

        self.line_SA = nn.Linear(cfg.state_size + action_size, cfg.belief_size)
        self.rnn = nn.GRUCell(cfg.belief_size, cfg.belief_size)
        self.line_Bpri = nn.Linear(cfg.belief_size, cfg.hidden_size)
        self.line_Spri = nn.Linear(cfg.hidden_size, 2 * cfg.state_size)
        self.line_Bpos = nn.Linear(cfg.belief_size + cfg.embedding_size, cfg.hidden_size)
        self.line_Spos = nn.Linear(cfg.hidden_size, 2 * cfg.state_size)

    def forward(self, s_0: Tensor, Action: Tensor, b_0: Tensor, Observation: Optional[Tensor] = None,
                Mask: Optional[Tensor] = None):
        """
        Operates over (previous) state, (previous) actions, (previous) belief, (previous) masks, and (current)
        observations. Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state
        that gets sliced off):
        # t :  0  1  2  3  4  5
        # o :    -X--X--X--X--X-
        # a : -X--X--X--X--X-
        # m : -X--X--X--X--X-
        # pb: -X-
        # ps: -X-
        # b : -x--X--X--X--X--X-
        # s : -x--X--X--X--X--X-

        :param s_0: previous state
        :param Action: actions
        :param b_0: previous belief
        :param Observation: observations
        :param Mask: non-terminal masks
        :return: B, S_prior, MU_prior, STD_prior, (S_post, MU_post, STD_post)
        """
        # init lists for hidden states
        T = Action.size(1) + 1  # minibatchsize and number timesteps
        B = [torch.empty(0)] * T
        S_pri, S_pos = [torch.empty(0)] * T, [torch.empty(0)] * T
        MU_pri, MU_pos = [torch.empty(0)] * T, [torch.empty(0)] * T
        STD_pri, STD_pos = [torch.empty(0)] * T, [torch.empty(0)] * T

        # Set first belief-state and state to previous belief and state
        B[0], S_pri[0], S_pos[0] = b_0, s_0, s_0

        # Loop over chunk size
        for t in range(T - 1):
            # If observation is known, use posterior over s_t, else use prior over s_t
            s_t = S_pri[t] if Observation is None else S_pos[t]
            s_t = s_t if Mask is None else s_t * Mask[:, t]  # apply masks

            # Compute belief
            B[t + 1] = self.rnn(self.func(self.line_SA(torch.cat([s_t, Action[:, t]], dim=1))), B[t])

            # Compute prior for s_t+1 by applying transition dynamics with reparameterization trick
            MU_pri[t + 1], h = torch.chunk(self.line_Spri(self.func(self.line_Bpri(B[t + 1]))), 2, dim=1)
            STD_pri[t + 1] = F.softplus(h) + cfg.min_std_dev
            S_pri[t + 1] = MU_pri[t + 1] + STD_pri[t + 1] * torch.randn_like(MU_pri[t + 1])

            if Observation is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1
                MU_pos[t + 1], h = torch.chunk(
                    self.line_Spos(self.func(self.line_Bpos(torch.cat([B[t + 1], Observation[:, t_ + 1]], dim=1)))), 2,
                    dim=1)
                STD_pos[t + 1] = F.softplus(h) + cfg.min_std_dev
                S_pos[t + 1] = MU_pos[t + 1] + STD_pos[t + 1] * torch.randn_like(MU_pos[t + 1])

        # Return new hidden states
        h = [torch.stack(B[1:], dim=1),
             torch.stack(S_pri[1:], dim=1),
             torch.stack(MU_pri[1:], dim=1),
             torch.stack(STD_pri[1:], dim=1)]
        if Observation is not None:
            h += [torch.stack(S_pos[1:], dim=1),
                  torch.stack(MU_pos[1:], dim=1),
                  torch.stack(STD_pos[1:], dim=1)]
        return h


class Encoder(nn.Module):
    def __init__(self, observation_size: int):
        super(Encoder, self).__init__()
        self.func = getattr(F, cfg.activation_function)

        self.line_1 = nn.Linear(observation_size, cfg.hidden_size)
        self.line_2 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.line_3 = nn.Linear(cfg.hidden_size, cfg.embedding_size)

    def forward(self, O):
        """
        Embed observation

        :param O: observations
        :return: embeddings
        """
        h = self.func(self.line_1(O))
        h = self.func(self.line_2(h))
        return self.line_3(h)


# o_t ~ p(o_t | b_t, s_t)
class Decoder(nn.Module):
    def __init__(self, observation_size: int):
        super(Decoder, self).__init__()
        self.func = getattr(F, cfg.activation_function)

        self.line_1 = nn.Linear(cfg.belief_size + cfg.state_size, cfg.hidden_size)
        self.line_2 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.line_3 = nn.Linear(cfg.hidden_size, observation_size)

    def forward(self, B, S):
        """
        predict observation from deterministic and stochastic state

        :param B: deterministic state
        :param S: stochastic state
        :return: observation
        """
        h = self.func(self.line_1(torch.cat([B, S], dim=1)))
        h = self.func(self.line_2(h))
        return self.line_3(h)


class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.func = getattr(F, cfg.activation_function)

        self.line_1 = nn.Linear(cfg.belief_size + cfg.state_size, cfg.hidden_size)
        self.line_2 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.line_3 = nn.Linear(cfg.hidden_size, 1)

    def forward(self, B: Tensor, S: Tensor) -> Tensor:
        """
        predict reward

        :param B: determinsitic state
        :param S: stochastic state
        :return: reward
        """
        h = self.func(self.line_1(torch.cat([B, S], dim=1)))
        h = self.func(self.line_2(h))
        return self.line_3(h).squeeze(dim=1)


class WorldModel:
    def __init__(self, observation_size: int, action_size: int):
        super(WorldModel, self).__init__()

        devices = list(range(torch.cuda.device_count()))
        self.t_model = nn.DataParallel(TransitionModel(action_size).cuda(), device_ids=devices)
        self.e_model = nn.DataParallel(Encoder(observation_size).cuda(), device_ids=devices)
        self.d_model = nn.DataParallel(Decoder(observation_size).cuda(), device_ids=devices)
        self.r_model = nn.DataParallel(RewardModel().cuda(), device_ids=devices)

        # Initialization
        if cfg.models is '':
            self.t_model.apply(init_weights)
            self.d_model.apply(init_weights)
            self.r_model.apply(init_weights)
            self.e_model.apply(init_weights)
        else:
            self.t_model.load_state_dict(torch.load(os.path.join(cfg.models, 't_model.pth')))
            self.d_model.load_state_dict(torch.load(os.path.join(cfg.models, 'd_model.pth')))
            self.r_model.load_state_dict(torch.load(os.path.join(cfg.models, 'r_model.pth')))
            self.e_model.load_state_dict(torch.load(os.path.join(cfg.models, 'e_model.pth')))


    def save_state_dicts(self, save_loc):
        pth = os.path.join(save_loc, 'model')
        os.makedirs(pth, exist_ok=True)
        torch.save(self.t_model.state_dict(), os.path.join(pth, 't_model.pth'))
        torch.save(self.d_model.state_dict(), os.path.join(pth, 'd_model.pth'))
        torch.save(self.r_model.state_dict(), os.path.join(pth, 'r_model.pth'))
        torch.save(self.e_model.state_dict(), os.path.join(pth, 'e_model.pth'))

    def train(self):
        self.t_model.train()
        self.e_model.train()
        self.d_model.train()
        self.r_model.train()

    def eval(self):
        self.t_model.train()
        self.e_model.train()
        self.d_model.train()
        self.r_model.train()

    def parameters(self):
        return (list(self.t_model.parameters()) + list(self.e_model.parameters()) +
                list(self.d_model.parameters()) + list(self.r_model.parameters()))
