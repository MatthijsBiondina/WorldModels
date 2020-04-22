from copy import deepcopy

import random
import time
from math import ceil
from statistics import mean

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal
from torch.optim import Optimizer
from tqdm import tqdm

import src.utils.config as cfg
from src.data.memory import ExperienceReplay

from src.environments.general.environment_template import Environment
from src.models.worldmodel.world_model import WorldModel, bottle
from src.utils.tools import poem, pyout, make_video, chunks, soft_update


class Trainer:
    def __init__(self, environment: Environment, world_model: WorldModel, planner: nn.Module, policy: nn.Module):
        self.env = environment
        self.wm = world_model
        self.planner = planner
        self.policy = policy
        self.param_list = list(self.wm.parameters())
        self.wm_optimizer = optim.Adam(self.param_list, lr=0 if cfg.learning_rate_schedule != 0 else cfg.learning_rate,
                                       eps=cfg.adam_epsilon)
        self.plcy_optimizer = optim.Adam(self.policy.parameters(),
                                         lr=0 if cfg.learning_rate_schedule != 0 else cfg.learning_rate,
                                         eps=cfg.adam_epsilon)
        self.action_noise = cfg.action_noise if cfg.action_noise_schedule == 0 else 1.

    def collect_interval(self, metrics: dict, D: ExperienceReplay, epoch: int):
        self.wm.eval()
        self.planner.eval()
        self.policy.eval()
        with torch.no_grad():
            o, r_tot = torch.tensor(self.env.reset(), dtype=torch.float32), 0
            b, s_post = torch.zeros(1, cfg.belief_size), torch.zeros(1, cfg.state_size)
            a = torch.zeros(1, self.env.action_size)
            for t in tqdm(range(ceil(cfg.max_episode_length / cfg.action_repeat)),
                          desc=poem(f"{epoch} Collection"), leave=False):
                b, _, _, _, s_post, _, _ = self.wm.t_model(s_post, a.unsqueeze(dim=1), b,
                                                           self.wm.e_model(o.unsqueeze(dim=0)).unsqueeze(dim=0))
                b, s_post = b.squeeze(dim=1), s_post.squeeze(dim=1)  # remove time dimension

                a = torch.clamp(self.policy(b, s_post).cpu() + self.action_noise * torch.randn_like(a), -1., 1.)

                o_, r, done = self.env.step(a.view(self.env.action_size).numpy())

                D.push(o, a.view(self.env.action_size), r, done)

                r_tot += r
                o = torch.tensor(o_, dtype=torch.float32)
                if done:
                    break
        if cfg.action_noise_schedule != 0:
            self._linearly_ramping_an()
        metrics['steps'].append(t if len(metrics['steps']) == 0 else t + metrics['steps'][-1])
        metrics['episodes'].append(epoch)
        metrics['rewards'].append(r_tot)

    def test_interval(self, metrics: dict, save_loc: str, epoch: int):
        self.wm.eval()
        self.planner.eval()
        self.policy.eval()
        frames = []
        with torch.no_grad():
            o, r_tot = torch.tensor(self.env.reset(), dtype=torch.float32), 0
            frames.append(self.env.render())
            b, s_post = torch.zeros(1, cfg.belief_size), torch.zeros(1, cfg.state_size)
            a = torch.zeros(1, self.env.action_size)
            for t in tqdm(range(ceil(cfg.max_episode_length / cfg.action_repeat)),
                          desc=poem(f"{epoch} Test Run"), leave=False):
                b, _, _, _, s_post, _, _ = self.wm.t_model(s_post, a.unsqueeze(dim=1), b,
                                                           self.wm.e_model(o.unsqueeze(dim=0)).unsqueeze(dim=0))
                b, s_post = b.squeeze(dim=1), s_post.squeeze(dim=1)

                a = self.policy(b, s_post).cpu()

                o, r, done = self.env.step(a.view(self.env.action_size).numpy())
                frames.append(self.env.render())
                r_tot += r
                o = torch.tensor(o, dtype=torch.float32)
                if done:
                    break
        metrics['t_scores'].append(r_tot)
        make_video(frames, save_loc, epoch)

    def train_interval(self, metrics: dict, D: ExperienceReplay, epoch: int, global_prior, free_nats):
        self.wm.train()
        self.policy.train()
        self._train_worldmodel(metrics, D, epoch, global_prior, free_nats)
        self._train_policy(metrics, D, epoch, global_prior, free_nats)

    def _train_worldmodel(self, metrics: dict, D: ExperienceReplay, epoch: int, global_prior, free_nats):
        losses = []
        for _ in tqdm(range(cfg.collect_interval_worm), desc=poem(f"{epoch} Train Interval"), leave=False):
            # self.optimizer.zero_grad()
            O, A, R, M = D.sample()

            b_0 = torch.zeros(cfg.batch_size, cfg.belief_size)
            s_0 = torch.zeros(cfg.batch_size, cfg.state_size)

            # Y := B, S_pri, MU_pri, STD_pri, S_pos, MU_pos, STD_pos
            Y = self.wm.t_model(s_0, A[:, :-1], b_0, bottle(self.wm.e_model, (O[:, 1:],)), M[:, :-1])

            o_loss, r_loss, kl_loss = self._reconstruction_loss(Y, O, R, free_nats, global_prior)

            if cfg.overshooting_kl_beta != 0:
                kl_loss += self._latent_overshooting(Y, A, M, free_nats)

            if cfg.learning_rate_schedule != 0:
                self._linearly_ramping_lr(self.wm_optimizer)

            self.wm_optimizer.zero_grad()
            (o_loss + r_loss + kl_loss).backward()
            nn.utils.clip_grad_norm_(self.param_list, cfg.grad_clip_norm, norm_type=2)
            self.wm_optimizer.step()

            losses.append([o_loss.item(), r_loss.item(), kl_loss.item()])

        o_loss, r_loss, kl_loss = tuple(zip(*losses))
        metrics['o_loss'].append(mean(o_loss))
        metrics['r_loss'].append(mean(r_loss))
        metrics['kl_loss'].append(mean(kl_loss))

    def _train_policy(self, metrics: dict, D: ExperienceReplay, epoch: int, global_prior, free_nats):
        losses = []
        policy_ = deepcopy(self.policy.module)
        optimizer = optim.Adam(policy_.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
        O, A, _, _ = D.get_last()
        B, S_pos = torch.zeros(O.size(1) + 1, cfg.belief_size), torch.zeros(O.size(1) + 1, cfg.state_size)
        with torch.no_grad():
            a_0 = torch.zeros(1, 1, A.size(2))
            b_0 = torch.zeros(1, cfg.belief_size).cuda()
            s_0 = torch.zeros(1, cfg.state_size).cuda()
            B, _, _, _, S_pos, _, _ = self.wm.t_model(s_0, torch.cat((a_0, A), dim=1)[:, :-1, :], b_0,
                                                      self.wm.e_model(O))
            B = torch.cat((b_0.unsqueeze(0), B), 1).squeeze(0)
            S_pos = torch.cat((s_0.unsqueeze(0), S_pos), 1).squeeze(0)

            A_tgt = torch.zeros(B.size(0), A.size(-1))
            for ii in tqdm(list(chunks(list(range(A_tgt.size(0))), cfg.batch_size)),
                           desc=poem(f"{epoch} Query Expert"), leave=False):
                A_tgt[ii] = self.planner(B[ii], S_pos[ii])
        A_tgt = A_tgt.cuda()

        for _ in tqdm(range(cfg.collect_interval_plcy), desc=poem(f"{epoch} Policy Train"), leave=False):
            ii = random.sample(range(A_tgt.size(0)), cfg.batch_size)
            A_pred = policy_(B[ii], S_pos[ii])
            loss = F.mse_loss(A_pred, A_tgt[ii], reduction='mean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        metrics['im_loss'].append(mean(losses))

        soft_update(self.policy.module, policy_, cfg.linear_policy_update)


    def _train_policy_(self, metrics: dict, D: ExperienceReplay, epoch: int, global_prior, free_nats):
        losses = []
        O, A, _, M = D.get_all()
        B, S_pos = torch.zeros(O.size(0), 1, cfg.belief_size), torch.zeros(O.size(0), 1, cfg.state_size)
        with torch.no_grad():
            for ii in tqdm(range(1, B.size(0)), desc=poem(f"{epoch} Expert Belief State"), leave=False):
                if M[ii]:
                    B[ii], _, _, _, S_pos[ii], _, _ = self.wm.t_model(S_pos[ii - 1], A[ii - 1].unsqueeze(1), B[ii - 1],
                                                                      self.wm.e_model(O[ii]).unsqueeze(0))
        B, S_pos = B.squeeze(1), S_pos.squeeze(1)
        idxs = chunks(sorted(list(range(B.size(0))), key=lambda x: random.random()), cfg.batch_size)
        for II in tqdm(idxs, desc=poem(f"{epoch} Policy Train"), leave=False):
            with torch.no_grad():
                A_tgt = self.planner(B[II], S_pos[II])
            A_pred = self.policy(B[II], S_pos[II])
            loss = (F.mse_loss(A_pred, A_tgt.cuda(), reduction='mean'))
            self.plcy_optimizer.zero_grad()
            loss.backward()
            self.plcy_optimizer.step()
            losses.append(loss.item())

        metrics['im_loss'].append(mean(losses))

    def _reconstruction_loss(self, Y, O, R, free_nats, global_prior):
        B, S_pri, MU_pri, STD_pri, S_pos, MU_pos, STD_pos = Y
        o_loss = F.mse_loss(bottle(self.wm.d_model, (B, S_pos)), O[:, 1:].cuda(), reduction='none').sum(2).mean()
        r_loss = F.mse_loss(bottle(self.wm.r_model, (B, S_pos)), R[:, :-1].cuda(), reduction='none').mean()
        kl_loss = torch.max(kl_divergence(Normal(MU_pos, STD_pos), Normal(MU_pri, STD_pri)).sum(2), free_nats).mean()
        if cfg.global_kl_beta != 0:
            kl_loss += cfg.global_kl_beta * kl_divergence(Normal(MU_pos, STD_pos), global_prior).sum(2).mean()
        return o_loss, r_loss, kl_loss

    def _latent_overshooting(self, Y, A_, M_, free_nats):
        B_, S_pri_, _, _, _, MU_pos_, STD_pos_ = Y

        overshooting_vars = []
        for t in range(1, cfg.chunk_size - 1):
            d = min(t + cfg.overshooting_distance, cfg.chunk_size - 1)
            t_, d_ = t - 1, d - 1
            seq_pad = [0, 0, 0, t - d + cfg.overshooting_distance, 0, 0]
            overshooting_vars.append((
                F.pad(A_[:, t:d], seq_pad),
                F.pad(M_[:, t:d], seq_pad),
                B_[:, t_],
                S_pri_[:, t_],
                F.pad(MU_pos_[:, t_ + 1: d_ + 1].detach(), seq_pad),
                F.pad(STD_pos_[:, t_ + 1:d_ + 1].detach(), seq_pad, value=1),
                F.pad(torch.ones(B_.size(0), d - t, cfg.state_size).cuda(), seq_pad)
            ))
        overshooting_vars = tuple(zip(*overshooting_vars))

        A = torch.cat(overshooting_vars[0], dim=0)
        M = torch.cat(overshooting_vars[1], dim=0)
        b0 = torch.cat(overshooting_vars[2], dim=0)
        s0 = torch.cat(overshooting_vars[3], dim=0)
        MU_pos = torch.cat(overshooting_vars[4], dim=0)
        STD_pos = torch.cat(overshooting_vars[5], dim=0)
        seq_masks = torch.cat(overshooting_vars[6], dim=0)

        _, _, MU_pri, STD_pri = self.wm.t_model(s0, A, b0, None, M)

        loss = (kl_divergence(Normal(MU_pos, STD_pos), Normal(MU_pri, STD_pri)) * seq_masks).sum(2)
        loss = torch.max(loss, free_nats).mean()
        loss = (1 / cfg.overshooting_distance) * cfg.overshooting_kl_beta * (cfg.chunk_size - 1) * loss

        return loss

    def _latent_overshooting_(self, Y, A_, M_, free_nats):
        t0 = time.time()
        B_, S_pri_, _, _, _, MU_pos_, STD_pos_ = Y
        bs, cs = (cfg.chunk_size - 2) * cfg.batch_size, cfg.overshooting_distance

        A = torch.zeros(bs, cs, A_.size(2))
        M = torch.zeros(bs, cs, 1)
        b0 = torch.zeros(bs, cfg.belief_size)
        s0 = torch.zeros(bs, cfg.state_size)
        MU_pos = torch.zeros(bs, cs, cfg.state_size)
        STD_pos = torch.ones(bs, cs, cfg.state_size)
        seq_masks = torch.zeros(bs, cs, cfg.state_size)

        for ii, t in enumerate(range(1, cfg.chunk_size - 1)):
            d = min(t + cfg.overshooting_distance, cfg.chunk_size - 1)
            t_, d_ = t - 1, d - 1
            bs, cs = cfg.batch_size, d - t
            A[ii * bs:(ii + 1) * bs, :cs] = A_[:, t:d]
            M[ii * bs:(ii + 1) * bs, :cs] = M_[:, t:d]
            b0[ii * bs:(ii + 1) * bs] = B_[:, t_]
            s0[ii * bs:(ii + 1) * bs] = S_pri_[:, t_]
            MU_pos[ii * bs:(ii + 1) * bs, :cs] = MU_pos_[:, t_ + 1:d_ + 1].detach()
            STD_pos[ii * bs:(ii + 1) * bs, :cs] = STD_pos_[:, t_ + 1:d_ + 1].detach()
            seq_masks[ii * bs:(ii + 1) * bs, :cs] = torch.ones(bs, cs, cfg.state_size)

        _, _, MU_pri, STD_pri = self.wm.t_model(s0, A, b0, None, M)

        loss = (kl_divergence(Normal(MU_pos.cuda(), STD_pos.cuda()), Normal(MU_pri, STD_pri)) * seq_masks.cuda()).sum(2)
        loss = torch.max(loss, free_nats).mean()
        loss = (1 / cfg.overshooting_distance) * cfg.overshooting_kl_beta * (cfg.chunk_size - 1) * loss
        aa = time.time() - t0

        return loss

    def _linearly_ramping_an(self):
        self.action_noise = max(self.action_noise - cfg.action_noise_schedule, cfg.action_noise)

    def _linearly_ramping_lr(self, optimizer: Optimizer):
        for group in optimizer.param_groups:
            group['lr'] = min(group['lr'] + cfg.learning_rate / cfg.learning_rate_schedule, cfg.learning_rate)
