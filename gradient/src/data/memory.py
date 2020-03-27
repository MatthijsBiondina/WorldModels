import json

import os

import random
import numpy as np
import src.utils.config as cfg

import torch
from torch import Tensor


class ExperienceReplay:

    def __init__(self, save_loc: str, load_loc: str = None):
        self.save_loc = os.path.join(save_loc,'memory')
        self.store_sync = True
        self.position = 0
        self.full = False
        # data arrays for: (O)bservations, (A)ctions, (R)ewards, *non*-terminal (M)asks
        self.O: np.ndarray = None
        self.A: np.ndarray = None
        self.R: np.ndarray = None
        self.M: np.ndarray = None
        if load_loc is not None:
            self._load(load_loc)

    def _load(self, path):
        self.O = np.load(os.path.join(path, 'observations.npy'))
        self.A = np.load(os.path.join(path, 'actions.npy'))
        self.R = np.load(os.path.join(path, 'rewards.npy'))
        self.M = np.load(os.path.join(path, 'masks.npy'))
        with open(os.path.join(path, 'attributes.json'), 'r') as f:
            d = json.load(f)
            self.position = d['position']
            self.full = d['full']

    def _save(self):
        os.makedirs(os.path.join(self.save_loc), exist_ok=True)
        np.save(os.path.join(self.save_loc, 'observations.npy'), self.O)
        np.save(os.path.join(self.save_loc, 'actions.npy'), self.A)
        np.save(os.path.join(self.save_loc, 'rewards.npy'), self.R)
        np.save(os.path.join(self.save_loc, 'masks.npy'), self.M)
        with open(os.path.join(self.save_loc, 'attributes.json'), 'w+') as f:
            json.dump({'position': self.position, 'full': self.full}, f)


    def push(self, o: Tensor, a: Tensor, r: float, done: bool) -> None:
        """
        Append new transition tuple to memory

        :param o: observation
        :param a: action
        :param r: reward
        :param done: terminal state
        :return:
        """
        self.store_sync = False
        if self.O is None:
            self._init_storage_matrices(o, a)
        self.O[self.position], self.A[self.position], self.R[self.position], self.M[self.position] = o, a, r, not done
        self.position = (self.position + 1) % cfg.experience_size
        self.full = self.full or self.position == 0

    def sample(self):
        """
        Returns a batch of sequence chunks uniformly sampled from the memory

        :return: observations, actions, rewards, masks
        """
        if not self.store_sync:
            self._save()
        return self._retrieve_batch(np.array([self._sample_idx() for _ in range(cfg.batch_size)]))

    @property
    def len(self):
        """
        :return: length of memory
        """
        if self.full:
            return cfg.experience_size
        else:
            return self.position

    def _init_storage_matrices(self, o: Tensor, a: Tensor) -> None:
        """
        PRIVATE - initialize storage matrices dynamically based on observation

        :param o: example observation
        :param a: example action
        :return: n/a
        """
        self.O = np.empty((cfg.experience_size, o.size(0)), dtype=np.float32)
        self.A = np.empty((cfg.experience_size, a.size(0)), dtype=np.float32)
        self.R = np.empty((cfg.experience_size,), dtype=np.float32)
        self.M = np.empty((cfg.experience_size,), dtype=np.float32)

    def _sample_idx(self) -> np.ndarray:
        """
        Returns an index for a valid single sequence chunk uniformly sampled from the memory

        :return: sampled index
        """
        idxs, valid_idx = None, False
        while not valid_idx:
            idx = np.random.randint(0, cfg.experience_size if self.full else self.position - cfg.chunk_size)
            idxs = np.arange(idx, idx + cfg.chunk_size) % cfg.experience_size
            valid_idx = self.position not in idxs
        return idxs

    def _retrieve_batch(self, idxs: np.ndarray):
        vec_idxs = idxs.reshape(-1)
        o_batch = torch.tensor(self.O[vec_idxs]).view(cfg.batch_size, cfg.chunk_size, *self.O.shape[1:])
        a_batch = torch.tensor(self.A[vec_idxs]).view(cfg.batch_size, cfg.chunk_size, -1)
        r_batch = torch.tensor(self.R[vec_idxs]).view(cfg.batch_size, cfg.chunk_size)
        m_batch = torch.tensor(
            [np.all(self.M[vec_idxs].reshape(cfg.batch_size, cfg.chunk_size, 1)[:, :ii + 1], axis=1)
             for ii in range(cfg.chunk_size)], dtype=torch.float32).transpose(0, 1)

        return o_batch, a_batch, r_batch, m_batch
