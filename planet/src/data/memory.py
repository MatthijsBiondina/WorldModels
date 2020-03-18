import random
import numpy as np
import src.utils.config as cfg

import torch
from torch import Tensor


class ExperienceReplay:

    def __init__(self):
        self.position = 0
        self.full = False
        # data arrays for: (O)bservations, (A)ctions, (R)ewards, *non*-terminal (M)asks
        self.O: np.ndarray = None
        self.A: np.ndarray = None
        self.R: np.ndarray = None
        self.M: np.ndarray = None

    def push(self, o: Tensor, a: Tensor, r: float, done: bool) -> None:
        """
        Append new transition tuple to memory

        :param o: observation
        :param a: action
        :param r: reward
        :param done: terminal state
        :return:
        """
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
        self.O = np.empty((cfg.experience_size, o.size(0)), dtype=np.float16)
        self.A = np.empty((cfg.experience_size, a.size(0)), dtype=np.float16)
        self.R = np.empty((cfg.experience_size,), dtype=np.float16)
        self.M = np.empty((cfg.experience_size,), dtype=np.float16)

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
        vec_idxs = idxs.transpose().reshape(-1)
        o_batch = torch.tensor(self.O[vec_idxs]).view(cfg.chunk_size, cfg.batch_size, *self.O.shape[1:])
        a_batch = torch.tensor(self.A[vec_idxs]).view(cfg.chunk_size, cfg.batch_size, -1)
        r_batch = torch.tensor(self.R[vec_idxs]).view(cfg.chunk_size, cfg.batch_size)
        m_batch = torch.tensor(
            [np.all(self.M[vec_idxs].reshape(cfg.chunk_size, cfg.batch_size, 1)[:ii + 1, :], axis=0)
             for ii in range(cfg.chunk_size)])
        return o_batch, a_batch, r_batch, m_batch
