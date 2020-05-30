import atexit
import time
from multiprocessing import sharedctypes, Queue

import numpy as np
import torch
import src.utils.config as cfg
from src.environments.general.environment_template import Environment
from src.environments.realworld.camera import Camera, Processor
from src.environments.realworld.serial_comm import SerialCommunication
from src.utils.tools import clip, pyout


def clean_action(a):
    if isinstance(a, torch.Tensor):
        while len(a.shape) > 1:
            a = a.squeeze()
        a = int(round(clip(a.cpu().numpy()[0], -1., 1.) * 127))
    elif isinstance(a, np.ndarray):
        a = int(round(clip(a[0], -1., 1.) * 127))
    else:
        a = int(round(clip(a, -1., 1.) * 127))
    return a


def reward_function(state):
    return state[2]


class Pendulum(Environment):

    def __init__(self, save_loc: str):
        super().__init__(save_loc)
        atexit.register(self.exitfunc)
        self.buffer = np.ctypeslib.as_ctypes(np.zeros((100, 480, 640, 3), dtype=np.uint8))
        self.buffer_pointer = sharedctypes.RawArray(self.buffer._type_, self.buffer)
        self.queue = Queue()
        self.T0 = time.time()
        self.cam = Camera(self.buffer_pointer, self.queue)
        self.prc = Processor(self.buffer_pointer, self.queue, self.T0)
        self.usb = SerialCommunication(cfg.action_delay)
        self.prev_upd8, self.prev_act = time.time(), None
        self.wait_startup()
        self.T_s = time.time()
        self.t = 0
        self.reset()

    def step(self, action, absolute=False):
        r, state = 0., None
        a = clean_action(action)  # standardize action format
        for _ in range(cfg.action_repeat):
            while time.time() - cfg.action_delay < self.prev_upd8:  # wait for minimum action delay
                pass
            u = self.auto_brakes(a, absolute)
            self.usb.send(u)
            self.prev_upd8 = time.time()
            state, self.T_s = self.get_state()
            r += reward_function(state)
            self.t += 1
        return state, r, self.t >= 1000

    def reset(self):
        self.usb.send(0)
        self.pid(0.)
        self.T0 = time.time()
        self.t = 0
        return self.get_state()[0]

    def close(self):
        self.exitfunc()

    def render(self):
        return np.full((480, 640, 3), 288, dtype=np.uint8)

    def sample_random_action(self) -> np.ndarray:
        return np.random.randn(1)

    @property
    def obs_size(self) -> int:
        return 5

    @property
    def action_size(self):
        return 1

    def pid(self, targ_x=0., timeout=0.10, mod=1.):
        K_p, K_i, K_d = mod * 50., 0.6, 10.
        s0, t0 = self.get_state()[0], time.time()
        while (self.get_state()[0] == s0).all() and time.time() - 60 < t0:
            time.sleep(0.01)
        ii, cnt = 0, 0
        E = np.zeros((100,), dtype=np.float32)
        t0 = time.time()
        while time.time() - timeout < t0:
            E[ii] = self.get_state()[0][0] - targ_x

            u = K_p * E[ii]
            u += K_i * sum(E)
            u += K_d * (E[ii] - E[(ii - 1) % E.size])

            if abs(u) < 2.8:
                cnt += 1
                if cnt > 10:
                    break
            else:
                cnt = 0

            self.step(-u / 127, absolute=True)
        self.step(0, absolute=True)

    def get_state(self):
        prc_state = self.prc.get_state()
        usb_state = self.usb.get_state()
        return np.concatenate((prc_state[:4], usb_state / 127), axis=0), prc_state[4]

    def exitfunc(self):
        self.usb.exit()
        self.cam.exit()
        self.prc.exit()

    def wait_startup(self):
        while not self.prc.ready():
            time.sleep(0.5)

    def auto_brakes(self, a, absolute=False):
        state, _ = self.get_state()
        max_v, min_v = (0.5 - state[0] / 2) * 170., (0.5 + state[0] / 2) * -170.
        max_v, min_v = max_v if max_v >= 0 else max_v * 10, min_v if min_v <= 0 else min_v * 10
        return clip(a if absolute else int(round(state[-1] * 127)) + a, min_v, max_v)
