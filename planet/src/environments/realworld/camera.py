import sys
import time
from math import atan2
from multiprocessing import sharedctypes, Process, Queue, Lock, Pipe
from statistics import median, StatisticsError

import pyrealsense2 as rs
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import cv2

from src.utils.tools import pyout, poem


def get_keypoint(img, xrange, lower_bound, upper_bound):
    mask = cv2.inRange(img[:, xrange[0]:xrange[1], :], lower_bound, upper_bound)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5, 5))
    try:
        y, x = tuple(int(median(x)) for x in np.where(mask))
        x += xrange[0]
        return x, y
    except StatisticsError:
        return None


def color_mask(img, lower_bound, upper_bound):
    mask = cv2.inRange(img, lower_bound, upper_bound)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5, 5))
    return mask


class Camera:
    def __init__(self, buffer_pointer: sharedctypes.RawArray, queue: Queue):
        self.global_buffer = buffer_pointer
        self.global_queue = queue
        self.local_queue = Queue()
        self.process = Process(target=self.run)
        self.process.start()

    def run(self):
        quit_ = False
        while not quit_:
            try:
                cam = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
                cam.start(config)
                pyout("CAMERA: ... (OK)")
                buffer = np.ctypeslib.as_array(self.global_buffer)
                buffer_size = buffer.shape[0]
                idx = 0
                while not quit_:
                    frames = cam.wait_for_frames()
                    buffer[idx, :, :, :] = np.asanyarray(frames.get_color_frame().get_data())
                    self.global_queue.put((idx, time.time()))
                    idx = (idx + 1) % buffer_size
                    quit_ = not self.local_queue.empty()
                cam.stop()
            except RuntimeError as e:
                pyout("CAMERA: ... (NOT FOUND)")
                self._reset_realsense(wait=True)
        self._reset_realsense()

    def _reset_realsense(self, wait=False):
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()
        if wait:
            for _ in tqdm(range(30), desc=poem("Resetting Realsense Camera")):
                time.sleep(1)
                if not self.local_queue.empty():
                    sys.exit(0)

    def exit(self):
        self.local_queue.put(0)
        self.process.join()


class Processor:
    IMG_SIZE = (480, 640, 3)
    MASKER_PARAMS = (((0, 320), (70, 50, 100), (90, 255, 255)),
                     ((320, 640), (70, 50, 100), (90, 255, 255)),
                     ((0, 640), (150, 100, 100), (170, 255, 255)),
                     ((0, 640), (90, 170, 100), (110, 255, 255)))

    def __init__(self, buffer_pointer: sharedctypes.RawArray, queue: Queue, time_0):
        self.buffer_pointer = buffer_pointer
        self.queue = queue
        self.access_lock = Lock()
        self.masker_queues, self.masker_pipes = [None, None, None, None], [None, None, None, None]
        self.child_conns, self.masker_processes = [None, None, None, None], [None, None, None, None]
        self._init_maskers()

        self.g_idx, self.m_idx = 0, 0
        self.gl_lst, self.gr_lst, self.m_lst = np.full(50, 320 - 50), np.full(50, 320 + 50), np.full(10, 320)
        self.gl, self.gr, self.m, self.b = (0, 0), (0, 0), (0, 0), (0, 0)

        self.state_vector = np.ctypeslib.as_ctypes(np.zeros((5,), dtype=np.float32))
        self.state_pointer = sharedctypes.RawArray(self.state_vector._type_, self.state_vector)
        self.state = np.ctypeslib.as_array(self.state_pointer)

        self.cartpos, self.pend_x, self.pend_y, self.angvel, self.old_angle = 0., 0., -1., 0., 0.
        self.T0, self.T_s = time_0, time.time()

        self.stop_queue, self.ready_queue = Queue(), Queue()
        self.main_process = Process(target=self.run, args=(self.access_lock,))
        self.main_process.start()

    def run(self, lock):
        buffer = np.ctypeslib.as_array(self.buffer_pointer)
        state_vector = np.ctypeslib.as_array(self.state_pointer)
        quit_, ready, T, t0 = False, False, 0, None
        while not quit_:
            idx, T_ = self.get_frame()
            if t0 is None:
                t0 = time.time()
            if idx is None:
                break

            buffer[idx, :, :, :] = cv2.cvtColor(buffer[idx, :, :, :], cv2.COLOR_BGR2HSV)
            gl, gr, m, b = self.smooth(*self.get_keypoints(idx))
            self.set_state(lock, state_vector, gl, gr, m, b, T_ - T, T_)
            T, quit_ = T_, not self.stop_queue.empty()
            if not ready:
                ready = self.disp(buffer[idx, :, :, :], t0)

    def _init_maskers(self):
        for ii, (xrange, lower_bound, upper_bound) in enumerate(self.MASKER_PARAMS):
            self.masker_queues[ii] = Queue()
            self.masker_pipes[ii], self.child_conns[ii] = Pipe()
            self.masker_processes[ii] = Process(target=self.masker_run,
                                                args=(xrange, lower_bound, upper_bound, self.masker_queues[ii],
                                                      self.child_conns[ii]))
            self.masker_processes[ii].start()

    def masker_run(self, xrange, lower_bound, upper_bound, queue, pipe):
        buffer = np.ctypeslib.as_array(self.buffer_pointer)
        quit_ = False
        while not quit_:
            idx = pipe.recv()
            img = buffer[idx, :, :, :]
            coords = get_keypoint(img, xrange, lower_bound, upper_bound)
            pipe.send(coords)
            quit_ = not queue.empty()
        pipe.close()

    def get_keypoints(self, idx):
        for pipe in self.masker_pipes:
            pipe.send(idx)
        for ii, kp in enumerate(('gl', 'gr', 'm', 'b')):
            exec(f"{kp} = self.masker_pipes[{ii}].recv()")
            exec(f"self.{kp} = {kp} if {kp} is not None else self.{kp}")
        return self.gl, self.gr, self.m, self.b

    def smooth(self, gl, gr, m, b):
        self.gl_lst[self.g_idx], self.gr_lst[self.g_idx] = gl[0], gr[0]
        self.g_idx = (self.g_idx + 1) % self.gl_lst.shape[0]
        self.gl, self.gr = (int(np.mean(self.gl_lst)), gl[1]), (int(np.mean(self.gr_lst)), gr[1])
        return self.gl, self.gr, m, b

    def get_frame(self):
        while self.queue.empty() and self.stop_queue.empty():
            pass
        idx, T = None, None
        while not self.queue.empty():
            idx, T = self.queue.get()
        return idx, T

    def get_state(self):
        self.access_lock.acquire()
        retval = deepcopy(self.state)
        self.access_lock.release()
        return retval

    def set_state(self, lock, state_vector, gl, gr, m, b, dt, t):
        pixels_per_meter = gr[0] - gl[0]
        self.cartpos = ((m[0] - gl[0]) / pixels_per_meter - 0.5) * 2
        new_angle = atan2(m[1] - b[1], b[0] - m[0])
        self.pend_x, self.pend_y = np.cos(new_angle), np.sin(new_angle)
        self.angvel = min(1., max(-1., (((new_angle - self.old_angle) + np.pi) % (2 * np.pi) - np.pi) / dt / 10))
        self.old_angle = new_angle
        self.T_s = t
        lock.acquire()
        state_vector[0], state_vector[1], state_vector[2] = self.cartpos, self.pend_x, self.pend_y
        state_vector[3], state_vector[4] = self.angvel, self.T_s - 1579794877
        lock.release()

    def disp(self, img, t0):
        # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        # cv2.imshow("Camera Check", img)
        # if cv2.waitKey(100) == ord('q'):
        #     sys.exit(0)
        if time.time() - 5 > t0:
            self.ready_queue.put(0)
            # cv2.destroyAllWindows()
            return True
        else:
            return False

    def ready(self):
        return not self.ready_queue.empty()

    def exit(self):
        self.stop_queue.put(0)
        for q in self.masker_queues:
            q.put(0)
        for pipe in self.masker_pipes:
            pipe.close()
        for p in self.masker_processes:
            p.terminate()
        for p in self.masker_processes:
            p.join()
        self.main_process.join()
