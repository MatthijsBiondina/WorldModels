import sys
import time
from multiprocessing import Queue, Process, sharedctypes
import pyrealsense2 as rs
import numpy as np
from tqdm import tqdm


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
                print("CAMERA: ... (OK)")
                buff = np.ctypeslib.as_array(self.global_buffer)
                buffer_size = buff.shape[0]
                idx = 0
                while not quit_:
                    frames = cam.wait_for_frames()
                    buff[idx, :, :, :] = np.asanyarray(frames.get_color_frame().get_data())
                    self.global_queue.put((idx, time.time()))
                    idx = (idx + 1) % buffer_size
                    quit_ = not self.local_queue.empty()
                cam.stop()
            except RuntimeError as e:
                print("CAMERA: ... (NOT FOUND)")
                self._reset_realsense(wait=True)
        self._reset_realsense()

    def _reset_realsense(self, wait=False):
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()
        if wait:
            for _ in tqdm(range(30), desc="Resetting Realsense Camera"):
                time.sleep(1)
                if not self.local_queue.empty():
                    sys.exit(0)

    def exit(self):
        self.local_queue.put(0)
        self.process.join()

if __name__ == '__main__':
    buffer = np.ctypeslib.as_ctypes(np.zeros((100, 480, 640, 3), dtype=np.uint8))
    buffer_pointer = sharedctypes.RawArray(buffer._type_, buffer)
    queue = Queue()
    camera = Camera(buffer_pointer, queue)