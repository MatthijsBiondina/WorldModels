import atexit
import traceback

import os
import sys
import time
from multiprocessing import Lock, sharedctypes, Queue, Process
import numpy as np
import serial
from copy import deepcopy

from src.utils.tools import pyout


class SerialCommunication:
    def __init__(self, action_delay):
        atexit.register(self.exit)
        self.lock = Lock()
        self.velocity_c = np.ctypeslib.as_ctypes(np.zeros((1,), dtype=np.float32))
        self.vel_pointer = sharedctypes.RawArray(self.velocity_c._type_, self.velocity_c)
        self.velocity = np.ctypeslib.as_array(self.vel_pointer)
        self.send_queue, self.stop_queue, self.err_queue = Queue(), Queue(), Queue()
        self.action_delay, self.timestamp = action_delay, time.time()
        self.process = Process(target=self.run)
        self.process.start()

    def run(self):
        usb = self.connect()
        velocity=np.ctypeslib.as_array(self.vel_pointer)
        quit_ = False
        while not quit_:
            try:
                if self.send_queue.empty():
                    if usb.in_waiting:
                        vel = int(usb.readline())
                        self.lock.acquire()
                        velocity[0] = vel
                        self.lock.release()
                else:
                    val = self.send_queue.get()
                    usb.write(str(max(-127, min(127, val))).encode())
                quit_ = not self.stop_queue.empty()
            except Exception:
                traceback.print_exc()
                self.err_queue.put(0)
                sys.exit(0)
            pass
        usb.close()

    def get_state(self):
        if not self.err_queue.empty():
            sys.exit(0)
        self.lock.acquire()
        retval = deepcopy(self.velocity)
        self.lock.release()
        return retval

    def send(self, val):
        while time.time() - self.action_delay < self.timestamp:
            pass
        # self.send_queue.put(max(-127, min(127, int(round(val)))))
        self.send_queue.put(0)
        self.timestamp = time.time()

    def connect(self):
        try:
            for prt in os.listdir('/dev'):
                if 'ttyACM' in prt:
                    portname = os.path.join('/dev', prt)
                    pyout(f"ARDUINO: {prt} (OK)")
                    break
            return serial.Serial(portname, 9600)
        except UnboundLocalError as e:
            print("ARDUINO: ... (NOT FOUND)")
            traceback.print_exc()
        sys.exit(0)

    def exit(self):
        self.send_queue.put(0)
        time.sleep(0.1)
        self.stop_queue.put(0)
        self.process.join()
