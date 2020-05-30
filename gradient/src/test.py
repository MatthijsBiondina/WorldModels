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