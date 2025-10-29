import cv2
import asyncio
from threading import Thread
import time
import logging


class AsyncFrameCapture:
    def __init__(self, source, frame_interval=0.1, queue_size=10):
        self.frame_interval = frame_interval
        self.source = source
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.cap = cv2.VideoCapture(source)
        self.running = False

    async def start_capture(self):
        self.running = True
        loop = asyncio.get_event_loop()
        Thread(target=self._capture_frames, args=(loop,), daemon=True).start()

    def _capture_frames(self, loop):
        last_time = time.time()
        while self.running and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                current_time = time.time()
                if not ret:
                    if self.source.startswith('rtsp'):
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.source)
                    else:
                        self.running = False
                        continue
                if self.source.startswith('file'):
                    time.sleep(0.1)
                    if not self.queue.full():
                        loop.call_soon_threadsafe(self.queue.put_nowait, frame)
                elif current_time - last_time >= self.frame_interval:
                    last_time = current_time
                    if not self.queue.full():
                        loop.call_soon_threadsafe(self.queue.put_nowait, frame)
            except Exception as e:
                logging.error(f"Error capturing frame: {e}")
        self.cap.release()

    async def get_frame(self):
        await asyncio.sleep(0)
        if not self.running:
            return None
        return await self.queue.get()

    def get_fps(self):
        if not self.running:
            return None
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_width_and_height(self):
        if not self.running:
            return None, None
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    async def is_running(self):
        await asyncio.sleep(0)
        return self.running
