import cv2
import asyncio
from threading import Thread
import time
import logging


class AsyncFrameCapture:
    def __init__(self, source, frame_interval=0.1, queue_size=10):
        self.frame_interval = frame_interval
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
                    break
                if current_time - last_time >= self.frame_interval:
                    last_time = current_time
                    if not self.queue.full():
                        loop.call_soon_threadsafe(self.queue.put_nowait, frame)
            except Exception as e:
                logging.error(f"Error capturing frame: {e}")

    async def get_frame(self):
        await asyncio.sleep(0)
        return await self.queue.get()
