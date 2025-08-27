# Arkus test camera position:
# 252.6 cm height

import asyncio
import cv2
import numpy as np
import logging

# Door camera parameters:
# TODO: user calibrate_camera_2

# Laptop web cam
# fx, fy, cx, cy = (np.float64(651.4091046043549), np.float64(653.2336134231156), np.float64(318.32692349087426), np.float64(234.8164852829485))

class PositionWorker:
    def __init__(self, num_workers=2):
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)

    async def locate_position(self, rect=None, ):
        async with self.semaphore:
            return await asyncio.to_thread(self._locate_position)

    def _locate_position(self):
        pass