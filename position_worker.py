# Arkus test camera position:
# 252.6 cm height

import asyncio
import cv2
import numpy as np
import logging

# Door camera parameters:
# fx, fy, cx, cy = (np.float64(581.685594957565), np.float64(579.6227773128813), np.float64(639.5000000004447), np.float64(359.4999999990846))
# fx, fy, cx, cy = (np.float64(636.4083958013254), np.float64(702.358399179285), np.float64(970.3430712575822), np.float64(447.94837834538697))
# fx, fy, cx, cy = (np.float64(636.4083958013254), np.float64(702.358399179285), np.float64(970.3430712575822), np.float64(447.94837834538697))
# fx, fy, cx, cy = (np.float64(434.0708275610782), np.float64(435.3281639557073), np.float64(639.4999999917694), np.float64(359.50000000946426))

# Laptop web cam
# fx, fy, cx, cy = (np.float64(777.1974361141605), np.float64(778.8196369288024), np.float64(319.50000001093974), np.float64(239.49999998490622))
# fx, fy, cx, cy = (np.float64(620.3821303011521), np.float64(618.9357863151845), np.float64(319.5000000048588), np.float64(239.50000001303846))

class PositionWorker:
    def __init__(self, num_workers=2):
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)

    async def locate_position(self, rect=None, ):
        async with self.semaphore:
            return await asyncio.to_thread(self._locate_position)

    def _locate_position(self):
        pass