import asyncio
import cv2
from ultralytics import YOLO
import face_recognition
import numpy as np
import logging


class ProcessorWorker:
    def __init__(self, num_workers=2):
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)
        self.yolo_model = YOLO("yolo11m.pt")

    async def process_frame(self, frame):
        async with self.semaphore:
            return await asyncio.to_thread(self._process_frame, frame)

    def _process_frame(self, frame):
        res_rects = []
        res_circles = []
        results = self.yolo_model(frame)
        face_locations = []
        for result in results:
            for box in result.boxes:
                if box.cls != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                res_rects.append((x1, y1, x2, y2))
                person_frame = frame[y1:y2, x1:x2]
                person_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
                rgb_small_frame = np.ascontiguousarray(person_frame[:, :, ::-1])

                face_small_locations = face_recognition.face_locations(rgb_small_frame)
                face_small_encodings = face_recognition.face_encodings(rgb_small_frame, face_small_locations)

                if len(face_locations) > 0:
                    sx1, sy1, sx2, sy2 = face_small_locations[0]
                    face_locations.append((x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2))
        # # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # hist = cv2.equalizeHist(gray_frame)
        # ksize = 31
        # blurred_gray_frame = cv2.GaussianBlur(hist, (ksize, ksize), cv2.BORDER_DEFAULT)
        # height, width = blurred_gray_frame.shape[:2]
        # minR = round(width / 65)
        # maxR = round(width / 25)
        # minDis = round(width / 7)
        # circles = cv2.HoughCircles(blurred_gray_frame, cv2.HOUGH_GRADIENT, 1, minDis, param1=14, param2=25, minRadius=minR, maxRadius=maxR)
        # if circles is not None:
        #     circles = np.round(circles[0, :]).astype("int")
        #     for (x, y, r) in circles:
        #         res_circles.append((x, y, r))
        return res_rects, res_circles, face_locations