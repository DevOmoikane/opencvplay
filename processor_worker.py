import asyncio
import cv2
from google.protobuf.any import is_type
from ultralytics import YOLO
import face_recognition
import numpy as np
from rich import print
import logging
import cvzone
import time


class ProcessorWorker:
    def __init__(self, num_workers=2):
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)
        self.yolo_model = YOLO("yolo12n.pt")

    async def process_frame(self, frame):
        async with self.semaphore:
            return await asyncio.to_thread(self._process_frame, frame)

    def _process_frame(self, frame):
        res_rects = []
        res_circles = []
        results = self.yolo_model.track(frame, persist=True, classes=[0])
        person_locations = []
        person_frames = []
        face_locations = []
        face_images = []
        object_tracking = {}
        for result in results:
            for box in result.boxes:
                if box.cls != 0 or not box.is_track:
                    continue
                # print(f"box = {box}")
                id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_locations.append((x1, y1, x2, y2))
                object_tracking[id] = {
                    "cx": (x1+x2)/2,
                    "cy": (y1+y2)/2,
                    "lx": x2,
                    "ly": y2,
                    "x": x1,
                    "y": y1,
                    "time": time.time()
                }
                person_frame = frame[y1:y2, x1:x2].copy()
                rgb_small_frame = person_frame[:, :, ::-1]

                face_small_locations = face_recognition.face_locations(rgb_small_frame)
                # # face_small_encodings = face_recognition.face_encodings(rgb_small_frame, face_small_locations)
                #
                if len(face_small_locations) > 0:
                    sy1, sx1, sy2, sx2 = face_small_locations[0]
                    if sx1 > sx2:
                        sx1, sx2 = sx2, sx1
                    if sy1 > sy2:
                        sy1, sy2 = sy2, sy1
                    face_locations.append((x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2))
                #     h, w, _ = person_frame.shape
                #     face_frame = rgb_small_frame[sy1:sy2, sx1:sx2]
                #     face_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR)
                #     face_images.append(face_frame)
                #     person_frames.append(person_frame)

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
        return res_rects, res_circles, person_locations, object_tracking, face_locations, person_frames, face_images