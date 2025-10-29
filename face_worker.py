import asyncio
import traceback
import os

import cv2
import face_recognition
from ultralytics import YOLO
import numpy as np
import logging


class FaceFinderWorker:
    def __init__(self, num_workers=2):
        logging.info("Initializing FaceFinderWorker")
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)
        self.yolo_model = YOLO('yolo11s.pt')

    async def process_frame_async(self, frame):
        async with self.semaphore:
            return await asyncio.to_thread(self.process_frame, frame)

    def _find_faces(self, frame, callback=None):
        try:
            results = self.yolo_model(frame)
            face_locations = []
            face_encodings = []
            for result in results:
                logging.debug(f"Found {len(result.boxes)} objects")
                for box in result.boxes:
                    if int(box.cls)==0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        logging.debug(f"Found person at {x1},{y1} to {x2},{y2}")
                        person_frame = frame[y1:y2, x1:x2]
                        person_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
                        rgb_small_frame = np.ascontiguousarray(person_frame[:, :, ::-1])

                        face_small_locations = face_recognition.face_locations(rgb_small_frame)
                        face_small_encodings = face_recognition.face_encodings(rgb_small_frame, face_small_locations)

                        if len(face_small_locations) > 0:
                            face_locations.append(face_small_locations[0])
                            face_encodings.append(face_small_encodings[0])

            if callback is not None:
                callback(frame, face_locations, face_encodings)
            return face_locations, face_encodings
        except Exception as e:
            logging.error(f"FaceFinderWorker Error: {e}")
            traceback.print_exc()

    def process_frame(self, frame, callback=None):
        return self._find_faces(frame, callback)

    def process_image(self, image, callback=None):
        if type(image) is str:
            image_data = face_recognition.load_image_file(image)
        else:
            image_data = image

        return self._find_faces(image_data, callback)
