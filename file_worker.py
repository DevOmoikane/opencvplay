import asyncio
import aiofiles
import os
import logging
import cv2
import uuid


class FileWorker:
    def __init__(self, storage_path, num_workers=2):
        logging.info("Initializing FileWorker")
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)
        self.storage_path = storage_path

    def define_paths(self):
        pass

    async def save_frame(self, frame, frame_id=None):
        async with self.semaphore:
            return await asyncio.to_thread(self._save_frame, frame, frame_id)

    def _save_frame(self, frame, frame_id=None):
        if frame is None:
            return None
        try:
            if frame_id is None:
                frame_id = uuid.uuid4()
            cv2.imwrite(os.path.join(self.storage_path, f"{frame_id}.jpg"), frame)
        except Exception as e:
            logging.error(f"FileWorker Error: {e}")
            frame_id = None
        return frame_id

    async def save_file(self, file, file_name=None):
        async with self.semaphore:
            return await asyncio.to_thread(self._save_file, file, file_name)

    def _save_file(self, file, file_name=None):
        if file is None:
            return None
        try:
            if file_name is None:
                file_name = uuid.uuid4()
            file_path = os.path.join(self.storage_path, f"{file_name}.jpg")
            with open(file_path, 'wb') as f:
                f.write(file.read())
        except Exception as e:
            logging.error(f"FileWorker Error: {e}")
            file_name = None
        return file_name

    async def read_file(self, file_path):
        async with self.semaphore:
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
