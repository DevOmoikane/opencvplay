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

    async def save_frame(self, frame):
        async with self.semaphore:
            return await asyncio.to_thread(self._save_frame, frame)

    def _save_frame(self, frame):
        file_id = uuid.uuid4()
        cv2.imwrite(os.path.join(self.storage_path, f"{file_id}.jpg"), frame)
        return file_id

    async def save_file(self, file, file_name=None):
        async with self.semaphore:
            return await asyncio.to_thread(self._save_file, file, file_name)

    def _save_file(self, file, file_name=None):
        if file_name is None:
            file_name = uuid.uuid4()
        file_path = os.path.join(self.storage_path, f"{file_name}.jpg")
        with open(file_path, 'wb') as f:
            f.write(file.read())
        return file_name

    async def read_file(self, file_path):
        async with self.semaphore:
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
