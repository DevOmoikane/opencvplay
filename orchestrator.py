import asyncio
import logging
import cv2
import numpy as np

from capture_worker import AsyncFrameCapture
from file_worker import FileWorker
from processor_worker import ProcessorWorker

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(funcName)s:%(lineno)d - %(message)s')

async def main():
    processor = ProcessorWorker()
    file_worker = FileWorker("./")
    capture = AsyncFrameCapture("rtsp://arkuscamera:arkus%40123@172.30.2.102/stream1")
    # capture = AsyncFrameCapture(0)
    await capture.start_capture()
    new_file = True
    while True:
        frame = await capture.get_frame()
        if frame is not None:
            # logging.info(f"Frame captured: {frame.shape}")
            rects, circles, faces = await processor.process_frame(frame)
            await asyncio.sleep(0)
            for rect in rects:
                x1, y1, x2, y2 = rect
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for circle in circles:
                x, y, r = circle
                cv2.circle(frame, (x, y), r, (0, 0, 255), 2)
            for face in faces:
                x1, y1, x2, y2 = face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Stopping capture worker...")
