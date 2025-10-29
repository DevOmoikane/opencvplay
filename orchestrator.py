import asyncio
import logging
import uuid

import cv2
import numpy as np

from capture_worker import AsyncFrameCapture
from face_worker import FaceFinderWorker
from file_worker import FileWorker
from processor_worker import ProcessorWorker

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(funcName)s:%(lineno)d - %(message)s')
logging.getLogger("ultralytics").setLevel(logging.ERROR)

NUM_POINTS = 5

def draw_image_on_frame(frame, overlay_image, position):
    x, y = position
    frame[y:y+overlay_image.shape[0], x:x+overlay_image.shape[1]] = overlay_image
    return frame


def resize_frame(frame, max_height=720):
    height, width = frame.shape[:2]
    if height > max_height:
        ratio = max_height / height
        new_width = int(width * ratio)
        frame = cv2.resize(frame, (new_width, max_height))
    return frame

def draw_and_predict_curve(frame, points, degree=2, prediction_length=5, image_size=(500, 500)):
    points = np.array(points, dtype=np.float32)
    x = points[:, 0]
    y = points[:, 1]

    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)

    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly(x_fit)

    x_pred = np.linspace(max(x), max(x) + prediction_length, 100)
    y_pred = poly(x_pred)

    for i in range(len(x)):
        cv2.circle(frame, (int(x[i]), int(y[i])), 5, (0, 128, 128), -1)

    curve_points = np.array([(int(x_fit[i]), int(y_fit[i])) for i in range(len(x_fit))], np.int32)
    cv2.polylines(frame, [curve_points], False, (0, 255, 0), 2)

    pred_points = np.array([(int(x_pred[i]), int(y_pred[i])) for i in range(len(x_pred))], np.int32)
    cv2.polylines(frame, [pred_points], False, (0, 0, 255), 2)


async def main():
    processor = ProcessorWorker()
    file_worker = FileWorker("./faces/")
    # capture = AsyncFrameCapture("rtsp://arkuscamera:arkus%40123@172.30.2.102/stream1")
    # capture = AsyncFrameCapture("rtsp://mindaccess:camara%40123@172.30.1.15/Preview_02_main")
    # capture = AsyncFrameCapture(0)
    capture = AsyncFrameCapture("file:///media/israel/data/resources/videos/arkus_sample/camera_frontal_aboveheadheight_unprocessed.mp4")
    # capture = AsyncFrameCapture("file:///media/israel/data/resources/videos/arkus_n1/1756419470822_0.mp4")
    await capture.start_capture()
    video_out = None
    new_file = True
    person_tracking = {}
    process_running = True
    started_writing = False
    while process_running:
        show_image = False
        cap_running = await capture.is_running()
        if not cap_running:
            process_running = False
            break
        frame = await capture.get_frame()
        if not started_writing:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            fps = capture.get_fps()
            w, h = capture.get_width_and_height()
            video_out = cv2.VideoWriter('output_video_x.mp4', fourcc, fps, (w, h))
            started_writing = True
        if frame is not None:
            # logging.info(f"Frame captured: {frame.shape}")
            rects, circles, person_locations, object_tracking, face_locations, persons, faces = await processor.process_frame(frame)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            for rect in rects:
                x1, y1, x2, y2 = rect
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 0), 2)
            for circle in circles:
                x, y, r = circle
                cv2.circle(frame, (x, y), r, (128, 128, 0), 2)
            for index, person in enumerate(person_locations):
                x1, y1, x2, y2 = person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            for id, object_data in object_tracking.items():
                x, y = object_data["cx"], object_data["cy"]
                time = object_data["time"]
                if id in person_tracking:
                    trajectory = np.array(person_tracking[id]["trajectory"])
                    end_point = None
                    if len(trajectory) >= NUM_POINTS:
                        # Calculate smooth direction using last few points
                        recent_points = trajectory[-NUM_POINTS:]
                        #recent_times = person_tracking[id]["times"][-NUM_POINTS:]
                        draw_and_predict_curve(frame, recent_points, degree=5, prediction_length=15, image_size=(frame.shape[1], frame.shape[0]))

                if id not in person_tracking:
                    person_tracking[id] = {
                        "trajectory": [],
                        "times": []
                    }
                person_tracking[id]["trajectory"].append(np.array([x, y]))
                person_tracking[id]["times"].append(time)
                if len(person_tracking[id]["trajectory"]) > NUM_POINTS*5:  # Keep last NUM_POINTS*5 points
                    person_tracking[id]["trajectory"].pop(0)
                    person_tracking[id]["times"].pop(0)

                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {id}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            for face in face_locations:
                x1, y1, x2, y2 = face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            position = (10,10)
            for index, face in enumerate(faces):
                h, w, _ = face.shape
                position = position[0], position[1] + h + 10
                show_image = True
                # small_frame = resize_frame(frame)
                id = uuid.uuid4()
                # await file_worker.save_frame(face, f"{id}_face")
                # await file_worker.save_frame(persons[index], f"{id}_person")
                # await file_worker.save_frame(frame, f"{id}_frame")
                frame = draw_image_on_frame(frame, face, position)
                # cv2.imshow("Face", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            # if new_file:
            #     await file_worker.save_frame(frame)
            #     new_file = False
            if started_writing:
                video_out.write(frame)
            frame = resize_frame(frame, 500)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                process_running = False
                break
    if started_writing:
        video_out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Stopping capture worker...")
