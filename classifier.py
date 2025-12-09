import os
import shutil
import time
import traceback
import cv2
import click
import numpy as np
from ultralytics import YOLO
import difPy
import logging
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

MAIN_SUBFOLDER = "imgs"
PERSON_PATH = os.path.join(MAIN_SUBFOLDER, "person")
NSFW_PATH = os.path.join(MAIN_SUBFOLDER, "nsfw")
ANIME_PATH = os.path.join(MAIN_SUBFOLDER, "anime")
REVIEW_PATH = os.path.join(MAIN_SUBFOLDER, "review")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(funcName)s:%(lineno)d - %(message)s')
logging.getLogger("ultralytics").setLevel(logging.ERROR)

@click.command()
@click.option("--source-dir", '-s', "source_dir", default="./", show_default=True, help="Folder to search for images.")
def main(**kwargs):
    #person_yolo = YOLO('models/yolo11s.pt')
    person_yolo = YOLO('models/woman_face.pt')
    #anime_yolo = YOLO('models/yolov8x6_animeface.pt')
    anime_yolo = YOLO('models/anime.pt')
    nsfw_yolo = YOLO('models/erax-anti-nsfw-yolo11m-v1.1.pt')
    #nsfw_yolo = YOLO('models/yolov8m_pp13.pt')
    #nsfw_yolo = YOLO('models/yolov8m_as03.pt')

    # create destination folders
    os.makedirs(PERSON_PATH, exist_ok=True)
    os.makedirs(NSFW_PATH, exist_ok=True)
    os.makedirs(ANIME_PATH, exist_ok=True)
    os.makedirs(REVIEW_PATH, exist_ok=True)

    # get file list from source directory
    source_dir = kwargs['source_dir']
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.completed:>6}"),
        transient=False,
    ) as progress:
        task = progress.add_task("[purple]Processing...", total=len(files))
        try:
            for file in files:
                img = cv2.imread(os.path.join(source_dir, file))
                scores = {
                    "person": 0.0,
                    "nsfw": 0.0,
                    "nsfw2": 0.0,
                    "anime": 0.0
                }
                found = False
                results = anime_yolo(img)
                for result in results:
                    logging.debug(f"result: {result.boxes}")
                    if len(result.boxes) > 0:
                        # progress.console.log(f"Found anime: class= {result.boxes[0].cls} with confidence={result.boxes[0].conf}")
                        scores["anime"] = result.boxes[0].conf
                        found = True
                        break
                results = person_yolo(img)
                for result in results:
                    logging.debug(f"result: {result.boxes}")
                    if len(result.boxes) > 0:
                        # progress.console.log(f"Found person: class= {result.boxes[0].cls} with confidence={result.boxes[0].conf}")
                        scores["person"] = result.boxes[0].conf
                        found = True
                        break
                results = nsfw_yolo(img)
                for result in results:
                    logging.debug(f"result: {result.boxes}")
                    if len(result.boxes) > 0: #and result.boxes[0].cls != 0:
                        # progress.console.log(f"Found nsfw: class= {result.boxes[0].cls} with confidence={result.boxes[0].conf}")
                        scores["nsfw"] = result.boxes[0].conf
                        found = True
                        break
                # depending on the highest score choose the folder to move
                if found:
                    folder = max(scores, key=scores.get)
                else:
                    folder = "review"
                new_path = os.path.join(os.path.join(MAIN_SUBFOLDER,folder), file)
                #os.rename(os.path.join(source_dir, file), new_path)
                shutil.copyfile(os.path.join(source_dir, file), new_path)
                # progress.console.log(f"Moved to {new_path}")
                progress.advance(task, advance=1)

        except Exception as e:
            progress.console.log(f"Error: {e}")

if __name__ == "__main__":
    main()
