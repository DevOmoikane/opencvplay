# Python
import os
import time
import random
import traceback
import threading
import queue
from dataclasses import dataclass

import cv2
import requests
import click
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import numpy as np
from ultralytics import YOLO
import difPy
import logging
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(funcName)s:%(lineno)d - %(message)s')
logging.getLogger("ultralytics").setLevel(logging.ERROR)

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolo11s.onnx")
YOLO_CFG = os.environ.get("YOLO_CFG", "yolo11s.cfg")
YOLO_NAMES = os.environ.get("YOLO_NAMES", "coco.names")

PERSON_CLASS_IDS = {0}
CONF_THRESH = 0.35
NMS_THRESH = 0.4

POST_SELECTOR = "div.xrvj5dj.xd0jker"
PROFILE_SELECTOR = "span.xjp7ctv div a.x1i10hfl.xjbqb8w.x1ejq31n.x18oe1m7.x1sy0etr.xstzfhl.x972fbf.x10w94by.x1qhh985.x14e42zd.x9f619.x1ypdohk.xt0psk2.x3ct3a4.xdj266r.x14z9mp.xat24cr.x1lziwak.xexx8yu.xyri2b.x18d9i69.x1c1uobl.x16tdsg8.x1hl2dhg.xggy1nq.x1a2a7pz.xp07o12.xzmqwrg.x1citr7e.x1kdxza.xt0b8zv"
IMAGE_SELECTOR_OLD = "img.xl1xv1r.x9f619.x1lliihq.xmz0i5r.x193iq5w.xuiwhb7.x1g40iwv.x47corl.x87ps6o.x1obq294.x5a5i1n.xde0f50.x15x8krk"
IMAGE_SELECTOR_ALT = "img.xl1xv1r.x9f619.x1lliihq.xmz0i5r.x193iq5w.xuiwhb7.x1g40iwv.x47corl.x87ps6o.x1obq294.x5a5i1n.xde0f50.x15x8krk.x1ey2m1c.xtijo5x.x1o0tod.x10l6tqk.x13vifvy.x5yr21d.xh8yej3"
IMAGE_SELECTOR_NOT_FOUND = "img.xl1xv1r.x9f619.x1lliihq.xmz0i5r.x193iq5w.xp3xoqj.x1g40iwv.x47corl.x87ps6o.x1obq294.x5a5i1n.xde0f50.x15x8krk"
IMAGE_SELECTOR_GENERAL = "img.xl1xv1r.x9f619.x1lliihq.xmz0i5r.x193iq5w"
VIDEO_SELECTOR = "video.x1lliihq.x5yr21d.xh8yej3"

def create_driver(headless: bool = True) -> webdriver.Chrome:
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def scroll_page(driver: webdriver.Chrome, scroll_pause=2.0):
    last_height = driver.execute_script("return document.body.scrollHeight")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(scroll_pause)
    new_height = driver.execute_script("return document.body.scrollHeight")
    return new_height != last_height

def load_netscape_cookies(driver: webdriver.Chrome, base_url: str, cookie_file_path: str):
    logging.info(f"Loading cookies from {cookie_file_path}")
    if not os.path.exists(cookie_file_path):
        raise FileNotFoundError(f"Cookie file not found: {cookie_file_path}")

    driver.get(base_url)

    with open(cookie_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 7:
                parts = line.split()
                if len(parts) != 7:
                    continue
            domain, flag, path, secure, expiry, name, value = parts
            secure_bool = secure.upper() == "TRUE"
            cookie_dict = {
                "name": name,
                "value": value,
                "domain": domain if domain.startswith(".") else domain,
                "path": path,
                "secure": secure_bool,
            }
            try:
                if expiry.isdigit():
                    cookie_dict["expiry"] = int(expiry)
            except Exception:
                pass
            try:
                driver.add_cookie(cookie_dict)
            except Exception:
                pass

def load_girl_page(driver: webdriver.Chrome, base_url: str):
    driver.get(base_url)
    time.sleep(3)
    max_scrolls = 20
    scroll_count = 0
    try:
        while scroll_count < max_scrolls:
            if not scroll_page(driver):
                break
            scroll_count += 1
    except Exception as e:
        logging.warning(f"Error while scrolling: {e}")

def remove_duplicates(array1: list[str], array2: list[str]) -> list[str]:
    return [item for item in array2 if item not in array1]

def find_image_urls(driver: webdriver.Chrome, base_url: str) -> list[str]:
    imgs = driver.find_elements(By.CSS_SELECTOR, IMAGE_SELECTOR_GENERAL)
    urls = []
    for img in imgs:
        srcset = img.get_attribute("srcset") or ""
        src = ""
        if srcset:
            srcsetarray = srcset.split(',')
            src_dict = {}
            for pair in srcsetarray:
                link, size_str = pair.split(' ', 1)
                size = int(size_str.replace('w', ''))
                src_dict[size] = link
            bigger = max(src_dict.keys())
            src = src_dict[bigger]
        else:
            src = img.get_attribute("src") or ""
            if not src:
                continue
        if src.startswith("http"):
            urls.append(src)
        else:
            full = urljoin(base_url, src)
            urls.append(full)
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped

def find_video_urls(driver: webdriver.Chrome, base_url: str) -> list[str]:
    videos = driver.find_elements(By.CSS_SELECTOR, VIDEO_SELECTOR)
    urls = []
    for video in videos:
        src = video.get_attribute("src") or ""
        if not src:
            continue
        if src.startswith("http"):
            urls.append(src)
        else:
            full = urljoin(base_url, src)
            urls.append(full)
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped

def download_image_bytes(url: str, session: requests.Session) -> bytes | None:
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
            return resp.content
    except Exception:
        return None
    return None

def download_link(url: str, session: requests.Session, output_path: str = "./"):
    try:
        with session.get(url, stream=True) as resp:
            resp.raise_for_status()
            with open(os.path.join(output_path, os.path.basename(urlparse(url).path)), "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        logging.info(f"[ERROR] {url}")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_processed(file_path: str) -> set[str]:
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def append_processed(file_path: str, url: str):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(url + "\n")

# ------------------------------
# Queue types and downloader thread
# ------------------------------
@dataclass
class DownloadItem:
    url: str
    kind: str  # "image" or "video"

def downloader_thread_fn(
    q: "queue.Queue[DownloadItem]",
    stop_event: threading.Event,
    session: requests.Session,
    output_dir: str,
    processed_log: str,
    processed_set: set[str],
    progress: Progress,
    download_task_id: int,
    yolo_model: YOLO,
):
    while not stop_event.is_set():
        try:
            item: DownloadItem = q.get(timeout=0.5)
        except queue.Empty:
            continue  # keep waiting forever until stop_event is set

        url = item.url
        kind = item.kind

        try:
            if url in processed_set:
                progress.update(download_task_id, advance=1)
                q.task_done()
                continue

            if kind == "video":
                # download video as-is
                download_link(url, session, output_dir)
                append_processed(processed_log, url)
                processed_set.add(url)
                progress.update(download_task_id, advance=1)
                q.task_done()
                continue

            # kind == "image" -> person filtering
            delay = random.uniform(0.10, 1.0)
            time.sleep(delay)

            data = download_image_bytes(url, session)
            if not data:
                append_processed(processed_log, url)
                processed_set.add(url)
                progress.update(download_task_id, advance=1)
                q.task_done()
                continue

            image_array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image_array is None:
                append_processed(processed_log, url)
                processed_set.add(url)
                progress.update(download_task_id, advance=1)
                q.task_done()
                continue

            results = yolo_model.predict(image_array)
            has_person = False
            for result in results:
                for box in result.boxes:
                    if int(box.cls) in PERSON_CLASS_IDS:
                        has_person = True
                        break
                if has_person:
                    break

            if has_person:
                # keep original name, add unique suffix if exists
                parsed = urlparse(url)
                base_name = os.path.basename(parsed.path) or "image.jpg"
                if not os.path.splitext(base_name)[1]:
                    base_name += ".jpg"
                # We keep your original approach to avoid overwriting by delegating to download_link
                download_link(url, session, output_dir)

            append_processed(processed_log, url)
            processed_set.add(url)
            progress.update(download_task_id, advance=1)
        except Exception:
            logging.info(f"[ERROR] processing {url}")
        finally:
            try:
                if not q.empty():
                    q.task_done()
            except Exception:
                pass

# ------------------------------
# Dynamic page loader that feeds queue immediately
# ------------------------------
def load_girl_page_dynamic_to_queue(
    driver: webdriver.Chrome,
    base_url: str,
    found_progress: Progress,
    found_task_id: int,
    enqueued_set: set[str],
    q: "queue.Queue[DownloadItem]",
) -> tuple[int, int]:
    driver.get(base_url)
    time.sleep(3)
    image_count = 0
    video_count = 0
    not_found_counter = 0

    try:
        while True:
            images = find_image_urls(driver, base_url)
            videos = find_video_urls(driver, base_url)

            # enqueue only new ones
            new_images = [u for u in images if u not in enqueued_set]
            new_videos = [u for u in videos if u not in enqueued_set]

            for u in new_images:
                enqueued_set.add(u)
                q.put(DownloadItem(url=u, kind="image"))
                image_count += 1
                found_progress.update(found_task_id, advance=1)

            for u in new_videos:
                enqueued_set.add(u)
                q.put(DownloadItem(url=u, kind="video"))
                video_count += 1
                found_progress.update(found_task_id, advance=1)

            if len(new_images) == 0 and len(new_videos) == 0:
                not_found_counter += 1
            else:
                not_found_counter = 0

            if not_found_counter >= 10:
                break

            if not scroll_page(driver):
                break
    except Exception as e:
        logging.warning(f"Error while scrolling: {e}")

    return image_count, video_count

# ------------------------------
# Main workflow
# ------------------------------
@click.command()
@click.option("--base-url", required=False, default="https://www.threads.com/@", help="Base URL for image URLs.")
@click.option("--girls-list-file", required=False, default="grls.txt")
@click.option("--url", required=False, default=None, help="Target webpage URL containing images.")
@click.option("--cookies-file", "cookies_file", required=False, default="threads_cookies.txt", help="Netscape-format cookies txt path.")
@click.option("--output-dir", "output_dir", default="saved_people", show_default=True, help="Folder to save images with people.")
@click.option("--processed-log", "processed_log", default="processed.txt", show_default=True, help="File storing processed image URLs.")
@click.option("--headless/--no-headless", default=True, show_default=True, help="Run browser headless.")
@click.option("--yolo-model", default="yolo11s.pt")
def main(base_url, girls_list_file, url, cookies_file, output_dir, processed_log, headless, yolo_model):
    ensure_dir(output_dir)

    logging.info(f"Loading YOLO...")
    model = YOLO(yolo_model)

    logging.info(f"Starting Selenium...")
    driver = create_driver(headless=headless)

    # Shared state
    processed = load_processed(processed_log)
    enqueued: set[str] = set()  # avoid double-enqueue within run
    session = requests.Session()

    # Queue and downloader thread
    work_queue: "queue.Queue[DownloadItem]" = queue.Queue(maxsize=0)
    stop_event = threading.Event()

    # Progress UI with two concurrent tasks
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.completed}"),
        transient=False,
    ) as progress:
        found_task = progress.add_task("[magenta]Found (images+videos)", total=None)
        download_task = progress.add_task("[green]Downloading", total=None)

        # Start downloader thread
        downloader_thread = threading.Thread(
            target=downloader_thread_fn,
            args=(work_queue, stop_event, session, output_dir, processed_log, processed, progress, download_task, model),
            daemon=True,
        )
        downloader_thread.start()

        total_found = 0

        try:
            if url is not None:
                load_netscape_cookies(driver, url, cookies_file)
                logging.info(f"Collecting from: {url}")
                img_count, vid_count = load_girl_page_dynamic_to_queue(
                    driver, url, progress, found_task, enqueued, work_queue
                )
                total_found += img_count + vid_count
                logging.info(f"Found {img_count} image(s).")
                logging.info(f"Found {vid_count} videos(s).")
            elif base_url is not None and girls_list_file is not None:
                logging.info(f"Collecting images from file list")
                load_netscape_cookies(driver, "https://www.threads.com", cookies_file)
                with open(girls_list_file, 'r') as f:
                    for line in f:
                        girl = line.strip()
                        if girl:
                            try:
                                girl_url = base_url + girl + "/media"
                                logging.info(f"Collecting from: {girl_url}")
                                img_count, vid_count = load_girl_page_dynamic_to_queue(
                                    driver, girl_url, progress, found_task, enqueued, work_queue
                                )
                                total_found += img_count + vid_count
                                logging.info(f"Found {img_count} image(s) for {girl}")
                                logging.info(f"Found {vid_count} videos(s) for {girl}")
                            except Exception:
                                logging.exception("Error while collecting for user")

            # We don’t stop the downloader just because the queue is empty.
            # But at the end of the program we should stop it explicitly.
            # Wait for current queue to drain if you want to finish downloads before exit:
            work_queue.join()

        except Exception:
            logging.error(traceback.format_exc())
        finally:
            try:
                driver.quit()
            except Exception:
                pass

            # Explicitly stop the downloader thread
            stop_event.set()
            downloader_thread.join(timeout=5)

    # Post-processing: duplicates and similar
    dif = difPy.build(output_dir)
    duplicates = difPy.search(dif, similarity="duplicates")
    similar = difPy.search(dif, similarity="similar")
    try:
        duplicates.delete(silent_del=True)
    except Exception:
        pass
    try:
        similar.delete(silent_del=True)
    except Exception:
        pass

if __name__ == "__main__":
    main()
