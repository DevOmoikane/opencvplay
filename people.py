# Python
import os
import time
import random
import traceback

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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(funcName)s:%(lineno)d - %(message)s')
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ------------------------------
# YOLO (OpenCV DNN) configuration
# ------------------------------
# Provide your YOLO weights/config files. Example below uses YOLOv3 COCO (person class id=0)
# You can replace with any YOLO model supported by OpenCV DNN (ONNX or Darknet).
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolo11s.onnx")
YOLO_CFG = os.environ.get("YOLO_CFG", "yolo11s.cfg")
YOLO_NAMES = os.environ.get("YOLO_NAMES", "coco.names")

PERSON_CLASS_IDS = {0}  # in COCO, class 0 is 'person'
CONF_THRESH = 0.35
NMS_THRESH = 0.4

# ------------------------------
# Selenium helpers
# ------------------------------
POST_SELECTOR = "div.xrvj5dj.xd0jker"
PROFILE_SELECTOR = "span.xjp7ctv div a.x1i10hfl.xjbqb8w.x1ejq31n.x18oe1m7.x1sy0etr.xstzfhl.x972fbf.x10w94by.x1qhh985.x14e42zd.x9f619.x1ypdohk.xt0psk2.x3ct3a4.xdj266r.x14z9mp.xat24cr.x1lziwak.xexx8yu.xyri2b.x18d9i69.x1c1uobl.x16tdsg8.x1hl2dhg.xggy1nq.x1a2a7pz.xp07o12.xzmqwrg.x1citr7e.x1kdxza.xt0b8zv"
IMAGE_SELECTOR = "img.xl1xv1r.x9f619.x1lliihq.xmz0i5r.x193iq5w.xuiwhb7.x1g40iwv.x47corl.x87ps6o.x1obq294.x5a5i1n.xde0f50.x15x8krk"
IMAGE_SELECTOR_ALT = "img.xl1xv1r.x9f619.x1lliihq.xmz0i5r.x193iq5w.xuiwhb7.x1g40iwv.x47corl.x87ps6o.x1obq294.x5a5i1n.xde0f50.x15x8krk.x1ey2m1c.xtijo5x.x1o0tod.x10l6tqk.x13vifvy.x5yr21d.xh8yej3"
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
    """Scrolls the page and returns True if new content was loaded"""
    last_height = driver.execute_script("return document.body.scrollHeight")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(scroll_pause)  # Wait for content to load
    new_height = driver.execute_script("return document.body.scrollHeight")
    return new_height != last_height


def load_netscape_cookies(driver: webdriver.Chrome, base_url: str, cookie_file_path: str):
    """
    Loads cookies in Netscape format into the Selenium driver.
    Cookies must match the domain; we first open base_url so we can set cookies.
    """

    logging.info(f"Loading cookies from {cookie_file_path}")
    if not os.path.exists(cookie_file_path):
        raise FileNotFoundError(f"Cookie file not found: {cookie_file_path}")

    driver.get(base_url)  # required before adding cookies

    with open(cookie_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 7:
                # Some files may use spaces instead of tabs; try splitting on whitespace
                parts = line.split()
                if len(parts) != 7:
                    continue
            domain, flag, path, secure, expiry, name, value = parts
            # Selenium requires booleans and ints for some fields
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
                # Ignore cookies that don't match current domain/subdomain
                pass

def load_girl_page(driver: webdriver.Chrome, base_url: str):
    # Refresh to apply cookies
    driver.get(base_url)

    # Wait for initial page load
    time.sleep(3)

    # Scroll until no new content loads or max attempts reached
    max_scrolls = 20
    scroll_count = 0
    try:
        while scroll_count < max_scrolls:
            if not scroll_page(driver):
                break  # No new content loaded
            scroll_count += 1
    except Exception as e:
        logging.warning(f"Error while scrolling: {e}")

def remove_duplicates(array1: list[str], array2: list[str]) -> list[str]:
    return [item for item in array2 if item not in array1]

def load_girl_page_dynamic(driver: webdriver.Chrome, base_url: str) -> tuple[list[str], list[str]]:
    driver.get(base_url)
    time.sleep(3)
    scroll_count = 0
    image_array = []
    video_array = []
    try:
        not_found_counter = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[cyan]{task.completed}"),
            transient=False,
        ) as progress:
            task = progress.add_task("[red]Obtaining links...", total=None)
            while True:
                images = find_image_urls(driver, base_url)
                videos = find_video_urls(driver, base_url)
                images = remove_duplicates(image_array, images)
                videos = remove_duplicates(video_array, videos)
                progress.update(task, completed=(len(image_array)+len(video_array)))
                if len(images)==0 and len(videos)==0:
                    not_found_counter += 1
                if not_found_counter >= 10:
                    break
                if len(images) > 0:
                    image_array.extend(images)
                if len(videos) > 0:
                    video_array.extend(videos)
                if not scroll_page(driver):
                    break
    except Exception as e:
        logging.warning(f"Error while scrolling: {e}")
    return image_array, video_array

# ------------------------------
# Image scraping helpers
# ------------------------------
def find_image_urls(driver: webdriver.Chrome, base_url: str) -> list[str]:
    # imgs = driver.find_elements(By.TAG_NAME, "img")
    imgs = driver.find_elements(By.CSS_SELECTOR, IMAGE_SELECTOR)
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
        # Resolve relative URLs
        if src.startswith("http"):
            urls.append(src)
        else:
            full = urljoin(base_url, src)
            urls.append(full)
    # Deduplicate while preserving order
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
        # Resolve relative URLs
        if src.startswith("http"):
            urls.append(src)
        else:
            full = urljoin(base_url, src)
            urls.append(full)
    # Deduplicate while preserving order
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

# ------------------------------
# Persistence for processed URLs
# ------------------------------
def load_processed(file_path: str) -> set[str]:
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def append_processed(file_path: str, url: str):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(url + "\n")

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
    # net, names, backend = load_yolo()
    model = YOLO(yolo_model)

    logging.info(f"Starting Selenium...")
    driver = create_driver(headless=headless)
    try:
        image_urls = []
        video_urls = []
        if url is not None:
            load_netscape_cookies(driver, url, cookies_file)
            logging.info(f"Collecting from: {url}")
            image_urls, video_urls = load_girl_page_dynamic(driver, url)
            logging.info(f"Found {len(image_urls)} image(s).")
            logging.info(f"Found {len(video_urls)} videos(s).")
        elif base_url is not None and girls_list_file is not None:
            logging.info(f"Collecting images from file list")
            load_netscape_cookies(driver, "https://www.threads.com", cookies_file)
            with open(girls_list_file, 'r') as f:
                for line in f:
                    girl = line.strip()
                    if girl is not None:
                        try:
                            girl_url = base_url + girl
                            logging.info(f"Collecting from: {girl_url}")
                            images, videos = load_girl_page_dynamic(driver, girl_url)
                            logging.info(f"Found {len(images)} image(s) for {girl}")
                            logging.info(f"Found {len(videos)} videos(s) for {girl}")
                            image_urls.extend(images)
                            video_urls.extend(videos)
                        except Exception:
                            pass

        processed = load_processed(processed_log)
        session = requests.Session()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.completed:>6}"),
            transient=False,
        ) as progress:
            task = progress.add_task("[red]Downloading...", total=len(image_urls) + len(video_urls))
            try:
                for video_url in video_urls:
                    download_link(video_url, session, output_dir)
                    progress.update(task, advance=1)
            except Exception as e:
                logging.info(f"[ERROR] {video_url}")

            for idx, img_url in enumerate(image_urls, start=1):
                progress.update(task, advance=1)
                if img_url in processed:
                    continue

                # Random delay 2–5 seconds
                delay = random.uniform(0.10, 1.0)
                time.sleep(delay)

                data = download_image_bytes(img_url, session)
                if not data:
                    append_processed(processed_log, img_url)
                    continue

                # Decode image
                image_array = cv2.imdecode(
                    np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image_array is None:
                    append_processed(processed_log, img_url)
                    continue

                results = model.predict(image_array)
                has_person = False
                for result in results:
                    for box in result.boxes:
                        if int(box.cls) in PERSON_CLASS_IDS:
                            has_person = True
                            break
                    if has_person:
                        break
                if has_person:
                    parsed = urlparse(img_url)
                    base_name = os.path.basename(parsed.path) or f"image_{idx}.jpg"
                    # Ensure extension
                    if not os.path.splitext(base_name)[1]:
                        base_name += ".jpg"
                    save_path = os.path.join(output_dir, base_name)
                    # If duplicate name, append index
                    root, ext = os.path.splitext(save_path)
                    c = 1
                    while os.path.exists(save_path):
                        save_path = f"{root}_{c}{ext}"
                        c += 1
                    try:
                        # cv2.imwrite(save_path, image_array)
                        download_link(img_url, session, output_dir)
                        # logging.info(f"[SAVED] {save_path}")
                    except Exception as e:
                        logging.info(f"[ERROR] {save_path}")

                # Mark as processed either way
                append_processed(processed_log, img_url)

        logging.info("[DONE] Processing complete.")
    except Exception as e:
        # traceback.print_exc()
        logging.error(traceback.format_exc())
    finally:
        try:
            driver.quit()
        except Exception:
            pass

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
