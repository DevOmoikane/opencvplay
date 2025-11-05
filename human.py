import base64
import os
import time
import json
import signal
import shutil
import random
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from rich import print

import httpx
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import (
    NoSuchElementException, WebDriverException, TimeoutException, StaleElementReferenceException
)
from urllib.parse import urlparse
from PIL import Image, UnidentifiedImageError

# Optional: YOLO and DeepFace imports
# You can swap models based on your choice.
# For YOLO person detection using ultralytics:
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


APP_MAIN_URL = "https://www.threads.com"  # TODO: set
DATA_DIR = "./data"
ORIGINAL_DIR = "./original"
HUMAN_DIR = "./human"
DB_PATH = os.path.join(DATA_DIR, "state.db")

POST_SELECTOR = "div.xrvj5dj.xd0jker"
PROFILE_SELECTOR = "span.xjp7ctv div a.x1i10hfl.xjbqb8w.x1ejq31n.x18oe1m7.x1sy0etr.xstzfhl.x972fbf.x10w94by.x1qhh985.x14e42zd.x9f619.x1ypdohk.xt0psk2.x3ct3a4.xdj266r.x14z9mp.xat24cr.x1lziwak.xexx8yu.xyri2b.x18d9i69.x1c1uobl.x16tdsg8.x1hl2dhg.xggy1nq.x1a2a7pz.xp07o12.xzmqwrg.x1citr7e.x1kdxza.xt0b8zv"
IMAGE_SELECTOR = "img.xl1xv1r.x9f619.x1lliihq.xmz0i5r.x193iq5w.xuiwhb7.x1g40iwv.x47corl.x87ps6o.x1obq294.x5a5i1n.xde0f50.x15x8krk"

DEEPFACE_URL = "http://10.73.19.130:5100"

COOKIES_FILE = "threads_cookies.txt"

def convert_to_jpeg_if_needed(path: str) -> str:
    """
    If the file at 'path' is not JPEG or PNG, convert it to JPEG and return the (possibly new) path.
    For PNG with alpha, if converting, we flatten to a white background.
    """
    try:
        with Image.open(path) as im:
            fmt = (im.format or "").upper()
            if fmt in ("JPEG", "JPG", "PNG"):
                # Nothing to do
                return path

            # Convert to JPEG
            rgb = im.convert("RGB")  # handles most modes (including palette/webp etc.)
            new_path = os.path.splitext(path)[0] + ".jpg"
            rgb.save(new_path, format="JPEG", quality=90)
            try:
                os.remove(path)
            except Exception:
                pass
            return new_path
    except UnidentifiedImageError:
        # File isn't an image PIL understands, leave as-is
        return path
    except Exception:
        # Be defensive: if anything fails, keep original
        return path
    
def file_to_base64(file_path):
    """
    Loads a file from the given path and returns its content as a Base64 encoded string.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The Base64 encoded string of the file's content, or None if an error occurs.
    """
    try:
        with open(file_path, "rb") as file:  # Open the file in binary read mode
            encoded_bytes = base64.b64encode(file.read())  # Encode the file content
            encoded_string = encoded_bytes.decode("ascii")  # Decode bytes to a UTF-8 string
            return "data:image/jpeg;base64," + encoded_string
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def parse_netscape_cookies(path: str):
    """
    Parses a Netscape cookies.txt and yields dicts compatible with Selenium add_cookie.
    Netscape columns:
    domain, flag, path, secure, expiration, name, value
    Lines starting with # are comments (except #HttpOnly_ prefix before domain).
    """
    cookies = []
    if not os.path.exists(path):
        print(f"Warning: cookies file not found at {path}")
        return cookies

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                # Handle #HttpOnly_ prefix
                if line.startswith("#HttpOnly_"):
                    # Next non-comment read should include the domain without this marker.
                    # But some exports place the actual cookie on the same line without tabs,
                    # so we just skip here; real cookies typically are not on commented lines.
                    pass
                continue
            parts = line.split("\t")
            if len(parts) != 7:
                continue  # skip malformed
            domain, flag, path, secure, expiration, name, value = parts
            # expiration could be 0 or unix timestamp
            try:
                expiry = int(expiration)
            except ValueError:
                expiry = None

            cookie = {
                "domain": domain.lstrip("."),   # Selenium accepts domain without leading dot
                "path": path or "/",
                "secure": secure.upper() == "TRUE",
                "name": name,
                "value": value,
            }
            if expiry and expiry != 0:
                cookie["expiry"] = expiry
            cookies.append(cookie)
    return cookies


def domain_matches(cookie_domain: str, url_host: str) -> bool:
    """
    Cookie domain (without leading dot) matches host if host == domain or host endswith '.' + domain
    """
    cookie_domain = cookie_domain.lstrip(".")
    url_host = url_host.lstrip(".")
    if url_host == cookie_domain:
        return True
    return url_host.endswith("." + cookie_domain)


def add_cookies_for_url(driver, url: str, cookies: list):
    """
    Navigates to base of url host if needed, then adds all matching cookies for that host.
    We must visit a page on that domain before adding cookies.
    """
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.hostname or ''}/"
    # Navigate once per host
    try:
        if driver.current_url.startswith(base) is False:
            driver.get(base)
            rand_sleep(0.6, 1.2)
    except Exception:
        # If get fails (e.g., blocked), still try once
        driver.get(base)
        rand_sleep(0.6, 1.2)

    host = parsed.hostname or ""
    added = 0
    for c in cookies:
        if domain_matches(c.get("domain", ""), host):
            try:
                driver.add_cookie(c)
                added += 1
            except Exception:
                # Some cookies may fail (e.g., SameSite restrictions), ignore
                continue
    if added:
        # Refresh to apply cookies for this host context
        try:
            driver.refresh()
            rand_sleep(0.6, 1.2)
        except Exception:
            pass

class DeepFace:
    @staticmethod
    def analyze(img_path, actions, enforce_detection=False):
        """Analyze facial and body features"""
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")
        
        base64_img = file_to_base64(img_path)
        
        if actions is None or len(actions) == 0:
            actions = ['age', 'gender', 'race', 'emotion']

        response = httpx.post(
            f"{DEEPFACE_URL}/analyze",
            json={
                "img": base64_img,
                "actions": actions,
                "enforce_detection": enforce_detection,
            },
            timeout=None
        )
        
        if response.status_code == 200:
            rj = response.json()
            return rj
        
        raise Exception(f"Response wat not succesfull {response.status_code}: {response.text} ")
    
    @staticmethod
    def verify(img1_path, img2_path, enforce_detection=False):
        """Verify similarity between two images"""
        base64_img1 = file_to_base64(img1_path)
        base64_img2 = file_to_base64(img2_path)
        
        response = httpx.post(
            f"{DEEPFACE_URL}/verify",
            json={
                "img1": base64_img1,
                "img2": base64_img2
            },
            timeout=None
        )
        
        if response.status_code == 200:
            return response.json()
        
        return None
    
    @staticmethod
    def represent(img_path, model_name, enforce_detection=False):
        """Extract facial features"""
        if not os.path.exists(img_path):
            return None
        
        base64_img = file_to_base64(img_path)
        
        response = httpx.post(
            f"{DEEPFACE_URL}/represent",
            json={
                "img": base64_img,
                "model_name": model_name,
                "enforce_detection": enforce_detection,
            },
            timeout=None
        )

        if response.status_code == 200:
            return response.json()
        
        return None

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(HUMAN_DIR, exist_ok=True)


def rand_sleep(a: float, b: float):
    time.sleep(random.uniform(a, b))


@contextmanager
def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()


def init_db():
    with db_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            status TEXT DEFAULT 'pending' -- pending, done, failed
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT UNIQUE,
            source_url TEXT,        -- image URL
            profile_url TEXT,       -- associated profile
            page_type TEXT,         -- 'feed' or 'profile'
            status TEXT DEFAULT 'queued', -- queued, downloading, downloaded, processing, human, nonhuman, failed
            meta TEXT               -- JSON for extra info
        )
        """)


def enqueue_image(file_name: str, source_url: str, profile_url: Optional[str], page_type: str, meta: Optional[dict] = None):
    with db_conn() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO images (file_name, source_url, profile_url, page_type, status, meta)
            VALUES (?, ?, ?, ?, 'queued', ?)
        """, (file_name, source_url, profile_url or "", page_type, json.dumps(meta or {})))


def upsert_profile(url: str):
    with db_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO profiles (url, status) VALUES (?, 'pending')", (url,))


def next_profile_to_process() -> Optional[str]:
    with db_conn() as conn:
        cur = conn.execute("SELECT url FROM profiles WHERE status='pending' ORDER BY id ASC LIMIT 1")
        row = cur.fetchone()
        return row[0] if row else None


def set_profile_status(url: str, status: str):
    with db_conn() as conn:
        conn.execute("UPDATE profiles SET status=? WHERE url=?", (status, url))


def next_image_for_download() -> Optional[Tuple[int, str, str, str]]:
    with db_conn() as conn:
        cur = conn.execute("""
            SELECT id, source_url, file_name, page_type FROM images
            WHERE status='queued'
            ORDER BY id ASC
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            return None
        conn.execute("UPDATE images SET status='downloading' WHERE id=?", (row[0],))
        return row  # (id, source_url, file_name, page_type)


def set_image_status_by_id(image_id: int, status: str):
    with db_conn() as conn:
        conn.execute("UPDATE images SET status=? WHERE id=?", (status, image_id))


def next_image_for_processing() -> Optional[Tuple[int, str, str]]:
    with db_conn() as conn:
        cur = conn.execute("""
            SELECT id, file_name, page_type FROM images
            WHERE status='downloaded'
            ORDER BY id ASC
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            return None
        conn.execute("UPDATE images SET status='processing' WHERE id=?", (row[0],))
        return row  # (id, file_name, page_type)


def set_image_result(image_id: int, status: str):
    with db_conn() as conn:
        conn.execute("UPDATE images SET status=? WHERE id=?", (status, image_id))


class ImageDownloader(threading.Thread):
    def __init__(self, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    def run(self):
        while not self.stop_event.is_set():
            job = next_image_for_download()
            if not job:
                rand_sleep(0.5, 1.5)
                continue
            image_id, src_url, file_name, _ = job
            dst_path = os.path.join(ORIGINAL_DIR, file_name)
            try:
                # Single image at a time by design (one downloader thread).
                r = self.session.get(src_url, timeout=20, stream=True)
                r.raise_for_status()
                with open(dst_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 64):
                        if chunk:
                            f.write(chunk)
                final_path = convert_to_jpeg_if_needed(dst_path)
                if final_path != dst_path:
                    # Overwrite original filename with converted content
                    try:
                        shutil.copy2(final_path, dst_path)
                        if os.path.exists(final_path) and final_path != dst_path:
                            os.remove(final_path)
                    except Exception:
                        # If overwrite fails, we'll just keep the converted file and mark as downloaded anyway.
                        pass
                set_image_status_by_id(image_id, "downloaded")
                rand_sleep(0.7, 1.3)  # human-like pacing
            except Exception:
                set_image_status_by_id(image_id, "failed")
                rand_sleep(1.0, 2.0)


class ImageProcessor(threading.Thread):
    def __init__(self, stop_event: threading.Event, use_yolo: bool = True, use_deepface: bool = True):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.use_deepface = use_deepface
        self.yolo_model = None
        try:
            # A small model is fine (e.g., yolov8n)
            self.yolo_model = YOLO("./human.pt")  # ensure weights are available
        except Exception:
            raise Exception("YOLO model not found. Install YOLO with 'pip install ultralytics' and download weights.")

    def contains_person(self, image_path: str) -> bool:
        # Prefer YOLO person detection; fall back to DeepFace 'detectFace' if YOLO missing.
        try:
            results = self.yolo_model.predict(image_path, imgsz=640, verbose=True)
            for res in results:
                for box, cls in zip(res.boxes, res.boxes.cls):
                    # Class 0 is person for COCO
                    if int(cls.item()) == 0:
                        return True
            return False
        except Exception:
            pass

    def classify_gender(self, image_path: str) -> Optional[str]:
        if not self.use_deepface:
            return None
        try:
            # DeepFace analyze returns gender in 'gender' key for models that provide it.
            analysis = DeepFace.analyze(img_path=image_path, actions=["gender"], enforce_detection=False)
            # analysis can be list or dict depending on version
            print(analysis)
            if isinstance(analysis, list) and analysis:
                analysis = analysis[0]
            gender = analysis.get("gender")
            if isinstance(gender, dict):
                # Choose the label with max probability
                return max(gender, key=gender.get)
            if isinstance(gender, str):
                return gender
            return None
        except Exception:
            return None

    def run(self):
        while not self.stop_event.is_set():
            job = next_image_for_processing()
            if not job:
                rand_sleep(0.5, 1.2)
                continue
            image_id, file_name, _ = job
            path = os.path.join(ORIGINAL_DIR, file_name)
            try:
                has_person = self.contains_person(path)
                if has_person:
                    # Optional gender classification
                    _ = self.classify_gender(path)
                    dst = os.path.join(HUMAN_DIR, file_name)
                    shutil.copy2(path, dst)
                    set_image_result(image_id, "human")
                else:
                    set_image_result(image_id, "nonhuman")
                rand_sleep(0.5, 1.0)
            except Exception:
                set_image_result(image_id, "failed")
                rand_sleep(0.7, 1.5)


@dataclass
class PostData:
    profile_name: str
    profile_url: str
    image_urls: List[str]


class SeleniumBase:
    def __init__(self):
        opts = Options()
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--window-size=1200,900")
        # Headful looks more human; consider not using headless:
        opts.add_argument("--headless=new")
        self.driver = webdriver.Chrome(options=opts)
        self.driver.implicitly_wait(5)
        self._cookies_list = parse_netscape_cookies(COOKIES_FILE)

    def ensure_cookies_for(self, url: str):
        if not self._cookies_list:
            return
        try:
            add_cookies_for_url(self.driver, url, self._cookies_list)
        except Exception:
            pass

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass

    def human_scroll(self, step_px=900, max_idle_scrolls=10):
        # last_height = self.driver.execute_script("return document.body.scrollHeight")
        # viewport_height = self.driver.execute_script("return window.innerHeight")
        # idle_scrolls = 0
        # while True:
        #     self.driver.execute_script(f"window.scrollBy(0, {viewport_height-last_height+200});")
        #     rand_sleep(0.6, 1.4)
        #     new_height = self.driver.execute_script("return document.body.scrollHeight")
        #     if new_height <= last_height:
        #         idle_scrolls += 1
        #     else:
        #         idle_scrolls = 0
        #     last_height = new_height
        #     if idle_scrolls >= max_idle_scrolls:
        #         break
        try:
            self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        except Exception:
            pass

    def get_new_posts(self) -> List[PostData]:
        # You must adjust selectors here to match the site.
        posts = []
        try:
            post_elements = self.driver.find_elements(By.CSS_SELECTOR, POST_SELECTOR)  # TODO
            for el in post_elements:
                try:
                    profile_link = el.find_element(By.CSS_SELECTOR, PROFILE_SELECTOR)
                    profile_name = profile_link.text.strip()
                    profile_url = profile_link.get_attribute("href")
                    img_elements = el.find_elements(By.CSS_SELECTOR, IMAGE_SELECTOR)  # TODO
                    image_urls = []
                    for img in img_elements:
                        src = img.get_attribute("src")
                        if src and src.startswith("http"):
                            image_urls.append(src)
                    if profile_url and image_urls:
                        posts.append(PostData(profile_name, profile_url, image_urls))
                except (NoSuchElementException, StaleElementReferenceException):
                    continue
        except Exception:
            pass
        return posts


class SeleniumFeedCrawler(SeleniumBase):
    def crawl_feed_incremental(self, until_interrupted: threading.Event):
        self.ensure_cookies_for(APP_MAIN_URL)
        self.driver.get(APP_MAIN_URL)
        rand_sleep(2.0, 3.0)

        seen_images: set[str] = set()
        seen_profiles: set[str] = set()
        while not until_interrupted.is_set():
            posts = self.get_new_posts()
            for post in posts:
                if post.profile_url:
                    upsert_profile(post.profile_url)
                    seen_profiles.add(post.profile_url)
                for img_url in post.image_urls:
                    if img_url in seen_images:
                        continue
                    # Create filename that relates to profile
                    file_name = self._make_file_name(post.profile_url, img_url)
                    enqueue_image(file_name, img_url, post.profile_url, page_type="feed", meta={"profile_name": post.profile_name})
                    seen_images.add(img_url)
                    # Sleep a bit between image discoveries to look natural
                    rand_sleep(0.2, 0.5)

            # Scroll a bit, but not infinite – break only on user interrupt
            self.human_scroll(step_px=random.randint(700, 1200), max_idle_scrolls=3)

    @staticmethod
    def _make_file_name(profile_url: str, image_url: str) -> str:
        # Sanitize profile and image identifiers for filename
        def slug(s: str) -> str:
            return "".join(c for c in s if c.isalnum() or c in ("-", "_")).strip("_")[:50]
        p = slug(profile_url.replace("https://", "").replace("http://", "").replace("/", "_"))
        h = slug(str(abs(hash(image_url))))
        return f"{p}__{h}.jpg"


class SeleniumProfileCrawler(SeleniumBase):
    def crawl_profile_images(self, profile_url: str):
        try:
            self.ensure_cookies_for(profile_url)
            self.driver.get(profile_url)
            rand_sleep(1.5, 2.5)
            seen_images: set[str] = set()
            while True:
                posts = self.get_new_posts()
                new_found = 0
                for post in posts:
                    for img_url in post.image_urls:
                        if img_url in seen_images:
                            continue
                        file_name = self._make_file_name(profile_url, img_url)
                        enqueue_image(file_name, img_url, profile_url, page_type="profile", meta={"profile_name": post.profile_name})
                        seen_images.add(img_url)
                        new_found += 1
                        rand_sleep(0.2, 0.5)
                prev_count = len(seen_images)
                self.human_scroll(step_px=random.randint(700, 1200), max_idle_scrolls=3)
                # If no new images detected after a few scrolls, exit
                if new_found == 0:
                    break
        except WebDriverException:
            pass

    @staticmethod
    def _make_file_name(profile_url: str, image_url: str) -> str:
        def slug(s: str) -> str:
            return "".join(c for c in s if c.isalnum() or c in ("-", "_")).strip("_")[:50]
        p = slug(profile_url.replace("https://", "").replace("http://", "").replace("/", "_"))
        h = slug(str(abs(hash(image_url))))
        return f"{p}__{h}.jpg"


class CrawlerController:
    def __init__(self):
        self.interrupt_event = threading.Event()
        self.downloader_stop = threading.Event()
        self.processor_stop = threading.Event()
        self.downloader = ImageDownloader(self.downloader_stop)
        self.processor = ImageProcessor(self.processor_stop)
        self.phase = "feed"  # 'feed' -> 'profiles'
        self._register_signal_handler()

    def _register_signal_handler(self):
        def on_sigint(signum, frame):
            print("Interrupt received. Switching to profiles phase once feed loop exits...")
            self.phase = "profiles"
            self.interrupt_event.set()
        signal.signal(signal.SIGINT, on_sigint)

    def run(self):
        init_db()
        # Start background workers
        self.downloader.start()
        self.processor.start()

        # Phase 1: Feed crawling until user interrupts
        feed = SeleniumFeedCrawler()
        try:
            print("Starting main feed crawl. Press Ctrl+C to switch to Profiles phase.")
            feed.crawl_feed_incremental(self.interrupt_event)
        finally:
            feed.close()

        # Phase 2: Process profiles one by one
        print("Processing saved profiles...")
        while True:
            profile_url = next_profile_to_process()
            if not profile_url:
                print("No more profiles to process. Waiting for remaining queue to finish...")
                break
            set_profile_status(profile_url, "processing")
            prof = SeleniumProfileCrawler()
            try:
                prof.crawl_profile_images(profile_url)
                set_profile_status(profile_url, "done")
            except Exception:
                set_profile_status(profile_url, "failed")
            finally:
                prof.close()
            # Short pause between profiles
            rand_sleep(1.0, 2.0)

        # Let background workers finish outstanding work for a while
        try:
            print("Draining queues. Press Ctrl+C again to stop workers and exit.")
            while True:
                # If no queued or downloaded items left, we can stop
                with db_conn() as conn:
                    q = conn.execute("SELECT COUNT(*) FROM images WHERE status IN ('queued','downloading','downloaded','processing')").fetchone()[0]
                if q == 0:
                    break
                rand_sleep(1.5, 2.5)
        except KeyboardInterrupt:
            pass

        # Shutdown workers
        self.downloader_stop.set()
        self.processor_stop.set()
        print("Shutting down workers...")
        self.downloader.join(timeout=5)
        self.processor.join(timeout=5)
        print("All done.")


if __name__ == "__main__":
    CrawlerController().run()
