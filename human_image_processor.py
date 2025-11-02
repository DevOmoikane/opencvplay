# Python
import asyncio
import base64
import json
from urllib.parse import urljoin
import aiohttp
import logging
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from queue import Queue
import threading
import time
import random
import os
from PIL import Image
import io
import pickle
import signal
from typing import List, Dict, Optional, Set
from rich.logging import RichHandler
import ollama
from ultralytics import YOLO
import difPy
from dotenv import dotenv_values
from dataclasses import dataclass


config = {
    **dotenv_values(".env"),
    **os.environ,
}

@dataclass
class ImageResult:
    profile_url: str
    image_url: str

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolo11s.onnx")
YOLO_CFG = os.environ.get("YOLO_CFG", "yolo11s.cfg")
YOLO_NAMES = os.environ.get("YOLO_NAMES", "coco.names")

PERSON_CLASS_IDS = {0}  # in COCO, class 0 is 'person'
CONF_THRESH = 0.35
NMS_THRESH = 0.4

POST_SELECTOR = "div.xrvj5dj.xd0jker"
PROFILE_SELECTOR = "span.xjp7ctv div a.x1i10hfl.xjbqb8w.x1ejq31n.x18oe1m7.x1sy0etr.xstzfhl.x972fbf.x10w94by.x1qhh985.x14e42zd.x9f619.x1ypdohk.xt0psk2.x3ct3a4.xdj266r.x14z9mp.xat24cr.x1lziwak.xexx8yu.xyri2b.x18d9i69.x1c1uobl.x16tdsg8.x1hl2dhg.xggy1nq.x1a2a7pz.xp07o12.xzmqwrg.x1citr7e.x1kdxza.xt0b8zv"
IMAGE_SELECTOR = "img.xl1xv1r.x9f619.x1lliihq.xmz0i5r.x193iq5w.xuiwhb7.x1g40iwv.x47corl.x87ps6o.x1obq294.x5a5i1n.xde0f50.x15x8krk"

class HumanLikeImageProcessor:
    def __init__(
        self,
        max_workers: int = 5,
        download_delay: tuple = (2, 5),
        scroll_delay: tuple = (3, 7),
        headless: bool = True,
        cookies_file: str = None,
    ):
        self.max_workers = max_workers
        self.download_delay = download_delay
        self.scroll_delay = scroll_delay
        self.headless = headless
        self.cookies_file = cookies_file
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None  # for workers

        # Async streaming queues/state
        self.image_async_queue: Optional[asyncio.Queue] = None  # for URLs
        self.processed_queue = Queue()  # keep as sync if external code expects it
        self._producer_done = asyncio.Event()  # marks the end of production
        self._stop_event = threading.Event()
        self._shutdown_initiated = False

        # Sets to avoid duplicates and track progress
        self._seen_images_lock = threading.Lock()
        self._seen_images: Set[str] = set()

        self.model = YOLO("yolo11s.pt")

        # Browser
        self.driver = None

        # Logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        # Signals and browser
        self._setup_signal_handlers()
        self._setup_browser()

    # Python
    # Python
    def _enqueue_url_blocking(self, src: str, timeout: float | None = None) -> bool:
        """
        Enqueue src into the asyncio queue from a non-async thread.
        Blocks this thread until the item is put (backpressure) or timeout.
        Returns True if enqueued, False if stopped or timeout.
        """
        if self._event_loop is None or self.image_async_queue is None:
            return False
        if self._stop_event.is_set():
            return False

        async def _put_once():
            await self.image_async_queue.put(src)
            return True

        fut = asyncio.run_coroutine_threadsafe(_put_once(), self._event_loop)
        try:
            return fut.result(timeout=timeout)
        except Exception:
            return False

    def _setup_signal_handlers(self):
        def signal_handler(signum, frame):
            self.logger.info("Received interrupt signal. Initiating graceful shutdown...")
            self.initiate_shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _setup_browser(self):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        if self.cookies_file and os.path.exists(self.cookies_file):
            self._load_cookies()

    def _load_cookies(self):
        try:
            with open(self.cookies_file, "r", encoding="utf-8") as file:
                for line in file:
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
                        self.driver.get("https://threads.com")
                        self.driver.add_cookie(cookie_dict)
                    except Exception as e:
                        self.logger.warning(f"Could not add cookie: {e}")
                        continue
            self.logger.info(f"Loaded cookies from {self.cookies_file}")
        except Exception as e:
            self.logger.error(f"Error loading cookies: {e}")

    def _human_delay(self, delay_range: tuple):
        delay = random.uniform(*delay_range)
        end_time = time.time() + delay
        while time.time() < end_time and not self._stop_event.is_set():
            time.sleep(0.1)

    def continuous_scroll_and_collect_streaming(self, url: str, max_scrolls: int = 100):
        """
        Producer: Scrolls and enqueues images into the async queue as they appear.
        Runs in the main thread (Selenium). The consumer workers are asyncio tasks.
        """
        self.logger.info(f"Starting continuous scroll on: {url}")
        try:
            self.driver.get(url)
            self._human_delay((3, 5))
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            scroll_count = 0
            consecutive_no_new_images = 0
            total_enqueued = 0

            # first scroll up to refresh new items
            self._scroll_up()
            self._human_delay((10, 20))

            while (
                not self._stop_event.is_set()
                and scroll_count < max_scrolls
                and consecutive_no_new_images < 5
            ):
                # Collect current images
                new_images = self._collect_visible_images()
                #TODO: also use the profile url to get all the images in that url
                # Determine truly new URLs and enqueue
                new_count = 0
                for imageresult in new_images:
                    if self._mark_seen_if_new(imageresult.image_url):
                        # Put into async queue via asyncio.run_coroutine_threadsafe if needed.
                        # Here we assume we are called from the same thread that owns the loop via loop.call_soon_threadsafe
                        # Block until there is space (applies backpressure)
                        if self._enqueue_url_blocking(imageresult.image_url, timeout=1200):
                            new_count += 1
                        else:
                            # If we couldn’t enqueue (timeout or shutdown), stop producing further
                            self.logger.debug("Failed to enqueue (timeout or shutdown); stopping producer loop")
                            self._stop_event.set()
                            break
                if new_count > 0:
                    total_enqueued += new_count
                    consecutive_no_new_images = 0
                    self.logger.info(
                        f"Scroll {scroll_count + 1}: Enqueued {new_count} new images (Total enqueued: {total_enqueued})"
                    )
                else:
                    consecutive_no_new_images += 1
                    self.logger.info(f"Scroll {scroll_count + 1}: No new images found")

                if not self._scroll_down():
                    self.logger.info("Reached bottom of page")
                    break

                scroll_count += 1
                self._human_delay(self.scroll_delay)

            self.logger.info(f"Scroll session completed. Total enqueued: {total_enqueued}")
        except Exception as e:
            self.logger.error(f"Error during continuous scroll: {e}")
        finally:
            # Signal that producer is done
            if self._event_loop is not None:
                self._event_loop.call_soon_threadsafe(self._producer_done.set)
            else:
                try:
                    self._producer_done.set()
                except RuntimeError:
                    # If no running loop (e.g., shutdown), set directly for safety if awaited elsewhere
                    pass

    def _mark_seen_if_new(self, url: str) -> bool:
        with self._seen_images_lock:
            if url in self._seen_images:
                return False
            self._seen_images.add(url)
            return True

    def _collect_visible_images(self) -> set:
        try:
            visible_images = set()
            # images = self.driver.find_elements(By.TAG_NAME, "img")
            post_elems = self.driver.find_elements(By.CSS_SELECTOR, POST_SELECTOR)
            for post_elem in post_elems:
                profile_elem = post_elem.find_element(By.CSS_SELECTOR, PROFILE_SELECTOR)
                profile_url = urljoin("https://www.threads.com/", profile_elem.get_attribute("href"))
                images = post_elem.find_elements(By.CSS_SELECTOR, IMAGE_SELECTOR)
                for img in images:
                    try:
                        if img.is_displayed():
                            src = img.get_attribute("src")
                            if src:
                                visible_images.add(ImageResult(src, profile_url))
                    except Exception:
                        continue
            return visible_images
        except Exception as e:
            self.logger.warning(f"Error collecting visible images: {e}")
            return set()

    def _scroll_up(self) -> bool:
        try:
            current_position = self.driver.execute_script("return window.pageYOffset")
            scroll_distance = random.randint(700, 800)
            self.driver.execute_script(
                f"window.scrollTo({{top: {current_position - scroll_distance}, behavior: 'smooth'}});"
            )
            return True
        except Exception as e:
            self.logger.warning(f"Error scrolling: {e}")
            return False

    def _scroll_down(self) -> bool:
        try:
            current_position = self.driver.execute_script("return window.pageYOffset")
            viewport_height = self.driver.execute_script("return window.innerHeight")
            total_height = self.driver.execute_script("return document.body.scrollHeight")
            if current_position + viewport_height >= total_height:
                return False
            scroll_distance = random.randint(300, 800)
            self.driver.execute_script(
                f"window.scrollTo({{top: {current_position + scroll_distance}, behavior: 'smooth'}});"
            )
            return True
        except Exception as e:
            self.logger.warning(f"Error scrolling: {e}")
            return False

    async def download_image(self, image_url: str) -> Optional[bytes]:
        if self._stop_event.is_set():
            return None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        self.logger.info(f"Downloaded image: {image_url}")
                        return image_data
                    else:
                        self.logger.warning(
                            f"Failed to download image: {image_url}, Status: {response.status}"
                        )
                        return None
        except Exception as e:
            self.logger.error(f"Error downloading image {image_url}: {e}")
            return None

    def resize_image(self, image_data: bytes, max_size: tuple = (800, 600)) -> bytes:
        try:
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=85)
            return output.getvalue()
        except Exception as e:
            self.logger.error(f"Error resizing image: {e}")
            return image_data
        
    def save_image(self, image_data: bytes, filename: str):
        try:
            # check the directory
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                # if directory does not exist, create it
                os.makedirs(directory)
            image = Image.open(io.BytesIO(image_data))
            with open(filename, 'wb') as f:
                image.save(f, format="JPEG", quality=85)
        except Exception as e:
            self.logger.error(f"Error saving image: {e}")
            return

    async def image_to_base64(self, image_data: bytes) -> str:
        try:
            return "data:image/jpeg;base64," + base64.b64encode(image_data).decode("ascii")
        except Exception as e:
            self.logger.error(f"Error converting image to base64: {e}")
            return ""

    async def is_a_person_in_image(self, image_data: bytes) -> bool:
        try:
            image_array = cv2.imdecode(
                np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image_array is None:
                return False
            
            num_people=0
            results = self.model.predict(image_array)
            has_person = False
            for result in results:
                for box in result.boxes:
                    if int(box.cls) in PERSON_CLASS_IDS:
                        has_person = True
                        num_people = num_people + 1
            return has_person and num_people == 1
        except Exception as e:
            self.logger.error(f"Error checking if image contains a person: {e}")
            return False

    async def process_with_deepface(self, image_data: bytes) -> Optional[Dict]:
        if self._stop_event.is_set():
            return None
        try:
            url = f"{config['DEEPFACE_URL']}/analyze"
            async with aiohttp.ClientSession() as session:
                json_data = {
                    "img": await self.image_to_base64(image_data),
                    "actions": ["age", "gender", "race", "emotion"],
                    "enforce_detection": False,
                }
                logging.debug(json_data)
                async with session.post(url, json=json_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        logging.debug(result)
                        return result
                    else:
                        self.logger.error(f"DeepFace API error: {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Error calling DeepFace: {e}")
            return None

    async def describe_with_ollama(self, deepface_result: Dict, image_data: bytes = None) -> Optional[str]:
        if self._stop_event.is_set():
            return None
        try:
            url = f"{config['OLLAMA_URL']}/api/generate"
            prompt = f"""
            Based on this facial analysis:
            - Age: {deepface_result.get('age', 'unknown')}
            - Gender: {deepface_result.get('gender', 'unknown')}
            - Emotion: {deepface_result.get('dominant_emotion', 'unknown')}
            - Race: {deepface_result.get('dominant_race', 'unknown')}

            Provide a natural, human-like description of this person.
            """
            result = ollama.generate(model="qwen3-vl:4b", prompt=prompt, stream=False, images=[image_data] if image_data else None)
            return result.response
        except Exception as e:
            self.logger.error(f"Error calling Ollama: {e}")
            return None

    async def process_single_image(self, image_url: str):
        if self._stop_event.is_set():
            return None
        self.logger.info(f"Processing image: {image_url}")
        image_data = await self.download_image(image_url)
        if not image_data:
            return None
        
        image = Image.open(io.BytesIO(image_data))
        if image.height < 400 or image.width < 400:
            return None

        new_file_name = 'woman_' + str(time.time())

        has_person = await self.is_a_person_in_image(image_data)
        if not has_person:
            return None
        
        self.save_image(image_data, os.path.join('original', new_file_name + '.jpg'))
        
        self._human_delay(self.download_delay)

        resized_image = self.resize_image(image_data)
        deepface_result = await self.process_with_deepface(resized_image)
        if not deepface_result:
            return None
        
        if deepface_result['results'][0]['dominant_gender'] == 'Woman':
            # save the image with the filename as current time
            new_file_path = os.path.join('women', new_file_name + '.jpg')
            self.save_image(image_data, new_file_path)
        else:
            return None

        # description = await self.describe_with_ollama(deepface_result, resized_image)
        result = {
            "image_url": image_url,
            "deepface_analysis": deepface_result,
            # "description": description,
            "processed_at": time.time(),
        }
        # save result to a text file as json
        directory = os.path.dirname(os.path.join('women', new_file_name + '.txt'))
        if not os.path.exists(directory):
            # if directory does not exist, create it
            os.makedirs(directory)
        with open(os.path.join('women', new_file_name + '.txt'), 'a') as f:
            f.write(json.dumps(result) + '\n')

        self.processed_queue.put(result)
        return result

    async def _worker(self, worker_id: int):
        """
        Consumer worker: pulls image URLs from the async queue and processes them.
        Exits when producer is done AND queue is empty OR shutdown is initiated.
        """
        self.logger.info(f"Worker {worker_id} started")
        try:
            while not self._stop_event.is_set():
                try:
                    # Use timeout so we can check stop flags periodically
                    url = await asyncio.wait_for(self.image_async_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # If producer done and queue empty, exit
                    if self._producer_done.is_set() and self.image_async_queue.empty():
                        break
                    continue

                try:
                    await self.process_single_image(url)
                except Exception as e:
                    self.logger.error(f"Worker {worker_id} error processing {url}: {e}")
                finally:
                    self.image_async_queue.task_done()

                # If shutdown initiated, optionally allow finishing current task and break
                if self._shutdown_initiated:
                    break

        finally:
            self.logger.info(f"Worker {worker_id} exiting")

    # Python
    def initiate_shutdown(self):
        if self._shutdown_initiated:
            return
        self._shutdown_initiated = True
        self._stop_event.set()
        # Wake any waiters on producer_done so workers can exit their wait path
        try:
            if self._event_loop is not None:
                self._event_loop.call_soon_threadsafe(self._producer_done.set)
            else:
                self._producer_done.set()
        except Exception:
            pass
        # Best-effort: if queue exists and is bounded, try to wake blocked put/get
        # No sentinel consumers needed because workers exit when _producer_done and queue empty
        self.logger.info("Shutdown initiated. Waiting for processing to complete...")


    # Python
    async def wait_for_completion(self, timeout: int = 300):
        start = time.time()

        # Wait for producer to announce completion or shutdown
        try:
            await asyncio.wait_for(
                asyncio.shield(self._producer_done.wait()),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Timeout waiting for producer to finish")

        # Then wait for queue to drain, but bail out faster if shutdown
        while time.time() - start < timeout:
            if self.image_async_queue.empty():
                break
            if self._shutdown_initiated and time.time() - start > 10:
                # After a brief grace, exit even if not fully drained
                self.logger.info("Shutdown: exiting before queue fully drained")
                break
            await asyncio.sleep(0.2)
        else:
            self.logger.warning("Timeout reached while waiting for queue to drain")

    async def process_single_page_continuous(self, url: str):
        """
        New streaming version:
        - Launch async workers
        - Start producer (scroll+enqueue) in a thread, so Selenium can block there
        - Workers process items as they arrive
        - Wait for producer done and queue drained (or shutdown)
        Returns a list of results accumulated in processed_queue.
        """
        self.logger.info(f"Starting continuous processing for: {url}")

        # Prepare per-run state
        self.image_async_queue = asyncio.Queue(maxsize=self.max_workers * 4)
        self._producer_done = asyncio.Event()
        self._seen_images = set()

        # Start workers
        workers = [asyncio.create_task(self._worker(i + 1)) for i in range(self.max_workers)]

        # Run producer in thread to avoid blocking event loop (Selenium is sync)
        loop = asyncio.get_running_loop()
        self._event_loop = loop
        producer_future = loop.run_in_executor(
            None, self.continuous_scroll_and_collect_streaming, url
        )

        # Python
        try:
            # Wait for producer to finish scrolling (producer sets _producer_done in finally)
            await asyncio.wrap_future(asyncio.ensure_future(producer_future))
        except Exception as e:
            self.logger.error(f"Producer encountered an error: {e}")
        finally:
            # Ensure producer completion flag is set
            self._producer_done.set()

        # Wait for workers to finish up (queue drained or shutdown)
        # If shutdown was initiated, use a shorter timeout to exit promptly.
        timeout = 30 if self._shutdown_initiated else 1200
        await self.wait_for_completion(timeout=timeout)

        # Cancel lingering workers if any still running
        for w in workers:
            if not w.done():
                w.cancel()
        # Gather workers safely
        for w in workers:
            try:
                await w
            except asyncio.CancelledError:
                pass

        # Clear the loop reference to avoid reuse across runs
        self._event_loop = None


        # Collect results accumulated so far
        results = []
        while not self.processed_queue.empty():
            results.append(self.processed_queue.get())

        if self._shutdown_initiated:
            self.logger.info("Shutdown was initiated during processing")

        self.logger.info(f"Streaming processing complete. Results: {len(results)}")
        return results

    def close(self):
        self.logger.info("Closing resources...")
        self.initiate_shutdown()
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
        self.logger.info("Shutdown complete")


# Usage example
async def main():
    processor = HumanLikeImageProcessor(
        max_workers=1,
        headless=True,
        cookies_file="threads_cookies.txt",  # Optional
    )

    try:
        url = "https://www.threads.com"
        results = await processor.process_single_page_continuous(url)
        for result in results:
            logging.info(f"Image: {result['image_url']}")
            # logging.info(f"Description: {result['description']}")
            logging.info("-" * 50)
    except KeyboardInterrupt:
        logging.error("\nKeyboard interrupt received in main...")
    finally:
        try:
            await asyncio.sleep(0.2)
        except Exception:
            pass
        processor.close()


if __name__ == "__main__":
    asyncio.run(main())
