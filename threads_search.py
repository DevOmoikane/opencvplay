import traceback
import cv2
import httpx
import requests
import os
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import numpy as np
from ultralytics import YOLO
import difPy
import logging
from aiomultiprocess import Worker


logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(funcName)s:%(lineno)d - %(message)s')


class ThreadsSearch:

    DEFAULT_SEARCH_CLASS_IDS = {0} # Persons

    def __init__(self, config: dict | None = None):
        self.search_class_ids = self.DEFAULT_SEARCH_CLASS_IDS
        self.driver = None
        self.config = config or {}

    def _create_driver(self, headless: bool = True):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--window-size=1920,1080")
        self.driver = webdriver.Chrome(options=chrome_options)

    def _scroll_page(self, scroll_pause=2.0) -> bool:
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        new_height = self.driver.execute_script("return document.body.scrollHeight")
        return new_height != last_height
    
    def _load_netscape_cookies(self, base_url: str, cookie_file_path: str):
        if not os.path.exists(cookie_file_path):
            raise FileNotFoundError(f"Cookie file not found: {cookie_file_path}")
        
        self.driver.get(base_url)

