import argparse

import cv2
import time
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILE', type=str, nargs=1)
    parser.add_argument('-u', '--url', type=str, required=False)
    parser.add_argument('-d', '--device', type=int, required=False)
    options = parser.parse_args()

    if options.url is None and options.device is None:
        return

    cap = None
    if options.url is not None:
        cap = cv2.VideoCapture(options.url)
    elif options.device is not None:
        cap = cv2.VideoCapture(options.device)
    if cap.isOpened():
        success, frame = cap.read()
        current_time = time.time()
        cv2.imwrite(f"{options.filename[0]}", frame)


if __name__ == "__main__":
    main()
