from __future__ import print_function

import time

import numpy as np
import cv2
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='calibrate camera intrinsics using OpenCV')

    parser.add_argument('-f', '--filenames', metavar='IMAGE', nargs='+',
                        required=False,
                        help='input image files')

    parser.add_argument('-r', '--rows', metavar='N', type=int,
                        default=9,
                        help='# of chessboard corners in vertical direction')

    parser.add_argument('-c', '--cols', metavar='N', type=int,
                        default=6,
                        help='# of chessboard corners in horizontal direction')

    parser.add_argument('-s', '--size', metavar='NUM', type=float,
                        default=1.0,
                        help='chessboard square size in user-chosen units (should not affect results)')

    parser.add_argument('-d', '--show-detections',
                        action='store_true',
                        help='show detections in window')

    parser.add_argument('-i', '--input-device',
                        required=False,
                        type=str,
                        help='input device to use for capturing images')

    options = parser.parse_args()

    if options.rows < options.cols:
        patternsize = (options.cols, options.rows)
    else:
        patternsize = (options.rows, options.cols)

    sz = options.size

    x = np.arange(patternsize[0]) * sz
    y = np.arange(patternsize[1]) * sz

    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = np.zeros_like(xgrid)
    opoints = np.dstack((xgrid, ygrid, zgrid)).reshape((-1, 1, 3)).astype(np.float32)

    win = 'Calibrate'
    cv2.namedWindow(win)

    ipoints = []

    def calibrate(frame, imagesize=None):
        if frame is None:
            return False, None, None

        cursize = (frame.shape[1], frame.shape[0])

        if imagesize is None:
            imagesize = cursize
        else:
            assert imagesize == cursize

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        retval, corners = cv2.findChessboardCorners(gray, patternsize)

        if options.show_detections:
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            cv2.drawChessboardCorners(display, patternsize, corners, retval)
            cv2.imshow(win, display)
            while cv2.waitKey(5) not in range(128): pass

        return retval, corners, imagesize

    imagesize = None

    if options.filenames is not None:
        for filename in options.filenames:

            rgb = cv2.imread(filename)

            retval, corners, imagesize = calibrate(rgb, imagesize)

            if retval:
                ipoints.append(corners)

    elif options.input_device is not None:
        device = options.input_device
        if device.isdigit():
            device = int(options.input_device)
        cap = cv2.VideoCapture(device)
        ret = False
        frame = None
        frame_interval = 0.5
        last_time = time.time()
        while True and cap.isOpened():
            ret, frame = cap.read()
            current_time = time.time()
            if not ret:
                break
            if current_time - last_time >= frame_interval:
                last_time = current_time
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame
                retval, corners, imagesize = calibrate(frame, imagesize)
                display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                cv2.drawChessboardCorners(display, patternsize, corners, retval)
                cv2.imshow(win, display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        if ret and frame is not None:
            retval, corners, imagesize = calibrate(frame)
            if retval:
                ipoints.append(corners)

    flags = (cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K1 |
             cv2.CALIB_FIX_K2 |
             cv2.CALIB_FIX_K3 |
             cv2.CALIB_FIX_K4 |
             cv2.CALIB_FIX_K5 |
             cv2.CALIB_FIX_K6)

    opoints = [opoints] * len(ipoints)

    if len(ipoints) == 0:
        print('no images found')
        return

    retval, K, dcoeffs, rvecs, tvecs = cv2.calibrateCamera(opoints, ipoints, imagesize,
                                                           cameraMatrix=None,
                                                           distCoeffs=np.zeros(5),
                                                           flags=flags)

    assert (np.all(dcoeffs == 0))

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    params = (fx, fy, cx, cy)

    print()
    print('all units below measured in pixels:')
    print('  fx = {}'.format(K[0, 0]))
    print('  fy = {}'.format(K[1, 1]))
    print('  cx = {}'.format(K[0, 2]))
    print('  cy = {}'.format(K[1, 2]))
    print()
    print('pastable into Python:')
    print('  fx, fy, cx, cy = {}'.format(repr(params)))
    print()


if __name__ == '__main__':
    main()
