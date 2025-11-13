import cv2
import numpy as np

template = cv2.imread('template.png', cv2.IMREAD_GRAYSCALE)
frame = cv2.imread('frame.jpg', cv2.IMREAD_GRAYSCALE)

best_val, best_loc, best_scale = 0, None, 1.0

for scale in np.logspace(-2, 0, 50, base=2):  # e.g., scales from 1/4x to 1x
    resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if resized.shape[0] >= frame.shape[0] or resized.shape[1] >= frame.shape[1]:
        continue
    result = cv2.matchTemplate(frame, resized, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val > best_val:
        best_val, best_loc, best_scale = max_val, max_loc, scale

if best_val > 0.75:
    h, w = (np.array(template.shape[::-1]) * best_scale).astype(int)
    top_left = best_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, top_left, bottom_right, (0, 255, 0), 2)
    print(f"Found at scale={best_scale:.2f}, score={best_val:.3f}")
    cv2.imshow('Detection', out)
    cv2.waitKey(0)
