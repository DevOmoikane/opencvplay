import cv2
import numpy as np


# Load images
template = cv2.imread("template.png", cv2.IMREAD_UNCHANGED)
frame = cv2.imread("frame.jpg", cv2.IMREAD_COLOR)

scales = np.linspace(0.5, 1.5, 20)
best_val = 0
best_loc = None
best_scale = None

gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

for scale in scales:
    resized = cv2.resize(gray_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    result = cv2.matchTemplate(gray_frame, resized, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val > best_val:
        best_val = max_val
        best_loc = max_loc
        best_scale = scale

if best_val > 0.8:
    h, w = (np.array(gray_template.shape[::-1]) * best_scale).astype(int)
    top_left = best_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    print(f"Best match scale={best_scale:.2f}, score={best_val:.2f}")

cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
