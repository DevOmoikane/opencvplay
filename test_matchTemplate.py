import cv2
import numpy as np

# Load images
template = cv2.imread("template.png", cv2.IMREAD_UNCHANGED)
frame = cv2.imread("frame.jpg", cv2.IMREAD_COLOR)

# Convert to grayscale
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Match template
result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)

# Find location with highest match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

threshold = 0.8
if max_val > threshold:
    top_left = max_loc
    h, w = gray_template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    print(f"Match found at {top_left}, score={max_val:.2f}")

cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
