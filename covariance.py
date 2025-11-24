import cv2
import numpy as np

def calculate_gradient_covariance(image_path):
    # 1. Load the image and calculate Luminance (Grayscale conversion)
    # OpenCV loads in BGR format by default. Convert to grayscale.
    # OpenCV's cv2.cvtColor uses the standard ITU-R BT.709-5 luminance coefficients
    # (0.2126*R + 0.7152*G + 0.0722*B) implicitly.
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    # Convert to grayscale (luminance)
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Convert to float for gradient calculation to handle negative values and precision
    L = np.float64(L)

    # 2. Calculate Gradients (Gx and Gy)
    # Use the Sobel operator to approximate partial derivatives
    # cv2.Sobel(src, ddepth, dx, dy, ksize)
    # dx=1, dy=0 for Gx; dx=0, dy=1 for Gy
    Gx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)

    # 3. Flatten the gradients into matrix M
    # Reshape Gx and Gy into 1D vectors and stack them as columns
    # N is the total number of pixels
    N = Gx.size
    # v(x,y) = [Gx(x,y), Gy(x,y)]^T
    M = np.vstack((Gx.ravel(), Gy.ravel())).T  # M is an N x 2 matrix

    # 4. Calculate the Covariance Matrix C = (1/N) * M^T * M
    # NumPy's np.cov function can calculate this directly.
    # We use rowvar=False to indicate that variables are columns, and bias=True to divide by N
    C = np.cov(M, rowvar=False, bias=True)

    return C

# Example usage:
# Replace 'your_image.jpg' with the path to your image file
covariance_matrix = calculate_gradient_covariance('your_image.jpg')

if covariance_matrix is not None:
    print("Gradient Covariance Matrix (C):")
    print(covariance_matrix)
    # C[0, 0] is the variance of Gx
    # C[1, 1] is the variance of Gy
    # C[0, 1] and C[1, 0] are the covariance between Gx and Gy

