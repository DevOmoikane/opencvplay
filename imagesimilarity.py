import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import math


class ImageSimilarityDetector:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract_features(self, image_path):
        """Extract ORB features from image"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None

        # Detect keypoints and descriptors
        keypoints, descriptors = self.orb.detectAndCompute(img, None)
        return keypoints, descriptors

    def compare_images(self, img1_path, img2_path, threshold=0.7):
        """Compare two images using feature matching"""
        kp1, desc1 = self.extract_features(img1_path)
        kp2, desc2 = self.extract_features(img2_path)

        if desc1 is None or desc2 is None:
            return 0.0

        # Match features
        matches = self.bf.match(desc1, desc2)

        # Calculate similarity score
        if len(matches) > 0:
            similarity = len(matches) / min(len(desc1), len(desc2))
            return similarity
        return 0.0

    def find_similar_in_directory(self, directory_path, similarity_threshold=0.6):
        """Find similar images in a directory"""
        image_files = [f for f in os.listdir(directory_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        similar_pairs = []

        print(f"Found {len(image_files)} images")
        for i in range(len(image_files)):
            for j in range(i + 1, len(image_files)):
                print(f"Comparing [{i}]{image_files[i]} and [{j}]{image_files[j]}")
                img1_path = os.path.join(directory_path, image_files[i])
                img2_path = os.path.join(directory_path, image_files[j])

                similarity = self.compare_images(img1_path, img2_path)
                print(f"Similarity: {similarity}")

                if similarity >= similarity_threshold:
                    similar_pairs.append({
                        'image1': image_files[i],
                        'image2': image_files[j],
                        'similarity': similarity
                    })

        print(f"Finished : Found {len(similar_pairs)} similar images")
        return similar_pairs


# Usage
# detector = ImageSimilarityDetector()
# similar_images = detector.find_similar_in_directory('/path/to/images')