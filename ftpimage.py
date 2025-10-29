import ftplib
import tempfile
import os
from imagesimilarity import ImageSimilarityDetector
from rich import print


class FTPImageSimilarity:
    def __init__(self, ftp_host, ftp_user, ftp_pass):
        self.ftp_host = ftp_host
        self.ftp_user = ftp_user
        self.ftp_pass = ftp_pass
        self.detector = ImageSimilarityDetector()  # or PHashSimilarityDetector()

    def download_ftp_images(self, ftp_directory, local_temp_dir):
        """Download images from FTP to temporary directory"""
        try:
            ftp = ftplib.FTP(self.ftp_host)
            ftp.login(self.ftp_user, self.ftp_pass)
            ftp.cwd(ftp_directory)

            files = []
            ftp.retrlines('NLST', files.append)

            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

            downloaded_paths = []
            print(f"Found {len(image_files)} images")
            for idx, img_file in enumerate(image_files):
                print(f"Downloading ({idx}) {img_file}")
                local_path = os.path.join(local_temp_dir, img_file)
                if not os.path.exists(local_path):
                    with open(local_path, 'wb') as f:
                        ftp.retrbinary(f'RETR {img_file}', f.write)
                downloaded_paths.append(local_path)

            ftp.quit()
            return downloaded_paths

        except Exception as e:
            print(f"FTP Error: {e}")
            return []

    def find_similar_on_ftp(self, ftp_directory, similarity_threshold=0.6):
        """Find similar images on FTP server"""
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"temp dir: {temp_dir}")
            # Download images from FTP
            image_paths = self.download_ftp_images(ftp_directory, temp_dir)

            if not image_paths:
                return []

            # Find similar images
            similar_pairs = []

            print(f"Found {len(image_paths)} images")
            for i in range(len(image_paths)):
                for j in range(i + 1, len(image_paths)):
                    print(f"Check similarity: ({i}) {image_paths[i]} - ({j}) {image_paths[j]}")
                    similarity = self.detector.compare_images(
                        image_paths[i], image_paths[j]
                    )

                    if similarity >= similarity_threshold:
                        img1_name = os.path.basename(image_paths[i])
                        img2_name = os.path.basename(image_paths[j])

                        similar_pairs.append({
                            'image1': img1_name,
                            'image2': img2_name,
                            'similarity': similarity
                        })

            print(f"Found {len(similar_pairs)} similar images")
            return similar_pairs


# Usage for FTP
# ftp_detector = FTPImageSimilarity('ftp.example.com', 'username', 'password')
# similar_images = ftp_detector.find_similar_on_ftp('/images/directory')

if __name__ == "__main__":
    print("Starting download")
    ftp_detector = FTPImageSimilarity('10.73.19.117', 'data', 'k4m1z4m4')
    print("Starting comparison")
    similar_images = ftp_detector.find_similar_on_ftp('/Resources/Images/img', 0.9)
    print(f"Found {len(similar_images)} similar images")
    print(similar_images)
