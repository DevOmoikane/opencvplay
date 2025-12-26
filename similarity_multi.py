from itertools import combinations
from pathlib import Path
import os
import shutil
from multiprocessing import Pool, cpu_count
import cv2

import numpy as np
from PIL import Image, ImageOps

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import click


# -----------------------------
# Parallel-friendly, array-based helpers
# -----------------------------

def summarise_array(path: str) -> tuple[str, np.ndarray]:
    """
    Load image at path, convert to RGB, resize to 16x16, return (path, uint8 array).
    This is designed to run in a worker process.
    """
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB").resize((16, 16), resample=Image.BILINEAR), dtype=np.uint8)
    return path, arr


def orientations_arrays(arr: np.ndarray) -> list[np.ndarray]:
    """
    Generate orientation variants for the given summary array (16x16x3 uint8).
    Returns list of arrays: [r0, r90, r180, r270, flip_h, flip_v]
    """
    r0 = arr
    r90 = np.rot90(arr, k=1)
    r180 = np.rot90(arr, k=2)
    r270 = np.rot90(arr, k=3)
    flip_h = np.flip(arr, axis=1)  # mirror left-right
    flip_v = np.flip(arr, axis=0)  # flip top-bottom
    return [r0, r90, r180, r270, flip_h, flip_v]


def difference_arrays(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean absolute difference per channel, normalized to [0,1].
    a, b: (16,16,3) uint8
    """
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))  # avoid uint8 wrap
    avg = diff.mean()  # 0..255
    return float(avg / 255.0)


def min_difference_any_orientation_arrays(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the minimal difference across rotations/flips of b.
    """
    return min(difference_arrays(a, v) for v in orientations_arrays(b))


def compare_pair(args: tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]) -> tuple[tuple[str, str], float]:
    """
    Compare two summaries, allowing for orientation changes on the second.
    Returns ((path1, path2), diff).
    """
    (f1, a1), (f2, a2) = args
    d = min_difference_any_orientation_arrays(a1, a2)
    key = tuple(sorted((f1, f2)))
    return key, d

def orb_similarity(images: tuple[str, str]) -> tuple[str, str, float]:
    img1_path, img2_path = images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Use ORB detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0 # Handle cases with no features

    # Use Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by their distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate similarity score (e.g., number of good matches)
    # A higher number of good matches means higher similarity
    # A simple metric: proportion of good matches among keypoints in the smaller image
    min_keypoints = min(len(kp1), len(kp2))
    if min_keypoints == 0:
        return 0
    
    # You might filter for good matches (distance < threshold)
    good_matches = [m for m in matches if m.distance < 75] # Distance threshold example
    
    similarity_score = len(good_matches) / min_keypoints
    return img1_path, img2_path, similarity_score


# -----------------------------
# IO helpers
# -----------------------------

def move_smallest_image(img_path_a: str, img_path_b: str, dest_path: str):
    """
    Move the smaller of the two original images into dest_path.
    """
    os.makedirs(dest_path, exist_ok=True)
    with Image.open(img_path_a) as img1, Image.open(img_path_b) as img2:
        size1 = img1.width * img1.height
        size2 = img2.width * img2.height
    if size1 < size2:
        shutil.move(img_path_a, dest_path)
    else:
        shutil.move(img_path_b, dest_path)


# -----------------------------
# Main exploration with multiprocessing
# -----------------------------
@click.command()
# @click.option('--path', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), default=Path('saved_people'), help='Directory to explore for images.')
@click.argument('paths', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), nargs=-1)
@click.option('--dup-threshold', type=float, default=0.005, help='Difference threshold below which images are considered duplicates.')
@click.option('--dest-dir', type=str, default='women/', help='Directory to move smaller duplicate images into.')
def explore_directory(paths: tuple[Path], dup_threshold: float = 0.005, dest_dir: str = "women/") -> None:
    """
    Find images in a directory and compare them all using multiprocessing.
    Steps:
      1) Parallel summarize images to small arrays.
      2) Parallel compare unique unordered pairs.
      3) In main process, move the smaller image for pairs under the threshold.
    """
    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    files: list[str] = []
    for p in paths:
        for pat in exts:
            files.extend(str(f) for f in p.glob(f'*{pat}'))

    if not files:
        print("No images found.")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.completed}"),
        transient=False
    ) as progress:
        sum_task = progress.add_task("[magenta]Summarizing...", total=len(files))

        # Step 1: parallel summaries
        summaries: list[tuple[str, np.ndarray]] = []
        with Pool(processes=cpu_count()) as pool:
            for result in pool.imap_unordered(summarise_array, files, chunksize=16):
                summaries.append(result)
                progress.update(sum_task, advance=1)

        # Step 2: parallel pairwise comparisons (unique unordered pairs)
        pairs_iter = combinations(summaries, 2)

        # To show a progress bar, we need a count. For large N this is fine;
        # if memory is tight, we could stream without materializing.
        n_pairs = len(summaries) * (len(summaries) - 1) // 2
        diff_task = progress.add_task("[magenta]Comparing...", total=n_pairs)

        diffs: dict[tuple[str, str], float] = {}
        with Pool(processes=cpu_count()) as pool:
            for key, d in pool.imap_unordered(compare_pair, pairs_iter, chunksize=64):
                diffs[key] = d
                progress.update(diff_task, advance=1)

    print()
    print("Duplicates found:")
    print("======================")
    for (f1, f2), d in diffs.items():
        if d < dup_threshold:
            print((f1, f2), d)
            try:
                move_smallest_image(f1, f2, dest_dir)
            except Exception:
                # Swallow move errors to keep the run going (e.g., permission, race conditions)
                pass

    # with Progress(
    #     SpinnerColumn(),
    #     TextColumn("[progress.description]{task.description}"),
    #     BarColumn(),
    #     TaskProgressColumn(),
    #     TextColumn("[cyan]{task.completed}"),
    #     transient=False
    # ) as progress:
    #     n_pairs = len(files) * (len(files) - 1) // 2
    #     orb_task = progress.add_task("[magenta]Computing ORB similarities...", total=n_pairs)
    #     pairs_iter = combinations(files, 2)
    #     with Pool(processes=cpu_count()) as pool:
    #         for result in pool.imap_unordered(orb_similarity, pairs_iter, chunksize=16):
    #             img1_path, img2_path, similarity_score = result
    #             if similarity_score > (1 - dup_threshold):
    #                 # print(f"Duplicate found: ({img1_path}, {img2_path}) with similarity score: {similarity_score:.4f}")
    #                 try:
    #                     move_smallest_image(img1_path, img2_path, dest_dir)
    #                 except Exception:
    #                     pass
    #             progress.update(orb_task, advance=1)



if __name__ == "__main__":
    try:
        explore_directory()
    except KeyboardInterrupt:
        print("Interrupted by user.")
