import cupy as cp
from cupyx.scipy.ndimage import zoom
from itertools import product
from pathlib import Path
import os
import shutil
import numpy as np

from PIL import Image, ImageChops

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn


def summarise(img: Image.Image, progress: Progress, task: int) -> Image.Image:
    """Summarise an image into a 16 x 16 image."""
    resized = img.resize((16, 16))
    progress.update(task, advance=1)
    return resized

def move_smallest_image(img_path_a: str, img_path_b: str, dest_path: str):
    with Image.open(img_path_a) as img1, Image.open(img_path_b) as img2:
        size1 = img1.width * img1.height
        size2 = img2.width * img2.height
        if size1 < size2:
            shutil.move(img_path_a, dest_path)
        elif size2 <= size1:
            shutil.move(img_path_b, dest_path)

def difference(img1: Image.Image, img2: Image.Image) -> float:
    """Find the difference between two images."""

    diff = ImageChops.difference(img1, img2)

    acc = 0
    width, height = diff.size
    for w, h in product(range(width), range(height)):
        r, g, b = diff.getpixel((w, h))
        acc += (r + g + b) / 3

    average_diff = acc / (width * height)
    normalised_diff = average_diff / 255
    return normalised_diff

def fast_difference(img1: Image.Image, img2: Image.Image) -> float:
    """Vectorized version - replaces your difference() function."""
    # Convert to numpy arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # Calculate mean absolute difference
    diff = np.mean(np.abs(arr1 - arr2)) / 255.0
    return float(diff)

def gpu_summarise(img: Image.Image, progress: Progress, task: int) -> cp.ndarray:
    """Summarise image to 16x16 on GPU."""
    # Convert to array and transfer to GPU
    img_array = cp.asarray(img)
    
    # Resize on GPU using interpolation
    if img_array.ndim == 3:  # RGB image
        # Downscale using GPU-accelerated interpolation
        scale_factor = 16 / img_array.shape[0]
        summary = zoom(img_array, (scale_factor, scale_factor, 1), order=1)
    else:  # Grayscale
        scale_factor = 16 / img_array.shape[0]
        summary = zoom(img_array, (scale_factor, scale_factor), order=1)
    progress.update(task, advance=1)
    return summary

def gpu_difference(gpu_img1: cp.ndarray, gpu_img2: cp.ndarray) -> float:
    """Calculate difference between two GPU arrays."""
    # Absolute difference on GPU
    diff = cp.abs(gpu_img1 - gpu_img2)
    
    # Mean difference on GPU (much faster than CPU loop)
    if diff.ndim == 3:  # RGB
        avg_diff = cp.mean(diff) / 255.0
    else:  # Grayscale
        avg_diff = cp.mean(diff) / 255.0
    
    return float(avg_diff.get())  # Transfer result back to CPU

def explore_directory(path: Path) -> None:
    """Find images in a directory and compare them all."""

    files = (
        list(path.glob("*.jpg")) + list(path.glob("*.jpeg")) + list(path.glob("*.png"))
    )
    diffs = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.completed}"),
        transient=False
    ) as progress:
        summarice_task = progress.add_task("[magenta]Processing...", total=len(files))
        diff_task = progress.add_task("[magenta]Diff...", total=len(files)*len(files))

        summaries = [(file, gpu_summarise(Image.open(file), progress, summarice_task)) for file in files]

        for (f1, sum1), (f2, sum2) in product(summaries, repeat=2):
            progress.update(diff_task, advance=1)
            key = tuple(sorted([str(f1), str(f2)]))
            if f1 == f2 or key in diffs:
                continue

            diff = gpu_difference(sum1, sum2)
            # print(key, diff)
            diffs[key] = diff

    # print()
    # print("Near-duplicates found:")
    # print("======================")
    # for key, diff in diffs.items():
    #     if diff < 0.07:
    #         print(key, diff)

    print()
    print("Duplicates found:")
    print("======================")
    for key, diff in diffs.items():
        if diff < 0.005:
            print(key, diff)
            move_smallest_image(key[0], key[1], "women/")


if __name__ == "__main__":
    explore_directory(Path("saved_people"))
