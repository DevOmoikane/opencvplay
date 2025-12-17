from itertools import product
from pathlib import Path
import os
import shutil

from PIL import Image, ImageChops, ImageOps

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn


def summarise(img: Image.Image, progress: Progress, task: int) -> Image.Image:
    """Summarise an image into a 16 x 16 image."""
    resized = img.convert("RGB").resize((16, 16), resample=Image.BILINEAR)
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

def orientations(img: Image.Image) -> list[Image.Image]:
    """Return a list of image variants"""
    # Ensure RGB to keep channels consistent
    base = img.convert("RGB")
    variants = []

    # Rotations
    r0 = base
    r90 = base.rotate(90, expand=True)
    r180 = base.rotate(180, expand=True)
    r270 = base.rotate(270, expand=True)

    # Flips (on the base orientation is enough, since rotations will cover the rest)
    flip_h = ImageOps.mirror(base)      # horizontal flip
    flip_v = ImageOps.flip(base)        # vertical flip

    # You can also include flipped+rotated variants for full symmetry coverage
    # but the set below is usually sufficient and cheaper.
    candidates = [r0, r90, r180, r270, flip_h, flip_v]

    # Resize to 16x16 to match summarise() output
    variants = [c.resize((16, 16)) for c in candidates]
    return variants

def min_difference_any_orientation(sum1: Image.Image, sum2: Image.Image) -> float:
    """Compare two summaries allowing for rotation/flip by checking min difference."""
    variants = orientations(sum2)
    return min(difference(sum1, v) for v in variants)

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

        summaries = [(file, summarise(Image.open(file), progress, summarice_task)) for file in files]

        for (f1, sum1), (f2, sum2) in product(summaries, repeat=2):
            progress.update(diff_task, advance=1)
            key = tuple(sorted([str(f1), str(f2)]))
            if f1 == f2 or key in diffs:
                continue

            diff = min_difference_any_orientation(sum1, sum2)
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
            try:
                move_smallest_image(key[0], key[1], "women/")
            except Exception as e:
                pass


if __name__ == "__main__":
    try:
        explore_directory(Path("saved_people"))
    except KeyboardInterrupt:
        print("Interrupted by user.")
