from itertools import combinations
from pathlib import Path
import os
import shutil
from multiprocessing import Pool, cpu_count, get_context

from PIL import Image, ImageChops, ImageOps

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn


def summarise_file(file_path: str) -> tuple[str, bytes]:
    """Load an image and summarise it into a 16x16 RGB image. Returns (path, raw_bytes)."""
    with Image.open(file_path) as img:
        resized = img.convert("RGB").resize((16, 16), resample=Image.BILINEAR)
        # Store as raw bytes to keep payload small and pickle-friendly
        return file_path, resized.tobytes()


def image_from_bytes(data: bytes) -> Image.Image:
    """Reconstruct a 16x16 RGB image from raw bytes."""
    return Image.frombytes("RGB", (16, 16), data)


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
    # Slight micro-optimization: load pixels once
    px = diff.load()
    for w in range(width):
        for h in range(height):
            r, g, b = px[w, h]
            acc += (r + g + b) / 3

    average_diff = acc / (width * height)
    normalised_diff = average_diff / 255
    return normalised_diff


def orientations(img: Image.Image) -> list[Image.Image]:
    """Return a list of image variants"""
    base = img.convert("RGB")
    # Rotations
    r0 = base
    r90 = base.rotate(90, expand=True)
    r180 = base.rotate(180, expand=True)
    r270 = base.rotate(270, expand=True)
    # Flips
    flip_h = ImageOps.mirror(base)
    flip_v = ImageOps.flip(base)
    candidates = [r0, r90, r180, r270, flip_h, flip_v]
    # Already 16x16 in our usage; ensure size anyway
    variants = [c.resize((16, 16)) for c in candidates]
    return variants


def min_difference_any_orientation(sum1_bytes: bytes, sum2_bytes: bytes) -> float:
    """Compare two summaries allowing for rotation/flip by checking min difference."""
    sum1 = image_from_bytes(sum1_bytes)
    sum2 = image_from_bytes(sum2_bytes)
    variants = orientations(sum2)
    return min(difference(sum1, v) for v in variants)


def pair_diff_task(args: tuple[str, bytes, str, bytes]) -> tuple[tuple[str, str], float]:
    """Worker task to compute min-diff for a pair."""
    f1, s1, f2, s2 = args
    d = min_difference_any_orientation(s1, s2)
    key = tuple(sorted([f1, f2]))
    return key, d


def explore_directory(path: Path, processes: int | None = None) -> None:
    """Find images in a directory and compare them all using multiprocessing."""
    files = (
        list(path.glob("*.jpg"))
        + list(path.glob("*.jpeg"))
        + list(path.glob("*.png"))
    )

    if not files:
        print("No images found.")
        return

    # Resolve to absolute string paths for pickling stability
    file_paths = [str(f.resolve()) for f in files]

    processes = processes or max(1, cpu_count() - 1)
    diffs: dict[tuple[str, str], float] = {}

    # Use spawn context explicitly for macOS safety
    ctx = get_context("spawn")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.completed}"),
        transient=False,
    ) as progress:
        # 1) Summarise in parallel
        summarise_task = progress.add_task("[magenta]Summarising...", total=len(file_paths))

        def _on_sum_result(_):
            progress.update(summarise_task, advance=1)

        with ctx.Pool(processes=processes) as pool:
            async_results = [pool.apply_async(summarise_file, (fp,), callback=_on_sum_result) for fp in file_paths]
            summaries_list = [r.get() for r in async_results]

        # Map path -> summary bytes
        summaries = dict(summaries_list)

        # 2) Generate unique pairs and compute diffs in parallel
        pairs = list(combinations(file_paths, 2))
        diff_task = progress.add_task("[magenta]Comparing...", total=len(pairs))

        def _pair_args():
            for f1, f2 in pairs:
                yield (f1, summaries[f1], f2, summaries[f2])

        def _on_diff_result(_):
            progress.update(diff_task, advance=1)

        with ctx.Pool(processes=processes) as pool:
            async_results = [pool.apply_async(pair_diff_task, (args,), callback=_on_diff_result) for args in _pair_args()]
            for r in async_results:
                key, d = r.get()
                diffs[key] = d

    print()
    print("Duplicates found:")
    print("======================")
    for key, diff in diffs.items():
        if diff < 0.005:
            print(key, diff)
            try:
                os.makedirs("women", exist_ok=True)
                move_smallest_image(key[0], key[1], "women/")
            except Exception:
                pass


if __name__ == "__main__":
    try:
        # Adjust processes if you like, e.g. processes=4
        explore_directory(Path("saved_people"))
    except KeyboardInterrupt:
        print("Interrupted by user.")
