from itertools import combinations
from pathlib import Path
import os
import shutil
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import click
from collections import defaultdict
from typing import Tuple, List, Dict, Iterable, Set


# -----------------------------
# Parallel-friendly, array-based helpers
# -----------------------------

def summarise_array(path: str) -> tuple[str, np.ndarray]:
    """
    Load image at path, convert to RGB, resize to 16x16, return (path, uint8 array).
    This is designed to run in a worker process.
    """
    try:
        with Image.open(path) as img:
            arr = np.array(img.convert("RGB").resize((16, 16), resample=Image.BILINEAR), dtype=np.uint8)
        return path, arr
    except Exception as e:
        # If image cannot be opened, return None
        print(f"Error loading image {path}: {e}")
        return path, None

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


def canonicalize_orientation(arr: np.ndarray) -> np.ndarray:
    """
    Make orientation-invariant representation by picking the lexicographically smallest
    orientation among the 6 variants, treating the bytes as the key.
    """
    variants = orientations_arrays(arr)
    # Choose the variant with the minimum byte-string representation (stable and fast)
    keys = [v.tobytes() for v in variants]
    idx = int(np.argmin([k for k in keys]))
    return variants[idx]


def to_gray(arr: np.ndarray) -> np.ndarray:
    """
    Convert 16x16x3 RGB uint8 to 16x16 uint8 grayscale using standard weights.
    """
    # weights sum to 1; we cast to uint8 at the end
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)
    return gray


def ahash(gray: np.ndarray, hash_size: int = 8) -> int:
    """
    Compute average hash (aHash) from a small grayscale image.
    - Resizes to hash_size x hash_size
    - Sets bits for pixels above mean.
    Returns a 64-bit integer for hash_size=8.
    """
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = float(small.mean())
    bits = (small.flatten() > avg).astype(np.uint8)
    # Pack bits into an int
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming_distance64(a: int, b: int) -> int:
    x = a ^ b
    # builtin popcount in Python 3.8+: int.bit_count()
    return x.bit_count()


def difference_arrays(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean absolute difference per channel, normalized to [0,1].
    a, b: (16,16,3) uint8
    """
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))  # avoid uint8 wrap
    avg = diff.mean()  # 0..255
    return float(avg / 255.0)


# -----------------------------
# Candidate generation via hashing
# -----------------------------

def build_buckets(
    items: List[tuple[str, np.ndarray]],
    hash_size: int = 8,
    bucket_hamming_radius: int = 1
) -> Dict[int, List[int]]:
    """
    Build perceptual-hash buckets. Each item is (path, canonical_rgb_16x16).
    We compute aHash from 16x16 grayscale and place indices into bins keyed by hash.

    For radius > 0, we also create neighbor bins on the fly during candidate enumeration,
    not by duplicating entries into multiple bins (to keep memory low).
    """
    hashes: List[int] = []
    for _, arr_rgb in items:
        gray = to_gray(arr_rgb)
        h = ahash(gray, hash_size=hash_size)
        hashes.append(h)

    # Store indices for exact hash bins
    bins: Dict[int, List[int]] = defaultdict(list)
    for idx, h in enumerate(hashes):
        bins[h].append(idx)

    return bins


def iter_candidate_pairs(
    bins: Dict[int, List[int]],
    items: List[tuple[str, np.ndarray]],
    hash_size: int = 8,
    bucket_hamming_radius: int = 1
) -> Iterable[tuple[int, int]]:
    """
    Yield unique candidate pairs (i, j) with i < j by grouping within hash bins and
    nearby bins (Hamming distance <= bucket_hamming_radius).

    We avoid O(N^2) by only comparing within small groups. A radius of 1 or 2 is typical.
    """
    # Precompute all hashes for items to check neighboring bins
    hashes = []
    for _, arr_rgb in items:
        h = ahash(to_gray(arr_rgb), hash_size=hash_size)
        hashes.append(h)

    seen: Set[tuple[int, int]] = set()

    # Generate neighbor hash masks for flipping up to R bits out of hash_size*hash_size bits.
    # For hash_size=8, 64 bits; generating all combinations for radius 1 or 2 is feasible
    # if done lazily per hash.
    def neighbors(h: int, radius: int) -> Iterable[int]:
        if radius <= 0:
            yield h
            return
        # Radius 1 neighbors: flip each bit once
        yield h
        if radius >= 1:
            for bit in range(hash_size * hash_size):
                yield h ^ (1 << bit)
        if radius >= 2:
            # For radius 2, flip every pair of bits — can be large (C(64,2)=2016); use with care.
            # If you need radius=2 often, consider truncating the hash (e.g., to 32 bits) first.
            for b1 in range(hash_size * hash_size):
                hb1 = h ^ (1 << b1)
                # Only b2 > b1 to avoid duplicates
                for b2 in range(b1 + 1, hash_size * hash_size):
                    yield hb1 ^ (1 << b2)

    # Iterate every item, then enumerate index pairs within its bin and neighbor bins
    for i in range(len(items)):
        hi = hashes[i]
        for hnb in neighbors(hi, bucket_hamming_radius):
            if hnb not in bins:
                continue
            for j in bins[hnb]:
                if j <= i:
                    continue
                key = (i, j)
                if key not in seen:
                    seen.add(key)
                    yield key


# -----------------------------
# IO helpers
# -----------------------------

def move_smallest_image(img_path_a: str, img_path_b: str, dest_path: str):
    """
    Move the smaller of the two original images into dest_path.
    """
    os.makedirs(dest_path, exist_ok=True)
    try:
        with Image.open(img_path_a) as img1, Image.open(img_path_b) as img2:
            size1 = img1.width * img1.height
            size2 = img2.width * img2.height
    except Exception:
        # If any image cannot be opened, skip
        return
    if size1 < size2:
        shutil.move(img_path_a, dest_path)
    else:
        shutil.move(img_path_b, dest_path)


# -----------------------------
# Parallel pair comparison using indices (reduced IPC)
# -----------------------------

_global_items: List[tuple[str, np.ndarray]] = []  # (path, canonical_rgb_16x16)


def _init_worker(items: List[tuple[str, np.ndarray]]):
    global _global_items
    _global_items = items


def _compare_pair_idx(pair: tuple[int, int]) -> tuple[tuple[str, str], float]:
    """
    Compare two canonicalized arrays by mean absolute difference (no orientation sweep).
    Returns ((path1, path2), diff).
    """
    i, j = pair
    (f1, a1) = _global_items[i]
    (f2, a2) = _global_items[j]
    # Orientation already canonicalized — single diff is enough
    diff = difference_arrays(a1, a2)
    key = (f1, f2) if f1 <= f2 else (f2, f1)
    return key, diff


# -----------------------------
# Main command
# -----------------------------
@click.command()
@click.argument('paths', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), nargs=-1)
@click.option('--dup-threshold', type=float, default=0.005, help='Difference threshold below which images are considered duplicates.')
@click.option('--dest-dir', type=str, default='women/', help='Directory to move smaller duplicate images into.')
@click.option('--hash-size', type=int, default=8, help='aHash size (hash is hash_size*hash_size bits). 8 => 64-bit.')
@click.option('--bucket-radius', type=int, default=1, help='Hamming radius for candidate buckets (0, 1, or 2).')
@click.option('--recursive/--no-recursive', default=False, help='Search for images recursively within provided directories.')
def explore_directory(
    paths: tuple[Path],
    dup_threshold: float = 0.005,
    dest_dir: str = "women/",
    hash_size: int = 8,
    bucket_radius: int = 1,
    recursive: bool = False
) -> None:
    """
    Find similar images efficiently:
      1) Parallel summarize images to 16x16 RGB.
      2) Canonicalize orientation once.
      3) Bucket by perceptual hash (aHash), generate only candidate pairs.
      4) Parallel compare candidate pairs.
      5) Move smaller image among near-duplicates under dup-threshold.
    """
    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    files: list[str] = []
    for p in paths:
        for pat in exts:
            iterator = p.rglob(f'*{pat}') if recursive else p.glob(f'*{pat}')
            files.extend(str(f) for f in iterator)

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
        sum_task = progress.add_task("[magenta]Summarizing + canonicalizing...", total=len(files))

        # Step 1: parallel summaries
        summaries: list[tuple[str, np.ndarray]] = []
        with Pool(processes=cpu_count()) as pool:
            for path, arr in pool.imap_unordered(summarise_array, files, chunksize=16):
                # Step 2: canonicalize orientation once (fast, small arrays)
                can = canonicalize_orientation(arr)
                summaries.append((path, can))
                progress.update(sum_task, advance=1)

        if not summaries:
            print("No valid images after load.")
            return

        # Step 3: bucket by aHash
        bucket_task = progress.add_task("[magenta]Bucketing (aHash)...", total=1)
        bins = build_buckets(summaries, hash_size=hash_size, bucket_hamming_radius=bucket_radius)
        progress.update(bucket_task, advance=1)

        # Step 4: candidate generation
        cand_task = progress.add_task("[magenta]Generating candidates...", total=1)
        candidates = list(iter_candidate_pairs(bins, summaries, hash_size=hash_size, bucket_hamming_radius=bucket_radius))
        progress.update(cand_task, advance=1)

        # If buckets are strict, candidates may be far fewer than N^2
        compare_task = progress.add_task("[magenta]Comparing candidates...", total=len(candidates))

        diffs: dict[tuple[str, str], float] = {}
        # Use a pool with initializer to avoid shipping arrays each task
        with Pool(processes=cpu_count(), initializer=_init_worker, initargs=(summaries,)) as pool:
            for key, d in pool.imap_unordered(_compare_pair_idx, candidates, chunksize=128):
                diffs[key] = d
                progress.update(compare_task, advance=1)

    print()
    print("Duplicates found:")
    print("======================")
    moved = 0
    for (f1, f2), d in diffs.items():
        if d < dup_threshold:
            print((f1, f2), d)
            try:
                move_smallest_image(f1, f2, dest_dir)
                moved += 1
            except Exception:
                # Swallow move errors to keep the run going (e.g., permission, race conditions)
                pass
    print(f"Moved {moved} images to {dest_dir}")


if __name__ == "__main__":
    try:
        explore_directory()
    except KeyboardInterrupt:
        print("Interrupted by user.")
