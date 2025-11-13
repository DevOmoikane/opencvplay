import os
import cv2
import math
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from resynthesizer import resynthesize
from rich.progress import Progress
import argparse

# ----------------------------
# Config
# ----------------------------
# TEMPLATE_PATH = "template.png"
# FRAME_PATH = "frame.jpg"

# Scale search
SCALES = np.logspace(-1.0, 0.3, 14, base=2.0)  # ~0.5x to ~1.23x, 14 scales
MIN_SIZE = 16  # minimum side for scaled template

# Sliding window stride control (coarse-to-fine)
COARSE_STRIDE_RATIO = 0.5   # stride = max(1, int(size * ratio))
REFINE_STRIDE_RATIO = 0.25  # local refinement stride

# Batching
BATCH_SIZE = 128  # number of crops processed per forward pass

# Coarse prefilter using classical template matching to shortlist candidates per scale
USE_COARSE_PREFILTER = True
COARSE_TOPK_PER_SCALE = 25  # collect this many best candidates per scale
GLOBAL_TOPK = 60            # then keep only global top-K before embedding

# Visualization
DRAW_COLOR = (0, 255, 0)
DRAW_THICKNESS = 2

# Device (optional GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Model and transforms
# ----------------------------
weights = ResNet18_Weights.DEFAULT
_backbone = resnet18(weights=weights)
# Feature extractor up to avgpool (removes final FC)
feature_extractor = nn.Sequential(*list(_backbone.children())[:-1]).to(DEVICE).eval()

# Transform to 224x224 (ImageNet normalization)
_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def embed512(img_bgr: np.ndarray) -> torch.Tensor:
    # img_bgr: HxWx3 (OpenCV)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = _transform(img_rgb).unsqueeze(0).to(DEVICE, dtype=torch.float32)  # [1, 3, 224, 224]
    feat = feature_extractor(x).squeeze(-1).squeeze(-1)  # [1, 512]
    feat = torch.nn.functional.normalize(feat, dim=1)    # normalize across channel dim
    feat = feat.squeeze(0).contiguous()                  # -> [512]
    return feat  # [512]

@torch.no_grad()
def embed512_batch(crops_bgr: list[np.ndarray]) -> torch.Tensor:
    # crops_bgr: list of HxWx3 np arrays
    if not crops_bgr:
        return torch.empty(0, 512, device=DEVICE)
    rgb_tensors = []
    for c in crops_bgr:
        img_rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        rgb_tensors.append(_transform(img_rgb))
    batch = torch.stack(rgb_tensors, dim=0).to(DEVICE, dtype=torch.float32)  # [N, 3, 224, 224]
    feats = feature_extractor(batch).squeeze(-1).squeeze(-1)  # [N, 512]
    feats = torch.nn.functional.normalize(feats, dim=1)       # [N, 512]
    return feats

def resize_template(template: np.ndarray, s: float) -> tuple[np.ndarray, int, int]:
    th0, tw0 = template.shape[:2]
    tw_s = max(MIN_SIZE, int(round(tw0 * s)))
    th_s = max(MIN_SIZE, int(round(th0 * s)))
    t_s = cv2.resize(template, (tw_s, th_s), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return t_s, th_s, tw_s

def coarse_candidates(frame: np.ndarray, template: np.ndarray, scales: np.ndarray,
                      topk_per_scale: int, global_topk: int) -> list[tuple[float,int,int,int,int,float]]:
    """
    Returns a list of candidate boxes:
    (score, x, y, tw_s, th_s, s)
    Using cv2.matchTemplate for speed, then keeping global top-K.
    """
    H, W = frame.shape[:2]
    candidates = []
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for s in scales:
        t_s, th_s, tw_s = resize_template(template, s)
        if th_s > H or tw_s > W:
            continue
        gray_t = cv2.cvtColor(t_s, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_frame, gray_t, cv2.TM_CCOEFF_NORMED)
        flat = res.ravel()
        if flat.size == 0:
            continue
        k = min(topk_per_scale, flat.size)
        top_idx = np.argpartition(flat, -k)[-k:]
        top_idx = top_idx[np.argsort(flat[top_idx])][::-1]
        ys, xs = np.unravel_index(top_idx, res.shape)
        for y, x in zip(ys, xs):
            candidates.append((float(res[y, x]), int(x), int(y), tw_s, th_s, float(s)))

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[:global_topk]

# ---- helpers to estimate work for progress bar ----
def count_total_positions_prefilter(frame_w, frame_h, tw, th, wx, wy, ex, ey, stride_x, stride_y) -> int:
    xs = max(0, ((min(frame_w - tw, ex) - wx) // stride_x) + 1) if ex >= wx else 0
    ys = max(0, ((min(frame_h - th, ey) - wy) // stride_y) + 1) if ey >= wy else 0
    return int(xs * ys)

def count_total_positions_full(frame_w, frame_h, tw, th, stride_x, stride_y) -> int:
    xs = max(0, ((frame_w - tw) // stride_x) + 1) if frame_w >= tw else 0
    ys = max(0, ((frame_h - th) // stride_y) + 1) if frame_h >= th else 0
    return int(xs * ys)

def sliding_search_with_batch(frame: np.ndarray,
                              template: np.ndarray,
                              scales: np.ndarray,
                              batch_size: int,
                              coarse_stride_ratio: float,
                              refine_stride_ratio: float,
                              use_prefilter: bool) -> tuple[int,int,int,int,float,float]:
    """
    Returns best match:
    (x, y, x2, y2, scale, similarity)
    """
    H0, W0 = frame.shape[:2]
    best_sim = -1.0
    best_box = None

    # Precompute total number of crops for progress bar
    total_crops = 0
    cand = None
    if use_prefilter:
        cand = coarse_candidates(
            frame, template, scales,
            topk_per_scale=COARSE_TOPK_PER_SCALE,
            global_topk=GLOBAL_TOPK
        )
        for _, cx, cy, tw_s, th_s, s in cand:
            if th_s > H0 or tw_s > W0:
                continue
            wx = max(0, cx - tw_s)
            wy = max(0, cy - th_s)
            ex = min(W0 - tw_s, cx + tw_s)
            ey = min(H0 - th_s, cy + th_s)
            stride_x = max(1, int(tw_s * refine_stride_ratio))
            stride_y = max(1, int(th_s * refine_stride_ratio))
            total_crops += count_total_positions_prefilter(W0, H0, tw_s, th_s, wx, wy, ex, ey, stride_x, stride_y)
    else:
        for s in scales:
            t_s, th_s, tw_s = resize_template(template, s)
            if th_s > H0 or tw_s > W0:
                continue
            stride_x = max(1, int(tw_s * coarse_stride_ratio))
            stride_y = max(1, int(th_s * coarse_stride_ratio))
            total_crops += count_total_positions_full(W0, H0, tw_s, th_s, stride_x, stride_y)

    # Run search with a progress bar that advances per batch
    # transient=True hides the bar after completion; disable if you want it to persist.
    with Progress(transient=True) as progress:
        task = progress.add_task("Embedding crops", total=total_crops if total_crops > 0 else None)

        def process_batch(batch_crops, batch_boxes, emb_t):
            nonlocal best_sim, best_box
            if not batch_crops:
                return
            feats = embed512_batch(batch_crops)       # [N, 512]
            sims = feats @ emb_t                       # [N]
            sim_vals = sims.detach().cpu().numpy()
            for (bx1, by1, bx2, by2, bs), sv in zip(batch_boxes, sim_vals):
                if sv > best_sim:
                    best_sim = float(sv)
                    best_box = (bx1, by1, bx2, by2, bs)
            progress.update(task, advance=len(batch_crops))

        if use_prefilter:
            # 'cand' computed above
            for _, cx, cy, tw_s, th_s, s in cand:
                if th_s > H0 or tw_s > W0:
                    continue

                wx = max(0, cx - tw_s)
                wy = max(0, cy - th_s)
                ex = min(W0 - tw_s, cx + tw_s)
                ey = min(H0 - th_s, cy + th_s)

                stride_x = max(1, int(tw_s * refine_stride_ratio))
                stride_y = max(1, int(th_s * refine_stride_ratio))

                t_s = cv2.resize(template, (tw_s, th_s), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                emb_t = embed512(t_s)              # [512], guaranteed 1-D

                batch_crops = []
                batch_boxes = []
                for y in range(wy, ey + 1, stride_y):
                    max_x = min(W0 - tw_s, ex)
                    for x in range(wx, max_x + 1, stride_x):
                        crop = frame[y:y + th_s, x:x + tw_s]
                        batch_crops.append(crop)
                        batch_boxes.append((x, y, x + tw_s, y + th_s, s))
                        if len(batch_crops) == batch_size:
                            process_batch(batch_crops, batch_boxes, emb_t)
                            batch_crops, batch_boxes = [], []

                process_batch(batch_crops, batch_boxes, emb_t)

        else:
            for s in scales:
                t_s, th_s, tw_s = resize_template(template, s)
                if th_s > H0 or tw_s > W0:
                    continue

                stride_x = max(1, int(tw_s * coarse_stride_ratio))
                stride_y = max(1, int(th_s * coarse_stride_ratio))

                emb_t = embed512(t_s)              # [512], guaranteed 1-D

                batch_crops = []
                batch_boxes = []
                for y in range(0, H0 - th_s + 1, stride_y):
                    for x in range(0, W0 - tw_s + 1, stride_x):
                        crop = frame[y:y + th_s, x:x + tw_s]
                        batch_crops.append(crop)
                        batch_boxes.append((x, y, x + tw_s, y + th_s, s))
                        if len(batch_crops) == batch_size:
                            process_batch(batch_crops, batch_boxes, emb_t)
                            batch_crops, batch_boxes = [], []

                process_batch(batch_crops, batch_boxes, emb_t)

    if best_box is None:
        raise RuntimeError("No candidate found (check image sizes and content)")

    x1, y1, x2, y2, bs = best_box
    return x1, y1, x2, y2, bs, best_sim

def create_mask_from_coordinates(target_size, x1: int, y1: int, x2: int, y2: int, padding: int = 5) -> np.ndarray:
    mask = np.zeros(target_size[:2], dtype=np.uint8)
    top_left_padded = (max(0, x1 - padding), max(0, y1 - padding))
    bottom_right_padded = (min(target_size[1], x2 + padding), min(target_size[0], y2 + padding))
    cv2.rectangle(mask, top_left_padded, bottom_right_padded, 255, -1)
    return mask

# --------------------------------------------------------------------
# Simple filter on the detected region
# --------------------------------------------------------------------
def apply_region_filter(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Apply a filter to the detected region only.
    Here: a strong Gaussian blur as an example.
    """
    mask = create_mask_from_coordinates(frame.shape, x1, y1, x2, y2)
    result = resynthesize(frame, mask)
    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Template-based region filtering on multiple images")
    parser.add_argument("--template", required=True, help="Path to template image")
    parser.add_argument("--input-dir", required=True, help="Directory with input images")
    parser.add_argument("--output-dir", required=True, help="Directory to save processed images")
    parser.add_argument("--extensions", nargs="+", default=[".jpg", ".jpeg", ".png"],
                        help="Image extensions to process (default: .jpg .jpeg .png)")
    return parser.parse_args()

def is_valid_image_file(filename: str, exts: list[str]) -> bool:
    lower = filename.lower()
    return any(lower.endswith(e.lower()) for e in exts)

def process_image_file(template: np.ndarray,
                       image_path: str,
                       output_path: str) -> None:
    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[WARN] Could not read image: {image_path}")
        return

    try:
        x1, y1, x2, y2, scale, score = sliding_search_with_batch(
            frame=frame,
            template=template,
            scales=SCALES,
            batch_size=BATCH_SIZE,
            coarse_stride_ratio=COARSE_STRIDE_RATIO,
            refine_stride_ratio=REFINE_STRIDE_RATIO,
            use_prefilter=USE_COARSE_PREFILTER
        )
    except RuntimeError as e:
        print(f"[WARN] No match found for {image_path}: {e}")
        return

    # Clamp coordinates to image bounds
    x1 = max(0, min(x1, frame.shape[1] - 1))
    y1 = max(0, min(y1, frame.shape[0] - 1))
    x2 = max(0, min(x2, frame.shape[1] - 1))
    y2 = max(0, min(y2, frame.shape[0] - 1))

    print(f"[INFO] {os.path.basename(image_path)} -> "
          f"scale={scale:.4f}, score={score:.4f}, box=({x1},{y1},{x2},{y2})")

    # Apply filter on detected region
    frame = apply_region_filter(frame, x1, y1, x2, y2)

    # Optionally draw rectangle (uncomment if you want visual debug)
    # cv2.rectangle(frame, (x1, y1), (x2, y2), DRAW_COLOR, DRAW_THICKNESS)

    # Save to output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not cv2.imwrite(output_path, frame):
        print(f"[ERROR] Failed to write output: {output_path}")

def main():
    args = parse_args()

    template = cv2.imread(args.template, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Could not load template image: {args.template}")

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Walk through input directory and process images
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not is_valid_image_file(fname, args.extensions):
                continue

            in_path = os.path.join(root, fname)
            # Preserve subdirectory structure under output_dir
            rel_path = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)

            print(f"[PROCESS] {in_path}")
            process_image_file(template, in_path, out_path)

if __name__ == "__main__":
    main()
