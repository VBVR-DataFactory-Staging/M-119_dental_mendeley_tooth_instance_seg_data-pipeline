"""Video + overlay primitives for M-119 dental tooth instance segmentation."""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


# Distinct, well-separated BGR colors for tooth categories.
# Index aligned with the COCO category list (excluding the catch-all "dental").
CATEGORY_COLORS = {
    "canine":          (0,   255, 255),  # yellow
    "central incisor": (0,   165, 255),  # orange
    "lateral incisor": (203, 192, 255),  # pink
    "first premolar":  (255, 191, 0),    # deep sky blue
    "second premolar": (180, 105, 255),  # hot pink
    "first molar":     (0,   200, 0),    # green
    "second molar":    (255, 0,   255),  # magenta
    "third molar":     (255, 255, 0),    # cyan
    "dental":          (200, 200, 200),  # neutral fallback
}


def loop_frames(image: np.ndarray, n: int = 6) -> List[np.ndarray]:
    return [image.copy() for _ in range(n)]


def polygon_to_mask(poly_xy: Sequence[float], shape_hw: Tuple[int, int]) -> np.ndarray:
    """Rasterize a flat [x0,y0,x1,y1,...] polygon to a uint8 mask."""
    h, w = shape_hw
    pts = np.array(poly_xy, dtype=np.float32).reshape(-1, 2)
    pts_i = np.round(pts).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_i], 1)
    return mask


def overlay_mask_color(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 200, 0),
    alpha: float = 0.45,
    contour: bool = True,
) -> np.ndarray:
    """Blend a binary mask onto a BGR image, with an optional contour outline."""
    out = image.copy()
    if mask is None:
        return out
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return out
    overlay = np.zeros_like(out)
    overlay[binary == 1] = color
    out = cv2.addWeighted(out, 1.0, overlay, alpha, 0)
    if contour:
        cnts, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
    return out


def make_video(frames: List[np.ndarray], out_path: Path, fps: int = 6) -> None:
    """Write H.264 MP4 via ffmpeg (broad-compat, even-dim padded)."""
    if not frames:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    w2, h2 = w - (w % 2), h - (h % 2)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-vf", f"scale={w2}:{h2}",
        str(out_path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
