"""M-038: ISIC 2018 Skin Lesion Segmentation (Task 1).

2594 dermoscopy images with binary lesion masks.
Case D: independent images → circular rotation of N=16 samples at 10 fps.
Each task focuses on one dermoscopy image; the other frames in the loop are
other samples from the same batch (circular playback).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
from common import (
    DATA_ROOT, to_rgb, overlay_mask, write_task, COLORS, fit_square,
)

PID = "M-038"
TASK_NAME = "isic2018_skin_lesion_seg"
FPS = 10
LOOP_SIZE = 16

PROMPT = (
    "This is a dermoscopy image from the ISIC 2018 Skin Lesion Analysis challenge. "
    "Segment the skin lesion (melanoma or non-melanoma) and overlay a green semi-"
    "transparent mask plus contour on each frame that contains the lesion. "
    "The ground_truth video keeps only the frames containing the lesion."
)


def load_sample(img_dir: Path, mask_dir: Path, stem: str):
    img = cv2.imread(str(img_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
    m = cv2.imread(str(mask_dir / f"{stem}_segmentation.png"), cv2.IMREAD_GRAYSCALE)
    return img, (m > 0).astype(np.uint8)


def build_task(samples, focus_idx: int, task_idx: int):
    first_frames, last_frames, gt_frames, flags = [], [], [], []
    ordered = samples[focus_idx:] + samples[:focus_idx]
    frames = (ordered * ((LOOP_SIZE // len(ordered)) + 1))[:LOOP_SIZE]
    for img, mask in frames:
        rgb = fit_square(img, 512)
        # resize mask to match
        mask_sq = fit_square(mask * 255, 512)
        mask_sq = (mask_sq > 0).astype(np.uint8)
        ann = overlay_mask(rgb, mask_sq, color=COLORS["green"], alpha=0.4)
        first_frames.append(rgb)
        last_frames.append(ann)
        has = bool(mask_sq.any())
        flags.append(has)
        if has:
            gt_frames.append(ann)
    if not gt_frames:
        gt_frames = last_frames[:5]
    first_frame = first_frames[0]
    final_frame = last_frames[0]

    meta = {
        "task": "skin lesion segmentation",
        "dataset": "ISIC 2018 Task 1",
        "focus_sample_index": focus_idx,
        "loop_size": LOOP_SIZE,
        "modality": "dermoscopy",
        "fps_source": "manual (case D circular loop)",
        "source_split": "train",
    }
    return write_task(PID, TASK_NAME, task_idx,
                      first_frame, final_frame,
                      first_frames, last_frames, gt_frames,
                      PROMPT, meta, FPS)


def main():
    img_dir = DATA_ROOT / "_extracted" / "22_ISIC" / "ISIC2018_Task1-2_Training_Input"
    mask_dir = DATA_ROOT / "_extracted" / "22_ISIC" / "ISIC2018_Task1_Training_GroundTruth"
    stems = ["ISIC_0000000", "ISIC_0000001", "ISIC_0000003", "ISIC_0000004"]
    samples = [load_sample(img_dir, mask_dir, s) for s in stems]
    for task_idx in range(2):
        d = build_task(samples, focus_idx=task_idx, task_idx=task_idx)
        print(f"  wrote {d}")


if __name__ == "__main__":
    main()
