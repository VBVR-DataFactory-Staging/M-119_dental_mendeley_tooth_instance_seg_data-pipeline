"""M-119: Dental panoramic X-ray tooth segmentation.

Layout (after extract):
    _extracted/M-119_Dental/Dental segmentation_X-ray panoramic/Data Dental X_Ray_Panoramic/
        {train,test,valid}/{train,test,valid}_images/*.jpg
        {train,test,valid}/{train,test,valid}_mask/*.jpg  (binary tooth mask)
        Update_Test/{test_images,test_masks}/*.jpg
Case D single image, loop 4s, tooth green overlay.
"""
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np
from common import DATA_ROOT, write_task, COLORS, fit_square, overlay_mask

PID = "M-119"; TASK_NAME = "dental_mendeley_tooth_instance_seg"; FPS = 8
PROMPT = ("This is a panoramic dental X-ray from a Mendeley dataset. "
          "Segment the teeth region with a green overlay so each tooth is highlighted.")

def loop_frames(f, n): return [f.copy() for _ in range(n)]

def process_case(img_p, mask_p, idx):
    img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None: return None
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    img_r = fit_square(img, 512)
    mask_r = cv2.resize((mask > 0).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
    annot = overlay_mask(img_r, mask_r, color=COLORS["green"], alpha=0.40)
    n = FPS * 4
    meta = {"task": "Dental panoramic tooth segmentation", "dataset": "Dental Mendeley",
            "case_id": img_p.stem[:40], "modality": "panoramic dental X-ray",
            "classes": ["tooth"], "colors": {"tooth": "green"},
            "fps": FPS, "frames_per_video": n, "case_type": "D_single_image_loop"}
    return write_task(PID, TASK_NAME, idx, img_r, annot,
                      loop_frames(img_r, n), loop_frames(annot, n), loop_frames(annot, n),
                      PROMPT, meta, FPS)

def main():
    root = DATA_ROOT / "_extracted" / "M-119_Dental"
    # walk everything, split by "images" vs "mask" in path
    all_jpgs = list(root.rglob("*.jpg"))
    imgs, masks = [], []
    for p in all_jpgs:
        pstr = str(p).lower()
        if "/test_images/" in pstr or "/train_images/" in pstr or "/valid_images/" in pstr:
            imgs.append(p)
        elif "_mask" in pstr or "/masks/" in pstr:
            masks.append(p)
    mask_by_name = {p.name: p for p in masks}
    pairs = [(img, mask_by_name[img.name]) for img in imgs if img.name in mask_by_name]
    print(f"  {len(pairs)} Dental image+mask pairs")
    for i, (img, mask) in enumerate(pairs):
        d = process_case(img, mask, i)
        if d: print(f"  wrote {d}")

if __name__ == "__main__":
    main()
