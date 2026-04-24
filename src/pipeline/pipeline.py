"""TaskPipeline for M-119 dental Mendeley tooth instance segmentation.

Builds one VBVR sample per panoramic dental X-ray:

  - first_frame.png:   raw panoramic
  - final_frame.png:   panoramic with every annotated tooth overlaid in
                       its category color (with thin contour)
  - first_video.mp4:   raw panoramic looping (a few seconds)
  - last_video.mp4:    final overlay looping
  - ground_truth.mp4:  teeth revealed one-by-one in stable spatial order
                       (left→right by polygon centroid x), each new tooth
                       persisted on top of the previous overlay
  - prompt.txt:        the task instruction
  - metadata.json:     per-sample metadata (image id, tooth count, classes)

Source: COCO-format Secondpart_extracted/ (60 images, 1811 per-tooth
polygon annotations across 8 named tooth categories).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from core.pipeline import BasePipeline, TaskSample
from src.download.downloader import create_downloader
from .config import TaskConfig
from .transforms import (
    CATEGORY_COLORS,
    loop_frames,
    make_video,
    overlay_mask_color,
    polygon_to_mask,
)


# Display target — keep the panoramic aspect (~2:1) intact.
TARGET_W = 1024
TARGET_H = 512


def _resize_keep(img: np.ndarray, w: int = TARGET_W, h: int = TARGET_H) -> Tuple[np.ndarray, float, float]:
    H, W = img.shape[:2]
    sx = w / W
    sy = h / H
    out = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return out, sx, sy


def _scale_polygon(poly: List[float], sx: float, sy: float) -> List[float]:
    out = []
    for i in range(0, len(poly), 2):
        out.append(poly[i] * sx)
        out.append(poly[i + 1] * sy)
    return out


def _polygon_centroid(poly_xy: List[float]) -> Tuple[float, float]:
    pts = np.array(poly_xy, dtype=np.float32).reshape(-1, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _load_coco_split(split_dir: Path) -> Tuple[List[dict], List[dict], dict]:
    """Return (images, annotations, id→category_name) for one COCO split."""
    js = split_dir / "_annotations.coco.json"
    if not js.exists():
        return [], [], {}
    data = json.loads(js.read_text())
    cat_by_id = {c["id"]: c["name"] for c in data.get("categories", [])}
    return data.get("images", []), data.get("annotations", []), cat_by_id


def _gather_all_images(raw_dir: Path) -> List[dict]:
    """Walk every COCO split and return a list of per-image records.

    Each record = {
        "image_path": Path,
        "image_id":   str (split + file_name),
        "width":      int,
        "height":     int,
        "teeth":      [ {"category": str, "polygon": [x0,y0,...]}, ... ]
    }
    """
    base = raw_dir / "Secondpart_extracted"
    records: List[dict] = []
    for split in ("train", "valid", "test"):
        split_dir = base / split
        if not split_dir.exists():
            continue
        imgs, anns, cat_by_id = _load_coco_split(split_dir)
        anns_by_image: dict = {}
        for a in anns:
            anns_by_image.setdefault(a["image_id"], []).append(a)
        for im in imgs:
            img_path = split_dir / "imgs" / im["file_name"]
            if not img_path.exists():
                continue
            teeth = []
            for a in anns_by_image.get(im["id"], []):
                segs = a.get("segmentation") or []
                if not segs:
                    continue
                # Use the largest polygon for this tooth instance.
                poly = max(segs, key=len)
                if len(poly) < 6:
                    continue
                cat = cat_by_id.get(a["category_id"], "dental")
                teeth.append({"category": cat, "polygon": list(poly)})
            if not teeth:
                continue
            records.append({
                "image_path": img_path,
                "image_id":   f"{split}_{Path(im['file_name']).stem}",
                "width":      int(im.get("width", 0)),
                "height":     int(im.get("height", 0)),
                "teeth":      teeth,
            })
    # Stable order: by image_id
    records.sort(key=lambda r: r["image_id"])
    return records


class TaskPipeline(BasePipeline):
    def __init__(self, config: Optional[TaskConfig] = None):
        super().__init__(config or TaskConfig())
        self.downloader = create_downloader(self.config)

    def download(self) -> Iterator[dict]:
        # We yield once — the actual sample list is built in run().
        yield from self.downloader.iter_samples(limit=self.config.num_samples)

    # ----------------------------------------------------------- per-sample
    def _process_record(self, rec: dict, idx: int, out_root: Path) -> Optional[TaskSample]:
        img = cv2.imread(str(rec["image_path"]), cv2.IMREAD_COLOR)
        if img is None:
            return None

        img_r, sx, sy = _resize_keep(img)
        H, W = img_r.shape[:2]

        # Sort teeth left→right by centroid x so the reveal order is stable.
        teeth = []
        for t in rec["teeth"]:
            poly = _scale_polygon(t["polygon"], sx, sy)
            cx, _cy = _polygon_centroid(poly)
            teeth.append({
                "category": t["category"],
                "polygon":  poly,
                "cx":       cx,
            })
        teeth.sort(key=lambda t: t["cx"])

        # Final overlay: every tooth, each in its category color, contoured.
        final = img_r.copy()
        for t in teeth:
            color = CATEGORY_COLORS.get(t["category"], CATEGORY_COLORS["dental"])
            mask = polygon_to_mask(t["polygon"], (H, W))
            final = overlay_mask_color(final, mask, color=color, alpha=0.45,
                                        contour=True)

        # Reveal video: start from raw, add one tooth per frame group, keep
        # previous overlays. Hold each step for `hold` frames; pad start/end.
        fps = self.config.fps
        hold = max(1, fps // 2)  # ~0.5s per tooth at fps=6 → 3 frames
        head_pad = fps             # 1s pause at the bare image
        tail_pad = fps * 2         # 2s hold on the fully-revealed image

        gt_frames: List[np.ndarray] = []
        gt_frames.extend(loop_frames(img_r, head_pad))

        cur = img_r.copy()
        for t in teeth:
            color = CATEGORY_COLORS.get(t["category"], CATEGORY_COLORS["dental"])
            mask = polygon_to_mask(t["polygon"], (H, W))
            cur = overlay_mask_color(cur, mask, color=color, alpha=0.45,
                                      contour=True)
            gt_frames.extend(loop_frames(cur, hold))

        gt_frames.extend(loop_frames(final, tail_pad))

        # first_video.mp4: raw panoramic loop (~2s)
        first_frames = loop_frames(img_r, fps * 2)
        # last_video.mp4: final overlay loop (~2s)
        last_frames = loop_frames(final, fps * 2)

        sid = f"task_{idx:04d}"
        out_dir = out_root / sid
        out_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_dir / "first_frame.png"), img_r)
        cv2.imwrite(str(out_dir / "final_frame.png"), final)
        make_video(first_frames, out_dir / "first_video.mp4", fps)
        make_video(last_frames,  out_dir / "last_video.mp4",  fps)
        make_video(gt_frames,    out_dir / "ground_truth.mp4", fps)

        prompt = self.config.task_prompt
        (out_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

        # Per-tooth-class counts for metadata.
        class_counts: dict = {}
        for t in teeth:
            class_counts[t["category"]] = class_counts.get(t["category"], 0) + 1

        meta = {
            "image_id":      rec["image_id"],
            "modality":      "panoramic dental X-ray",
            "dataset":       "Mendeley Dental Panoramic (Secondpart COCO subset)",
            "tooth_count":   len(teeth),
            "class_counts":  class_counts,
            "fps":           fps,
            "frames_per_video_gt":   len(gt_frames),
            "frames_per_video_loop": fps * 2,
            "video_size_wh": [W, H],
            "case_type":     "D_single_image_progressive_reveal",
        }
        (out_dir / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        return TaskSample(
            task_id=sid,
            domain=self.config.domain,
            prompt=prompt,
            first_image=img_r,
            final_image=final,
            first_video=str(out_dir / "first_video.mp4"),
            last_video=str(out_dir / "last_video.mp4"),
            ground_truth_video=str(out_dir / "ground_truth.mp4"),
            metadata=meta,
        )

    # --------------------------------------------------------------- driver
    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        # Not used directly — run() bypasses BasePipeline.run because we
        # need to walk the COCO splits ourselves.
        return None

    def run(self) -> List[TaskSample]:
        # Trigger the S3 sync exactly once (consumes the 1-shot iterator).
        for _ in self.download():
            pass
        raw_dir = Path(self.config.raw_dir)

        records = _gather_all_images(raw_dir)
        print(f"Found {len(records)} annotated panoramic images")

        limit = self.config.num_samples
        if limit is not None:
            records = records[:limit]
            print(f"Limiting to {len(records)} samples (--num-samples {limit})")

        out_root = Path(self.config.output_dir) / f"{self.config.domain}_task"
        out_root.mkdir(parents=True, exist_ok=True)

        samples: List[TaskSample] = []
        for i, rec in enumerate(records):
            try:
                s = self._process_record(rec, i, out_root)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}] FAILED {rec['image_id']}: {exc}")
                continue
            if s is None:
                continue
            samples.append(s)
            if (i + 1) % 5 == 0 or i == len(records) - 1:
                print(f"  [{i + 1}/{len(records)}] processed {rec['image_id']} "
                      f"(teeth={rec.get('teeth') and len(rec['teeth'])})")

        print(f"Done — wrote {len(samples)} samples to {out_root}")
        return samples
