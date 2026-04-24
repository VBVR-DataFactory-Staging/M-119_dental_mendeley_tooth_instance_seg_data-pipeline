"""Raw-data downloader for M-119 (dental Mendeley panoramic).

Pulls only what's needed:
  - Secondpart_extracted/{train,test,valid}/_annotations.coco.json
  - Secondpart_extracted/{train,test,valid}/imgs/*.jpg

The Secondpart_extracted/ tree is the COCO-format gold-standard subset we
pre-extracted from Secondpart.rar (60 images, 1811 per-tooth polygon
annotations across 9 named categories).
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional

from core.download import download_from_s3


SECONDPART_PREFIX = "Secondpart_extracted/"


class TaskDownloader:
    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config.raw_dir)

    def ensure_raw(self):
        # Sync only the Secondpart_extracted subtree — that's the
        # COCO-annotated subset we use.
        target = self.raw_dir / "Secondpart_extracted"
        if not target.exists() or not any(target.rglob("*.jpg")):
            print(
                f"Raw data not found at {target}, syncing from "
                f"s3://{self.config.s3_bucket}/{self.config.s3_prefix}{SECONDPART_PREFIX} ..."
            )
            download_from_s3(
                bucket_name=self.config.s3_bucket,
                s3_prefix=self.config.s3_prefix + SECONDPART_PREFIX,
                local_dir=target,
            )
        else:
            print(f"Raw data already present at {target}, skipping sync.")

    def iter_samples(self, limit: Optional[int] = None) -> Iterator[dict]:
        self.ensure_raw()
        yield {"raw_dir": str(self.raw_dir)}


def create_downloader(config) -> TaskDownloader:
    return TaskDownloader(config)
