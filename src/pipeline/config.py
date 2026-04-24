"""Pipeline configuration for M-119 (dental_mendeley_tooth_instance_seg)."""
from pathlib import Path
from pydantic import Field
from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Configuration for M-119 pipeline."""

    domain: str = Field(default="dental_mendeley_tooth_instance_seg")

    s3_bucket: str = Field(
        default="med-vr-datasets",
        description="S3 bucket containing the raw M-119 data",
    )
    s3_prefix: str = Field(
        default="M-119/dental_mendeley/",
        description="S3 key prefix for the dataset raw data",
    )
    fps: int = Field(
        default=6,
        description="Frames per second for the generated videos",
    )
    raw_dir: Path = Field(
        default=Path("raw"),
        description="Local directory for downloaded raw data",
    )
    task_prompt: str = Field(
        default=(
            "This is a panoramic dental X-ray. Segment each visible tooth as a "
            "distinct instance. Reveal the teeth one by one with a unique color "
            "per tooth class (canine, central incisor, lateral incisor, "
            "first/second/third molar, first/second premolar) until every "
            "tooth in the panoramic view is highlighted."
        ),
        description="The task instruction shown to the reasoning model.",
    )
