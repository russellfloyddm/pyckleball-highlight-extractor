"""Configuration management for the pickleball highlight extractor."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import yaml


@dataclass
class ScoringWeights:
    """Configurable weights for the highlight scoring formula."""

    rally_duration: float = 0.35
    shot_count: float = 0.25
    ball_speed: float = 0.15
    movement: float = 0.15
    audio: float = 0.10

    def __post_init__(self) -> None:
        total = (
            self.rally_duration
            + self.shot_count
            + self.ball_speed
            + self.movement
            + self.audio
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.4f}"
            )


@dataclass
class DetectionConfig:
    """Object detection and tracking configuration."""

    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.2
    yolo_iou: float = 0.5
    tracker: str = "bytetrack"
    ball_class_id: int = 32  # COCO sports ball
    player_class_id: int = 0  # COCO person
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"


@dataclass
class AudioConfig:
    """Audio analysis configuration."""

    sample_rate: int = 22050
    hop_length: int = 512
    frame_duration: float = 1.0  # seconds per audio frame
    excitement_threshold: float = 0.6
    min_spike_duration: float = 0.5  # seconds


@dataclass
class RallyConfig:
    """Rally detection configuration."""

    min_rally_duration: float = 3.0  # seconds
    max_gap_duration: float = 3.0  # seconds to allow between shots
    min_shots: int = 1
    min_detection_streak: int = 2
    detection_window: int = 5
    min_movement_to_start: float = 0.0
    # Lower floor to keep distant-camera rallies alive; very small ball motion
    # can project to single-digit px/s between frames. Trade-off: this can be
    # more sensitive to slow camera shake or non-ball motion (increase this
    # value if you observe false positives on static/noisy footage).
    min_ball_speed: float = 8.0  # pixels/second
    service_anchor_window: float = 1.0  # seconds
    service_anchor_position_std: float = 8.0  # pixels
    # Serve anchoring uses a moderate spike so low-FPS/mobile footage still
    # marks serve starts without requiring very high per-frame speed jumps.
    service_anchor_min_speed_spike: float = 40.0  # pixels/second
    long_rally_bonus_shots: Dict[int, float] = field(
        default_factory=lambda: {10: 0.1, 20: 0.2, 30: 0.35}
    )


@dataclass
class HighlightConfig:
    """Highlight selection and clip configuration."""

    score_threshold: float = 0.4
    merge_gap: float = 10.0  # seconds - merge highlights within this gap
    clip_before: float = 5.0  # seconds before highlight
    clip_after: float = 5.0  # seconds after highlight
    output_format: str = "mp4"
    ffmpeg_crf: int = 23  # FFmpeg constant rate factor (lower = better quality)
    ffmpeg_preset: str = "medium"
    add_transitions: bool = True
    transition_duration: float = 0.5  # seconds
    title_screen: bool = False
    title_text: str = "Pickleball Highlights"
    title_duration: float = 3.0  # seconds


@dataclass
class AppConfig:
    """Top-level application configuration."""

    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    rally: RallyConfig = field(default_factory=RallyConfig)
    highlight: HighlightConfig = field(default_factory=HighlightConfig)
    log_level: str = "INFO"
    max_workers: int = 2
    batch_size: int = 32  # frames per processing batch


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults.

    Args:
        config_path: Path to the YAML configuration file. If None or the file
                     does not exist, default configuration is returned.

    Returns:
        Populated AppConfig instance.
    """
    if config_path is None or not os.path.isfile(config_path):
        return AppConfig()

    with open(config_path, "r", encoding="utf-8") as fh:
        raw: Dict = yaml.safe_load(fh) or {}

    scoring_raw = raw.get("scoring", {})
    detection_raw = raw.get("detection", {})
    audio_raw = raw.get("audio", {})
    rally_raw = raw.get("rally", {})
    highlight_raw = raw.get("highlight", {})

    # Long-rally bonuses require int keys after YAML parsing
    if "long_rally_bonus_shots" in rally_raw:
        rally_raw["long_rally_bonus_shots"] = {
            int(k): float(v)
            for k, v in rally_raw["long_rally_bonus_shots"].items()
        }

    return AppConfig(
        scoring=ScoringWeights(**scoring_raw),
        detection=DetectionConfig(**detection_raw),
        audio=AudioConfig(**audio_raw),
        rally=RallyConfig(**rally_raw),
        highlight=HighlightConfig(**highlight_raw),
        log_level=raw.get("log_level", "INFO"),
        max_workers=int(raw.get("max_workers", 2)),
        batch_size=int(raw.get("batch_size", 32)),
    )
