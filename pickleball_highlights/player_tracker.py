"""Player detection and movement tracking using YOLOv8."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from pickleball_highlights.config import DetectionConfig
from pickleball_highlights.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PlayerDetection:
    """Single-frame detection of one player."""

    track_id: int
    frame_idx: int
    timestamp: float
    x: float  # centre x
    y: float  # centre y
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


@dataclass
class PlayerTrackState:
    """Per-player accumulated tracking state."""

    track_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    def add(self, x: float, y: float, timestamp: float) -> None:
        self.positions.append((x, y))
        self.timestamps.append(timestamp)

    @property
    def total_distance(self) -> float:
        """Total pixels travelled by this player."""
        if len(self.positions) < 2:
            return 0.0
        dist = 0.0
        for (x1, y1), (x2, y2) in zip(self.positions, self.positions[1:]):
            dist += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return dist

    @property
    def average_speed(self) -> float:
        """Average speed in pixels/second."""
        if len(self.positions) < 2 or len(self.timestamps) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[0]
        return self.total_distance / dt if dt > 0 else 0.0

    def reset(self) -> None:
        self.positions.clear()
        self.timestamps.clear()


class PlayerTracker:
    """Detect and track players across frames using YOLOv8.

    Falls back to frame-differencing motion estimation when the model is not
    available.

    Args:
        config: Detection configuration.
    """

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._model = None
        self._tracks: Dict[int, PlayerTrackState] = {}
        self._next_id = 0
        self._device = self._resolve_device(config.device)
        self._load_model()
        self._prev_frame_gray: Optional[np.ndarray] = None
        self._prev_optical_flow: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self, frame_idx: int, timestamp: float, frame: np.ndarray
    ) -> List[PlayerDetection]:
        """Detect and update players in one frame.

        Args:
            frame_idx: Frame index.
            timestamp: Frame timestamp in seconds.
            frame: BGR image array.

        Returns:
            List of PlayerDetection for this frame.
        """
        if self._model is None:
            return self._optical_flow_fallback(frame_idx, timestamp, frame)

        try:
            results = self._model.track(
                frame,
                classes=[self.config.player_class_id],
                conf=self.config.yolo_confidence,
                iou=self.config.yolo_iou,
                device=self._device,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Player tracking inference error: %s", exc)
            return []

        return self._parse_results(results, frame_idx, timestamp)

    def get_tracks(self) -> Dict[int, PlayerTrackState]:
        """Return all accumulated player track states."""
        return self._tracks

    def get_combined_movement_score(self, frame_count: int) -> float:
        """Return a normalised movement score [0, 1] for this segment.

        Args:
            frame_count: Number of frames in the segment (used for normalisation).

        Returns:
            Normalised movement intensity.
        """
        if not self._tracks or frame_count == 0:
            return 0.0

        total_speed = sum(t.average_speed for t in self._tracks.values())
        # Normalise: 300 px/s per player is considered "full" movement
        norm = total_speed / (300.0 * max(len(self._tracks), 1))
        return min(norm, 1.0)

    def reset(self) -> None:
        """Clear accumulated tracking state."""
        self._tracks.clear()
        self._next_id = 0
        self._prev_frame_gray = None
        self._prev_optical_flow = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.config.yolo_model)
            logger.info("Loaded YOLOv8 model for player tracking.")
        except ImportError:
            logger.warning(
                "ultralytics not installed – player tracking will use "
                "optical flow fallback."
            )
            self._model = None
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load model for player tracking: %s", exc)
            self._model = None

    def _parse_results(
        self, results, frame_idx: int, timestamp: float
    ) -> List[PlayerDetection]:
        detections: List[PlayerDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                track_id_tensor = box.id
                tid = (
                    int(track_id_tensor[0])
                    if track_id_tensor is not None
                    else self._next_id
                )
                if track_id_tensor is None:
                    self._next_id += 1

                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if tid not in self._tracks:
                    self._tracks[tid] = PlayerTrackState(track_id=tid)
                self._tracks[tid].add(cx, cy, timestamp)

                detections.append(
                    PlayerDetection(
                        track_id=tid,
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        x=cx,
                        y=cy,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                    )
                )
        return detections

    def _optical_flow_fallback(
        self, frame_idx: int, timestamp: float, frame: np.ndarray
    ) -> List[PlayerDetection]:
        """Return a synthetic 'player' detection based on dense optical flow."""
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray
            return []

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_frame_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        self._prev_frame_gray = gray

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        # Find the centroid of high-motion areas as a proxy for players
        mask = (magnitude > 2.0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[PlayerDetection] = []
        # Keep at most 2 largest contours (2 players)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        for idx, cnt in enumerate(sorted_contours):
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            tid = idx
            if tid not in self._tracks:
                self._tracks[tid] = PlayerTrackState(track_id=tid)
            self._tracks[tid].add(cx, cy, timestamp)
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append(
                PlayerDetection(
                    track_id=tid,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    x=cx,
                    y=cy,
                    bbox=(float(x), float(y), float(x + w), float(y + h)),
                    confidence=min(area / 5000.0, 1.0),
                )
            )
        return detections

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
