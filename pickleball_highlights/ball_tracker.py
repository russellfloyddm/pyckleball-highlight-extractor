"""Ball detection and tracking using YOLOv8 and trajectory analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from pickleball_highlights.config import DetectionConfig
from pickleball_highlights.utils import get_logger

logger = get_logger(__name__)


@dataclass
class BallDetection:
    """Single-frame ball detection result."""

    frame_idx: int
    timestamp: float
    x: float  # centre x (pixels)
    y: float  # centre y (pixels)
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class BallTrackState:
    """Accumulated ball tracking state across multiple frames."""

    detections: List[BallDetection] = field(default_factory=list)
    speeds: List[float] = field(default_factory=list)  # pixels/second

    def add(self, det: BallDetection) -> None:
        if self.detections:
            prev = self.detections[-1]
            dt = det.timestamp - prev.timestamp
            if dt > 0:
                dist = math.hypot(det.x - prev.x, det.y - prev.y)
                self.speeds.append(dist / dt)
        self.detections.append(det)

    @property
    def average_speed(self) -> float:
        return float(np.mean(self.speeds)) if self.speeds else 0.0

    @property
    def max_speed(self) -> float:
        return float(np.max(self.speeds)) if self.speeds else 0.0

    def reset(self) -> None:
        self.detections.clear()
        self.speeds.clear()


class BallTracker:
    """Detect and track the pickleball across video frames.

    Uses YOLOv8 for per-frame detection and applies a simple nearest-neighbour
    association to link detections across frames.  Falls back gracefully when
    ``ultralytics`` is not installed.

    Args:
        config: Detection configuration.
    """

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self._model = None
        self._state = BallTrackState()
        self._device = self._resolve_device(config.device)
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self, frame_idx: int, timestamp: float, frame: np.ndarray
    ) -> Optional[BallDetection]:
        """Run detection on a single BGR frame.

        Args:
            frame_idx: Index of the current frame.
            timestamp: Frame timestamp in seconds.
            frame: BGR image array.

        Returns:
            BallDetection if the ball is found, otherwise None.
        """
        if self._model is None:
            return self._optical_flow_fallback(frame_idx, timestamp, frame)

        try:
            results = self._model.predict(
                frame,
                classes=[self.config.ball_class_id],
                conf=self.config.yolo_confidence,
                iou=self.config.yolo_iou,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("YOLOv8 inference error: %s", exc)
            return None

        best = self._best_detection(results, frame_idx, timestamp)
        if best is not None:
            self._state.add(best)
        return best

    def get_state(self) -> BallTrackState:
        """Return the current accumulated tracking state."""
        return self._state

    def reset(self) -> None:
        """Reset accumulated tracking state (call between rallies)."""
        self._state.reset()
        self._prev_frame_gray: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.config.yolo_model)
            logger.info("Loaded YOLOv8 model: %s", self.config.yolo_model)
        except ImportError:
            logger.warning(
                "ultralytics not installed – ball tracking will use optical "
                "flow fallback."
            )
            self._model = None
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load YOLOv8 model (%s) – using fallback.", exc)
            self._model = None

        self._prev_frame_gray: Optional[np.ndarray] = None

    def _best_detection(
        self, results, frame_idx: int, timestamp: float
    ) -> Optional[BallDetection]:
        """Extract the highest-confidence ball detection from YOLO results."""
        best_conf = -1.0
        best_det: Optional[BallDetection] = None

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.config.yolo_confidence:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if conf > best_conf:
                    best_conf = conf
                    best_det = BallDetection(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        x=cx,
                        y=cy,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                    )
        return best_det

    def _optical_flow_fallback(
        self, frame_idx: int, timestamp: float, frame: np.ndarray
    ) -> Optional[BallDetection]:
        """Simple motion-based pseudo-detection when YOLO is unavailable.

        Returns a synthetic detection at the centroid of the strongest motion
        region, which is sufficient for rough rally/motion detection.
        """
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray
            return None

        diff = cv2.absdiff(gray, self._prev_frame_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self._prev_frame_gray = gray

        if not contours:
            return None

        # Pick the largest contour as a proxy for the ball/most-moving object
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        # Reject tiny motion blobs (sensor noise/compression artifacts).
        if area < 100:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        min_side = min(w, h)
        if min_side <= 0:
            return None
        # A ball-like region should be near-round, not highly elongated.
        aspect_ratio = max(w, h) / min_side
        if aspect_ratio > 2.0:
            return None

        m = cv2.moments(largest)
        if m["m00"] == 0:
            return None
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]

        det = BallDetection(
            frame_idx=frame_idx,
            timestamp=timestamp,
            x=cx,
            y=cy,
            confidence=min(area / 2000.0, 0.6),
            bbox=(cx - 5, cy - 5, cx + 5, cy + 5),
        )
        self._state.add(det)
        return det

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
