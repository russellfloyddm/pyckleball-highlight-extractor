"""Pose estimation for detecting celebrations, lunges, dives, and saves."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from pickleball_highlights.utils import get_logger

logger = get_logger(__name__)


class PoseEvent(Enum):
    """High-level pose events relevant to highlight detection."""

    CELEBRATION = "celebration"
    LUNGE = "lunge"
    DIVE = "dive"
    RAISED_PADDLE = "raised_paddle"
    NORMAL = "normal"


@dataclass
class PoseResult:
    """Pose estimation result for a single frame."""

    frame_idx: int
    timestamp: float
    event: PoseEvent
    confidence: float
    keypoints: Optional[np.ndarray] = None  # shape (N, 3) – x, y, score


class PoseEstimator:
    """Estimate player poses to detect exciting moments.

    Uses MediaPipe Pose as the primary backend, falling back to a
    heuristic keypoint analysis when MediaPipe is unavailable.

    Args:
        device: Inference device (ignored for MediaPipe, used for YOLO pose).
    """

    # MediaPipe landmark indices
    _MP_LEFT_WRIST = 15
    _MP_RIGHT_WRIST = 16
    _MP_LEFT_ANKLE = 27
    _MP_RIGHT_ANKLE = 28
    _MP_LEFT_HIP = 23
    _MP_RIGHT_HIP = 24
    _MP_LEFT_SHOULDER = 11
    _MP_RIGHT_SHOULDER = 12
    _MP_NOSE = 0

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._pose = None
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self, frame_idx: int, timestamp: float, frame: np.ndarray
    ) -> Optional[PoseResult]:
        """Run pose estimation on a single BGR frame.

        Args:
            frame_idx: Frame index.
            timestamp: Frame timestamp in seconds.
            frame: BGR image array.

        Returns:
            PoseResult or None if no person is found.
        """
        if self._pose is not None:
            return self._mediapipe_process(frame_idx, timestamp, frame)
        return self._heuristic_fallback(frame_idx, timestamp, frame)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass
            self._pose = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            import mediapipe as mp  # type: ignore

            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe Pose loaded successfully.")
        except ImportError:
            logger.warning(
                "mediapipe not installed – pose estimation will use heuristic fallback."
            )
            self._pose = None
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load MediaPipe Pose: %s", exc)
            self._pose = None

    def _mediapipe_process(
        self, frame_idx: int, timestamp: float, frame: np.ndarray
    ) -> Optional[PoseResult]:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]
        kps = np.array(
            [[lm.x * w, lm.y * h, lm.visibility] for lm in landmarks],
            dtype=np.float32,
        )

        event = self._classify_pose(kps, h, w)
        conf = float(np.mean(kps[:, 2]))
        return PoseResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            event=event,
            confidence=conf,
            keypoints=kps,
        )

    def _heuristic_fallback(
        self, frame_idx: int, timestamp: float, frame: np.ndarray
    ) -> Optional[PoseResult]:
        """Lightweight fallback using colour/motion heuristics."""
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_magnitude = float(np.std(gray.astype(np.float32)))
        # Very rough proxy: high intensity in upper half → raised arm/paddle
        h = frame.shape[0]
        upper = gray[: h // 2, :]
        lower = gray[h // 2 :, :]
        upper_activity = float(np.std(upper.astype(np.float32)))
        lower_activity = float(np.std(lower.astype(np.float32)))

        if upper_activity > 60 and upper_activity > 1.3 * lower_activity:
            event = PoseEvent.RAISED_PADDLE
            conf = min(upper_activity / 100.0, 1.0)
        elif diff_magnitude > 70:
            event = PoseEvent.LUNGE
            conf = min(diff_magnitude / 100.0, 1.0)
        else:
            event = PoseEvent.NORMAL
            conf = 0.3

        return PoseResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            event=event,
            confidence=conf,
        )

    def _classify_pose(
        self, kps: np.ndarray, frame_h: int, frame_w: int
    ) -> PoseEvent:
        """Classify the dominant pose event from keypoints."""
        left_wrist = kps[self._MP_LEFT_WRIST]
        right_wrist = kps[self._MP_RIGHT_WRIST]
        left_shoulder = kps[self._MP_LEFT_SHOULDER]
        right_shoulder = kps[self._MP_RIGHT_SHOULDER]
        left_hip = kps[self._MP_LEFT_HIP]
        right_hip = kps[self._MP_RIGHT_HIP]
        left_ankle = kps[self._MP_LEFT_ANKLE]
        right_ankle = kps[self._MP_RIGHT_ANKLE]

        # Raised paddle / celebration: wrists above shoulders
        wrists_above_shoulders = (
            left_wrist[1] < left_shoulder[1] and left_wrist[2] > 0.5
        ) or (right_wrist[1] < right_shoulder[1] and right_wrist[2] > 0.5)

        # Lunge: large horizontal displacement of ankles
        ankle_spread = abs(left_ankle[0] - right_ankle[0])
        hip_width = abs(left_hip[0] - right_hip[0])
        is_lunge = ankle_spread > 2.5 * hip_width and (
            left_ankle[2] > 0.4 and right_ankle[2] > 0.4
        )

        # Dive: body close to ground (hips near bottom quarter of frame)
        hip_y = (left_hip[1] + right_hip[1]) / 2
        is_dive = hip_y > 0.75 * frame_h

        if wrists_above_shoulders:
            return PoseEvent.CELEBRATION
        if is_dive:
            return PoseEvent.DIVE
        if is_lunge:
            return PoseEvent.LUNGE
        return PoseEvent.NORMAL
