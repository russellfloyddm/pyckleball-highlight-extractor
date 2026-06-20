"""Rally detection: identify start/end of rallies and count shots."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple

import numpy as np

from pickleball_highlights.ball_tracker import BallDetection, BallTrackState
from pickleball_highlights.config import RallyConfig
from pickleball_highlights.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Rally:
    """Represents a detected rally segment."""

    start_time: float
    end_time: float
    shot_count: int
    max_ball_speed: float  # pixels/second
    avg_ball_speed: float  # pixels/second
    movement_score: float  # player movement intensity [0, 1]

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def duration_score(self) -> float:
        """Normalised rally duration score [0, 1] (20 s → 1.0)."""
        return min(self.duration / 20.0, 1.0)

    @property
    def shot_count_score(self) -> float:
        """Normalised shot count score [0, 1] (20 shots → 1.0)."""
        return min(self.shot_count / 20.0, 1.0)

    @property
    def long_rally_bonus(self) -> float:
        """Bonus score for long rallies."""
        if self.shot_count >= 30:
            return 0.35
        if self.shot_count >= 20:
            return 0.20
        if self.shot_count >= 10:
            return 0.10
        return 0.0


class RallyDetector:
    """Detect rallies from ball tracking data using temporal windowing.

    Maintains a sliding window over ball detections to identify contiguous
    periods of play.

    Args:
        config: Rally detection configuration.
    """

    def __init__(self, config: RallyConfig) -> None:
        self.config = config
        self._active_rally_start: Optional[float] = None
        self._active_rally_shots: int = 0
        self._last_ball_time: Optional[float] = None
        self._shot_times: List[float] = []
        self._ball_speeds: List[float] = []
        self._completed_rallies: List[Rally] = []
        self._movement_score: float = 0.0
        self._recent_detections: Deque[bool] = deque(
            maxlen=max(self.config.detection_window, 1)
        )
        self._pre_rally_positions: Deque[Tuple[float, float, float]] = deque()
        self._audio_analyzer: Optional[Any] = None

    def set_audio_analyzer(self, audio_analyzer: Optional[Any]) -> None:
        """Attach an optional audio analyzer for shot corroboration."""
        self._audio_analyzer = audio_analyzer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        timestamp: float,
        ball_det: Optional[BallDetection],
        ball_state: BallTrackState,
        movement_score: float = 0.0,
    ) -> Optional[Rally]:
        """Process a single frame and potentially emit a completed rally.

        Args:
            timestamp: Current frame timestamp in seconds.
            ball_det: Ball detection for this frame (or None).
            ball_state: Current ball tracking state.
            movement_score: Normalised player movement score for this frame.

        Returns:
            A Rally object if a rally just ended, otherwise None.
        """
        self._recent_detections.append(ball_det is not None)

        if ball_det is not None:
            self._record_pre_rally_position(timestamp, ball_det.x, ball_det.y)

            latest_speed = ball_state.speeds[-1] if ball_state.speeds else 0.0

            # Ball visible – potentially in a rally
            if self._active_rally_start is None and self._should_start_rally(
                timestamp, latest_speed, movement_score
            ):
                self._start_rally(timestamp)

            if self._active_rally_start is not None:
                self._update_rally(timestamp, ball_state, movement_score)
        else:
            # Ball not visible
            if self._active_rally_start is not None:
                last_ball_time = (
                    self._last_ball_time
                    if self._last_ball_time is not None
                    else timestamp
                )
                gap = timestamp - last_ball_time
                if gap > self.config.max_gap_duration:
                    return self._end_rally(timestamp)
        return None

    def flush(self, final_timestamp: float) -> Optional[Rally]:
        """Flush any in-progress rally at the end of the video.

        Args:
            final_timestamp: Timestamp of the last processed frame.

        Returns:
            Final Rally if one was in progress, otherwise None.
        """
        if self._active_rally_start is not None:
            return self._end_rally(final_timestamp)
        return None

    def get_rallies(self) -> List[Rally]:
        """Return all completed rallies detected so far."""
        return list(self._completed_rallies)

    def reset(self) -> None:
        """Reset all state."""
        self._active_rally_start = None
        self._active_rally_shots = 0
        self._last_ball_time = None
        self._shot_times.clear()
        self._ball_speeds.clear()
        self._completed_rallies.clear()
        self._movement_score = 0.0
        self._recent_detections.clear()
        self._pre_rally_positions.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _start_rally(self, timestamp: float) -> None:
        self._active_rally_start = timestamp
        self._active_rally_shots = 0
        self._shot_times.clear()
        self._ball_speeds.clear()
        self._movement_score = 0.0
        self._last_ball_time = timestamp
        self._pre_rally_positions.clear()
        logger.debug("Rally started at %.2fs", timestamp)

    def _update_rally(
        self,
        timestamp: float,
        ball_state: BallTrackState,
        movement_score: float,
    ) -> None:
        """Count shots using direction-reversal heuristic."""
        if ball_state.speeds:
            latest_speed = ball_state.speeds[-1]
            if latest_speed >= self.config.min_ball_speed:
                self._ball_speeds.append(latest_speed)
                self._last_ball_time = timestamp

                # Shot detection: significant speed peak (proxy for paddle contact)
                if len(self._ball_speeds) >= 3:
                    prev_speed, peak_speed = self._ball_speeds[-3], self._ball_speeds[-2]
                    audio_ok = self._audio_supports_shot(timestamp)
                    # Require a minimum absolute speed floor (20 px/s) to
                    # reject jitter/camera wobble even when config thresholds
                    # are tuned low for distant-camera footage.
                    # Double configured min speed so shot peaks must clearly
                    # exceed the baseline in-play threshold.
                    min_peak_speed = max(self.config.min_ball_speed * 2.0, 20.0)
                    visually_strong_peak = (
                        # 1.35x captures realistic shot acceleration in noisy
                        # tracks where per-frame speed is damped by blur.
                        peak_speed > prev_speed * 1.35 and peak_speed > min_peak_speed
                    )
                    if visually_strong_peak and (
                        # If audio does not corroborate, require a much larger
                        # 2.2x jump before counting a visual-only shot.
                        audio_ok or peak_speed > prev_speed * 2.2
                    ):
                        self._active_rally_shots += 1
                        self._shot_times.append(timestamp)

            elif self._last_ball_time is None:
                # Keep very early low-speed detections from ending the rally immediately.
                self._last_ball_time = timestamp

        elif self._last_ball_time is None:
            self._last_ball_time = timestamp

        # Maintain a running weighted average of movement
        alpha = 0.1
        self._movement_score = (
            alpha * movement_score + (1 - alpha) * self._movement_score
        )

    def _should_start_rally(
        self, timestamp: float, latest_speed: float, movement_score: float
    ) -> bool:
        if movement_score < self.config.min_movement_to_start:
            return False

        detection_hits = sum(self._recent_detections)
        if detection_hits < self.config.min_detection_streak:
            return False

        if self._was_recently_stationary(timestamp):
            return latest_speed >= self.config.service_anchor_min_speed_spike

        return True

    def _record_pre_rally_position(self, timestamp: float, x: float, y: float) -> None:
        self._pre_rally_positions.append((timestamp, x, y))
        earliest = timestamp - self.config.service_anchor_window
        while self._pre_rally_positions and self._pre_rally_positions[0][0] < earliest:
            self._pre_rally_positions.popleft()

    def _was_recently_stationary(self, timestamp: float) -> bool:
        if self.config.service_anchor_window <= 0:
            return False

        lower = timestamp - self.config.service_anchor_window
        recent_points = [
            (x, y)
            for t, x, y in self._pre_rally_positions
            if lower <= t <= timestamp
        ]
        if len(recent_points) < 3:
            return False

        arr = np.array(recent_points, dtype=np.float32)
        center = np.mean(arr, axis=0)
        radial_distances = np.linalg.norm(arr - center, axis=1)
        radial_std = float(np.std(radial_distances))
        return radial_std <= self.config.service_anchor_position_std

    def _audio_supports_shot(self, timestamp: float) -> bool:
        if self._audio_analyzer is None:
            return True

        try:
            frames = self._audio_analyzer.get_frames()
        except Exception as exc:
            logger.debug("Audio corroboration unavailable at %.2fs: %s", timestamp, exc)
            return True

        # Be permissive when audio extraction/analysis produced no usable
        # windows: treat audio as unavailable rather than blocking visual shots.
        if not frames:
            return True

        for frame in frames:
            if frame.start_time <= timestamp < frame.end_time:
                return bool(frame.is_spike)
        # Audio windows can miss frame timestamps at boundaries; do not block
        # visual shot evidence when audio is unavailable for this instant.
        return True

    def _end_rally(self, timestamp: float) -> Optional[Rally]:
        start = self._active_rally_start
        self._active_rally_start = None

        if start is None:
            return None

        duration = timestamp - start
        if duration < self.config.min_rally_duration:
            logger.debug("Discarding rally (duration %.2fs < min %.2fs)", duration, self.config.min_rally_duration)
            return None
        if self._active_rally_shots < self.config.min_shots:
            logger.debug("Discarding rally (shots %d < min %d)", self._active_rally_shots, self.config.min_shots)
            return None

        avg_speed = float(np.mean(self._ball_speeds)) if self._ball_speeds else 0.0
        max_speed = float(np.max(self._ball_speeds)) if self._ball_speeds else 0.0

        rally = Rally(
            start_time=start,
            end_time=timestamp,
            shot_count=self._active_rally_shots,
            max_ball_speed=max_speed,
            avg_ball_speed=avg_speed,
            movement_score=self._movement_score,
        )
        self._completed_rallies.append(rally)
        logger.info(
            "Rally detected: %.2fs–%.2fs (%.1fs, %d shots, avg speed=%.1f px/s)",
            rally.start_time,
            rally.end_time,
            rally.duration,
            rally.shot_count,
            rally.avg_ball_speed,
        )
        return rally
