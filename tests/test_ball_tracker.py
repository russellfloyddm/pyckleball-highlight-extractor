"""Tests for ball tracking logic."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pickleball_highlights.ball_tracker import BallDetection, BallTracker, BallTrackState
from pickleball_highlights.config import DetectionConfig


class TestBallTrackState:
    def test_empty_state(self):
        state = BallTrackState()
        assert state.average_speed == 0.0
        assert state.max_speed == 0.0

    def test_add_single_detection(self):
        state = BallTrackState()
        det = BallDetection(
            frame_idx=0,
            timestamp=0.0,
            x=100.0,
            y=100.0,
            confidence=0.9,
            bbox=(95.0, 95.0, 105.0, 105.0),
        )
        state.add(det)
        # No previous detection, so no speed computed
        assert state.average_speed == 0.0

    def test_add_two_detections_computes_speed(self):
        state = BallTrackState()
        det1 = BallDetection(0, 0.0, 0.0, 0.0, 0.9, (0, 0, 10, 10))
        det2 = BallDetection(1, 1.0, 300.0, 400.0, 0.9, (295, 395, 305, 405))
        state.add(det1)
        state.add(det2)
        # distance = sqrt(300^2 + 400^2) = 500, dt = 1.0 → speed = 500 px/s
        assert state.average_speed == pytest.approx(500.0)
        assert state.max_speed == pytest.approx(500.0)

    def test_reset_clears_state(self):
        state = BallTrackState()
        det1 = BallDetection(0, 0.0, 0.0, 0.0, 0.9, (0, 0, 10, 10))
        det2 = BallDetection(1, 1.0, 100.0, 100.0, 0.9, (95, 95, 105, 105))
        state.add(det1)
        state.add(det2)
        state.reset()
        assert state.detections == []
        assert state.speeds == []
        assert state.average_speed == 0.0


class TestBallTrackerFallback:
    """Test the optical-flow fallback (no YOLO required)."""

    def _make_tracker(self) -> BallTracker:
        config = DetectionConfig(device="cpu")
        tracker = BallTracker.__new__(BallTracker)
        tracker.config = config
        tracker._model = None  # Force fallback
        tracker._state = BallTrackState()
        tracker._device = "cpu"
        tracker._prev_frame_gray = None
        return tracker

    def test_first_frame_returns_none(self):
        tracker = self._make_tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.process_frame(0, 0.0, frame)
        assert result is None  # First frame sets prev, no detection yet

    def test_static_frame_returns_no_detection(self):
        tracker = self._make_tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.process_frame(0, 0.0, frame)
        # Same frame again → no motion
        result = tracker.process_frame(1, 0.033, frame)
        assert result is None

    def test_moving_object_returns_detection(self):
        tracker = self._make_tracker()
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a white square in frame2 to simulate motion
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[200:250, 300:350] = 255

        tracker.process_frame(0, 0.0, frame1)
        result = tracker.process_frame(1, 0.033, frame2)
        assert result is not None
        assert result.frame_idx == 1
        assert result.confidence > 0.0
        assert result.confidence <= 0.6

    def test_thin_motion_region_is_rejected(self):
        tracker = self._make_tracker()
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        # Very elongated motion region should be filtered by aspect ratio.
        frame2[220:225, 100:260] = 255

        tracker.process_frame(0, 0.0, frame1)
        result = tracker.process_frame(1, 0.033, frame2)
        assert result is None

    def test_reset_clears_prev_frame(self):
        tracker = self._make_tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.process_frame(0, 0.0, frame)
        tracker.reset()
        assert tracker._prev_frame_gray is None
        assert tracker._state.detections == []

    def test_yolo_miss_uses_optical_flow_fallback(self):
        class EmptyResult:
            boxes = []

        class FakeModel:
            def predict(self, *args, **kwargs):
                return [EmptyResult()]

        tracker = self._make_tracker()
        tracker._model = FakeModel()

        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[200:250, 300:350] = 255

        first = tracker.process_frame(0, 0.0, frame1)
        second = tracker.process_frame(1, 0.033, frame2)
        assert first is None
        assert second is not None
        assert second.frame_idx == 1


class TestBallTrackerDeviceResolution:
    def test_cpu_device(self):
        result = BallTracker._resolve_device("cpu")
        assert result == "cpu"

    def test_auto_device_fallback_to_cpu(self):
        # In test environment without GPU, should fall back to CPU
        result = BallTracker._resolve_device("auto")
        assert result in ("cpu", "cuda", "mps")

    def test_explicit_device_passthrough(self):
        result = BallTracker._resolve_device("cuda")
        assert result == "cuda"
