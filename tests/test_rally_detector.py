"""Tests for rally detection logic."""

from __future__ import annotations

import pytest

from pickleball_highlights.ball_tracker import BallDetection, BallTrackState
from pickleball_highlights.config import RallyConfig
from pickleball_highlights.rally_detector import Rally, RallyDetector


class DummyAudioFrame:
    def __init__(self, start_time: float, end_time: float, is_spike: bool) -> None:
        self.start_time = start_time
        self.end_time = end_time
        self.is_spike = is_spike


class DummyAudioAnalyzer:
    def __init__(self, frames):
        self._frames = frames

    def get_frames(self):
        return self._frames


def make_ball_det(frame_idx: int, timestamp: float, x: float = 100.0, y: float = 100.0) -> BallDetection:
    return BallDetection(
        frame_idx=frame_idx,
        timestamp=timestamp,
        x=x,
        y=y,
        confidence=0.9,
        bbox=(x - 5, y - 5, x + 5, y + 5),
    )


def make_state_with_speed(speed: float) -> BallTrackState:
    state = BallTrackState()
    state.speeds = [speed]
    return state


class TestRally:
    def test_duration(self):
        rally = Rally(
            start_time=10.0,
            end_time=30.0,
            shot_count=15,
            max_ball_speed=500.0,
            avg_ball_speed=300.0,
            movement_score=0.5,
        )
        assert rally.duration == pytest.approx(20.0)

    def test_duration_score_capped_at_one(self):
        rally = Rally(
            start_time=0.0,
            end_time=60.0,
            shot_count=30,
            max_ball_speed=700.0,
            avg_ball_speed=500.0,
            movement_score=0.8,
        )
        assert rally.duration_score == 1.0

    def test_duration_score_partial(self):
        rally = Rally(
            start_time=0.0,
            end_time=10.0,
            shot_count=5,
            max_ball_speed=400.0,
            avg_ball_speed=300.0,
            movement_score=0.4,
        )
        assert rally.duration_score == pytest.approx(0.5)

    def test_shot_count_score(self):
        rally = Rally(
            start_time=0.0,
            end_time=15.0,
            shot_count=10,
            max_ball_speed=400.0,
            avg_ball_speed=300.0,
            movement_score=0.4,
        )
        assert rally.shot_count_score == pytest.approx(0.5)

    def test_long_rally_bonus_30_shots(self):
        rally = Rally(
            start_time=0.0,
            end_time=30.0,
            shot_count=30,
            max_ball_speed=400.0,
            avg_ball_speed=300.0,
            movement_score=0.5,
        )
        assert rally.long_rally_bonus == pytest.approx(0.35)

    def test_long_rally_bonus_20_shots(self):
        rally = Rally(
            start_time=0.0,
            end_time=20.0,
            shot_count=20,
            max_ball_speed=400.0,
            avg_ball_speed=300.0,
            movement_score=0.5,
        )
        assert rally.long_rally_bonus == pytest.approx(0.20)

    def test_long_rally_bonus_10_shots(self):
        rally = Rally(
            start_time=0.0,
            end_time=15.0,
            shot_count=10,
            max_ball_speed=400.0,
            avg_ball_speed=300.0,
            movement_score=0.5,
        )
        assert rally.long_rally_bonus == pytest.approx(0.10)

    def test_no_long_rally_bonus(self):
        rally = Rally(
            start_time=0.0,
            end_time=5.0,
            shot_count=5,
            max_ball_speed=200.0,
            avg_ball_speed=150.0,
            movement_score=0.3,
        )
        assert rally.long_rally_bonus == 0.0


class TestRallyDetector:
    def _make_detector(self, **kwargs) -> RallyDetector:
        defaults = {
            "min_detection_streak": 1,
            "detection_window": 1,
            "min_movement_to_start": 0.0,
            "min_ball_speed": 0.0,
            "service_anchor_window": 0.0,
        }
        defaults.update(kwargs)
        config = RallyConfig(**defaults)
        return RallyDetector(config)

    def test_no_ball_no_rally(self):
        detector = self._make_detector()
        state = BallTrackState()
        for t in range(10):
            result = detector.update(float(t), None, state)
            assert result is None
        assert detector.get_rallies() == []

    def test_flush_empty(self):
        detector = self._make_detector()
        result = detector.flush(10.0)
        assert result is None

    def test_short_rally_discarded(self):
        """Rallies shorter than min_rally_duration should be discarded."""
        detector = self._make_detector(min_rally_duration=5.0, min_shots=2)
        state = make_state_with_speed(200.0)

        # Simulate 2 seconds of ball visibility then gap
        for i in range(4):
            det = make_ball_det(i, float(i) * 0.5)
            detector.update(float(i) * 0.5, det, state)

        # Gap > max_gap_duration triggers end
        result = detector.update(20.0, None, state)
        assert result is None
        assert len(detector.get_rallies()) == 0

    def test_valid_rally_detected(self):
        """A sufficient rally should be detected and returned."""
        config = RallyConfig(
            min_rally_duration=2.0,
            max_gap_duration=3.0,
            min_shots=0,  # disable shot requirement for this test
            min_detection_streak=1,
            detection_window=1,
            min_movement_to_start=0.0,
            min_ball_speed=0.0,
            service_anchor_window=0.0,
        )
        detector = RallyDetector(config)
        state = BallTrackState()
        state.speeds = [300.0, 350.0, 400.0, 350.0, 320.0]

        # 10 seconds of ball visibility (frames at 0.0, 0.5, …, 9.5)
        for i in range(20):
            det = make_ball_det(i, float(i) * 0.5)
            detector.update(float(i) * 0.5, det, state)

        # First frame with no ball, gap = 20.0 - 9.5 = 10.5 s > max_gap_duration=3.0
        # so the rally ends immediately; end_time is stamped at the current frame (20.0)
        result = detector.update(20.0, None, state)
        assert result is not None
        assert result.start_time == pytest.approx(0.0)
        assert result.end_time == pytest.approx(20.0)
        assert result.duration >= 2.0

    def test_detection_streak_required(self):
        detector = RallyDetector(
            RallyConfig(
                min_rally_duration=1.0,
                max_gap_duration=1.0,
                min_shots=0,
                min_detection_streak=3,
                detection_window=5,
                min_movement_to_start=0.0,
                min_ball_speed=0.0,
                service_anchor_window=0.0,
            )
        )
        state = make_state_with_speed(200.0)

        # Only two detections in window -> should not start.
        detector.update(0.0, make_ball_det(0, 0.0), state)
        detector.update(0.5, None, state)
        detector.update(1.0, make_ball_det(2, 1.0), state)
        assert detector._active_rally_start is None

        # Third detection in the window starts rally.
        detector.update(1.5, make_ball_det(3, 1.5), state)
        assert detector._active_rally_start is not None

    def test_movement_gate_blocks_start(self):
        detector = RallyDetector(
            RallyConfig(
                min_rally_duration=1.0,
                min_shots=0,
                min_detection_streak=1,
                detection_window=1,
                min_movement_to_start=0.2,
                min_ball_speed=0.0,
                service_anchor_window=0.0,
            )
        )
        state = make_state_with_speed(200.0)

        detector.update(0.0, make_ball_det(0, 0.0), state, movement_score=0.1)
        assert detector._active_rally_start is None

        detector.update(0.5, make_ball_det(1, 0.5), state, movement_score=0.3)
        assert detector._active_rally_start == pytest.approx(0.5)

    def test_min_ball_speed_required_to_keep_rally_active(self):
        detector = RallyDetector(
            RallyConfig(
                min_rally_duration=1.0,
                max_gap_duration=1.0,
                min_shots=0,
                min_detection_streak=1,
                detection_window=1,
                min_movement_to_start=0.0,
                min_ball_speed=50.0,
                service_anchor_window=0.0,
            )
        )
        slow_state = make_state_with_speed(10.0)
        fast_state = make_state_with_speed(100.0)

        detector.update(0.0, make_ball_det(0, 0.0), fast_state, movement_score=0.5)
        detector.update(0.5, make_ball_det(1, 0.5), slow_state, movement_score=0.5)
        detector.update(2.0, None, slow_state, movement_score=0.0)
        # Rally should have ended because last in-play speed update was at t=0.0
        assert detector._active_rally_start is None

    def test_service_anchor_waits_for_speed_spike(self):
        detector = RallyDetector(
            RallyConfig(
                min_rally_duration=1.0,
                max_gap_duration=1.0,
                min_shots=0,
                min_detection_streak=1,
                detection_window=3,
                min_movement_to_start=0.0,
                min_ball_speed=0.0,
                service_anchor_window=1.0,
                service_anchor_position_std=3.0,
                service_anchor_min_speed_spike=80.0,
            )
        )

        # Stationary pre-serve ball at nearly same position with low speed.
        low_speed = make_state_with_speed(20.0)
        for i, t in enumerate([0.0, 0.3, 0.6]):
            detector.update(
                t,
                make_ball_det(i, t, x=100.0 + (i * 0.2), y=100.0 + (i * 0.2)),
                low_speed,
                movement_score=0.2,
            )
        assert detector._active_rally_start is None

        # Speed spike indicates serve contact -> rally starts now.
        spike = make_state_with_speed(120.0)
        detector.update(0.9, make_ball_det(3, 0.9, x=105.0, y=104.0), spike, movement_score=0.2)
        assert detector._active_rally_start == pytest.approx(0.9)

    def test_audio_corroboration_gates_shot_count(self):
        detector = self._make_detector(min_shots=0)
        detector.set_audio_analyzer(
            DummyAudioAnalyzer([DummyAudioFrame(0.0, 10.0, is_spike=False)])
        )
        state = BallTrackState()

        detector.update(0.0, make_ball_det(0, 0.0), state, movement_score=0.3)
        state.speeds = [40.0]
        detector.update(0.1, make_ball_det(1, 0.1), state, movement_score=0.3)
        state.speeds = [100.0]
        detector.update(0.2, make_ball_det(2, 0.2), state, movement_score=0.3)
        state.speeds = [60.0]
        detector.update(0.3, make_ball_det(3, 0.3), state, movement_score=0.3)

        # Peak is present but audio did not spike.
        assert detector._active_rally_shots == 0

        detector.set_audio_analyzer(
            DummyAudioAnalyzer([DummyAudioFrame(0.0, 10.0, is_spike=True)])
        )
        state.speeds = [120.0]
        detector.update(0.4, make_ball_det(4, 0.4), state, movement_score=0.3)
        state.speeds = [70.0]
        detector.update(0.5, make_ball_det(5, 0.5), state, movement_score=0.3)
        assert detector._active_rally_shots >= 1

    def test_reset_clears_state(self):
        detector = self._make_detector()
        state = make_state_with_speed(200.0)
        for i in range(5):
            det = make_ball_det(i, float(i))
            detector.update(float(i), det, state)

        detector.reset()
        assert detector.get_rallies() == []
        assert detector._active_rally_start is None
