"""Tests for rally detection logic."""

from __future__ import annotations

import pytest

from pickleball_highlights.ball_tracker import BallDetection, BallTrackState
from pickleball_highlights.config import RallyConfig
from pickleball_highlights.rally_detector import Rally, RallyDetector


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
        config = RallyConfig(**kwargs) if kwargs else RallyConfig()
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

    def test_reset_clears_state(self):
        detector = self._make_detector()
        state = make_state_with_speed(200.0)
        for i in range(5):
            det = make_ball_det(i, float(i))
            detector.update(float(i), det, state)

        detector.reset()
        assert detector.get_rallies() == []
        assert detector._active_rally_start is None
