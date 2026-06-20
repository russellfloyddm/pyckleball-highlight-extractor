"""Tests for the highlight scoring module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from pickleball_highlights.audio_analyzer import AudioFrame
from pickleball_highlights.config import ScoringWeights
from pickleball_highlights.highlight_scorer import HighlightCandidate, HighlightScorer
from pickleball_highlights.rally_detector import Rally


def make_rally(
    start: float = 0.0,
    end: float = 20.0,
    shot_count: int = 15,
    max_ball_speed: float = 500.0,
    avg_ball_speed: float = 300.0,
    movement_score: float = 0.5,
) -> Rally:
    return Rally(
        start_time=start,
        end_time=end,
        shot_count=shot_count,
        max_ball_speed=max_ball_speed,
        avg_ball_speed=avg_ball_speed,
        movement_score=movement_score,
    )


def make_scorer(audio_frames=None) -> HighlightScorer:
    audio = MagicMock()
    audio.get_frames.return_value = audio_frames or []
    audio.get_score_at.return_value = 0.0
    weights = ScoringWeights()
    return HighlightScorer(weights, audio)


class TestHighlightScorer:
    def test_score_returns_candidate(self):
        scorer = make_scorer()
        rally = make_rally()
        candidate = scorer.score_rally(rally)
        assert isinstance(candidate, HighlightCandidate)
        assert 0.0 <= candidate.score <= 1.0

    def test_score_is_bounded(self):
        scorer = make_scorer()
        # Perfect rally (max everything)
        rally = make_rally(
            start=0.0,
            end=60.0,
            shot_count=50,
            avg_ball_speed=800.0,
            movement_score=1.0,
        )
        candidate = scorer.score_rally(rally)
        assert candidate.score <= 1.0

    def test_score_zero_rally(self):
        scorer = make_scorer()
        rally = make_rally(
            start=0.0,
            end=1.0,
            shot_count=0,
            avg_ball_speed=0.0,
            movement_score=0.0,
        )
        candidate = scorer.score_rally(rally)
        assert candidate.score >= 0.0

    def test_longer_rally_scores_higher(self):
        scorer = make_scorer()
        short = make_rally(start=0.0, end=5.0, shot_count=3, avg_ball_speed=100.0)
        long = make_rally(start=0.0, end=25.0, shot_count=20, avg_ball_speed=300.0)
        c_short = scorer.score_rally(short)
        c_long = scorer.score_rally(long)
        assert c_long.score > c_short.score

    def test_score_rallies_sorted_descending(self):
        scorer = make_scorer()
        rallies = [
            make_rally(start=0.0, end=5.0, shot_count=3),
            make_rally(start=10.0, end=30.0, shot_count=20),
            make_rally(start=40.0, end=50.0, shot_count=10),
        ]
        candidates = scorer.score_rallies(rallies)
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_audio_score_incorporated(self):
        # Rally with high audio excitement should score higher
        high_audio_frames = [
            AudioFrame(start_time=0.0, end_time=20.0, rms_energy=1.0, excitement_score=0.95, is_spike=True)
        ]
        low_audio_frames = [
            AudioFrame(start_time=0.0, end_time=20.0, rms_energy=0.1, excitement_score=0.05, is_spike=False)
        ]
        scorer_high = make_scorer(audio_frames=high_audio_frames)
        scorer_low = make_scorer(audio_frames=low_audio_frames)

        rally = make_rally()
        c_high = scorer_high.score_rally(rally)
        c_low = scorer_low.score_rally(rally)
        assert c_high.score > c_low.score

    def test_reason_long_rally(self):
        scorer = make_scorer()
        rally = make_rally(shot_count=25)
        candidate = scorer.score_rally(rally)
        assert "long_rally" in candidate.reason

    def test_reason_very_long_rally(self):
        scorer = make_scorer()
        rally = make_rally(shot_count=30)
        candidate = scorer.score_rally(rally)
        assert "very_long_rally" in candidate.reason

    def test_reason_fast_exchanges(self):
        scorer = make_scorer()
        rally = make_rally(avg_ball_speed=700.0)
        candidate = scorer.score_rally(rally)
        assert "fast_exchanges" in candidate.reason

    def test_long_rally_bonus_capped_at_one(self):
        scorer = make_scorer()
        rally = make_rally(
            start=0.0,
            end=60.0,
            shot_count=30,
            avg_ball_speed=800.0,
            movement_score=1.0,
        )
        candidate = scorer.score_rally(rally)
        assert candidate.score <= 1.0
