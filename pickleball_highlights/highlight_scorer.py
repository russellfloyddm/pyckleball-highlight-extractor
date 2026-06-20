"""Highlight scoring: compute weighted scores for each detected rally."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pickleball_highlights.audio_analyzer import AudioAnalyzer
from pickleball_highlights.config import ScoringWeights
from pickleball_highlights.rally_detector import Rally
from pickleball_highlights.utils import get_logger

logger = get_logger(__name__)


@dataclass
class HighlightCandidate:
    """A rally candidate with its computed highlight score."""

    rally: Rally
    score: float
    reason: str

    @property
    def start_time(self) -> float:
        return self.rally.start_time

    @property
    def end_time(self) -> float:
        return self.rally.end_time

    @property
    def duration(self) -> float:
        return self.rally.duration


class HighlightScorer:
    """Compute highlight scores for rally candidates.

    The score is a weighted combination of:
    - Rally duration
    - Shot count
    - Ball speed
    - Player movement
    - Audio excitement

    Args:
        weights: Configurable scoring weights.
        audio_analyzer: AudioAnalyzer instance for retrieving per-timestamp
                        audio scores.
        max_ball_speed_reference: Ball speed (px/s) considered "maximum" for
                                  normalisation purposes.
    """

    def __init__(
        self,
        weights: ScoringWeights,
        audio_analyzer: AudioAnalyzer,
        max_ball_speed_reference: float = 800.0,
    ) -> None:
        self.weights = weights
        self.audio = audio_analyzer
        self.max_ball_speed_reference = max_ball_speed_reference

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_rally(self, rally: Rally) -> HighlightCandidate:
        """Compute a highlight score for a single rally.

        Args:
            rally: The rally to score.

        Returns:
            HighlightCandidate with score and human-readable reason.
        """
        # 1. Rally duration score [0, 1]
        duration_score = rally.duration_score  # 20 s → 1.0

        # 2. Shot count score [0, 1]
        shot_score = rally.shot_count_score  # 20 shots → 1.0

        # 3. Ball speed score [0, 1]
        speed_score = min(rally.avg_ball_speed / self.max_ball_speed_reference, 1.0)

        # 4. Player movement score [0, 1] (already normalised in PlayerTracker)
        movement_score = min(rally.movement_score, 1.0)

        # 5. Audio excitement score (average over rally duration)
        audio_score = self._avg_audio_score(rally.start_time, rally.end_time)

        weighted_score = (
            self.weights.rally_duration * duration_score
            + self.weights.shot_count * shot_score
            + self.weights.ball_speed * speed_score
            + self.weights.movement * movement_score
            + self.weights.audio * audio_score
        )

        # Apply long-rally bonus (additive, capped at 1.0)
        bonus = rally.long_rally_bonus
        total_score = min(weighted_score + bonus, 1.0)

        reason = self._build_reason(rally, duration_score, shot_score, speed_score)

        logger.debug(
            "Rally %.2fs-%.2fs: duration=%.2f, shots=%.2f, speed=%.2f, "
            "movement=%.2f, audio=%.2f, bonus=%.2f → score=%.3f",
            rally.start_time,
            rally.end_time,
            duration_score,
            shot_score,
            speed_score,
            movement_score,
            audio_score,
            bonus,
            total_score,
        )

        return HighlightCandidate(rally=rally, score=total_score, reason=reason)

    def score_rallies(self, rallies: List[Rally]) -> List[HighlightCandidate]:
        """Score a list of rallies.

        Args:
            rallies: List of Rally objects.

        Returns:
            List of HighlightCandidate objects sorted by score descending.
        """
        candidates = [self.score_rally(r) for r in rallies]
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _avg_audio_score(self, start: float, end: float) -> float:
        """Return average audio excitement score over a time window."""
        frames = self.audio.get_frames()
        if not frames:
            return 0.0

        scores = [
            f.excitement_score
            for f in frames
            if f.start_time >= start and f.end_time <= end
        ]
        if not scores:
            # Fallback: find nearest frame
            return self.audio.get_score_at((start + end) / 2)
        import numpy as np
        return float(np.mean(scores))

    @staticmethod
    def _build_reason(
        rally: Rally,
        duration_score: float,
        shot_score: float,
        speed_score: float,
    ) -> str:
        reasons = []
        if rally.shot_count >= 30:
            reasons.append("very_long_rally")
        elif rally.shot_count >= 20:
            reasons.append("long_rally")
        elif rally.shot_count >= 10:
            reasons.append("medium_rally")
        if speed_score > 0.7:
            reasons.append("fast_exchanges")
        if rally.duration > 15:
            reasons.append("extended_point")
        if not reasons:
            reasons.append("highlight_moment")
        return ",".join(reasons)
