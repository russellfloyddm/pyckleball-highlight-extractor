"""Clip generation: extract highlight clips from the source video using FFmpeg."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from pickleball_highlights.config import HighlightConfig
from pickleball_highlights.highlight_scorer import HighlightCandidate
from pickleball_highlights.utils import get_logger
from pickleball_highlights.video_loader import VideoMetadata

logger = get_logger(__name__)


@dataclass
class GeneratedClip:
    """Metadata about a successfully generated highlight clip."""

    index: int
    source_path: str
    output_path: str
    start_time: float
    end_time: float
    padded_start: float
    padded_end: float
    duration: float
    score: float
    reason: str


class ClipGenerator:
    """Extract individual highlight clips from the source video.

    Clips are padded by ``clip_before`` and ``clip_after`` seconds (from the
    HighlightConfig) and encoded with FFmpeg.

    Args:
        config: Highlight configuration.
        output_dir: Directory where clip files will be written.
    """

    def __init__(self, config: HighlightConfig, output_dir: str) -> None:
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._verify_ffmpeg()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        candidates: List[HighlightCandidate],
        video_path: str,
        video_meta: VideoMetadata,
        threshold: float = 0.7,
    ) -> List[GeneratedClip]:
        """Generate clips for all candidates above the score threshold.

        Overlapping or near-adjacent highlights are merged before extraction.

        Args:
            candidates: Scored highlight candidates.
            video_path: Path to the source video file.
            video_meta: Video metadata (used for duration clamping).
            threshold: Minimum score required to generate a clip.

        Returns:
            List of GeneratedClip objects for successfully extracted clips.
        """
        # Filter and merge
        filtered = [c for c in candidates if c.score >= threshold]
        merged = self._merge_candidates(filtered, self.config.merge_gap)

        logger.info(
            "Generating %d clips (filtered from %d candidates with threshold %.2f)",
            len(merged),
            len(candidates),
            threshold,
        )

        clips: List[GeneratedClip] = []
        for idx, candidate in enumerate(merged, start=1):
            logger.info(
                "Extracting highlight %d/%d (score=%.3f, reason=%s)",
                idx,
                len(merged),
                candidate.score,
                candidate.reason,
            )
            clip = self._extract_clip(
                idx=idx,
                candidate=candidate,
                video_path=video_path,
                video_duration=video_meta.duration,
            )
            if clip is not None:
                clips.append(clip)

        return clips

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_clip(
        self,
        idx: int,
        candidate: HighlightCandidate,
        video_path: str,
        video_duration: float,
    ) -> Optional[GeneratedClip]:
        """Run FFmpeg to extract a single padded clip."""
        pad_start = max(0.0, candidate.start_time - self.config.clip_before)
        pad_end = min(video_duration, candidate.end_time + self.config.clip_after)
        duration = pad_end - pad_start

        filename = f"highlight_{idx:03d}.{self.config.output_format}"
        output_path = os.path.join(self.output_dir, filename)

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{pad_start:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-crf", str(self.config.ffmpeg_crf),
            "-preset", self.config.ffmpeg_preset,
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=600,
            )
            if result.returncode != 0:
                logger.error(
                    "FFmpeg clip extraction failed for clip %d (code %d): %s",
                    idx,
                    result.returncode,
                    result.stderr.decode(errors="replace")[:500],
                )
                return None
        except FileNotFoundError:
            logger.error(
                "FFmpeg not found.  Install FFmpeg and ensure it is on PATH."
            )
            return None
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out extracting clip %d.", idx)
            return None
        except Exception as exc:  # pragma: no cover
            logger.error("Error extracting clip %d: %s", idx, exc)
            return None

        clip = GeneratedClip(
            index=idx,
            source_path=video_path,
            output_path=output_path,
            start_time=candidate.start_time,
            end_time=candidate.end_time,
            padded_start=pad_start,
            padded_end=pad_end,
            duration=duration,
            score=candidate.score,
            reason=candidate.reason,
        )
        logger.info("Clip saved: %s (%.1fs)", filename, duration)
        return clip

    @staticmethod
    def _merge_candidates(
        candidates: List[HighlightCandidate], merge_gap: float
    ) -> List[HighlightCandidate]:
        """Merge candidates that are within *merge_gap* seconds of each other.

        When two candidates are merged, the higher score and combined reason
        are kept.
        """
        if not candidates:
            return []

        # Sort by start time
        sorted_cands = sorted(candidates, key=lambda c: c.start_time)
        merged: List[HighlightCandidate] = [sorted_cands[0]]

        for cand in sorted_cands[1:]:
            last = merged[-1]
            if cand.start_time - last.end_time <= merge_gap:
                # Merge: extend end_time, keep best score, combine reasons
                from pickleball_highlights.rally_detector import Rally

                merged_rally = Rally(
                    start_time=last.rally.start_time,
                    end_time=max(last.rally.end_time, cand.rally.end_time),
                    shot_count=last.rally.shot_count + cand.rally.shot_count,
                    max_ball_speed=max(
                        last.rally.max_ball_speed, cand.rally.max_ball_speed
                    ),
                    avg_ball_speed=(
                        last.rally.avg_ball_speed + cand.rally.avg_ball_speed
                    ) / 2,
                    movement_score=max(
                        last.rally.movement_score, cand.rally.movement_score
                    ),
                )
                from pickleball_highlights.highlight_scorer import HighlightCandidate as HC

                merged[-1] = HC(
                    rally=merged_rally,
                    score=max(last.score, cand.score),
                    reason=f"{last.reason},{cand.reason}",
                )
            else:
                merged.append(cand)

        return merged

    def _verify_ffmpeg(self) -> None:
        """Warn if FFmpeg is not installed."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=10,
            )
        except FileNotFoundError:
            logger.warning(
                "FFmpeg not found on PATH.  Clip generation will fail.  "
                "Install FFmpeg: https://ffmpeg.org/download.html"
            )
        except Exception:  # pragma: no cover
            pass
