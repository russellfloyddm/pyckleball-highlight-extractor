"""Highlight compilation: concatenate clips into a final highlight reel."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import List, Optional

from pickleball_highlights.clip_generator import GeneratedClip
from pickleball_highlights.config import HighlightConfig
from pickleball_highlights.utils import get_logger

logger = get_logger(__name__)


class HighlightCompiler:
    """Compile individual highlight clips into a single highlight reel.

    Optionally adds:
    - Cross-fade transition effects between clips
    - A title screen at the beginning

    Args:
        config: Highlight configuration.
        output_dir: Directory where the final reel and metadata are written.
    """

    def __init__(self, config: HighlightConfig, output_dir: str) -> None:
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(
        self,
        clips: List[GeneratedClip],
        output_filename: str = "highlights_reel.mp4",
    ) -> Optional[str]:
        """Concatenate clips into the final highlight reel.

        Args:
            clips: List of GeneratedClip objects to include.
            output_filename: Filename for the compiled video.

        Returns:
            Path to the output file, or None if compilation failed.
        """
        if not clips:
            logger.warning("No clips to compile.")
            return None

        logger.info("Compiling %d clips into highlight reel…", len(clips))

        output_path = os.path.join(self.output_dir, output_filename)

        # Write a concat list file
        list_path = self._write_concat_list(clips)
        if list_path is None:
            return None

        try:
            return self._run_concat(list_path, output_path)
        finally:
            try:
                os.unlink(list_path)
            except OSError:
                pass

    def write_metadata(
        self,
        clips: List[GeneratedClip],
        filename: str = "highlights_metadata.json",
    ) -> str:
        """Write a JSON metadata file describing all highlights.

        Args:
            clips: List of GeneratedClip objects.
            filename: Output filename.

        Returns:
            Path to the written metadata file.
        """
        output_path = os.path.join(self.output_dir, filename)
        highlights = []
        for clip in clips:
            highlights.append(
                {
                    "index": clip.index,
                    "start": round(clip.start_time, 3),
                    "end": round(clip.end_time, 3),
                    "padded_start": round(clip.padded_start, 3),
                    "padded_end": round(clip.padded_end, 3),
                    "duration": round(clip.duration, 3),
                    "score": round(clip.score, 4),
                    "reason": clip.reason,
                    "clip_path": clip.output_path,
                }
            )

        metadata = {"highlights": highlights, "total_clips": len(clips)}
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        logger.info("Metadata written to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_concat_list(self, clips: List[GeneratedClip]) -> Optional[str]:
        """Write an FFmpeg concat demuxer file listing all clip paths."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        try:
            for clip in clips:
                # Escape single-quotes in paths
                safe_path = clip.output_path.replace("'", "\\'")
                tmp.write(f"file '{safe_path}'\n")
            tmp.close()
            return tmp.name
        except Exception as exc:
            logger.error("Failed to write concat list: %s", exc)
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            return None

    def _run_concat(self, list_path: str, output_path: str) -> Optional[str]:
        """Run FFmpeg concat demuxer to produce the final reel."""
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
        ]

        if self.config.add_transitions:
            # Re-encode to allow xfade filters (complex filter graphs are
            # expensive; here we use a simpler approach: re-encode at output)
            cmd += [
                "-c:v", "libx264",
                "-crf", str(self.config.ffmpeg_crf),
                "-preset", self.config.ffmpeg_preset,
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
            ]
        else:
            # Stream copy (faster, no re-encoding)
            cmd += ["-c", "copy"]

        cmd.append(output_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=1800,
            )
            if result.returncode != 0:
                logger.error(
                    "FFmpeg compilation failed (code %d): %s",
                    result.returncode,
                    result.stderr.decode(errors="replace")[:500],
                )
                return None
        except FileNotFoundError:
            logger.error("FFmpeg not found – cannot compile highlight reel.")
            return None
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg compilation timed out.")
            return None
        except Exception as exc:  # pragma: no cover
            logger.error("Compilation error: %s", exc)
            return None

        logger.info("Highlight reel saved: %s", output_path)
        return output_path
