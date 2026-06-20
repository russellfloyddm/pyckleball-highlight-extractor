"""Video loading and metadata extraction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from pickleball_highlights.utils import get_logger

logger = get_logger(__name__)


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""

    path: str
    fps: float
    width: int
    height: int
    total_frames: int
    duration: float  # seconds
    codec: str


class VideoLoader:
    """Load and iterate video frames using OpenCV.

    Args:
        path: Path to the video file.
        start_time: Optional start time in seconds (skip earlier frames).
        end_time: Optional end time in seconds (stop after this time).
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

    def __init__(
        self,
        path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{ext}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        self.path = path
        self.start_time = start_time
        self.end_time = end_time
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> VideoMetadata:
        """Lazy-load and cache video metadata."""
        if self._metadata is None:
            self._metadata = self._read_metadata()
        return self._metadata

    def frames(
        self, batch_size: int = 1
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield (frame_index, timestamp_seconds, frame_bgr) tuples.

        Args:
            batch_size: Unused—provided for API consistency. Each call yields
                        one frame at a time.

        Yields:
            Tuple of (frame_index, timestamp_in_seconds, BGR_frame).
        """
        meta = self.metadata
        start_frame = (
            int(self.start_time * meta.fps) if self.start_time is not None else 0
        )
        end_frame = (
            int(self.end_time * meta.fps)
            if self.end_time is not None
            else meta.total_frames
        )

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        try:
            frame_idx = start_frame
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp = frame_idx / meta.fps
                yield frame_idx, timestamp, frame
                frame_idx += 1
        finally:
            cap.release()

    def read_frame_at(self, timestamp: float) -> Optional[np.ndarray]:
        """Read a single frame at the given timestamp (seconds).

        Args:
            timestamp: Time in seconds.

        Returns:
            BGR frame or None if the timestamp is out of range.
        """
        meta = self.metadata
        frame_idx = int(timestamp * meta.fps)
        if frame_idx >= meta.total_frames:
            return None

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            return frame if ret else None
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_metadata(self) -> VideoMetadata:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0.0
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
        finally:
            cap.release()

        meta = VideoMetadata(
            path=self.path,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            duration=duration,
            codec=codec.strip(),
        )
        logger.info(
            "Loaded video: %s (%.1f fps, %dx%d, %.1fs, %d frames)",
            os.path.basename(self.path),
            meta.fps,
            meta.width,
            meta.height,
            meta.duration,
            meta.total_frames,
        )
        return meta

    def __repr__(self) -> str:  # pragma: no cover
        return f"VideoLoader(path={self.path!r})"
