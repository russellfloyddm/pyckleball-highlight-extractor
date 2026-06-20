"""Models package - placeholder for custom trained model weights and loaders."""

from __future__ import annotations

import os
from typing import Optional


def get_model_path(filename: str, models_dir: Optional[str] = None) -> str:
    """Resolve the absolute path of a model file.

    The function first checks *models_dir* (or the directory containing this
    file when *models_dir* is not supplied), then falls back to searching in the
    current working directory.

    Args:
        filename: Model filename, e.g. ``"yolov8n.pt"``.
        models_dir: Directory to search first.  Defaults to the ``models/``
                    directory inside the package.

    Returns:
        Absolute path to the model file if found, otherwise returns *filename*
        unchanged (ultralytics will download it automatically).
    """
    if models_dir is None:
        models_dir = os.path.dirname(__file__)

    candidate = os.path.join(models_dir, filename)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)

    # Fallback: let the calling library (e.g. ultralytics) handle resolution.
    return filename
