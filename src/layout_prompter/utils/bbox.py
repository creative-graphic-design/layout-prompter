from __future__ import annotations

import numpy as np


def normalize_bboxes(bboxes: np.ndarray, w: int, h: int) -> np.ndarray:
    """Normalize bounding boxes to [0, 1] range."""
    assert bboxes.shape[1] == 4, "bboxes should be of shape (N, 4)"

    bboxes = bboxes.astype(np.float32)
    bboxes[:, 0::2] /= w
    bboxes[:, 1::2] /= h
    return bboxes
