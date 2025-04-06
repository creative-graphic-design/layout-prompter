import numpy as np


def normalize_bboxes(bboxes, w: int, h: int) -> np.ndarray:
    """Normalize bounding boxes to [0, 1] range."""
    assert bboxes.shape[1] == 4, "bboxes should be of shape (N, 4)"

    bboxes = bboxes.astype(np.float32)
    bboxes[:, 0::2] /= w
    bboxes[:, 1::2] /= h
    return bboxes


def decapsulate(bboxes: np.ndarray):
    if len(bboxes.shape) == 2:
        x1, y1, x2, y2 = bboxes.T
    else:
        # FIXME: Change torch impl. to numpy impl.
        # x1, y1, x2, y2 = bboxes.permute(2, 0, 1)
        raise NotImplementedError

    return x1, y1, x2, y2
