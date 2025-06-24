import numpy as np


def normalize_bboxes(bboxes: np.ndarray, w: int, h: int) -> np.ndarray:
    """Normalize bounding boxes to [0, 1] range."""
    assert bboxes.shape[1] == 4, "bboxes should be of shape (N, 4)"

    bboxes = bboxes.astype(np.float32)
    bboxes[:, 0::2] /= w
    bboxes[:, 1::2] /= h
    return bboxes


def decapsulate(bboxes: np.ndarray):
    if len(bboxes.shape) == 2:
        return bboxes.T
    else:
        return np.transpose(bboxes, (2, 0, 1))


def convert_ltwh_to_ltrb(bbox: np.ndarray):
    if len(bbox.shape) == 1:
        left, top, width, height = bbox
        right = left + width
        bottom = top + height
        return np.array((left, top, right, bottom))

    left, top, width, height = decapsulate(bbox)
    right = left + width
    bottom = top + height
    return np.stack([left, top, right, bottom], axis=-1)
