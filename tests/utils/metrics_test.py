import numpy as np
import pytest
from layout_prompter.utils import (
    compute_alignment,
    compute_overlap,
    convert_ltwh_to_ltrb,
)


@pytest.fixture
def bboxes() -> np.ndarray:
    return np.array(
        [
            [10, 8, 81, 13],
            [5, 118, 90, 16],
            [8, 134, 85, 12],
            [5, 29, 24, 5],
            [30, 117, 55, 20],
            [2, 133, 128, 15],
            [17, 6, 68, 19],
        ]
    )


@pytest.fixture
def labels() -> np.ndarray:
    return np.array(
        [
            "logo",
            "text",
            "text",
            "text",
            "underlay",
            "underlay",
            "underlay",
        ]
    )


def test_compute_alignment(bboxes: np.ndarray, labels: np.ndarray):
    bboxes = convert_ltwh_to_ltrb(bboxes)
    bboxes = bboxes[None, :, :]

    labels = np.array(
        ["logo", "text", "text", "text", "underlay", "underlay", "underlay"]
    )
    labels = labels[None, :]
    padmsk = np.ones_like(labels, dtype=bool)

    ali_score = compute_alignment(bboxes, padmsk)
    assert ali_score == 0.09902102579427789


def test_compute_overlap(bboxes: np.ndarray, labels: np.ndarray):
    bboxes = convert_ltwh_to_ltrb(bboxes)
    bboxes = bboxes[None, :, :]

    labels = np.array(
        ["logo", "text", "text", "text", "underlay", "underlay", "underlay"]
    )
    labels = labels[None, :]
    padmsk = np.ones_like(labels, dtype=bool)

    ove_score = compute_overlap(bboxes, padmsk)
    assert ove_score == 0.7431144070688704
