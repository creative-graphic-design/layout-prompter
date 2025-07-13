import numpy as np
import pytest
from layout_prompter.utils import (
    compute_alignment,
    compute_overlap,
)
from layout_prompter.models.layout_data import Bbox


def convert_ltwh_to_ltrb(bboxes: np.ndarray) -> np.ndarray:
    """Helper function to convert LTWH format to LTRB format using Bbox class"""
    result = []
    for bbox_data in bboxes:
        bbox = Bbox(
            left=int(bbox_data[0]),
            top=int(bbox_data[1]),
            width=int(bbox_data[2]),
            height=int(bbox_data[3]),
        )
        ltrb = bbox.to_ltrb()
        result.append([ltrb[0], ltrb[1], ltrb[2], ltrb[3]])
    return np.array(result)


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
    assert ali_score == pytest.approx(0.09902102579427789)


def test_compute_overlap(bboxes: np.ndarray, labels: np.ndarray):
    bboxes = convert_ltwh_to_ltrb(bboxes)
    bboxes = bboxes[None, :, :]

    labels = np.array(
        ["logo", "text", "text", "text", "underlay", "underlay", "underlay"]
    )
    labels = labels[None, :]
    padmsk = np.ones_like(labels, dtype=bool)

    ove_score = compute_overlap(bboxes, padmsk)
    assert ove_score == pytest.approx(0.7431144070688704)


def test_compute_alignment_simple():
    """Test alignment computation with simple bboxes"""
    # Create simple aligned bboxes
    simple_bboxes = np.array(
        [
            [0, 0, 10, 10],  # left-aligned at x=0
            [0, 20, 10, 30],  # left-aligned at x=0
            [20, 0, 30, 10],  # left-aligned at x=20
        ]
    )

    bboxes_ltrb = convert_ltwh_to_ltrb(simple_bboxes)
    bboxes_ltrb = bboxes_ltrb[None, :, :]

    labels = np.array(["text", "text", "text"])[None, :]
    padmsk = np.ones_like(labels, dtype=bool)

    ali_score = compute_alignment(bboxes_ltrb, padmsk)
    assert isinstance(ali_score, float)
    assert ali_score >= 0.0


def test_compute_overlap_simple():
    """Test overlap computation with simple bboxes"""
    # Create overlapping bboxes
    overlap_bboxes = np.array(
        [
            [0, 0, 20, 20],  # 20x20 box
            [10, 10, 20, 20],  # 10x10 box overlapping with first
            [30, 30, 10, 10],  # separate 10x10 box
        ]
    )

    bboxes_ltrb = convert_ltwh_to_ltrb(overlap_bboxes)
    bboxes_ltrb = bboxes_ltrb[None, :, :]

    labels = np.array(["text", "text", "text"])[None, :]
    padmsk = np.ones_like(labels, dtype=bool)

    ove_score = compute_overlap(bboxes_ltrb, padmsk)
    assert isinstance(ove_score, float)
    assert ove_score >= 0.0


def test_empty_bboxes():
    """Test metrics with empty bbox arrays"""
    empty_bboxes = np.empty((1, 0, 4))
    # empty_labels = np.empty((1, 0), dtype=str)  # Not used in this test
    empty_mask = np.empty((1, 0), dtype=bool)

    # compute_alignment should raise ValueError due to reduction operation
    with pytest.raises(ValueError, match="reduction operation minimum"):
        compute_alignment(empty_bboxes, empty_mask)

    # compute_overlap handles empty arrays differently and returns a value
    ove_score = compute_overlap(empty_bboxes, empty_mask)
    assert isinstance(ove_score, float)
    # Should be 0.0 or NaN for empty arrays
    assert ove_score == 0.0 or np.isnan(ove_score)


def test_single_bbox():
    """Test metrics with single bbox"""
    single_bbox = np.array([[10, 10, 20, 20]])

    bboxes_ltrb = convert_ltwh_to_ltrb(single_bbox)
    bboxes_ltrb = bboxes_ltrb[None, :, :]

    labels = np.array(["text"])[None, :]
    padmsk = np.ones_like(labels, dtype=bool)

    ali_score = compute_alignment(bboxes_ltrb, padmsk)
    ove_score = compute_overlap(bboxes_ltrb, padmsk)

    assert isinstance(ali_score, float)
    assert isinstance(ove_score, float)
    assert ali_score >= 0.0
    assert ove_score >= 0.0
