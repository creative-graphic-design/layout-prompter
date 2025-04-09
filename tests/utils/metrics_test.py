import numpy as np
import pytest
import torch

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


def torch_compute_overlap(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.3 Overlapping Loss

    bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0)
    bbox = bbox.permute(2, 0, 1)

    l1, t1, r1, b1 = bbox.unsqueeze(-1)
    l2, t2, r2, b2 = bbox.unsqueeze(-2)
    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = torch.maximum(l1, l2)
    r_min = torch.minimum(r1, r2)
    t_max = torch.maximum(t1, t2)
    b_min = torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max), torch.zeros_like(a1[0]))

    diag_mask = torch.eye(a1.size(1), dtype=torch.bool, device=a1.device)
    ai = ai.masked_fill(diag_mask, 0)

    ar = ai / a1
    ar = torch.from_numpy(np.nan_to_num(ar.numpy()))
    score = torch.from_numpy(
        np.nan_to_num((ar.sum(dim=(1, 2)) / mask.float().sum(-1)).numpy())
    )
    return (score).mean().item()


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
