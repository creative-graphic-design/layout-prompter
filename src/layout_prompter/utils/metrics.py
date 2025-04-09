import numpy as np


def compute_alignment(bbox: np.ndarray, mask: np.ndarray) -> float:
    # Attribute-conditioned Layout GAN
    # 3.6.4 Alignment Loss

    bbox = bbox.transpose(2, 0, 1)
    xl, yt, xr, yb = bbox
    xc = (xr + xl) / 2
    yc = (yt + yb) / 2
    X = np.stack([xl, xc, xr, yt, yc, yb], axis=1)

    X = X[:, :, :, None] - X[:, :, None, :]
    idx = np.arange(X.shape[2])
    X[:, :, idx, idx] = 1.0
    X = np.abs(X).transpose(0, 2, 1, 3)
    X[~mask] = 1.0
    X = X.min(-1).min(-1)
    X[X == 1.0] = 0.0

    X = -np.log(1 - X)
    score = np.nan_to_num((X.sum(-1) / mask.astype(float).sum(-1)))

    return score.mean().item()


def compute_overlap(bbox: np.ndarray, mask: np.ndarray) -> float:
    # Attribute-conditioned Layout GAN
    # 3.6.3 Overlapping Loss

    bbox[bbox == ~mask[:, :, None]] = 0
    bbox = bbox.transpose(2, 0, 1)

    l1, t1, r1, b1 = bbox[:, :, :, None]
    l2, t2, r2, b2 = bbox[:, :, None, :]
    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = np.maximum(l1, l2)
    r_min = np.minimum(r1, r2)
    t_max = np.maximum(t1, t2)
    b_min = np.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = np.where(cond, (r_min - l_max) * (b_min - t_max), np.zeros_like(a1[0]))

    diag_mask = np.eye(a1.shape[1], dtype=bool)
    ai = ai * ~diag_mask

    ar = ai / a1
    ar = np.nan_to_num(ar)
    score = np.nan_to_num((ar.sum(axis=(1, 2)) / mask.astype(float).sum(-1)))
    return score.mean().item()
