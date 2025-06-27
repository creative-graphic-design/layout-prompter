import numpy as np
from layout_prompter.utils import convert_ltwh_to_ltrb


def test_convert_ltwh_to_ltrb():
    ltwh_bbox = np.array([10, 20, 30, 40])
    ltrb_bbox = convert_ltwh_to_ltrb(ltwh_bbox)
    assert np.array_equal(ltrb_bbox, np.array([10, 20, 40, 60]))

    ltwh_bbox = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
    ltrb_bbox = convert_ltwh_to_ltrb(ltwh_bbox)
    assert np.array_equal(ltrb_bbox, np.array([[10, 20, 40, 60], [50, 60, 120, 140]]))
