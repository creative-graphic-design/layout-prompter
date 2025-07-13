import pytest
from layout_prompter.models.layout_data import Bbox, NormalizedBbox


def test_bbox_to_ltrb():
    """Test Bbox class to_ltrb conversion"""
    bbox = Bbox(left=10, top=20, width=30, height=40)
    ltrb = bbox.to_ltrb()
    assert ltrb == (10, 20, 40, 60)  # left, top, right, bottom


def test_bbox_to_ltwh():
    """Test Bbox class to_ltwh conversion"""
    bbox = Bbox(left=10, top=20, width=30, height=40)
    ltwh = bbox.to_ltwh()
    assert ltwh == (10, 20, 30, 40)  # left, top, width, height


def test_normalized_bbox_to_ltrb():
    """Test NormalizedBbox class to_ltrb conversion"""
    bbox = NormalizedBbox(left=0.1, top=0.2, width=0.3, height=0.4)
    ltrb = bbox.to_ltrb()
    expected = (0.1, 0.2, 0.4, 0.6)  # left, top, right, bottom
    assert ltrb == pytest.approx(expected)


def test_normalized_bbox_to_ltwh():
    """Test NormalizedBbox class to_ltwh conversion"""
    bbox = NormalizedBbox(left=0.1, top=0.2, width=0.3, height=0.4)
    ltwh = bbox.to_ltwh()
    expected = (0.1, 0.2, 0.3, 0.4)  # left, top, width, height
    assert ltwh == pytest.approx(expected)


def test_bbox_properties():
    """Test Bbox right and bottom properties"""
    bbox = Bbox(left=10, top=20, width=30, height=40)
    assert bbox.right == 40  # 10 + 30
    assert bbox.bottom == 60  # 20 + 40


def test_normalized_bbox_properties():
    """Test NormalizedBbox right and bottom properties"""
    bbox = NormalizedBbox(left=0.1, top=0.2, width=0.3, height=0.4)
    assert bbox.right == pytest.approx(0.4)  # 0.1 + 0.3
    assert bbox.bottom == pytest.approx(0.6)  # 0.2 + 0.4


def test_bbox_validation():
    """Test Bbox validation for non-negative values"""
    # Valid bbox
    bbox = Bbox(left=0, top=0, width=100, height=100)
    assert bbox.left == 0
    assert bbox.top == 0
    assert bbox.width == 100
    assert bbox.height == 100

    # Invalid bbox with negative values should raise validation error
    with pytest.raises(Exception):  # Pydantic validation error
        Bbox(left=-10, top=0, width=100, height=100)


def test_normalized_bbox_validation():
    """Test NormalizedBbox validation for [0,1] range"""
    # Valid normalized bbox
    bbox = NormalizedBbox(left=0.0, top=0.0, width=1.0, height=1.0)
    assert bbox.left == 0.0
    assert bbox.top == 0.0
    assert bbox.width == 1.0
    assert bbox.height == 1.0

    # Invalid normalized bbox with out-of-range values should raise validation error
    with pytest.raises(Exception):  # Pydantic validation error
        NormalizedBbox(left=1.5, top=0.0, width=0.5, height=0.5)
