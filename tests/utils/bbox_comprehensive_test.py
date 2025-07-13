import numpy as np
import pytest

from layout_prompter.models.layout_data import Bbox, NormalizedBbox
from layout_prompter.utils.bbox import (
    normalize_bboxes,
)


class TestBboxUtilsComprehensive:
    def test_normalize_bboxes_basic(self):
        bboxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        w, h = 100, 200

        result = normalize_bboxes(bboxes, w, h)

        expected = np.array(
            [[0.1, 0.1, 0.3, 0.2], [0.5, 0.3, 0.7, 0.4]], dtype=np.float32
        )
        assert np.allclose(result, expected)
        assert result.dtype == np.float32

    def test_normalize_bboxes_single_bbox(self):
        bboxes = np.array([[100, 50, 200, 150]])
        w, h = 400, 300

        result = normalize_bboxes(bboxes, w, h)

        expected = np.array([[0.25, 1 / 6, 0.5, 0.5]], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_normalize_bboxes_zero_dimensions(self):
        bboxes = np.array([[0, 0, 0, 0]])
        w, h = 100, 100

        result = normalize_bboxes(bboxes, w, h)

        expected = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_normalize_bboxes_full_dimensions(self):
        bboxes = np.array([[0, 0, 100, 200]])
        w, h = 100, 200

        result = normalize_bboxes(bboxes, w, h)

        expected = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_normalize_bboxes_invalid_shape(self):
        # Test with wrong number of coordinates
        bboxes = np.array([[10, 20, 30]])  # Only 3 coordinates
        w, h = 100, 100

        with pytest.raises(
            AssertionError, match="bboxes should be of shape \\(N, 4\\)"
        ):
            normalize_bboxes(bboxes, w, h)

    def test_normalize_bboxes_empty_array(self):
        bboxes = np.empty((0, 4))
        w, h = 100, 100

        result = normalize_bboxes(bboxes, w, h)

        assert result.shape == (0, 4)
        assert result.dtype == np.float32

    def test_bbox_to_ltrb_conversion(self):
        """Test Bbox class to_ltrb method"""
        bbox = Bbox(left=10, top=20, width=30, height=40)

        result = bbox.to_ltrb()

        expected = (10, 20, 40, 60)  # left, top, right, bottom
        assert result == expected

    def test_bbox_to_ltwh_conversion(self):
        """Test Bbox class to_ltwh method"""
        bbox = Bbox(left=10, top=20, width=30, height=40)

        result = bbox.to_ltwh()

        expected = (10, 20, 30, 40)  # left, top, width, height
        assert result == expected

    def test_normalized_bbox_to_ltrb_conversion(self):
        """Test NormalizedBbox class to_ltrb method"""
        bbox = NormalizedBbox(left=0.1, top=0.2, width=0.3, height=0.4)

        result = bbox.to_ltrb()

        expected = (0.1, 0.2, 0.4, 0.6)  # left, top, right, bottom
        assert result == pytest.approx(expected)

    def test_bbox_properties(self):
        """Test Bbox class right and bottom properties"""
        bbox = Bbox(left=10, top=20, width=30, height=40)

        assert bbox.right == 40  # 10 + 30
        assert bbox.bottom == 60  # 20 + 40

    def test_normalized_bbox_properties(self):
        """Test NormalizedBbox class right and bottom properties"""
        bbox = NormalizedBbox(left=0.1, top=0.2, width=0.3, height=0.4)

        assert bbox.right == pytest.approx(0.4)  # 0.1 + 0.3
        assert bbox.bottom == pytest.approx(0.6)  # 0.2 + 0.4

    def test_bbox_zero_dimensions(self):
        """Test bbox with zero dimensions"""
        bbox = Bbox(left=0, top=0, width=0, height=0)

        assert bbox.to_ltrb() == (0, 0, 0, 0)
        assert bbox.to_ltwh() == (0, 0, 0, 0)

    def test_bbox_negative_coordinates_not_allowed(self):
        """Test that negative coordinates are not allowed for Bbox"""
        with pytest.raises(Exception):  # Pydantic validation error
            Bbox(left=-10, top=-20, width=30, height=40)

    def test_normalized_bbox_out_of_range_not_allowed(self):
        """Test that values outside [0,1] are not allowed for NormalizedBbox"""
        with pytest.raises(Exception):  # Pydantic validation error
            NormalizedBbox(left=1.5, top=0.2, width=0.3, height=0.4)

    def test_normalize_bboxes_edge_case_large_numbers(self):
        bboxes = np.array([[1000, 2000, 3000, 4000]])
        w, h = 10000, 20000

        result = normalize_bboxes(bboxes, w, h)

        expected = np.array([[0.1, 0.1, 0.3, 0.2]], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_normalize_bboxes_integer_input(self):
        # Test that integer input gets converted to float32
        bboxes = np.array([[10, 20, 30, 40]], dtype=np.int32)
        w, h = 100, 200

        result = normalize_bboxes(bboxes, w, h)

        assert result.dtype == np.float32
        expected = np.array([[0.1, 0.1, 0.3, 0.2]], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_bbox_discretization(self):
        """Test NormalizedBbox discretization to Bbox"""
        from layout_prompter.models.layout_data import CanvasSize

        normalized_bbox = NormalizedBbox(left=0.1, top=0.2, width=0.3, height=0.4)
        canvas_size = CanvasSize(width=100, height=150)

        discrete_bbox = normalized_bbox.discretize(canvas_size)

        assert isinstance(discrete_bbox, Bbox)
        assert discrete_bbox.left == 10  # 0.1 * 100
        assert discrete_bbox.top == 30  # 0.2 * 150
        assert discrete_bbox.width == 30  # 0.3 * 100
        assert discrete_bbox.height == 60  # 0.4 * 150
