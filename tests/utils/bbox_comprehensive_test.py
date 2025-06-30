import numpy as np
import pytest
from layout_prompter.utils.bbox import (
    convert_ltwh_to_ltrb,
    decapsulate,
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

    def test_decapsulate_2d_array(self):
        bboxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])

        result = decapsulate(bboxes)

        # Should transpose the array
        expected = bboxes.T
        assert np.array_equal(result, expected)
        assert result.shape == (4, 2)

    def test_decapsulate_3d_array(self):
        # Create a 3D array (batch_size, num_boxes, 4)
        bboxes = np.array(
            [
                [[10, 20, 30, 40], [50, 60, 70, 80]],
                [[90, 100, 110, 120], [130, 140, 150, 160]],
            ]
        )

        result = decapsulate(bboxes)

        # Should transpose with axis permutation (2, 0, 1)
        expected = np.transpose(bboxes, (2, 0, 1))
        assert np.array_equal(result, expected)
        assert result.shape == (4, 2, 2)

    def test_decapsulate_single_batch_3d(self):
        # Test with single batch dimension
        bboxes = np.array([[[10, 20, 30, 40]]])  # shape: (1, 1, 4)

        result = decapsulate(bboxes)

        expected = np.transpose(bboxes, (2, 0, 1))
        assert np.array_equal(result, expected)
        assert result.shape == (4, 1, 1)

    def test_convert_ltwh_to_ltrb_1d_array(self):
        bbox = np.array([10, 20, 30, 40])  # left, top, width, height

        result = convert_ltwh_to_ltrb(bbox)

        expected = np.array([10, 20, 40, 60])  # left, top, right, bottom
        assert np.array_equal(result, expected)

    def test_convert_ltwh_to_ltrb_2d_array(self):
        bboxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])

        result = convert_ltwh_to_ltrb(bboxes)

        expected = np.array([[10, 20, 40, 60], [50, 60, 120, 140]])
        assert np.array_equal(result, expected)
        assert result.shape == (2, 4)

    def test_convert_ltwh_to_ltrb_3d_array(self):
        # Test with batch dimension
        bboxes = np.array(
            [
                [[10, 20, 30, 40], [50, 60, 70, 80]],
                [[90, 100, 110, 120], [130, 140, 150, 160]],
            ]
        )

        result = convert_ltwh_to_ltrb(bboxes)

        # The function preserves the batch structure, so shape should be (2, 2, 4)
        expected_shape = (2, 2, 4)
        assert result.shape == expected_shape

        # Check conversion: left, top, width, height -> left, top, right, bottom
        # First batch, first bbox: [10, 20, 30, 40] -> [10, 20, 40, 60]
        assert np.array_equal(result[0, 0], [10, 20, 40, 60])
        # First batch, second bbox: [50, 60, 70, 80] -> [50, 60, 120, 140]
        assert np.array_equal(result[0, 1], [50, 60, 120, 140])
        # Second batch, first bbox: [90, 100, 110, 120] -> [90, 100, 200, 220]
        assert np.array_equal(result[1, 0], [90, 100, 200, 220])

    def test_convert_ltwh_to_ltrb_zero_dimensions(self):
        bbox = np.array([0, 0, 0, 0])

        result = convert_ltwh_to_ltrb(bbox)

        expected = np.array([0, 0, 0, 0])
        assert np.array_equal(result, expected)

    def test_convert_ltwh_to_ltrb_negative_coordinates(self):
        bbox = np.array([-10, -20, 30, 40])

        result = convert_ltwh_to_ltrb(bbox)

        expected = np.array([-10, -20, 20, 20])  # -10+30=20, -20+40=20
        assert np.array_equal(result, expected)

    def test_convert_ltwh_to_ltrb_float_values(self):
        bbox = np.array([10.5, 20.3, 30.7, 40.1])

        result = convert_ltwh_to_ltrb(bbox)

        expected = np.array([10.5, 20.3, 41.2, 60.4])
        assert np.allclose(result, expected)

    def test_convert_ltwh_to_ltrb_empty_array(self):
        bboxes = np.empty((0, 4))

        result = convert_ltwh_to_ltrb(bboxes)

        # For empty 2D array, the function preserves the empty structure
        expected_shape = (0, 4)  # Same shape but converted coordinates
        assert result.shape == expected_shape

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
