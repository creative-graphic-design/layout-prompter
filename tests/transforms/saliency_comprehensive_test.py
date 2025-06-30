import numpy as np
import pytest
from layout_prompter.transforms import SaliencyMapToBboxes
from PIL import Image


class TestSaliencyMapToBboxesComprehensive:
    @pytest.fixture
    def transformer_default(self) -> SaliencyMapToBboxes:
        return SaliencyMapToBboxes()

    @pytest.fixture
    def transformer_custom(self) -> SaliencyMapToBboxes:
        return SaliencyMapToBboxes(
            threshold=150, min_side=50, min_area=4000, is_filter_small_bboxes=True
        )

    @pytest.fixture
    def transformer_no_filter(self) -> SaliencyMapToBboxes:
        return SaliencyMapToBboxes(is_filter_small_bboxes=False)

    def test_is_small_bbox_width_height_filter(
        self, transformer_default: SaliencyMapToBboxes
    ):
        # Test small width and height
        small_bbox = [10, 20, 50, 60]  # Both width and height <= min_side (80)
        assert transformer_default.is_small_bbox(small_bbox) is True

        # Test large bbox
        large_bbox = [10, 20, 100, 100]  # Both width and height > min_side
        assert transformer_default.is_small_bbox(large_bbox) is False

    def test_is_small_bbox_area_filter(self, transformer_default: SaliencyMapToBboxes):
        # Test small area
        small_area_bbox = [10, 20, 100, 50]  # area = 5000 < min_area (6000)
        assert transformer_default.is_small_bbox(small_area_bbox) is True

        # Test large area
        large_area_bbox = [10, 20, 100, 80]  # area = 8000 > min_area (6000)
        assert transformer_default.is_small_bbox(large_area_bbox) is False

    def test_is_small_bbox_invalid_input(
        self, transformer_default: SaliencyMapToBboxes
    ):
        # Test invalid bbox format
        with pytest.raises(
            AssertionError, match="bbox must be a list or tuple of 4 integers"
        ):
            transformer_default.is_small_bbox([10, 20, 30])  # Only 3 values

        with pytest.raises(
            AssertionError, match="bbox must be a list or tuple of 4 integers"
        ):
            transformer_default.is_small_bbox([10, 20, 30, 40, 50])  # 5 values

    def test_get_filtered_bboxes_with_filtering(
        self, transformer_default: SaliencyMapToBboxes
    ):
        # Create mock contours that cv2.boundingRect can handle
        # Large contour - will create bbox [100, 100, 100, 100] (x, y, w, h)
        large_contour = np.array(
            [[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.int32
        )
        # Small contour - will create bbox [10, 10, 40, 40] which should be filtered out
        small_contour = np.array(
            [[[10, 10]], [[50, 10]], [[50, 50]], [[10, 50]]], dtype=np.int32
        )

        contours = (large_contour, small_contour)

        result = transformer_default.get_filtered_bboxes(contours)

        # Should only include the large bbox after filtering
        assert len(result) == 1
        # The bbox should be [x, y, width, height] format
        assert result[0][0] == 100  # x
        assert result[0][1] == 100  # y
        # cv2.boundingRect might give width/height as 101 due to inclusive bounds
        assert result[0][2] in [100, 101]  # width
        assert result[0][3] in [100, 101]  # height

    def test_get_filtered_bboxes_no_filtering(
        self, transformer_no_filter: SaliencyMapToBboxes
    ):
        # Create mock contours
        large_contour = np.array(
            [[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]]
        )
        small_contour = np.array([[[10, 10]], [[50, 10]], [[50, 50]], [[10, 50]]])

        contours = (large_contour, small_contour)

        result = transformer_no_filter.get_filtered_bboxes(contours)

        # Should include both bboxes when filtering is disabled
        assert len(result) == 2

    def test_get_filtered_bboxes_sorting(
        self, transformer_no_filter: SaliencyMapToBboxes
    ):
        # Create contours with different y positions to test sorting
        bottom_contour = np.array([[[10, 200]], [[60, 200]], [[60, 250]], [[10, 250]]])
        top_contour = np.array([[[10, 100]], [[60, 100]], [[60, 150]], [[10, 150]]])

        contours = (bottom_contour, top_contour)

        result = transformer_no_filter.get_filtered_bboxes(contours)

        # Should be sorted by y coordinate first
        assert len(result) == 2
        assert result[0][1] < result[1][1]  # First bbox should have smaller y

    def test_invoke_with_valid_saliency_map(
        self, transformer_default: SaliencyMapToBboxes
    ):
        # Create a simple grayscale saliency map
        img_array = np.zeros((200, 200), dtype=np.uint8)
        img_array[50:150, 50:150] = 255  # White square that exceeds threshold
        saliency_map = Image.fromarray(img_array, mode="L")

        result = transformer_default.invoke(saliency_map)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 4  # Should have 4 coordinates per bbox

    def test_invoke_with_low_threshold_map(
        self, transformer_default: SaliencyMapToBboxes
    ):
        # Create a saliency map with values below threshold
        img_array = np.full(
            (200, 200), 50, dtype=np.uint8
        )  # All values below default threshold (100)
        saliency_map = Image.fromarray(img_array, mode="L")

        result = transformer_default.invoke(saliency_map)

        # Should return None when no areas exceed threshold
        assert result is None

    def test_invoke_with_rgb_image_error(
        self, transformer_default: SaliencyMapToBboxes
    ):
        # Create RGB image (should fail assertion)
        img_array = np.zeros((200, 200, 3), dtype=np.uint8)
        rgb_image = Image.fromarray(img_array, mode="RGB")

        with pytest.raises(
            AssertionError, match="saliency map must be grayscale image"
        ):
            transformer_default.invoke(rgb_image)

    def test_custom_threshold_settings(self, transformer_custom: SaliencyMapToBboxes):
        # Test that custom settings are applied
        assert transformer_custom.threshold == 150
        assert transformer_custom.min_side == 50
        assert transformer_custom.min_area == 4000

        # Test small bbox detection with custom settings
        bbox_between_defaults = [10, 20, 60, 60]  # area = 3600, sides = 60
        assert transformer_custom.is_small_bbox(bbox_between_defaults) is True

    def test_invoke_empty_contours(self, transformer_default: SaliencyMapToBboxes):
        # Create a completely black saliency map
        img_array = np.zeros((200, 200), dtype=np.uint8)
        saliency_map = Image.fromarray(img_array, mode="L")

        result = transformer_default.invoke(saliency_map)

        # Should return None when no contours are found
        assert result is None

    def test_invoke_multiple_regions(self, transformer_no_filter: SaliencyMapToBboxes):
        # Create saliency map with multiple bright regions
        img_array = np.zeros((200, 200), dtype=np.uint8)
        img_array[20:80, 20:80] = 200  # First region
        img_array[120:180, 120:180] = 200  # Second region
        saliency_map = Image.fromarray(img_array, mode="L")

        result = transformer_no_filter.invoke(saliency_map)

        assert result is not None
        assert len(result) >= 2  # Should detect multiple regions

        # Verify sorting by y-coordinate then x-coordinate
        if len(result) > 1:
            for i in range(len(result) - 1):
                assert result[i][1] < result[i + 1][1] or (
                    result[i][1] == result[i + 1][1]
                    and result[i][0] <= result[i + 1][0]
                )
