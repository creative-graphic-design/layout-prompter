import numpy as np
import pytest
from layout_prompter.models import LayoutData, ProcessedLayoutData
from layout_prompter.transforms.lexicographic_sort import LexicographicSort


class TestLexicographicSort:
    @pytest.fixture
    def sorter(self) -> LexicographicSort:
        return LexicographicSort()

    @pytest.fixture
    def sample_layout_data(self) -> LayoutData:
        # Create bboxes with different positions for testing lexicographic sort
        # Format: [left, top, right, bottom]
        # bbox1: [0.1, 0.5, 0.3, 0.7] - top=0.5, left=0.1
        # bbox2: [0.2, 0.2, 0.4, 0.4] - top=0.2, left=0.2 (should be first)
        # bbox3: [0.1, 0.5, 0.3, 0.7] - top=0.5, left=0.1 (same as bbox1)
        return LayoutData(
            idx=0,
            bboxes=np.array(
                [[0.1, 0.5, 0.3, 0.7], [0.2, 0.2, 0.4, 0.4], [0.0, 0.5, 0.2, 0.7]]
            ),
            labels=np.array(["text1", "text2", "text3"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=np.array([[0.0, 0.0, 1.0, 1.0]]),
        )

    @pytest.fixture
    def sample_processed_data(self) -> ProcessedLayoutData:
        return ProcessedLayoutData(
            idx=1,
            bboxes=np.array([[0.3, 0.8, 0.5, 1.0], [0.1, 0.8, 0.3, 1.0]]),
            labels=np.array(["button1", "button2"]),
            gold_bboxes=np.array([[0.3, 0.8, 0.5, 1.0], [0.1, 0.8, 0.3, 1.0]]),
            orig_bboxes=np.array([[0.3, 0.8, 0.5, 1.0], [0.1, 0.8, 0.3, 1.0]]),
            orig_labels=np.array(["button1", "button2"]),
            canvas_size={"width": 102, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            discrete_bboxes=None,
            discrete_gold_bboxes=None,
            discrete_content_bboxes=None,
        )

    def test_invoke_with_layout_data_sorts_by_position(
        self, sorter: LexicographicSort, sample_layout_data: LayoutData
    ):
        """Test that bboxes are sorted by top then left coordinates."""
        result = sorter.invoke(sample_layout_data)

        assert isinstance(result, ProcessedLayoutData)
        assert result.idx == sample_layout_data.idx

        # Expected sort order by (top, left):
        # bbox2: [0.2, 0.2, 0.4, 0.4] - top=0.2, left=0.2 (first)
        # bbox3: [0.0, 0.5, 0.2, 0.7] - top=0.5, left=0.0 (second)
        # bbox1: [0.1, 0.5, 0.3, 0.7] - top=0.5, left=0.1 (third)
        expected_bboxes = np.array(
            [[0.2, 0.2, 0.4, 0.4], [0.0, 0.5, 0.2, 0.7], [0.1, 0.5, 0.3, 0.7]]
        )
        expected_labels = np.array(["text2", "text3", "text1"])

        assert np.array_equal(result.bboxes, expected_bboxes)
        assert np.array_equal(result.labels, expected_labels)
        assert np.array_equal(result.gold_bboxes, expected_bboxes)

    def test_invoke_with_processed_data_sorts_by_position(
        self, sorter: LexicographicSort, sample_processed_data: ProcessedLayoutData
    ):
        """Test that ProcessedLayoutData is sorted correctly by position."""
        result = sorter.invoke(sample_processed_data)

        assert isinstance(result, ProcessedLayoutData)
        assert result.idx == sample_processed_data.idx

        # Both bboxes have same top (0.8), so sort by left:
        # bbox2: [0.1, 0.8, 0.3, 1.0] - left=0.1 (first)
        # bbox1: [0.3, 0.8, 0.5, 1.0] - left=0.3 (second)
        expected_bboxes = np.array([[0.1, 0.8, 0.3, 1.0], [0.3, 0.8, 0.5, 1.0]])
        expected_labels = np.array(["button2", "button1"])

        assert np.array_equal(result.bboxes, expected_bboxes)
        assert np.array_equal(result.labels, expected_labels)
        assert np.array_equal(result.gold_bboxes, expected_bboxes)

    def test_invoke_single_element(self, sorter: LexicographicSort):
        """Test sorting with a single element."""
        layout_data = LayoutData(
            idx=2,
            bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            labels=np.array(["text"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=None,
        )

        result = sorter.invoke(layout_data)

        assert isinstance(result, ProcessedLayoutData)
        assert np.array_equal(result.labels, np.array(["text"]))
        assert np.array_equal(result.bboxes, np.array([[0.1, 0.2, 0.3, 0.4]]))

    def test_invoke_same_top_different_left(self, sorter: LexicographicSort):
        """Test sorting with same top coordinate but different left coordinates."""
        layout_data = LayoutData(
            idx=3,
            bboxes=np.array(
                [[0.8, 0.5, 1.0, 0.7], [0.2, 0.5, 0.4, 0.7], [0.5, 0.5, 0.7, 0.7]]
            ),
            labels=np.array(["right", "left", "middle"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=None,
        )

        result = sorter.invoke(layout_data)

        # Should be sorted by left coordinate: 0.2, 0.5, 0.8
        expected_bboxes = np.array(
            [[0.2, 0.5, 0.4, 0.7], [0.5, 0.5, 0.7, 0.7], [0.8, 0.5, 1.0, 0.7]]
        )
        expected_labels = np.array(["left", "middle", "right"])

        assert np.array_equal(result.bboxes, expected_bboxes)
        assert np.array_equal(result.labels, expected_labels)

    def test_invoke_different_top_same_left(self, sorter: LexicographicSort):
        """Test sorting with different top coordinates but same left coordinate."""
        layout_data = LayoutData(
            idx=4,
            bboxes=np.array(
                [[0.3, 0.8, 0.5, 1.0], [0.3, 0.2, 0.5, 0.4], [0.3, 0.5, 0.5, 0.7]]
            ),
            labels=np.array(["bottom", "top", "middle"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=None,
        )

        result = sorter.invoke(layout_data)

        # Should be sorted by top coordinate: 0.2, 0.5, 0.8
        expected_bboxes = np.array(
            [[0.3, 0.2, 0.5, 0.4], [0.3, 0.5, 0.5, 0.7], [0.3, 0.8, 0.5, 1.0]]
        )
        expected_labels = np.array(["top", "middle", "bottom"])

        assert np.array_equal(result.bboxes, expected_bboxes)
        assert np.array_equal(result.labels, expected_labels)

    def test_invoke_reading_order_pattern(self, sorter: LexicographicSort):
        """Test lexicographic sort with typical reading order pattern."""
        # Simulate a 2x2 grid layout
        layout_data = LayoutData(
            idx=5,
            bboxes=np.array(
                [
                    [0.5, 0.5, 1.0, 1.0],  # bottom-right
                    [0.0, 0.0, 0.5, 0.5],  # top-left
                    [0.0, 0.5, 0.5, 1.0],  # bottom-left
                    [0.5, 0.0, 1.0, 0.5],  # top-right
                ]
            ),
            labels=np.array(["bottom-right", "top-left", "bottom-left", "top-right"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=None,
        )

        result = sorter.invoke(layout_data)

        # Reading order: top-left, top-right, bottom-left, bottom-right
        expected_bboxes = np.array(
            [
                [0.0, 0.0, 0.5, 0.5],  # top-left (top=0.0, left=0.0)
                [0.5, 0.0, 1.0, 0.5],  # top-right (top=0.0, left=0.5)
                [0.0, 0.5, 0.5, 1.0],  # bottom-left (top=0.5, left=0.0)
                [0.5, 0.5, 1.0, 1.0],  # bottom-right (top=0.5, left=0.5)
            ]
        )
        expected_labels = np.array(
            ["top-left", "top-right", "bottom-left", "bottom-right"]
        )

        assert np.array_equal(result.bboxes, expected_bboxes)
        assert np.array_equal(result.labels, expected_labels)

    def test_invoke_without_content_data(self, sorter: LexicographicSort):
        """Test sorting when no content data is present."""
        layout_data = LayoutData(
            idx=6,
            bboxes=np.array([[0.5, 0.5, 0.7, 0.7], [0.1, 0.2, 0.3, 0.4]]),
            labels=np.array(["second", "first"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image=None,
            content_bboxes=None,
        )

        result = sorter.invoke(layout_data)

        assert isinstance(result, ProcessedLayoutData)
        assert result.content_bboxes is None
        assert result.encoded_image is None

        # Should be sorted by position: first bbox has top=0.2, second has top=0.5
        expected_bboxes = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.7, 0.7]])
        expected_labels = np.array(["first", "second"])

        assert np.array_equal(result.bboxes, expected_bboxes)
        assert np.array_equal(result.labels, expected_labels)

    def test_invoke_preserves_original_data(self, sorter: LexicographicSort):
        """Test that original data is preserved correctly."""
        layout_data = LayoutData(
            idx=7,
            bboxes=np.array([[0.5, 0.5, 0.7, 0.7], [0.1, 0.2, 0.3, 0.4]]),
            labels=np.array(["second", "first"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=np.array([[0.0, 0.0, 1.0, 1.0]]),
        )

        result = sorter.invoke(layout_data)

        # Original data should match input data (before sorting)
        expected_orig_bboxes = np.array([[0.5, 0.5, 0.7, 0.7], [0.1, 0.2, 0.3, 0.4]])
        expected_orig_labels = np.array(["second", "first"])
        assert np.array_equal(result.orig_bboxes, expected_orig_bboxes)
        assert np.array_equal(result.orig_labels, expected_orig_labels)

    def test_invoke_preserves_content_bboxes(self, sorter: LexicographicSort):
        """Test that content_bboxes are preserved during sorting."""
        layout_data = LayoutData(
            idx=8,
            bboxes=np.array([[0.5, 0.5, 0.7, 0.7], [0.1, 0.2, 0.3, 0.4]]),
            labels=np.array(["second", "first"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=np.array([[0.0, 0.0, 1.0, 1.0], [0.2, 0.3, 0.4, 0.5]]),
        )

        result = sorter.invoke(layout_data)

        # content_bboxes should be preserved unchanged
        expected_content_bboxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.2, 0.3, 0.4, 0.5]])
        assert np.array_equal(result.content_bboxes, expected_content_bboxes)

    def test_invoke_with_processed_data_resets_discrete_fields(
        self, sorter: LexicographicSort
    ):
        """Test that discrete fields are reset to None when processing."""
        processed_data = ProcessedLayoutData(
            idx=9,
            bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            labels=np.array(["text"]),
            gold_bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            orig_bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            orig_labels=np.array(["text"]),
            canvas_size={"width": 102, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=None,
            discrete_bboxes=np.array([[10, 20, 30, 40]], dtype=np.int32),
            discrete_gold_bboxes=np.array([[10, 20, 30, 40]], dtype=np.int32),
            discrete_content_bboxes=None,
        )

        result = sorter.invoke(processed_data)

        # Discrete fields should be reset to None (this is the actual behavior)
        assert result.discrete_bboxes is None
        assert result.discrete_gold_bboxes is None
        assert result.discrete_content_bboxes is None

    def test_name_property(self, sorter: LexicographicSort):
        """Test that the name property is correct."""
        assert sorter.name == "lexicographic-sort"

    def test_invoke_assertion_error_no_bboxes(self, sorter: LexicographicSort):
        """Test that assertion error is raised when bboxes is None."""
        layout_data = LayoutData(
            idx=10,
            bboxes=None,
            labels=np.array(["text"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image=None,
            content_bboxes=None,
        )

        with pytest.raises(AssertionError):
            sorter.invoke(layout_data)

    def test_invoke_assertion_error_no_labels(self, sorter: LexicographicSort):
        """Test that assertion error is raised when labels is None."""
        layout_data = LayoutData(
            idx=11,
            bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            labels=None,
            canvas_size={"width": 100, "height": 150},
            encoded_image=None,
            content_bboxes=None,
        )

        with pytest.raises(AssertionError):
            sorter.invoke(layout_data)

    def test_invoke_zero_coordinates(self, sorter: LexicographicSort):
        """Test sorting with zero coordinates."""
        layout_data = LayoutData(
            idx=12,
            bboxes=np.array(
                [[0.0, 0.0, 0.2, 0.2], [0.0, 0.1, 0.2, 0.3], [0.1, 0.0, 0.3, 0.2]]
            ),
            labels=np.array(["origin", "down", "right"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image=None,
            content_bboxes=None,
        )

        result = sorter.invoke(layout_data)

        # Expected order: (0,0), (0,1), (1,0) -> origin, down, right
        expected_bboxes = np.array(
            [[0.0, 0.0, 0.2, 0.2], [0.1, 0.0, 0.3, 0.2], [0.0, 0.1, 0.2, 0.3]]
        )
        expected_labels = np.array(["origin", "right", "down"])

        assert np.array_equal(result.bboxes, expected_bboxes)
        assert np.array_equal(result.labels, expected_labels)
