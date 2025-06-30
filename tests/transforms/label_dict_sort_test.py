import numpy as np
import pytest
from layout_prompter.models import LayoutData, ProcessedLayoutData
from layout_prompter.transforms.label_dict_sort import LabelDictSort


class TestLabelDictSort:
    @pytest.fixture
    def sorter(self) -> LabelDictSort:
        return LabelDictSort()

    @pytest.fixture
    def sample_layout_data(self) -> LayoutData:
        return LayoutData(
            idx=0,
            bboxes=np.array(
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 1.0, 0.2]]
            ),
            labels=np.array(["text", "button", "logo"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=np.array([[0.0, 0.0, 1.0, 1.0]]),
        )

    @pytest.fixture
    def sample_processed_data(self) -> ProcessedLayoutData:
        return ProcessedLayoutData(
            idx=1,
            bboxes=np.array([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]),
            labels=np.array(["text", "button"]),
            gold_bboxes=np.array([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]),
            orig_bboxes=np.array([[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]),
            orig_labels=np.array(["text", "button"]),
            canvas_size={"width": 102, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            discrete_bboxes=None,
            discrete_gold_bboxes=None,
            discrete_content_bboxes=None,
        )

    def test_invoke_with_layout_data_sorts_by_label(
        self, sorter: LabelDictSort, sample_layout_data: LayoutData
    ):
        """Test that labels are sorted alphabetically with corresponding bboxes."""
        result = sorter.invoke(sample_layout_data)

        assert isinstance(result, ProcessedLayoutData)
        assert result.idx == sample_layout_data.idx

        # Labels should be sorted: ["button", "logo", "text"]
        expected_labels = np.array(["button", "logo", "text"])
        assert np.array_equal(result.labels, expected_labels)

        # Bboxes should be reordered to match sorted labels
        # Original order: text[0.1,0.2,0.3,0.4], button[0.5,0.6,0.7,0.8], logo[0.9,0.1,1.0,0.2]
        # New order: button[0.5,0.6,0.7,0.8], logo[0.9,0.1,1.0,0.2], text[0.1,0.2,0.3,0.4]
        expected_bboxes = np.array(
            [[0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 1.0, 0.2], [0.1, 0.2, 0.3, 0.4]]
        )
        assert np.array_equal(result.bboxes, expected_bboxes)
        assert np.array_equal(result.gold_bboxes, expected_bboxes)

    def test_invoke_with_processed_data_sorts_by_label(
        self, sorter: LabelDictSort, sample_processed_data: ProcessedLayoutData
    ):
        """Test that ProcessedLayoutData is sorted correctly."""
        result = sorter.invoke(sample_processed_data)

        assert isinstance(result, ProcessedLayoutData)
        assert result.idx == sample_processed_data.idx

        # Labels should be sorted: ["button", "text"]
        expected_labels = np.array(["button", "text"])
        assert np.array_equal(result.labels, expected_labels)

        # Bboxes should be reordered: button[0.6,0.7,0.8,0.9], text[0.2,0.3,0.4,0.5]
        expected_bboxes = np.array([[0.6, 0.7, 0.8, 0.9], [0.2, 0.3, 0.4, 0.5]])
        assert np.array_equal(result.bboxes, expected_bboxes)
        assert np.array_equal(result.gold_bboxes, expected_bboxes)

    def test_invoke_without_content_data(self, sorter: LabelDictSort):
        """Test sorting when no content data is present."""
        layout_data = LayoutData(
            idx=2,
            bboxes=np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            labels=np.array(["zebra", "apple"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image=None,
            content_bboxes=None,
        )

        result = sorter.invoke(layout_data)

        assert isinstance(result, ProcessedLayoutData)
        assert result.content_bboxes is None
        assert result.encoded_image is None

        # Labels should be sorted: ["apple", "zebra"]
        expected_labels = np.array(["apple", "zebra"])
        assert np.array_equal(result.labels, expected_labels)

        # Bboxes should be reordered: apple[0.5,0.6,0.7,0.8], zebra[0.1,0.2,0.3,0.4]
        expected_bboxes = np.array([[0.5, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.4]])
        assert np.array_equal(result.bboxes, expected_bboxes)

    def test_invoke_single_element(self, sorter: LabelDictSort):
        """Test sorting with a single element."""
        layout_data = LayoutData(
            idx=3,
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

    def test_invoke_identical_labels(self, sorter: LabelDictSort):
        """Test sorting with identical labels."""
        layout_data = LayoutData(
            idx=4,
            bboxes=np.array(
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 1.0, 0.2]]
            ),
            labels=np.array(["text", "text", "text"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=None,
        )

        result = sorter.invoke(layout_data)

        assert isinstance(result, ProcessedLayoutData)
        assert np.array_equal(result.labels, np.array(["text", "text", "text"]))
        # Order should be preserved when labels are identical
        expected_bboxes = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 1.0, 0.2]]
        )
        assert np.array_equal(result.bboxes, expected_bboxes)

    def test_invoke_preserves_original_data(self, sorter: LabelDictSort):
        """Test that original data is preserved correctly."""
        layout_data = LayoutData(
            idx=5,
            bboxes=np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            labels=np.array(["text", "button"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=np.array([[0.0, 0.0, 1.0, 1.0]]),
        )

        result = sorter.invoke(layout_data)

        # Original data should match input data
        expected_orig_bboxes = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        expected_orig_labels = np.array(["text", "button"])
        assert np.array_equal(result.orig_bboxes, expected_orig_bboxes)
        assert np.array_equal(result.orig_labels, expected_orig_labels)

    def test_invoke_preserves_content_bboxes(self, sorter: LabelDictSort):
        """Test that content_bboxes are preserved during sorting."""
        layout_data = LayoutData(
            idx=6,
            bboxes=np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            labels=np.array(["text", "button"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=np.array([[0.0, 0.0, 1.0, 1.0], [0.2, 0.3, 0.4, 0.5]]),
        )

        result = sorter.invoke(layout_data)

        # content_bboxes should be preserved unchanged
        expected_content_bboxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.2, 0.3, 0.4, 0.5]])
        assert np.array_equal(result.content_bboxes, expected_content_bboxes)

    def test_invoke_with_processed_data_resets_discrete_fields(
        self, sorter: LabelDictSort
    ):
        """Test that discrete fields are reset to None when processing."""
        processed_data = ProcessedLayoutData(
            idx=7,
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

    def test_name_property(self, sorter: LabelDictSort):
        """Test that the name property is correct."""
        assert sorter.name == "label-dict-sort"

    def test_invoke_assertion_error_no_bboxes(self, sorter: LabelDictSort):
        """Test that assertion error is raised when bboxes is None."""
        layout_data = LayoutData(
            idx=8,
            bboxes=None,
            labels=np.array(["text"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image=None,
            content_bboxes=None,
        )

        with pytest.raises(AssertionError):
            sorter.invoke(layout_data)

    def test_invoke_assertion_error_no_labels(self, sorter: LabelDictSort):
        """Test that assertion error is raised when labels is None."""
        layout_data = LayoutData(
            idx=9,
            bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            labels=None,
            canvas_size={"width": 100, "height": 150},
            encoded_image=None,
            content_bboxes=None,
        )

        with pytest.raises(AssertionError):
            sorter.invoke(layout_data)
