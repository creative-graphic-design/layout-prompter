import pytest
from layout_prompter.models import LayoutData, ProcessedLayoutData, NormalizedBbox, Bbox, CanvasSize
from layout_prompter.transforms.lexicographic_sort import LexicographicSort


class TestLexicographicSort:
    @pytest.fixture
    def sorter(self) -> LexicographicSort:
        return LexicographicSort()

    @pytest.fixture
    def sample_layout_data(self) -> LayoutData:
        # Create bboxes with different positions for testing lexicographic sort
        # Note: NormalizedBbox uses left, top, width, height format
        # bbox1: left=0.1, top=0.5, width=0.2, height=0.2 - top=0.5, left=0.1
        # bbox2: left=0.2, top=0.2, width=0.2, height=0.2 - top=0.2, left=0.2 (should be first)
        # bbox3: left=0.0, top=0.5, width=0.2, height=0.2 - top=0.5, left=0.0
        return LayoutData(
            idx=0,
            bboxes=[
                NormalizedBbox(left=0.1, top=0.5, width=0.2, height=0.2),  # text1
                NormalizedBbox(left=0.2, top=0.2, width=0.2, height=0.2),  # text2
                NormalizedBbox(left=0.0, top=0.5, width=0.2, height=0.2),  # text3
            ],
            labels=["text1", "text2", "text3"],
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image="base64encoded",
            content_bboxes=[NormalizedBbox(left=0.0, top=0.0, width=1.0, height=1.0)],
        )

    @pytest.fixture
    def sample_processed_data(self) -> ProcessedLayoutData:
        return ProcessedLayoutData(
            idx=1,
            bboxes=[
                NormalizedBbox(left=0.3, top=0.8, width=0.2, height=0.2),  # button1
                NormalizedBbox(left=0.1, top=0.8, width=0.2, height=0.2),  # button2
            ],
            labels=["button1", "button2"],
            gold_bboxes=[
                NormalizedBbox(left=0.3, top=0.8, width=0.2, height=0.2),
                NormalizedBbox(left=0.1, top=0.8, width=0.2, height=0.2),
            ],
            orig_bboxes=[
                NormalizedBbox(left=0.3, top=0.8, width=0.2, height=0.2),
                NormalizedBbox(left=0.1, top=0.8, width=0.2, height=0.2),
            ],
            orig_labels=["button1", "button2"],
            orig_canvas_size=CanvasSize(width=102, height=150),
            canvas_size=CanvasSize(width=102, height=150),
            encoded_image="base64encoded",
            content_bboxes=[NormalizedBbox(left=0.1, top=0.2, width=0.1, height=0.2)],
            discrete_bboxes=None,
            discrete_gold_bboxes=None,
            discrete_content_bboxes=None,
        )

    def test_invoke_with_layout_data_sorts_by_position(
        self, sorter: LexicographicSort, sample_layout_data: LayoutData
    ):
        """Test that bboxes are sorted by left then top coordinates."""
        result = sorter.invoke(sample_layout_data)

        assert isinstance(result, ProcessedLayoutData)
        assert result.idx == sample_layout_data.idx

        # Expected sort order by (left, top) as per implementation:
        # bbox3: left=0.0, top=0.5 (first)
        # bbox1: left=0.1, top=0.5 (second) 
        # bbox2: left=0.2, top=0.2 (third)
        expected_bboxes = (
            NormalizedBbox(left=0.0, top=0.5, width=0.2, height=0.2),  # text3
            NormalizedBbox(left=0.1, top=0.5, width=0.2, height=0.2),  # text1
            NormalizedBbox(left=0.2, top=0.2, width=0.2, height=0.2),  # text2
        )
        expected_labels = ("text3", "text1", "text2")

        assert result.bboxes == expected_bboxes
        assert result.labels == expected_labels
        assert result.gold_bboxes == expected_bboxes

    def test_invoke_with_processed_data_sorts_by_position(
        self, sorter: LexicographicSort, sample_processed_data: ProcessedLayoutData
    ):
        """Test that ProcessedLayoutData is sorted correctly by position."""
        result = sorter.invoke(sample_processed_data)

        assert isinstance(result, ProcessedLayoutData)
        assert result.idx == sample_processed_data.idx

        # Both bboxes have same top (0.8), so sort by left:
        # button2: left=0.1 (first)
        # button1: left=0.3 (second)
        expected_bboxes = (
            NormalizedBbox(left=0.1, top=0.8, width=0.2, height=0.2),  # button2
            NormalizedBbox(left=0.3, top=0.8, width=0.2, height=0.2),  # button1
        )
        expected_labels = ("button2", "button1")

        assert result.bboxes == expected_bboxes
        assert result.labels == expected_labels
        assert result.gold_bboxes == expected_bboxes

    def test_invoke_single_element(self, sorter: LexicographicSort):
        """Test sorting with a single element."""
        layout_data = LayoutData(
            idx=2,
            bboxes=[NormalizedBbox(left=0.1, top=0.2, width=0.2, height=0.2)],
            labels=["text"],
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image="base64encoded",
            content_bboxes=None,
        )

        result = sorter.invoke(layout_data)

        assert isinstance(result, ProcessedLayoutData)
        assert result.labels == ("text",)
        assert result.bboxes == (NormalizedBbox(left=0.1, top=0.2, width=0.2, height=0.2),)

    def test_invoke_same_top_different_left(self, sorter: LexicographicSort):
        """Test sorting with same top coordinate but different left coordinates."""
        layout_data = LayoutData(
            idx=3,
            bboxes=[
                NormalizedBbox(left=0.8, top=0.5, width=0.2, height=0.2),  # right
                NormalizedBbox(left=0.2, top=0.5, width=0.2, height=0.2),  # left
                NormalizedBbox(left=0.5, top=0.5, width=0.2, height=0.2),  # middle
            ],
            labels=["right", "left", "middle"],
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image="base64encoded",
            content_bboxes=None,
        )

        result = sorter.invoke(layout_data)

        # Should be sorted by left coordinate: 0.2, 0.5, 0.8
        expected_bboxes = (
            NormalizedBbox(left=0.2, top=0.5, width=0.2, height=0.2),  # left
            NormalizedBbox(left=0.5, top=0.5, width=0.2, height=0.2),  # middle
            NormalizedBbox(left=0.8, top=0.5, width=0.2, height=0.2),  # right
        )
        expected_labels = ("left", "middle", "right")

        assert result.bboxes == expected_bboxes
        assert result.labels == expected_labels

    def test_invoke_different_top_same_left(self, sorter: LexicographicSort):
        """Test sorting with same left coordinate but different top coordinates."""
        layout_data = LayoutData(
            idx=4,
            bboxes=[
                NormalizedBbox(left=0.3, top=0.8, width=0.2, height=0.2),  # bottom
                NormalizedBbox(left=0.3, top=0.2, width=0.2, height=0.2),  # top
                NormalizedBbox(left=0.3, top=0.5, width=0.2, height=0.2),  # middle
            ],
            labels=["bottom", "top", "middle"],
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image="base64encoded",
            content_bboxes=None,
        )

        result = sorter.invoke(layout_data)

        # Should be sorted by top coordinate when left is same: 0.2, 0.5, 0.8
        expected_bboxes = (
            NormalizedBbox(left=0.3, top=0.2, width=0.2, height=0.2),  # top
            NormalizedBbox(left=0.3, top=0.5, width=0.2, height=0.2),  # middle
            NormalizedBbox(left=0.3, top=0.8, width=0.2, height=0.2),  # bottom
        )
        expected_labels = ("top", "middle", "bottom")

        assert result.bboxes == expected_bboxes
        assert result.labels == expected_labels

    def test_invoke_reading_order_pattern(self, sorter: LexicographicSort):
        """Test lexicographic sort with typical reading order pattern."""
        # Simulate a 2x2 grid layout
        layout_data = LayoutData(
            idx=5,
            bboxes=[
                NormalizedBbox(left=0.5, top=0.5, width=0.5, height=0.5),  # bottom-right
                NormalizedBbox(left=0.0, top=0.0, width=0.5, height=0.5),  # top-left
                NormalizedBbox(left=0.0, top=0.5, width=0.5, height=0.5),  # bottom-left
                NormalizedBbox(left=0.5, top=0.0, width=0.5, height=0.5),  # top-right
            ],
            labels=["bottom-right", "top-left", "bottom-left", "top-right"],
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image="base64encoded",
            content_bboxes=None,
        )

        result = sorter.invoke(layout_data)

        # Sort order by (left, top): (0,0), (0,5), (5,0), (5,5)
        expected_bboxes = (
            NormalizedBbox(left=0.0, top=0.0, width=0.5, height=0.5),  # top-left
            NormalizedBbox(left=0.0, top=0.5, width=0.5, height=0.5),  # bottom-left
            NormalizedBbox(left=0.5, top=0.0, width=0.5, height=0.5),  # top-right
            NormalizedBbox(left=0.5, top=0.5, width=0.5, height=0.5),  # bottom-right
        )
        expected_labels = (
            "top-left", "bottom-left", "top-right", "bottom-right"
        )

        assert result.bboxes == expected_bboxes
        assert result.labels == expected_labels

    def test_invoke_without_content_data(self, sorter: LexicographicSort):
        """Test sorting when no content data is present."""
        layout_data = LayoutData(
            idx=6,
            bboxes=[
                NormalizedBbox(left=0.5, top=0.5, width=0.2, height=0.2),  # second
                NormalizedBbox(left=0.1, top=0.2, width=0.2, height=0.2),  # first
            ],
            labels=["second", "first"],
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image=None,
            content_bboxes=None,
        )

        result = sorter.invoke(layout_data)

        assert isinstance(result, ProcessedLayoutData)
        assert result.content_bboxes is None
        assert result.encoded_image is None

        # Should be sorted by left coordinate: first has left=0.1, second has left=0.5
        expected_bboxes = (
            NormalizedBbox(left=0.1, top=0.2, width=0.2, height=0.2),  # first
            NormalizedBbox(left=0.5, top=0.5, width=0.2, height=0.2),  # second
        )
        expected_labels = ("first", "second")

        assert result.bboxes == expected_bboxes
        assert result.labels == expected_labels

    def test_invoke_preserves_original_data(self, sorter: LexicographicSort):
        """Test that original data is preserved correctly."""
        layout_data = LayoutData(
            idx=7,
            bboxes=[
                NormalizedBbox(left=0.5, top=0.5, width=0.2, height=0.2),  # second
                NormalizedBbox(left=0.1, top=0.2, width=0.2, height=0.2),  # first
            ],
            labels=["second", "first"],
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image="base64encoded",
            content_bboxes=[NormalizedBbox(left=0.0, top=0.0, width=1.0, height=1.0)],
        )

        result = sorter.invoke(layout_data)

        # Original data should match input data (before sorting)
        expected_orig_bboxes = [
            NormalizedBbox(left=0.5, top=0.5, width=0.2, height=0.2),  # second
            NormalizedBbox(left=0.1, top=0.2, width=0.2, height=0.2),  # first
        ]
        expected_orig_labels = ["second", "first"]
        assert result.orig_bboxes == expected_orig_bboxes
        assert result.orig_labels == expected_orig_labels

    def test_invoke_preserves_content_bboxes(self, sorter: LexicographicSort):
        """Test that content_bboxes are preserved during sorting."""
        layout_data = LayoutData(
            idx=8,
            bboxes=[
                NormalizedBbox(left=0.5, top=0.5, width=0.2, height=0.2),  # second
                NormalizedBbox(left=0.1, top=0.2, width=0.2, height=0.2),  # first
            ],
            labels=["second", "first"],
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image="base64encoded",
            content_bboxes=[
                NormalizedBbox(left=0.0, top=0.0, width=1.0, height=1.0),
                NormalizedBbox(left=0.2, top=0.3, width=0.2, height=0.2),
            ],
        )

        result = sorter.invoke(layout_data)

        # content_bboxes should be preserved unchanged
        expected_content_bboxes = [
            NormalizedBbox(left=0.0, top=0.0, width=1.0, height=1.0),
            NormalizedBbox(left=0.2, top=0.3, width=0.2, height=0.2),
        ]
        assert result.content_bboxes == expected_content_bboxes

    def test_invoke_with_processed_data_resets_discrete_fields(
        self, sorter: LexicographicSort
    ):
        """Test that discrete fields are reset to None when processing."""
        processed_data = ProcessedLayoutData(
            idx=9,
            bboxes=[NormalizedBbox(left=0.1, top=0.2, width=0.2, height=0.2)],
            labels=["text"],
            gold_bboxes=[NormalizedBbox(left=0.1, top=0.2, width=0.2, height=0.2)],
            orig_bboxes=[NormalizedBbox(left=0.1, top=0.2, width=0.2, height=0.2)],
            orig_labels=["text"],
            orig_canvas_size=CanvasSize(width=102, height=150),
            canvas_size=CanvasSize(width=102, height=150),
            encoded_image="base64encoded",
            content_bboxes=None,
            discrete_bboxes=[Bbox(left=10, top=20, width=20, height=20)],
            discrete_gold_bboxes=[Bbox(left=10, top=20, width=20, height=20)],
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
            labels=["text"],
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image=None,
            content_bboxes=None,
        )

        with pytest.raises(AssertionError):
            sorter.invoke(layout_data)

    def test_invoke_assertion_error_no_labels(self, sorter: LexicographicSort):
        """Test that assertion error is raised when labels is None."""
        layout_data = LayoutData(
            idx=11,
            bboxes=[NormalizedBbox(left=0.1, top=0.2, width=0.2, height=0.2)],
            labels=None,
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image=None,
            content_bboxes=None,
        )

        with pytest.raises(AssertionError):
            sorter.invoke(layout_data)

    def test_invoke_zero_coordinates(self, sorter: LexicographicSort):
        """Test sorting with zero coordinates."""
        layout_data = LayoutData(
            idx=12,
            bboxes=[
                NormalizedBbox(left=0.0, top=0.0, width=0.2, height=0.2),  # origin
                NormalizedBbox(left=0.0, top=0.1, width=0.2, height=0.2),  # down
                NormalizedBbox(left=0.1, top=0.0, width=0.2, height=0.2),  # right
            ],
            labels=["origin", "down", "right"],
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image=None,
            content_bboxes=None,
        )

        result = sorter.invoke(layout_data)

        # Expected order by (left, top): (0,0), (0,1), (1,0) -> origin, down, right
        expected_bboxes = (
            NormalizedBbox(left=0.0, top=0.0, width=0.2, height=0.2),  # origin
            NormalizedBbox(left=0.0, top=0.1, width=0.2, height=0.2),  # down
            NormalizedBbox(left=0.1, top=0.0, width=0.2, height=0.2),  # right
        )
        expected_labels = ("origin", "down", "right")

        assert result.bboxes == expected_bboxes
        assert result.labels == expected_labels
