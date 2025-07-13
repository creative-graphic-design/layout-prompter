import pytest

from layout_prompter.models import (
    CanvasSize,
    LayoutData,
    ProcessedLayoutData,
    NormalizedBbox,
)
from layout_prompter.transforms import DiscretizeBboxes


class TestDiscretizeBboxes:
    @pytest.fixture
    def discretizer(self) -> DiscretizeBboxes:
        return DiscretizeBboxes(target_canvas_size=CanvasSize(width=100, height=150))

    @pytest.fixture
    def sample_layout_data(self) -> LayoutData:
        return LayoutData(
            idx=0,
            bboxes=[
                NormalizedBbox(left=0.1, top=0.2, width=0.3, height=0.4),
                NormalizedBbox(left=0.5, top=0.6, width=0.2, height=0.1),
            ],
            labels=["text", "logo"],
            canvas_size=CanvasSize(width=80, height=120),
            encoded_image="base64encoded",
            content_bboxes=[NormalizedBbox(left=0.0, top=0.0, width=1.0, height=1.0)],
        )

    @pytest.fixture
    def sample_processed_data(self) -> ProcessedLayoutData:
        return ProcessedLayoutData(
            idx=1,
            bboxes=[NormalizedBbox(left=0.2, top=0.3, width=0.4, height=0.2)],
            labels=["text"],
            gold_bboxes=[NormalizedBbox(left=0.2, top=0.3, width=0.4, height=0.2)],
            orig_bboxes=[NormalizedBbox(left=0.15, top=0.25, width=0.45, height=0.25)],
            orig_labels=["text"],
            orig_canvas_size=CanvasSize(width=80, height=120),
            canvas_size=CanvasSize(width=102, height=153),
            encoded_image="base64encoded",
            content_bboxes=None,
            discrete_bboxes=None,
            discrete_gold_bboxes=None,
            discrete_content_bboxes=None,
        )

    def test_invoke_with_layout_data(
        self, discretizer: DiscretizeBboxes, sample_layout_data: LayoutData
    ):
        result = discretizer.invoke(sample_layout_data)

        # Check that it returns ProcessedLayoutData
        assert isinstance(result, ProcessedLayoutData)
        assert result.idx == sample_layout_data.idx

        # Check that original data is preserved
        assert result.bboxes == sample_layout_data.bboxes
        assert result.labels == sample_layout_data.labels
        assert result.encoded_image == sample_layout_data.encoded_image

        # Check that gold_bboxes and orig_bboxes are set correctly
        assert result.gold_bboxes == sample_layout_data.bboxes
        assert result.orig_bboxes == sample_layout_data.bboxes
        assert result.orig_labels == sample_layout_data.labels
        assert result.orig_canvas_size == sample_layout_data.canvas_size

        # Check that canvas_size is updated to target
        assert result.canvas_size == discretizer.target_canvas_size

        # Check that discrete values are generated
        assert result.discrete_bboxes is not None
        assert len(result.discrete_bboxes) == len(sample_layout_data.bboxes)
        assert result.discrete_gold_bboxes is not None
        assert len(result.discrete_gold_bboxes) == len(sample_layout_data.bboxes)
        assert result.discrete_content_bboxes is not None

    def test_invoke_with_processed_data(
        self, discretizer: DiscretizeBboxes, sample_processed_data: ProcessedLayoutData
    ):
        result = discretizer.invoke(sample_processed_data)

        # Check that it returns ProcessedLayoutData
        assert isinstance(result, ProcessedLayoutData)
        assert result.idx == sample_processed_data.idx

        # Check that original processed data is preserved
        assert result.bboxes == sample_processed_data.bboxes
        assert result.labels == sample_processed_data.labels
        assert result.gold_bboxes == sample_processed_data.gold_bboxes
        # Due to inheritance, ProcessedLayoutData is also an instance of LayoutData
        # So orig_bboxes gets set to copy.deepcopy(gold_bboxes) instead of input.orig_bboxes
        assert result.orig_bboxes == sample_processed_data.gold_bboxes
        assert result.orig_labels == sample_processed_data.orig_labels
        assert result.orig_canvas_size == sample_processed_data.orig_canvas_size

        # Check that canvas_size is updated to target
        assert result.canvas_size == discretizer.target_canvas_size

        # Check that discrete values are generated
        assert result.discrete_bboxes is not None
        assert result.discrete_gold_bboxes is not None

    def test_invoke_without_content_bboxes(self, discretizer: DiscretizeBboxes):
        layout_data = LayoutData(
            idx=0,
            bboxes=[NormalizedBbox(left=0.1, top=0.2, width=0.3, height=0.4)],
            labels=["text"],
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image="base64encoded",
            content_bboxes=None,
        )

        result = discretizer.invoke(layout_data)

        # Should handle None content_bboxes gracefully
        assert result.discrete_content_bboxes is None
        assert result.content_bboxes is None

    def test_discretization_values(self, discretizer: DiscretizeBboxes):
        """Test that discretization produces expected pixel values"""
        layout_data = LayoutData(
            idx=0,
            bboxes=[
                NormalizedBbox(left=0.0, top=0.0, width=1.0, height=1.0),  # Full canvas
                NormalizedBbox(
                    left=0.1, top=0.2, width=0.3, height=0.4
                ),  # Specific values
            ],
            labels=["full", "partial"],
            canvas_size=CanvasSize(width=50, height=75),
            encoded_image=None,
            content_bboxes=None,
        )

        result = discretizer.invoke(layout_data)

        # Check discrete bboxes have correct pixel values
        discrete_bboxes = result.discrete_bboxes
        assert discrete_bboxes is not None

        # First bbox should be full canvas size (0, 0, 100, 150)
        first_bbox = discrete_bboxes[0]
        assert first_bbox.left == 0
        assert first_bbox.top == 0
        assert first_bbox.width == 100
        assert first_bbox.height == 150

        # Second bbox should be discretized correctly
        second_bbox = discrete_bboxes[1]
        assert second_bbox.left == 10  # 0.1 * 100
        assert second_bbox.top == 30  # 0.2 * 150
        assert second_bbox.width == 30  # 0.3 * 100
        assert second_bbox.height == 60  # 0.4 * 150

    def test_invalid_input_no_bboxes(self, discretizer: DiscretizeBboxes):
        """Test that missing bboxes raises an assertion error"""
        layout_data = LayoutData(
            idx=0,
            bboxes=None,
            labels=["text"],
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image=None,
            content_bboxes=None,
        )

        with pytest.raises(AssertionError):
            discretizer.invoke(layout_data)

    def test_invalid_input_no_labels(self, discretizer: DiscretizeBboxes):
        """Test that missing labels raises an assertion error"""
        layout_data = LayoutData(
            idx=0,
            bboxes=[NormalizedBbox(left=0.1, top=0.2, width=0.3, height=0.4)],
            labels=None,
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image=None,
            content_bboxes=None,
        )

        with pytest.raises(AssertionError):
            discretizer.invoke(layout_data)
