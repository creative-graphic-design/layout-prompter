import numpy as np
import pytest
from layout_prompter.models import LayoutData, ProcessedLayoutData
from layout_prompter.transforms import DiscretizeBboxes


class TestDiscretizeBboxes:
    @pytest.fixture
    def discretizer(self) -> DiscretizeBboxes:
        return DiscretizeBboxes()

    @pytest.fixture
    def sample_layout_data(self) -> LayoutData:
        return LayoutData(
            idx=0,
            bboxes=np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            labels=np.array(["text", "logo"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=np.array([[0.0, 0.0, 1.0, 1.0]]),
        )

    @pytest.fixture
    def sample_processed_data(self) -> ProcessedLayoutData:
        return ProcessedLayoutData(
            idx=1,
            bboxes=np.array([[0.2, 0.3, 0.4, 0.5]]),
            labels=np.array(["text"]),
            gold_bboxes=np.array([[0.2, 0.3, 0.4, 0.5]]),
            orig_bboxes=np.array([[0.2, 0.3, 0.4, 0.5]]),
            orig_labels=np.array(["text"]),
            canvas_size={"width": 102, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=None,
            discrete_bboxes=None,
            discrete_gold_bboxes=None,
            discrete_content_bboxes=None,
        )

    def test_discretize_function(self, discretizer: DiscretizeBboxes):
        bboxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.1, 0.2, 0.3, 0.4]])
        width, height = 100, 150

        result = discretizer.discretize(bboxes, width, height)

        assert result.shape == (2, 4)
        assert result.dtype == np.int32
        # First bbox: [0, 0, 100, 150]
        assert np.array_equal(result[0], [0, 0, 100, 150])
        # Second bbox: [10, 30, 30, 60]
        assert np.array_equal(result[1], [10, 30, 30, 60])

    def test_discretize_clipping(self, discretizer: DiscretizeBboxes):
        # Test values outside [0, 1] range get clipped
        bboxes = np.array([[-0.1, -0.2, 1.1, 1.2]])
        width, height = 100, 150

        result = discretizer.discretize(bboxes, width, height)

        # Should be clipped to [0, 0, 1, 1] then discretized to [0, 0, 100, 150]
        assert np.array_equal(result[0], [0, 0, 100, 150])

    def test_continuize_function(self, discretizer: DiscretizeBboxes):
        bboxes = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
        width, height = 100, 150

        result = discretizer.continuize(bboxes, width, height)

        assert result.shape == (2, 4)
        assert result.dtype == np.float32
        # First bbox: [0.1, 0.133..., 0.3, 0.266...]
        expected_first = np.array([0.1, 20 / 150, 0.3, 40 / 150], dtype=np.float32)
        assert np.allclose(result[0], expected_first)

    def test_invoke_with_layout_data(
        self, discretizer: DiscretizeBboxes, sample_layout_data: LayoutData
    ):
        result = discretizer.invoke(sample_layout_data)

        assert isinstance(result, ProcessedLayoutData)
        assert result.idx == sample_layout_data.idx
        assert np.array_equal(result.bboxes, sample_layout_data.bboxes)
        assert np.array_equal(result.labels, sample_layout_data.labels)
        assert result.discrete_bboxes is not None
        assert result.discrete_gold_bboxes is not None
        assert result.discrete_content_bboxes is not None

    def test_invoke_with_processed_data(
        self, discretizer: DiscretizeBboxes, sample_processed_data: ProcessedLayoutData
    ):
        result = discretizer.invoke(sample_processed_data)

        assert isinstance(result, ProcessedLayoutData)
        assert result.idx == sample_processed_data.idx
        assert np.array_equal(result.gold_bboxes, sample_processed_data.gold_bboxes)
        assert result.discrete_bboxes is not None
        assert result.discrete_gold_bboxes is not None

    def test_invoke_without_content_bboxes(self, discretizer: DiscretizeBboxes):
        layout_data = LayoutData(
            idx=0,
            bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            labels=np.array(["text"]),
            canvas_size={"width": 100, "height": 150},
            encoded_image="base64encoded",
            content_bboxes=None,
        )

        result = discretizer.invoke(layout_data)

        assert result.discrete_content_bboxes is None

    def test_discretize_invalid_shape(self, discretizer: DiscretizeBboxes):
        # Test with wrong shape - should raise assertion error
        invalid_bboxes = np.array([[0.1, 0.2, 0.3]])  # Only 3 values instead of 4

        with pytest.raises(
            AssertionError, match="bboxes should be of shape \\(N, 4\\)"
        ):
            discretizer.discretize(invalid_bboxes, 100, 150)
