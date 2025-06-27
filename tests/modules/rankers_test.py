from unittest.mock import Mock

import pytest
from layout_prompter.models.serialized_data import (
    Coordinates,
    PosterLayoutSerializedData,
    PosterLayoutSerializedOutputData,
)
from layout_prompter.modules.rankers import LayoutPrompterRanker


@pytest.fixture
def sample_layouts():
    """Create sample layout data for testing."""
    layouts = [
        PosterLayoutSerializedData(
            class_name="text", coord=Coordinates(left=10, top=20, width=100, height=30)
        ),
        PosterLayoutSerializedData(
            class_name="logo", coord=Coordinates(left=50, top=60, width=80, height=40)
        ),
        PosterLayoutSerializedData(
            class_name="underlay",
            coord=Coordinates(left=0, top=0, width=200, height=150),
        ),
    ]
    return layouts


@pytest.fixture
def sample_serialized_data(sample_layouts):
    """Create sample serialized output data."""
    return [
        PosterLayoutSerializedOutputData(layouts=sample_layouts),
        PosterLayoutSerializedOutputData(
            layouts=sample_layouts[:2]
        ),  # Different layout
    ]


class TestLayoutPrompterRanker:
    def test_ranker_initialization(self):
        """Test ranker can be initialized with default parameters."""
        ranker = LayoutPrompterRanker()
        assert ranker.name == "layout-prompter-ranker"
        assert ranker.lam_ali == 0.2
        assert ranker.lam_ove == 0.2
        assert ranker.lam_iou == 0.6

    def test_lambda_params_validation(self):
        """Test lambda parameters validation."""
        # Test invalid case with parameters not summing to 1.0 (Pydantic ValidationError)
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            LayoutPrompterRanker(lam_ali=0.3, lam_ove=0.3, lam_iou=0.3)

    def test_valid_lambda_params(self):
        """Test valid lambda parameters."""
        # Test valid case with all three parameters
        ranker1 = LayoutPrompterRanker(lam_ali=0.1, lam_ove=0.4, lam_iou=0.5)
        assert ranker1.lam_ali == 0.1
        assert ranker1.lam_ove == 0.4
        assert ranker1.lam_iou == 0.5

        # Test valid case with lam_iou=0
        ranker2 = LayoutPrompterRanker(lam_ali=0.4, lam_ove=0.6, lam_iou=0.0)
        assert ranker2.lam_ali == 0.4
        assert ranker2.lam_ove == 0.6
        assert ranker2.lam_iou == 0.0

    def test_empty_input(self):
        """Test ranker handles empty input."""
        ranker = LayoutPrompterRanker()
        # Empty input causes ValueError when computing min/max on empty array
        with pytest.raises(ValueError):
            ranker.invoke([])

    def test_single_input(self, sample_serialized_data):
        """Test ranker handles single input without processing."""
        ranker = LayoutPrompterRanker()
        single_input = [sample_serialized_data[0]]
        result = ranker.invoke(single_input)
        assert result == single_input

    def test_multiple_inputs(self, sample_serialized_data):
        """Test ranker processes multiple inputs and returns them ranked."""
        ranker = LayoutPrompterRanker()
        result = ranker.invoke(sample_serialized_data)

        # Should return same number of items
        assert len(result) == len(sample_serialized_data)

        # Should return the same items (though potentially reordered)
        assert set(id(item) for item in result) == set(
            id(item) for item in sample_serialized_data
        )

    def test_multiple_inputs_with_iou_zero(self, sample_serialized_data):
        """Test ranker processes multiple inputs when lam_iou=0."""
        ranker = LayoutPrompterRanker(lam_ali=0.4, lam_ove=0.6, lam_iou=0.0)
        result = ranker.invoke(sample_serialized_data)

        # Should return same number of items
        assert len(result) == len(sample_serialized_data)

        # Should return the same items (though potentially reordered)
        assert set(id(item) for item in result) == set(
            id(item) for item in sample_serialized_data
        )

    def test_invalid_input_no_layouts(self):
        """Test ranker handles input without layouts attribute."""
        ranker = LayoutPrompterRanker()
        invalid_data = Mock(spec=[])  # Mock without layouts attribute

        # Should raise AttributeError when trying to access layouts
        with pytest.raises(AttributeError):
            ranker.invoke([invalid_data])

    def test_invalid_input_empty_layouts(self):
        """Test ranker handles input with empty layouts."""
        ranker = LayoutPrompterRanker()
        empty_data = PosterLayoutSerializedOutputData(layouts=[])

        # Should raise ValueError when trying to unpack empty bbox array
        with pytest.raises(ValueError):
            ranker.invoke([empty_data])

    def test_invalid_layout_attributes(self):
        """Test ranker handles layouts without required attributes."""
        ranker = LayoutPrompterRanker()

        # Create mock object that will fail when accessing coord attribute
        class InvalidLayout:
            pass

        invalid_data = Mock()
        invalid_data.layouts = [InvalidLayout()]

        with pytest.raises(AttributeError):
            ranker.invoke([invalid_data])

    def test_normalization_edge_cases(self, sample_layouts):
        """Test normalization handles edge cases like identical metrics."""
        ranker = LayoutPrompterRanker()

        # Create identical layouts to test zero-range normalization
        identical_data = [
            PosterLayoutSerializedOutputData(layouts=sample_layouts),
            PosterLayoutSerializedOutputData(layouts=sample_layouts),
        ]

        # Should handle division by zero in normalization gracefully
        result = ranker.invoke(identical_data)
        assert len(result) == 2

    def test_ranking_order(self, sample_serialized_data):
        """Test that ranker returns items in quality order."""
        ranker = LayoutPrompterRanker()
        result = ranker.invoke(sample_serialized_data)

        # The exact order depends on the metrics, but we can verify
        # that the function runs without error and returns correct length
        assert len(result) == len(sample_serialized_data)
        assert all(hasattr(item, "layouts") for item in result)

    def test_mixed_valid_empty_layouts(self, sample_layouts):
        """Test handling of mixed valid and empty layouts."""
        ranker = LayoutPrompterRanker()

        valid_data = PosterLayoutSerializedOutputData(layouts=sample_layouts)
        empty_data = PosterLayoutSerializedOutputData(layouts=[])

        # Empty layouts will cause ValueError when trying to unpack empty bbox
        with pytest.raises(ValueError):
            ranker.invoke([valid_data, empty_data])

    def test_all_empty_layouts(self):
        """Test behavior when all inputs have empty layouts."""
        ranker = LayoutPrompterRanker()

        empty_data1 = PosterLayoutSerializedOutputData(layouts=[])
        empty_data2 = PosterLayoutSerializedOutputData(layouts=[])

        # All inputs have empty layouts, should raise ValueError
        with pytest.raises(ValueError):
            ranker.invoke([empty_data1, empty_data2])
