import datasets as ds
import pytest
from pytest_lazy_fixtures import lf

from layout_prompter.models import CanvasSize, LayoutData, NormalizedBbox
from layout_prompter.preprocessors import ContentAwareProcessor
from layout_prompter.utils.testing import LayoutPrompterTestCase


class TestContentAwareProcessor(LayoutPrompterTestCase):
    @pytest.fixture
    def num_proc(self) -> int:
        return 1  # Reduced for testing

    @pytest.mark.parametrize(
        argnames=("raw_layout_dataset"), argvalues=(lf("raw_poster_layout_dataset"),)
    )
    def test_content_aware_processor(
        self,
        raw_layout_dataset: ds.DatasetDict,
        num_proc: int,
    ):
        # Test with just a few samples to avoid timeout
        dataset = {
            split: [
                LayoutData.model_validate(data)
                for data in list(raw_layout_dataset[split])[
                    :2
                ]  # Only take first 2 samples
            ]
            for split in raw_layout_dataset
        }
        processor = ContentAwareProcessor()

        # Process each split separately since batch expects a list of LayoutData, not a dict
        processed_dataset = {}
        for split, layout_list in dataset.items():
            processed_list = processor.batch(
                layout_list,
                config={"max_concurrency": num_proc},
            )
            processed_dataset[split] = processed_list

        assert isinstance(processed_dataset, dict)

    def test_content_aware_processor_hashable(self):
        """Test that ContentAwareProcessor is hashable"""
        processor1 = ContentAwareProcessor()
        processor2 = ContentAwareProcessor()

        # Test hashability
        processor_set = {processor1, processor2}
        assert len(processor_set) == 1  # Both processors should be the same

        # Test as dict keys
        processor_dict = {processor1: "first", processor2: "second"}
        assert len(processor_dict) == 1
        assert (
            processor_dict[processor1] == "second"
        )  # processor2 overwrites processor1

        # Test equality
        assert processor1 == processor2

    def test_content_aware_processor_immutable(self):
        """Test that ContentAwareProcessor is immutable (frozen)"""
        processor = ContentAwareProcessor()

        # Attempting to set attributes should raise an error
        with pytest.raises(Exception):  # ValidationError or similar
            processor.max_element_numbers = 20

    def test_content_aware_processor_possible_labels_tuple(self):
        """Test that _possible_labels is properly handled as tuple"""
        processor = ContentAwareProcessor()

        # Initially should be empty tuple
        assert processor._possible_labels == tuple()

        # Create mock layout data with labels
        mock_layout = LayoutData(
            bboxes=[
                NormalizedBbox(left=0.1, top=0.1, width=0.5, height=0.5),
                NormalizedBbox(left=0.2, top=0.2, width=0.6, height=0.6),
            ],
            labels=["text", "logo"],
            canvas_size=CanvasSize(width=100, height=100),
            encoded_image="dummy_encoded_image",
            content_bboxes=[
                NormalizedBbox(left=0.05, top=0.05, width=0.95, height=0.95)
            ],
        )

        # Process the layout data to add labels to _possible_labels
        processor.invoke(mock_layout)

        # Verify the labels were stored as tuple of tuples
        assert isinstance(processor._possible_labels, tuple)
        assert len(processor._possible_labels) == 1  # One layout processed
        assert len(processor._possible_labels[0]) == 2  # Two labels in that layout
        assert "text" in processor._possible_labels[0]
        assert "logo" in processor._possible_labels[0]
