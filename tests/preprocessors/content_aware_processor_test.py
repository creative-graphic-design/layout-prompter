import numpy as np
import pytest
from layout_prompter.models import LayoutData
from layout_prompter.preprocessors import ContentAwareProcessor
from layout_prompter.settings import CanvasSize
from layout_prompter.utils.testing import LayoutPrompterTestCase

import datasets as ds


class TestContentAwareProcessor(LayoutPrompterTestCase):
    @pytest.fixture
    def num_proc(self) -> int:
        return 1  # Reduced for testing

    def test_content_aware_processor(self, hf_dataset: ds.DatasetDict, num_proc: int):
        # Test with just a few samples to avoid timeout
        dataset = {
            split: [
                LayoutData.model_validate(data)
                for data in list(hf_dataset[split])[:2]  # Only take first 2 samples
            ]
            for split in hf_dataset
        }
        processor = ContentAwareProcessor(canvas_size=CanvasSize(width=100, height=100))

        # Process each split separately since batch expects a list of LayoutData, not a dict
        processed_dataset = {}
        for split, layout_list in dataset.items():
            processed_list = processor.batch(
                layout_list,
                config={"configurable": {"num_proc": num_proc}},
            )
            processed_dataset[split] = processed_list

        assert isinstance(processed_dataset, dict)

    def test_content_aware_processor_hashable(self):
        """Test that ContentAwareProcessor is hashable"""
        processor1 = ContentAwareProcessor(
            canvas_size=CanvasSize(width=100, height=100)
        )
        processor2 = ContentAwareProcessor(
            canvas_size=CanvasSize(width=100, height=100)
        )

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
        processor = ContentAwareProcessor(canvas_size=CanvasSize(width=100, height=100))

        # Attempting to set attributes should raise an error
        with pytest.raises(Exception):  # ValidationError or similar
            processor.max_element_numbers = 20

    def test_content_aware_processor_possible_labels_tuple(self):
        """Test that _possible_labels is properly handled as tuple"""
        processor = ContentAwareProcessor(canvas_size=CanvasSize(width=100, height=100))

        # Initially should be empty tuple
        assert processor._possible_labels == tuple()

        # Create mock layout data with labels
        mock_layout = LayoutData(
            bboxes=np.array([[10, 10, 50, 50], [20, 20, 60, 60]]),
            labels=np.array(["text", "logo"]),
            canvas_size=CanvasSize(width=100, height=100),
            encoded_image="dummy_encoded_image",
            content_bboxes=np.array([[5, 5, 95, 95]]),
        )

        # Process the layout data to add labels to _possible_labels
        processor.invoke(mock_layout)

        # Verify the labels were stored as tuple of tuples
        assert isinstance(processor._possible_labels, tuple)
        assert len(processor._possible_labels) == 1  # One layout processed
        assert len(processor._possible_labels[0]) == 2  # Two labels in that layout
        assert "text" in processor._possible_labels[0]
        assert "logo" in processor._possible_labels[0]
