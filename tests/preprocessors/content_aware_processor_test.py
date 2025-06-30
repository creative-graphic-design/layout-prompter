import pytest
from layout_prompter.models import LayoutData
from layout_prompter.preprocessors import ContentAwareProcessor
from layout_prompter.utils.testing import LayoutPrompterTestCase
from tqdm.auto import tqdm

import datasets as ds


class TestContentAwareProcessor(LayoutPrompterTestCase):
    @pytest.fixture
    def num_proc(self) -> int:
        return 32

    def test_content_aware_processor(self, hf_dataset: ds.DatasetDict, num_proc: int):
        dataset = {
            split: [
                LayoutData.model_validate(data)
                for data in tqdm(hf_dataset[split], desc=f"Processing for {split}")
            ]
            for split in hf_dataset
        }
        processor = ContentAwareProcessor()
        processed_dataset = processor.invoke(
            dataset,
            config={"configurable": {"num_proc": num_proc}},
        )
        assert isinstance(processed_dataset, dict)
