import pytest
from layout_prompter.models import ProcessedLayoutData
from layout_prompter.modules.selectors import ContentAwareSelector
from layout_prompter.modules.serializers import (
    ContentAwareSerializer,
    LayoutSerializerInput,
)
from layout_prompter.settings import PosterLayoutSettings
from layout_prompter.utils.testing import LayoutPrompterTestCase

import datasets as ds


class TestContentAwareSerializer(LayoutPrompterTestCase):
    @pytest.fixture
    def dataset(self) -> ds.DatasetDict:
        dataset_dir = self.FIXTURES_ROOT / "datasets" / "poster-layout" / "processed"
        dataset = ds.load_from_disk(dataset_dir)
        assert isinstance(dataset, ds.DatasetDict)
        return dataset

    @pytest.fixture
    def num_prompt(self) -> int:
        return 10

    def test_content_aware_serializer(self, dataset: ds.DatasetDict):
        settings = PosterLayoutSettings()
        selector = ContentAwareSelector(
            canvas_size=settings.canvas_size,
            examples=[ProcessedLayoutData(**example) for example in dataset["train"]],
        )
        test_data = ProcessedLayoutData(**dataset["test"][0])
        candidates = selector.select_examples(test_data)

        serializer = ContentAwareSerializer()

        prompt = serializer.invoke(
            input=LayoutSerializerInput(query=test_data, candidates=candidates)
        )
        for message in prompt.to_messages():
            message.pretty_print()
