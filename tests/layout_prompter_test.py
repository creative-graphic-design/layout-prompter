import datasets as ds
import pytest
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from layout_prompter.models import (
    MultipleSerializedOutputData,
    ProcessedLayoutData,
    SerializedOutputData,
)
from layout_prompter.modules.selectors import ContentAwareSelector
from layout_prompter.modules.serializers import ContentAwareSerializer, SerializerInput
from layout_prompter.settings import PosterLayoutSettings
from layout_prompter.testing import LayoutPrompterTestCase


class TestContentAwareGeneration(LayoutPrompterTestCase):
    @pytest.fixture
    def dataset(self) -> ds.DatasetDict:
        dataset_dir = self.FIXTURES_ROOT / "datasets" / "poster-layout" / "processed"
        dataset = ds.load_from_disk(dataset_dir)
        assert isinstance(dataset, ds.DatasetDict)
        return dataset

    @pytest.fixture
    def num_prompt(self) -> int:
        return 10

    @pytest.fixture
    def num_return(self) -> int:
        return 10

    def test_content_aware_generation(
        self, dataset: ds.DatasetDict, num_prompt: int, num_return: int
    ):
        settings = PosterLayoutSettings()

        selector = ContentAwareSelector(
            num_prompt=num_prompt,
            canvas_size=settings.canvas_size,
            examples=[ProcessedLayoutData(**example) for example in dataset["train"]],
        )
        test_data = ProcessedLayoutData(**dataset["test"][0])
        candidates = selector.select_examples(test_data)

        serializer = ContentAwareSerializer()
        llm = init_chat_model(model_provider="openai", model="gpt-4o", n=num_return)

        chain = serializer | llm.with_structured_output(SerializedOutputData)

        output = chain.invoke(
            input=SerializerInput(query=test_data, candidates=candidates)
        )

        # for output in chain.stream(
        #     input=SerializerInput(query=test_data, candidates=candidates)
        # ):
        #     print(output)

        breakpoint()
