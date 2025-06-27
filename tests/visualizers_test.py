import random

import pytest
from langchain.chat_models import init_chat_model
from layout_prompter.models import ProcessedLayoutData, SerializedOutputData
from layout_prompter.modules.selectors import ContentAwareSelector
from layout_prompter.modules.serializers import (
    ContentAwareSerializer,
    LayoutSerializerInput,
)
from layout_prompter.settings import PosterLayoutSettings
from layout_prompter.utils.testing import LayoutPrompterTestCase
from layout_prompter.visualizers import ContentAwareVisualizer, generate_color_palette

import datasets as ds


class TestContentAwareVisualizer(LayoutPrompterTestCase):
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

    @pytest.fixture
    def num_colors(self) -> int:
        return 3

    def test_generate_color_palette(self, num_colors: int):
        palette = generate_color_palette(num_colors)
        assert len(palette) == num_colors

    def test_content_aware_visualizer(
        self, dataset: ds.DatasetDict, num_prompt: int, num_return: int
    ):
        settings = PosterLayoutSettings()

        selector = ContentAwareSelector(
            num_prompt=num_prompt,
            canvas_size=settings.canvas_size,
            examples=[ProcessedLayoutData(**example) for example in dataset["train"]],
        )

        idx = random.choice(range(len(dataset["test"])))

        test_data = ProcessedLayoutData(**dataset["test"][idx])
        candidates = selector.select_examples(test_data)

        serializer = ContentAwareSerializer(
            layout_domain=settings.domain,
        )
        llm = init_chat_model(
            model_provider="openai",
            model="gpt-4o",
            n=num_return,
        )

        visualizer = ContentAwareVisualizer(
            canvas_size=settings.canvas_size,
            labels=settings.labels,
        )
        chain = (
            serializer | llm.with_structured_output(SerializedOutputData) | visualizer
        )
        image = chain.invoke(
            input=LayoutSerializerInput(query=test_data, candidates=candidates),
            config={
                "configurable": {
                    "resize_ratio": 2.0,
                    "bg_image": test_data.content_image,
                    "content_bboxes": test_data.discrete_content_bboxes,
                }
            },
        )
        image.save(f"generated_{idx}.png")
        image.save("generated.png")
