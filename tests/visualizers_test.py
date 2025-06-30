from typing import Dict, List, Type, cast

import pytest
from langchain.chat_models import init_chat_model
from layout_prompter.models import (
    LayoutData,
    LayoutSerializedData,
    LayoutSerializedOutputData,
    PosterLayoutSerializedData,
    PosterLayoutSerializedOutputData,
    ProcessedLayoutData,
)
from layout_prompter.modules.selectors import ContentAwareSelector
from layout_prompter.modules.serializers import (
    ContentAwareSerializer,
    LayoutSerializerInput,
)
from layout_prompter.preprocessors import ContentAwareProcessor
from layout_prompter.settings import PosterLayoutSettings, TaskSettings
from layout_prompter.utils.testing import LayoutPrompterTestCase
from layout_prompter.visualizers import ContentAwareVisualizer, generate_color_palette


class TestContentAwareVisualizer(LayoutPrompterTestCase):
    @pytest.fixture
    def processor(self) -> ContentAwareProcessor:
        return ContentAwareProcessor()

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

    @pytest.mark.parametrize(
        argnames=("settings", "input_schema", "output_schema"),
        argvalues=(
            (
                PosterLayoutSettings(),
                PosterLayoutSerializedData,
                PosterLayoutSerializedOutputData,
            ),
        ),
    )
    def test_content_aware_visualizer(
        self,
        layout_dataset: Dict[str, List[LayoutData]],
        processor: ContentAwareProcessor,
        settings: TaskSettings,
        input_schema: Type[LayoutSerializedData],
        output_schema: Type[LayoutSerializedOutputData],
        num_prompt: int,
        num_return: int,
    ):
        tng_dataset, tst_dataset = layout_dataset["train"], layout_dataset["test"]

        examples = cast(
            List[ProcessedLayoutData],
            processor.invoke(input=tng_dataset),
        )

        selector = ContentAwareSelector(
            num_prompt=num_prompt,
            canvas_size=settings.canvas_size,
            examples=examples,
        )

        # idx = random.choice(range(len(tst_dataset)))
        idx = 0
        test_data = cast(
            ProcessedLayoutData,
            processor.invoke(input=tst_dataset[idx]),
        )

        candidates = selector.select_examples(test_data)

        serializer = ContentAwareSerializer(
            layout_domain=settings.domain,
            schema=input_schema,
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
        chain = serializer | llm.with_structured_output(output_schema) | visualizer

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
