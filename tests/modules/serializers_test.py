from typing import Dict, List, Type, cast

import pytest
from layout_prompter.models import (
    LayoutData,
    LayoutSerializedData,
    PosterLayoutSerializedData,
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


class TestContentAwareSerializer(LayoutPrompterTestCase):
    @pytest.fixture
    def processor(self) -> ContentAwareProcessor:
        return ContentAwareProcessor()

    @pytest.mark.parametrize(
        argnames=("settings", "input_schema"),
        argvalues=(
            (
                PosterLayoutSettings(),
                PosterLayoutSerializedData,
            ),
        ),
    )
    def test_content_aware_serializer(
        self,
        layout_dataset: Dict[str, List[LayoutData]],
        processor: ContentAwareProcessor,
        settings: TaskSettings,
        input_schema: Type[LayoutSerializedData],
    ):
        tng_dataset, tst_dataset = layout_dataset["train"], layout_dataset["test"]

        examples = cast(
            List[ProcessedLayoutData],
            processor.invoke(input=tng_dataset),
        )
        selector = ContentAwareSelector(
            canvas_size=settings.canvas_size,
            examples=examples,
        )

        tst_data = tst_dataset[0]
        processed_test_data = cast(
            ProcessedLayoutData, processor.invoke(input=tst_data)
        )
        candidates = selector.select_examples(processed_test_data)

        serializer = ContentAwareSerializer(
            layout_domain=settings.domain,
            schema=input_schema,
        )
        prompt = serializer.invoke(
            input=LayoutSerializerInput(
                query=processed_test_data, candidates=candidates
            )
        )
        for message in prompt.to_messages():
            message.pretty_print()
