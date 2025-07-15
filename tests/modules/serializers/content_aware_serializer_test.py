from typing import Dict, List, Type, cast

import pytest
from langchain.smith.evaluation.progress import ProgressBarCallback
from pytest_lazy_fixtures import lf

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
from layout_prompter.transforms import DiscretizeBboxes
from layout_prompter.utils import get_num_workers
from layout_prompter.utils.testing import LayoutPrompterTestCase


class TestContentAwareSerializer(LayoutPrompterTestCase):
    @pytest.fixture
    def processor(self) -> ContentAwareProcessor:
        return ContentAwareProcessor()

    @pytest.fixture
    def bbox_discretizer(self) -> DiscretizeBboxes:
        return DiscretizeBboxes()

    @pytest.mark.parametrize(
        argnames=("layout_dataset", "settings", "input_schema"),
        argvalues=(
            (
                lf("poster_layout_dataset"),
                PosterLayoutSettings(),
                PosterLayoutSerializedData,
            ),
        ),
    )
    def test_content_aware_serializer(
        self,
        layout_dataset: Dict[str, List[LayoutData]],
        processor: ContentAwareProcessor,
        bbox_discretizer: DiscretizeBboxes,
        settings: TaskSettings,
        input_schema: Type[LayoutSerializedData],
    ):
        tng_dataset, tst_dataset = layout_dataset["train"], layout_dataset["test"]
        target_canvas_size = settings.canvas_size

        # Perform preprocessing on candidate examples and inference example
        candidate_examples = cast(
            List[ProcessedLayoutData],
            processor.batch(
                inputs=tng_dataset,
                config={
                    "configurable": {"target_canvas_size": target_canvas_size},
                    "max_concurrency": get_num_workers(max_concurrency=4),
                    "callbacks": [ProgressBarCallback(total=len(tng_dataset))],
                },
            ),
        )
        processed_test_data = cast(
            ProcessedLayoutData,
            processor.invoke(input=tst_dataset[0]),
        )

        # Apply the bbox discretizer to the candidate examples and the inference example
        candidate_examples = cast(
            List[ProcessedLayoutData],
            bbox_discretizer.batch(
                candidate_examples,
                config={"configurable": {"target_canvas_size": target_canvas_size}},
            ),
        )
        processed_test_data = bbox_discretizer.invoke(
            processed_test_data,
            config={"configurable": {"target_canvas_size": target_canvas_size}},
        )

        # Define the ContentAwareSerializer to select examples
        selector = ContentAwareSelector(examples=candidate_examples)
        selector_output = selector.select_examples(processed_test_data)

        # Define the ContentAwareSerializer to serialize the selected examples
        serializer = ContentAwareSerializer(layout_domain=settings.domain)

        # Apply the serializer to the examples
        prompt = serializer.invoke(
            input=LayoutSerializerInput(
                query=processed_test_data,
                candidates=selector_output.selected_examples,
            ),
            config={
                "configurable": {
                    "input_schema": input_schema,
                }
            },
        )
        for message in prompt.to_messages():
            message.pretty_print()
