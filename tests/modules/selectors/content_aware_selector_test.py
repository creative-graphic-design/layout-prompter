from typing import Dict, List, cast

import pytest
from langchain.smith.evaluation.progress import ProgressBarCallback
from pytest_lazy_fixtures import lf

from layout_prompter.models import LayoutData, ProcessedLayoutData
from layout_prompter.modules.selectors import ContentAwareSelector
from layout_prompter.preprocessors import ContentAwareProcessor
from layout_prompter.settings import PosterLayoutSettings, TaskSettings
from layout_prompter.transforms import DiscretizeBboxes
from layout_prompter.utils import get_num_workers
from layout_prompter.utils.testing import LayoutPrompterTestCase


class TestContentAwareSelector(LayoutPrompterTestCase):
    @pytest.fixture
    def processor(self) -> ContentAwareProcessor:
        return ContentAwareProcessor()

    @pytest.fixture
    def bbox_discretizer(self) -> DiscretizeBboxes:
        return DiscretizeBboxes()

    @pytest.fixture
    def num_prompt(self) -> int:
        return 10

    @pytest.mark.parametrize(
        argnames=("layout_dataset", "settings"),
        argvalues=(
            (
                lf("poster_layout_dataset"),
                PosterLayoutSettings(),
            ),
        ),
    )
    def test_content_aware_selector(
        self,
        layout_dataset: Dict[str, List[LayoutData]],
        settings: TaskSettings,
        processor: ContentAwareProcessor,
        bbox_discretizer: DiscretizeBboxes,
        num_prompt: int,
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
                config={
                    "configurable": {"target_canvas_size": target_canvas_size},
                },
            ),
        )
        processed_test_data = bbox_discretizer.invoke(
            processed_test_data,
            config={
                "configurable": {"target_canvas_size": target_canvas_size},
            },
        )

        # Perform selection using the ContentAwareSelector
        selector = ContentAwareSelector(examples=candidate_examples)
        selector_output = selector.select_examples(processed_test_data)

        assert len(selector_output.selected_examples) == num_prompt