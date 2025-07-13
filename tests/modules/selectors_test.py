from typing import Dict, List, cast

import pytest
from langchain.smith.evaluation.progress import ProgressBarCallback

from layout_prompter.models import LayoutData, ProcessedLayoutData
from layout_prompter.modules.selectors import ContentAwareSelector
from layout_prompter.preprocessors import ContentAwareProcessor
from layout_prompter.settings import PosterLayoutSettings, TaskSettings
from layout_prompter.utils import get_num_workers
from layout_prompter.utils.testing import LayoutPrompterTestCase


class TestContentAwareSelector(LayoutPrompterTestCase):
    @pytest.fixture
    def processor(self, settings: TaskSettings) -> ContentAwareProcessor:
        return ContentAwareProcessor(target_canvas_size=settings.canvas_size)

    @pytest.fixture
    def num_prompt(self) -> int:
        return 10

    @pytest.mark.parametrize(
        argnames="settings",
        argvalues=(PosterLayoutSettings(),),
    )
    def test_content_aware_selector(
        self,
        layout_dataset: Dict[str, List[LayoutData]],
        processor: ContentAwareProcessor,
        settings: TaskSettings,
        num_prompt: int,
    ):
        tng_dataset, tst_dataset = layout_dataset["train"], layout_dataset["test"]

        examples = cast(
            List[ProcessedLayoutData],
            processor.batch(
                inputs=tng_dataset,
                config={
                    "max_concurrency": get_num_workers(max_concurrency=4),
                    "callbacks": [ProgressBarCallback(total=len(tng_dataset))],
                },
            ),
        )
        selector = ContentAwareSelector(
            examples=examples,
        )
        test_data = cast(
            ProcessedLayoutData,
            processor.invoke(input=tst_dataset[0]),
        )
        candidates = selector.select_examples(test_data)

        assert len(candidates.selected_examples) == num_prompt
