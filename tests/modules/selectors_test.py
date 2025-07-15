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

        processor_chain = processor | bbox_discretizer

        examples = cast(
            List[ProcessedLayoutData],
            processor_chain.batch(
                inputs=tng_dataset,
                config={
                    "configurable": {"target_canvas_size": settings.canvas_size},
                    "max_concurrency": get_num_workers(max_concurrency=4),
                    "callbacks": [ProgressBarCallback(total=len(tng_dataset))],
                },
            ),
        )

        selector = ContentAwareSelector(examples=examples)

        test_data = cast(
            ProcessedLayoutData,
            processor_chain.invoke(
                input=tst_dataset[0],
                config={
                    "configurable": {"target_canvas_size": settings.canvas_size},
                },
            ),
        )
        candidates = selector.select_examples(test_data)

        assert len(candidates.selected_examples) == num_prompt
