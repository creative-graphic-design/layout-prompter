import random

import datasets as ds
import pytest
from langchain.chat_models import init_chat_model

from layout_prompter import LayoutPrompter
from layout_prompter.models import ProcessedLayoutData
from layout_prompter.modules.rankers import LayoutPrompterRanker
from layout_prompter.modules.selectors import ContentAwareSelector
from layout_prompter.modules.serializers import ContentAwareSerializer
from layout_prompter.settings import PosterLayoutSettings
from layout_prompter.utils.testing import LayoutPrompterTestCase
from layout_prompter.visualizers import ContentAwareVisualizer


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

    def test_content_aware_generation(self, dataset: ds.DatasetDict, num_prompt: int):
        settings = PosterLayoutSettings()

        examples = [ProcessedLayoutData(**example) for example in dataset["train"]]

        # idx = random.choice(range(len(dataset["test"])))
        idx = 309
        print(f"{idx=}")

        test_data = ProcessedLayoutData(**dataset["test"][idx])

        layout_prompter = LayoutPrompter(
            selector=ContentAwareSelector(
                num_prompt=num_prompt,
                canvas_size=settings.canvas_size,
                examples=examples,
            ),
            serializer=ContentAwareSerializer(
                layout_domain=settings.domain,
            ),
            llm=init_chat_model(
                model_provider="openai",
                model="gpt-4o",
            ),
            ranker=LayoutPrompterRanker(),
        )
        outputs = layout_prompter.invoke(input=test_data)

        visualizer = ContentAwareVisualizer(
            canvas_size=settings.canvas_size, labels=settings.labels
        )
        visualizer_config = {
            "resize_ratio": 2.0,
            "bg_image": test_data.content_image,
            "content_bboxes": test_data.discrete_content_bboxes,
        }

        save_dir = self.PROJECT_ROOT / "generated" / "content_aware"
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, output in enumerate(outputs):
            image = visualizer.invoke(
                output,
                config={"configurable": visualizer_config},
            )
            image.save(save_dir / f"{idx=},{i=}.png")
