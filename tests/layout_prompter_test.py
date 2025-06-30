from typing import Dict, List, Type, cast

import pytest
from langchain.chat_models import init_chat_model
from layout_prompter import LayoutPrompter
from layout_prompter.models import (
    LayoutData,
    LayoutSerializedData,
    LayoutSerializedOutputData,
    PosterLayoutSerializedData,
    PosterLayoutSerializedOutputData,
    ProcessedLayoutData,
    Rico25SerializedData,
    Rico25SerializedOutputData,
)
from layout_prompter.modules import (
    ContentAwareSelector,
    ContentAwareSerializer,
    LayoutPrompterRanker,
)
from layout_prompter.preprocessors import ContentAwareProcessor
from layout_prompter.settings import PosterLayoutSettings, Rico25Settings, TaskSettings
from layout_prompter.utils.testing import LayoutPrompterTestCase
from layout_prompter.visualizers import ContentAwareVisualizer


class TestLayoutPrompter(LayoutPrompterTestCase):
    @pytest.fixture
    def num_prompt(self) -> int:
        return 10

    @pytest.fixture
    def num_return(self) -> int:
        return 10

    @pytest.fixture
    def model_provider(self) -> str:
        return "openai"

    @pytest.fixture
    def model_id(self) -> str:
        return "gpt-4o"

    @pytest.mark.parametrize(
        argnames=("settings", "input_schema", "output_schema"),
        argvalues=(
            (
                Rico25Settings(),
                Rico25SerializedData,
                Rico25SerializedOutputData,
            ),
        ),
    )
    def test_content_agnostic_generation(
        self,
        layout_dataset: Dict[str, List[LayoutData]],
        num_prompt: int,
        num_return: int,
        model_provider: str,
        model_id: str,
        settings: TaskSettings,
        input_schema: Type[LayoutSerializedData],
        output_schema: Type[LayoutSerializedOutputData],
    ):
        pass

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
    def test_content_aware_generation(
        self,
        layout_dataset: Dict[str, List[LayoutData]],
        num_prompt: int,
        num_return: int,
        model_provider: str,
        model_id: str,
        settings: TaskSettings,
        input_schema: Type[LayoutSerializedData],
        output_schema: Type[LayoutSerializedOutputData],
    ):
        tng_dataset = layout_dataset["train"]
        tst_dataset = layout_dataset["test"]

        # Define the content-aware processor and process the data for candidates
        processor = ContentAwareProcessor()
        examples = cast(
            List[ProcessedLayoutData],
            processor.invoke(input=tng_dataset),
        )

        # Select a random test example
        # idx = random.choice(range(len(test_dataset)))
        idx = 443
        test_data = tst_dataset[idx]

        # Process the test data
        processed_test_data = cast(
            ProcessedLayoutData, processor.invoke(input=test_data)
        )

        # Define the LayoutPrompter
        layout_prompter = LayoutPrompter(
            selector=ContentAwareSelector(
                num_prompt=num_prompt,
                canvas_size=settings.canvas_size,
                examples=examples,
            ),
            serializer=ContentAwareSerializer(
                layout_domain=settings.domain,
                schema=input_schema,
            ),
            llm=init_chat_model(
                model_provider=model_provider,
                model=model_id,
            ),
            ranker=LayoutPrompterRanker(),
            schema=output_schema,
        )

        # Invoke the LayoutPrompter
        layout_prompter_config = {
            "num_return": num_return,
        }
        outputs = layout_prompter.invoke(
            input=processed_test_data, config={"configurable": layout_prompter_config}
        )

        # Define the visualizer
        visualizer = ContentAwareVisualizer(
            canvas_size=settings.canvas_size, labels=settings.labels
        )
        visualizer_config = {
            "resize_ratio": 2.0,
            "bg_image": test_data.content_image,
            "content_bboxes": processed_test_data.discrete_content_bboxes,
        }

        # Create the save directory
        save_dir = self.PROJECT_ROOT / "generated" / "content_aware"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the generated layout-rendering images
        for i, output in enumerate(outputs):
            image = visualizer.invoke(
                output,
                config={"configurable": visualizer_config},
            )
            image.save(save_dir / f"{idx=},{i=}.png")


class TestContentAgnosticGeneration(LayoutPrompterTestCase):
    @pytest.fixture
    def num_prompt(self) -> int:
        return 10

    @pytest.fixture
    def num_return(self) -> int:
        return 10

    @pytest.fixture
    def model_provider(self) -> str:
        return "openai"

    @pytest.fixture
    def model_id(self) -> str:
        return "gpt-4o"
