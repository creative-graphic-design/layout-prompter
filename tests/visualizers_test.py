from typing import Dict, List, Type, cast

import pytest
from langchain.chat_models import init_chat_model

from layout_prompter.models import (
    CanvasSize,
    Bbox,
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
    def processor(self, settings: TaskSettings) -> ContentAwareProcessor:
        return ContentAwareProcessor(target_canvas_size=settings.canvas_size)

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

    @pytest.mark.skip(reason="End-to-end test with OpenAI API - too slow for CI")
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
            processor.batch(inputs=tng_dataset),
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

        selector_output = selector.select_examples(test_data)

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
            input=LayoutSerializerInput(
                query=test_data, candidates=selector_output.selected_examples
            ),
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

    def test_content_aware_visualizer_init(self):
        """Test ContentAwareVisualizer initialization."""
        canvas_size = CanvasSize(width=100, height=150)
        labels = ["text", "logo", "underlay"]
        
        visualizer = ContentAwareVisualizer(
            canvas_size=canvas_size,
            labels=labels,
            schema=PosterLayoutSerializedOutputData,
        )
        
        assert visualizer.canvas_size == canvas_size
        assert visualizer.labels == labels
        assert visualizer.schema == PosterLayoutSerializedOutputData

    def test_content_aware_visualizer_get_sorted_layouts(self):
        """Test ContentAwareVisualizer layout sorting functionality."""
        canvas_size = CanvasSize(width=100, height=150)
        labels = ["text", "logo", "underlay"]
        
        visualizer = ContentAwareVisualizer(
            canvas_size=canvas_size,
            labels=labels,
            schema=PosterLayoutSerializedOutputData,
        )
        
        # Create test layouts with different sizes
        layouts = [
            LayoutSerializedData(
                class_name="small",
                bbox=Bbox(left=0, top=0, width=10, height=10),  # area = 100
            ),
            LayoutSerializedData(
                class_name="large", 
                bbox=Bbox(left=0, top=0, width=20, height=30),  # area = 600
            ),
            LayoutSerializedData(
                class_name="medium",
                bbox=Bbox(left=0, top=0, width=15, height=20),  # area = 300
            ),
        ]
        
        sorted_layouts = visualizer.get_sorted_layouts(layouts)
        
        # Should be sorted by area in descending order
        assert len(sorted_layouts) == 3
        assert sorted_layouts[0].class_name == "large"   # area = 600
        assert sorted_layouts[1].class_name == "medium"  # area = 300
        assert sorted_layouts[2].class_name == "small"   # area = 100
