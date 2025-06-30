import numpy as np
import pytest
from layout_prompter.models import (
    Coordinates,
    LayoutSerializedData,
    LayoutSerializedOutputData,
    PosterLayoutSerializedOutputData,
    ProcessedLayoutData,
)
from layout_prompter.settings import CanvasSize
from layout_prompter.visualizers import (
    ContentAgnosticVisualizer,
    ContentAgnosticVisualizerConfig,
    ContentAwareVisualizer,
    ContentAwareVisualizerConfig,
)
from PIL import Image


class TestVisualizerBase:
    @pytest.fixture
    def canvas_size(self) -> CanvasSize:
        return CanvasSize(width=100, height=150)

    @pytest.fixture
    def labels(self) -> list[str]:
        return ["text", "logo", "underlay"]

    @pytest.fixture
    def processed_layout_data(self) -> ProcessedLayoutData:
        return ProcessedLayoutData(
            idx=0,
            bboxes=np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            labels=np.array(["text", "logo"]),
            gold_bboxes=np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            orig_bboxes=np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            orig_labels=np.array(["text", "logo"]),
            discrete_bboxes=np.array(
                [[10, 30, 30, 60], [50, 90, 70, 120]], dtype=np.int32
            ),
            discrete_gold_bboxes=np.array(
                [[10, 30, 30, 60], [50, 90, 70, 120]], dtype=np.int32
            ),
            discrete_content_bboxes=None,
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image=None,
            content_bboxes=None,
        )

    @pytest.fixture
    def serialized_output_data(self) -> LayoutSerializedOutputData:
        return LayoutSerializedOutputData(
            layouts=[
                LayoutSerializedData(
                    class_name="text",
                    coord=Coordinates(left=10, top=30, width=30, height=60),
                ),
                LayoutSerializedData(
                    class_name="logo",
                    coord=Coordinates(left=50, top=90, width=70, height=120),
                ),
            ]
        )


class TestContentAgnosticVisualizer(TestVisualizerBase):
    @pytest.fixture
    def visualizer(
        self, canvas_size: CanvasSize, labels: list[str]
    ) -> ContentAgnosticVisualizer:
        return ContentAgnosticVisualizer(
            canvas_size=canvas_size,
            labels=labels,
            schema=PosterLayoutSerializedOutputData,
        )

    def test_convert_to_serialized_output_data(
        self,
        visualizer: ContentAgnosticVisualizer,
        processed_layout_data: ProcessedLayoutData,
    ):
        result = visualizer._convert_to_serialized_output_data(processed_layout_data)

        assert isinstance(result, LayoutSerializedOutputData)
        assert len(result.layouts) == 2

        # Check first layout
        layout1 = result.layouts[0]
        assert layout1.class_name == "text"
        assert layout1.coord.left == 10
        assert layout1.coord.top == 30
        assert layout1.coord.width == 30
        assert layout1.coord.height == 60

        # Check second layout
        layout2 = result.layouts[1]
        assert layout2.class_name == "logo"
        assert layout2.coord.left == 50
        assert layout2.coord.top == 90

    def test_convert_to_serialized_output_data_missing_labels(
        self, visualizer: ContentAgnosticVisualizer
    ):
        # Test with None labels - should raise assertion error
        processed_data = ProcessedLayoutData(
            idx=0,
            bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            labels=None,  # This should cause assertion error
            gold_bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            orig_bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            orig_labels=np.array(["text"]),
            discrete_bboxes=np.array([[10, 30, 30, 60]], dtype=np.int32),
            discrete_gold_bboxes=np.array([[10, 30, 30, 60]], dtype=np.int32),
            discrete_content_bboxes=None,
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image=None,
            content_bboxes=None,
        )

        with pytest.raises(AssertionError):
            visualizer._convert_to_serialized_output_data(processed_data)

    def test_convert_to_serialized_output_data_missing_discrete_bboxes(
        self, visualizer: ContentAgnosticVisualizer
    ):
        # Test with None discrete_bboxes - should raise assertion error
        processed_data = ProcessedLayoutData(
            idx=0,
            bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            labels=np.array(["text"]),
            gold_bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            orig_bboxes=np.array([[0.1, 0.2, 0.3, 0.4]]),
            orig_labels=np.array(["text"]),
            discrete_bboxes=None,  # This should cause assertion error
            discrete_gold_bboxes=np.array([[10, 30, 30, 60]], dtype=np.int32),
            discrete_content_bboxes=None,
            canvas_size=CanvasSize(width=100, height=150),
            encoded_image=None,
            content_bboxes=None,
        )

        with pytest.raises(AssertionError):
            visualizer._convert_to_serialized_output_data(processed_data)

    def test_get_sorted_layouts_by_area(self, visualizer: ContentAgnosticVisualizer):
        layouts = [
            LayoutSerializedData(
                class_name="small",
                coord=Coordinates(left=0, top=0, width=10, height=10),
            ),  # area = 100
            LayoutSerializedData(
                class_name="large",
                coord=Coordinates(left=0, top=0, width=20, height=30),
            ),  # area = 600
            LayoutSerializedData(
                class_name="medium",
                coord=Coordinates(left=0, top=0, width=15, height=20),
            ),  # area = 300
        ]

        result = visualizer.get_sorted_layouts(layouts)

        # Should be sorted by area in descending order
        assert len(result) == 3
        assert result[0].class_name == "large"  # area = 600
        assert result[1].class_name == "medium"  # area = 300
        assert result[2].class_name == "small"  # area = 100

    def test_get_sorted_layouts_empty_list(self, visualizer: ContentAgnosticVisualizer):
        result = visualizer.get_sorted_layouts([])
        assert result == []

    def test_get_sorted_layouts_same_area(self, visualizer: ContentAgnosticVisualizer):
        layouts = [
            LayoutSerializedData(
                class_name="first",
                coord=Coordinates(left=0, top=0, width=10, height=10),
            ),  # area = 100
            LayoutSerializedData(
                class_name="second",
                coord=Coordinates(left=0, top=0, width=5, height=20),
            ),  # area = 100
        ]

        result = visualizer.get_sorted_layouts(layouts)

        # Should maintain some order when areas are equal
        assert len(result) == 2
        assert {layout.class_name for layout in result} == {"first", "second"}

    def test_draw_layout_bboxes_basic(self, visualizer: ContentAgnosticVisualizer):
        # Create a test image
        image = Image.new("RGB", (200, 200), (255, 255, 255))

        layout = LayoutSerializedData(
            class_name="text", coord=Coordinates(left=10, top=20, width=50, height=30)
        )

        result_image = visualizer.draw_layout_bboxes(image, layout)

        # Should return a PIL Image
        assert isinstance(result_image, Image.Image)
        assert result_image.size == (200, 200)
        # Image should be modified (not same as original)
        assert result_image != image

    def test_draw_layout_bboxes_with_resize_ratio(
        self, visualizer: ContentAgnosticVisualizer
    ):
        image = Image.new("RGB", (200, 200), (255, 255, 255))

        layout = LayoutSerializedData(
            class_name="text", coord=Coordinates(left=10, top=20, width=50, height=30)
        )

        result_image = visualizer.draw_layout_bboxes(image, layout, resize_ratio=2.0)

        assert isinstance(result_image, Image.Image)
        # The actual bbox coordinates should be scaled by resize_ratio in the drawing

    def test_draw_layout_bboxes_unknown_class(
        self, visualizer: ContentAgnosticVisualizer
    ):
        image = Image.new("RGB", (200, 200), (255, 255, 255))

        layout = LayoutSerializedData(
            class_name="unknown_class",  # Not in labels list
            coord=Coordinates(left=10, top=20, width=50, height=30),
        )

        # Should raise ValueError when class not found in labels
        with pytest.raises(ValueError):
            visualizer.draw_layout_bboxes(image, layout)

    def test_invoke_with_processed_layout_data(
        self,
        visualizer: ContentAgnosticVisualizer,
        processed_layout_data: ProcessedLayoutData,
    ):
        config = {"configurable": ContentAgnosticVisualizerConfig().model_dump()}

        result = visualizer.invoke(processed_layout_data, config=config)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 150)  # Should match canvas size

    def test_invoke_with_serialized_output_data(
        self,
        visualizer: ContentAgnosticVisualizer,
        serialized_output_data: LayoutSerializedOutputData,
    ):
        config = {"configurable": ContentAgnosticVisualizerConfig().model_dump()}

        result = visualizer.invoke(serialized_output_data, config=config)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 150)

    def test_invoke_with_custom_resize_ratio(
        self,
        visualizer: ContentAgnosticVisualizer,
        serialized_output_data: LayoutSerializedOutputData,
    ):
        config = {
            "configurable": ContentAgnosticVisualizerConfig(
                resize_ratio=2.0
            ).model_dump()
        }

        result = visualizer.invoke(serialized_output_data, config=config)

        assert isinstance(result, Image.Image)
        assert result.size == (200, 300)  # Should be scaled by resize_ratio

    def test_invoke_with_custom_background_color(
        self,
        visualizer: ContentAgnosticVisualizer,
        serialized_output_data: LayoutSerializedOutputData,
    ):
        config = {
            "configurable": ContentAgnosticVisualizerConfig(
                bg_rgb_color=(255, 0, 0)
            ).model_dump()
        }

        result = visualizer.invoke(serialized_output_data, config=config)

        assert isinstance(result, Image.Image)
        # Background should be red, but we can't easily test the exact color without pixel inspection

    def test_invoke_with_none_config(
        self,
        visualizer: ContentAgnosticVisualizer,
        serialized_output_data: LayoutSerializedOutputData,
    ):
        # Should use default config when None is passed
        result = visualizer.invoke(serialized_output_data, config=None)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 150)


class TestContentAwareVisualizer(TestVisualizerBase):
    @pytest.fixture
    def bg_image(self) -> Image.Image:
        return Image.new("RGB", (100, 150), (128, 128, 128))

    @pytest.fixture
    def content_bboxes(self) -> np.ndarray:
        return np.array([[10, 20, 30, 40], [60, 80, 25, 35]])

    @pytest.fixture
    def visualizer(
        self, canvas_size: CanvasSize, labels: list[str]
    ) -> ContentAwareVisualizer:
        return ContentAwareVisualizer(
            canvas_size=canvas_size,
            labels=labels,
            schema=PosterLayoutSerializedOutputData,
        )

    def test_draw_content_bboxes_basic(
        self,
        visualizer: ContentAwareVisualizer,
        bg_image: Image.Image,
        content_bboxes: np.ndarray,
    ):
        result_image = visualizer.draw_content_bboxes(bg_image, content_bboxes)

        assert isinstance(result_image, Image.Image)
        assert result_image.size == bg_image.size
        # Image should be modified
        assert result_image != bg_image

    def test_draw_content_bboxes_with_resize_ratio(
        self,
        visualizer: ContentAwareVisualizer,
        bg_image: Image.Image,
        content_bboxes: np.ndarray,
    ):
        result_image = visualizer.draw_content_bboxes(
            bg_image, content_bboxes, resize_ratio=1.5
        )

        assert isinstance(result_image, Image.Image)
        # Coordinates should be scaled by resize_ratio

    def test_draw_content_bboxes_custom_font_color(
        self,
        visualizer: ContentAwareVisualizer,
        bg_image: Image.Image,
        content_bboxes: np.ndarray,
    ):
        result_image = visualizer.draw_content_bboxes(
            bg_image, content_bboxes, font_color=(255, 0, 0)
        )

        assert isinstance(result_image, Image.Image)

    def test_draw_content_bboxes_empty_array(
        self, visualizer: ContentAwareVisualizer, bg_image: Image.Image
    ):
        empty_bboxes = np.empty((0, 4))

        result_image = visualizer.draw_content_bboxes(bg_image, empty_bboxes)

        assert isinstance(result_image, Image.Image)
        # With no bboxes, image should be nearly identical (just copied)

    def test_invoke_with_processed_layout_data(
        self,
        visualizer: ContentAwareVisualizer,
        processed_layout_data: ProcessedLayoutData,
        bg_image: Image.Image,
        content_bboxes: np.ndarray,
    ):
        config = {
            "configurable": ContentAwareVisualizerConfig(
                bg_image=bg_image, content_bboxes=content_bboxes
            ).model_dump()
        }

        result = visualizer.invoke(processed_layout_data, config=config)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 150)

    def test_invoke_with_serialized_output_data(
        self,
        visualizer: ContentAwareVisualizer,
        serialized_output_data: LayoutSerializedOutputData,
        bg_image: Image.Image,
    ):
        config = {
            "configurable": ContentAwareVisualizerConfig(bg_image=bg_image).model_dump()
        }

        result = visualizer.invoke(serialized_output_data, config=config)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 150)

    def test_invoke_without_content_bboxes(
        self,
        visualizer: ContentAwareVisualizer,
        serialized_output_data: LayoutSerializedOutputData,
        bg_image: Image.Image,
    ):
        # Test with content_bboxes = None
        config = {
            "configurable": ContentAwareVisualizerConfig(
                bg_image=bg_image, content_bboxes=None
            ).model_dump()
        }

        result = visualizer.invoke(serialized_output_data, config=config)

        assert isinstance(result, Image.Image)
        # Should work fine without content bboxes

    def test_invoke_with_resize_ratio(
        self,
        visualizer: ContentAwareVisualizer,
        serialized_output_data: LayoutSerializedOutputData,
        bg_image: Image.Image,
    ):
        config = {
            "configurable": ContentAwareVisualizerConfig(
                bg_image=bg_image, resize_ratio=0.5
            ).model_dump()
        }

        result = visualizer.invoke(serialized_output_data, config=config)

        assert isinstance(result, Image.Image)
        assert result.size == (50, 75)  # Should be half the original size

    def test_invoke_with_different_bg_image_mode(
        self,
        visualizer: ContentAwareVisualizer,
        serialized_output_data: LayoutSerializedOutputData,
    ):
        # Test with RGBA background image
        rgba_image = Image.new("RGBA", (100, 150), (128, 128, 128, 255))
        config = {
            "configurable": ContentAwareVisualizerConfig(
                bg_image=rgba_image
            ).model_dump()
        }

        result = visualizer.invoke(serialized_output_data, config=config)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"  # Should be converted to RGB

    def test_invoke_image_resizing(
        self,
        visualizer: ContentAwareVisualizer,
        serialized_output_data: LayoutSerializedOutputData,
    ):
        # Test with background image of different size
        large_bg_image = Image.new("RGB", (200, 300), (128, 128, 128))
        config = {
            "configurable": ContentAwareVisualizerConfig(
                bg_image=large_bg_image
            ).model_dump()
        }

        result = visualizer.invoke(serialized_output_data, config=config)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 150)  # Should be resized to canvas size
