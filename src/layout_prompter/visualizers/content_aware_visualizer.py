from typing import Any, List, Optional, Tuple, Union

from langchain_core.runnables.config import RunnableConfig
from PIL import Image, ImageDraw

from layout_prompter.models import (
    Bbox,
    LayoutSerializedOutputData,
    ProcessedLayoutData,
)
from layout_prompter.typehints import PilImage

from .base import Visualizer, VisualizerConfig


class ContentAwareVisualizerConfig(VisualizerConfig):
    """Configuration for ContentAwareVisualizer."""

    bg_image: PilImage
    content_bboxes: Optional[List[Bbox]] = None


class ContentAwareVisualizer(Visualizer):
    name: str = "content-aware-visualizer"

    def draw_content_bboxes(
        self,
        image: PilImage,
        content_bboxes: List[Bbox],
        resize_ratio: float = 1.0,
        font_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> PilImage:
        image = image.copy()
        draw = ImageDraw.Draw(image, mode="RGBA")

        for bbox in content_bboxes:
            x1, y1, x2, y2 = bbox.to_ltrb()

            x1, y1, x2, y2 = (
                int(x1 * resize_ratio),
                int(y1 * resize_ratio),
                int(x2 * resize_ratio),
                int(y2 * resize_ratio),
            )

            draw.rectangle(
                xy=(x1, y1, x2, y2),
                fill=(0, 0, 0, 50),
                outline=(0, 0, 0, 100),
            )
            draw.text(xy=(x1, y1), text="content", fill=font_color)

        return image

    def invoke(
        self,
        input: Union[ProcessedLayoutData, LayoutSerializedOutputData],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> PilImage:
        # Load the runtime configuration
        conf = ContentAwareVisualizerConfig.from_runnable_config(config)

        # Convert the input to SerializedOutputData if needed
        if isinstance(input, ProcessedLayoutData):
            input = self._convert_to_serialized_output_data(input)

        # Resize the canvas based on the configuration
        canvas_w = int(self.canvas_size.width * conf.resize_ratio)
        canvas_h = int(self.canvas_size.height * conf.resize_ratio)

        # Prepare canvas image for drawing
        # Note here that copy the background image
        # to avoid race conditions in batch processing
        image = conf.bg_image.copy()
        image = image.convert("RGB")
        image = image.resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)

        # Get the sorted layouts by area of the bboxes
        layouts = self.get_sorted_layouts(input.layouts)

        # Draw the content bboxes if they passed
        image = (
            self.draw_content_bboxes(
                image=image,
                content_bboxes=conf.content_bboxes,
                resize_ratio=conf.resize_ratio,
                font_color=conf.bg_rgb_color,
            )
            if conf.content_bboxes is not None
            else image
        )

        # Draw the layout bboxes
        for layout in layouts:
            image = self.draw_layout_bboxes(
                image=image,
                layout=layout,
                resize_ratio=conf.resize_ratio,
                font_color=conf.bg_rgb_color,
            )

        return image
