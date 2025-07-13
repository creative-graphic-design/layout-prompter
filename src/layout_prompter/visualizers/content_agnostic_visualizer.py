from typing import Any, Optional, Union

from langchain_core.runnables.config import RunnableConfig
from PIL import Image

from layout_prompter.models import (
    LayoutSerializedOutputData,
    ProcessedLayoutData,
)
from layout_prompter.typehints import PilImage

from .base import Visualizer, VisualizerConfig


class ContentAgnosticVisualizerConfig(VisualizerConfig):
    """Configuration for ContentAgnosticVisualizer."""


class ContentAgnosticVisualizer(Visualizer):
    name: str = "content-agnostic-visualizer"

    def invoke(
        self,
        input: Union[ProcessedLayoutData, LayoutSerializedOutputData],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> PilImage:
        # Load the runtime configuration
        conf = ContentAgnosticVisualizerConfig.from_runnable_config(config)

        # Convert the input to SerializedOutputData if needed
        if isinstance(input, ProcessedLayoutData):
            input = self._convert_to_serialized_output_data(input)

        # Resize the canvas based on the configuration
        canvas_w = int(self.canvas_size.width * conf.resize_ratio)
        canvas_h = int(self.canvas_size.height * conf.resize_ratio)

        # Prepare canvas for drawing
        image = Image.new(
            "RGB",
            size=(canvas_w, canvas_h),
            color=conf.bg_rgb_color,
        )

        # Get the sorted layouts by area of the bboxes
        layouts = self.get_sorted_layouts(input.layouts)

        # Draw the layout bboxes
        for layout in layouts:
            image = self.draw_layout_bboxes(
                image=image,
                layout=layout,
                resize_ratio=conf.resize_ratio,
                font_color=conf.bg_rgb_color,
            )
        return image
