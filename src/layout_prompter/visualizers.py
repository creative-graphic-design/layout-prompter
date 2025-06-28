from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type, Union

import numpy as np
import pydantic_numpy.typing as pnd
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from PIL import Image, ImageDraw

from layout_prompter.models import (
    LayoutSerializedData,
    LayoutSerializedOutputData,
    ProcessedLayoutData,
)
from layout_prompter.settings import CanvasSize
from layout_prompter.typehints import PilImage
from layout_prompter.utils import Configuration, generate_color_palette


class VisualizerConfig(Configuration):
    """Base Configuration for Visualizer."""

    resize_ratio: float = 1.0
    bg_rgb_color: Tuple[int, int, int] = (255, 255, 255)


class ContentAgnosticVisualizerConfig(VisualizerConfig):
    """Configuration for ContentAgnosticVisualizer."""


class ContentAwareVisualizerConfig(VisualizerConfig):
    """Configuration for ContentAwareVisualizer."""

    bg_image: PilImage
    content_bboxes: Optional[pnd.NpNDArray] = None


@dataclass
class Visualizer(Runnable):
    canvas_size: CanvasSize
    labels: List[str]
    schema: Optional[Type[LayoutSerializedOutputData]] = None

    def _convert_to_serialized_output_data(
        self, processed_data: ProcessedLayoutData
    ) -> LayoutSerializedOutputData:
        assert (
            processed_data.labels is not None
            and processed_data.discrete_bboxes is not None
        )
        assert self.schema is not None, "Schema must be defined for serialization."

        serialized_output_data = {
            "layouts": [
                {
                    "class_name": class_name,
                    "coord": {
                        "left": bbox[0],
                        "top": bbox[1],
                        "width": bbox[2],
                        "height": bbox[3],
                    },
                }
                for class_name, bbox in zip(
                    processed_data.labels, processed_data.discrete_bboxes
                )
            ]
        }
        return self.schema(**serialized_output_data)

    def get_sorted_layouts(
        self, layouts: List[LayoutSerializedData]
    ) -> List[LayoutSerializedData]:
        """Sort layouts by area in descending order."""
        # Calculate the area of each layout
        areas = [layout.coord.width * layout.coord.height for layout in layouts]

        # Get the indices of the areas sorted in descending order
        indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)

        # Sort the layouts by area
        return [layouts[i] for i in indices]

    def draw_layout_bboxes(
        self,
        image: PilImage,
        layout: LayoutSerializedData,
        resize_ratio: float = 1.0,
        opacity: int = 100,
        font_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> PilImage:
        # Generate a color palette
        colors = generate_color_palette(len(self.labels))

        # Create a copy of the image and define a draw object
        image = image.copy()
        draw = ImageDraw.Draw(image, mode="RGBA")

        # Get the color for the layout class
        color = colors[self.labels.index(layout.class_name)]
        c_fill = color + (opacity,)

        # Draw the layout bbox on the canvas
        x1, y1, x2, y2 = (
            layout.coord.left,
            layout.coord.top,
            layout.coord.left + layout.coord.width,
            layout.coord.top + layout.coord.height,
        )
        x1, y1, x2, y2 = (
            int(x1 * resize_ratio),
            int(y1 * resize_ratio),
            int(x2 * resize_ratio),
            int(y2 * resize_ratio),
        )
        draw.rectangle(xy=(x1, y1, x2, y2), fill=c_fill, outline=color)

        # Draw the class name on the canvas
        draw.text(xy=(x1, y1), text=layout.class_name, fill=font_color)

        return image

    @abstractmethod
    def invoke(
        self,
        input: Union[ProcessedLayoutData, LayoutSerializedOutputData],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError


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


class ContentAwareVisualizer(Visualizer):
    name: str = "content-aware-visualizer"

    def draw_content_bboxes(
        self,
        image: PilImage,
        content_bboxes: np.ndarray,
        resize_ratio: float = 1.0,
        font_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> PilImage:
        image = image.copy()
        draw = ImageDraw.Draw(image, mode="RGBA")

        for bbox in content_bboxes:
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h

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
