from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type, Union

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from PIL import ImageDraw

from layout_prompter.models import (
    CanvasSize,
    LayoutSerializedData,
    LayoutSerializedOutputData,
    ProcessedLayoutData,
)
from layout_prompter.typehints import PilImage
from layout_prompter.utils import Configuration, generate_color_palette


class VisualizerConfig(Configuration):
    """Base Configuration for Visualizer."""

    resize_ratio: float = 1.0
    bg_rgb_color: Tuple[int, int, int] = (255, 255, 255)


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
                    "bbox": bbox.model_dump(),
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
        return list(
            sorted(
                layouts,
                # calculate area
                key=lambda layout: layout.bbox.width * layout.bbox.height,
                # sort by area in descending order
                reverse=True,
            )
        )

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
        x1, y1, x2, y2 = layout.bbox.to_ltrb()

        # Scale the coordinates based on the resize ratio
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
    ) -> PilImage:
        raise NotImplementedError

    def batch(
        self,
        inputs: Union[List[ProcessedLayoutData], List[LayoutSerializedOutputData]],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> List[PilImage]:
        return super().batch(
            inputs, config, return_exceptions=return_exceptions, **kwargs
        )
