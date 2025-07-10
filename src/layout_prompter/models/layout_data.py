from functools import cached_property
from typing import List, Optional, Tuple

import pydantic_numpy.typing as pnd
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from layout_prompter.settings import CanvasSize
from layout_prompter.typehints import PilImage
from layout_prompter.utils import base64_to_pil


class Bbox(BaseModel):
    left: float = Field(
        ge=0.0,
        le=1.0,
        description="Left coordinate of the normalized bounding box",
    )
    top: float = Field(
        ge=0.0,
        le=1.0,
        description="Top coordinate of the normalized bounding box",
    )
    width: float = Field(
        ge=0.0,
        le=1.0,
        description="Width of the normalized bounding box",
    )
    height: float = Field(
        ge=0.0,
        le=1.0,
        description="Height of the normalized bounding box",
    )

    @property
    def right(self) -> float:
        """Calculate the right coordinate of the normalized bounding box."""
        return self.left + self.width

    @property
    def bottom(self) -> float:
        """Calculate the bottom coordinate of the normalized bounding box."""
        return self.top + self.height


class LayoutData(BaseModel):
    idx: Optional[int] = Field(
        default=None,
        description="Index of the layout data",
    )

    bboxes: Optional[List[Bbox]] = Field(
        description="List of bounding boxes in normalized coordinates"
    )
    labels: Optional[List[str]] = Field(
        description="List of labels for the bounding boxes",
    )

    canvas_size: CanvasSize

    encoded_image: Optional[str]
    content_bboxes: Optional[List[Bbox]]

    @model_validator(mode="after")
    def validate_bboxes_and_labels(self) -> Self:
        if self.bboxes is not None and self.labels is not None:
            assert len(self.bboxes) == len(self.labels), (
                "The number of bounding boxes must match the number of labels."
            )
        return self

    @cached_property
    def content_image(self) -> PilImage:
        assert self.encoded_image is not None, (
            "Encoded image must be provided to get content image."
        )
        return base64_to_pil(self.encoded_image)

    def is_content_aware(self) -> bool:
        return self.encoded_image is not None or self.content_bboxes is not None


class ProcessedLayoutData(LayoutData):
    gold_bboxes: pnd.Np2DArray

    orig_bboxes: pnd.Np2DArray
    orig_labels: pnd.NpNDArray

    discrete_bboxes: Optional[List[Tuple[int, int, int, int]]]
    discrete_gold_bboxes: Optional[List[Tuple[int, int, int, int]]]
    discrete_content_bboxes: Optional[List[Tuple[int, int, int, int]]]
