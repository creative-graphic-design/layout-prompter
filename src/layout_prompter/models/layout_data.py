from functools import cached_property
from typing import Optional

import pydantic_numpy.typing as pnd
from pydantic import BaseModel, Field

from layout_prompter.settings import CanvasSize
from layout_prompter.typehints import PilImage
from layout_prompter.utils import base64_to_pil


class LayoutData(BaseModel):
    idx: Optional[int] = Field(
        default=None,
        description="Index of the layout data",
    )

    bboxes: Optional[pnd.Np2DArray]
    labels: Optional[pnd.NpNDArray]
    canvas_size: CanvasSize

    encoded_image: Optional[str]
    content_bboxes: Optional[pnd.Np2DArray]

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

    discrete_bboxes: Optional[pnd.Np2DArrayInt32]
    discrete_gold_bboxes: Optional[pnd.Np2DArrayInt32]
    discrete_content_bboxes: Optional[pnd.Np2DArrayInt32]
