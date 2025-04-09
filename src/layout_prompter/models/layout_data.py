from functools import cached_property
from typing import Optional, Union

import pydantic_numpy.typing as pnd
from pydantic import BaseModel, field_validator
from pydantic_numpy import np_array_pydantic_annotated_typing

from layout_prompter.settings import CanvasSize
from layout_prompter.typehints import PilImage
from layout_prompter.utils import base64_to_pil, pil_to_base64


class LayoutData(BaseModel):
    bboxes: pnd.Np2DArray
    labels: pnd.NpNDArray
    canvas_size: CanvasSize

    encoded_image: Optional[str]
    content_bboxes: Optional[pnd.Np2DArray]

    @cached_property
    def content_image(self) -> PilImage:
        assert self.encoded_image is not None
        return base64_to_pil(self.encoded_image)

    def is_content_aware(self) -> bool:
        return self.encoded_image is not None and self.content_bboxes is not None


class ProcessedLayoutData(LayoutData):
    gold_bboxes: pnd.Np2DArray

    orig_bboxes: pnd.Np2DArray
    orig_labels: pnd.NpNDArray

    discrete_bboxes: Optional[pnd.Np2DArrayInt32]
    discrete_gold_bboxes: Optional[pnd.Np2DArrayInt32]
    discrete_content_bboxes: Optional[pnd.Np2DArrayInt32]
